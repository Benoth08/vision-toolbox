"""Base de comparaison sans apprentissage (carte de reponse ou piece de reference), reglee sur la validation.

    python recipes/classical/scripts/baseline.py --config recipes/classical/configs/mvtec_bottle_reference.yaml
    python recipes/classical/scripts/baseline.py --config recipes/classical/configs/magnetic_tile.yaml

Produit split.csv, grid_val.csv (tous les reglages essayes sur val), metrics.json (test, lu une
seule fois) et examples.jpg (verite en vert, prediction en rouge) dans output_dir.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[str], X | None) partout

import argparse  # argparse : lit les options passees en ligne de commande (--config)
import json  # json : ecrit les mesures dans metrics.json
import logging  # logging : affiche des messages de suivi (mieux que print dans un script)
import shutil  # shutil : copie de fichiers (vers le dossier de la documentation)
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)

from vtb.classical.baseline import (  # briques de la methode classique (sans apprentissage) de la toolbox
    ClassicalConfig,  # le modele de configuration : la liste des cles autorisees dans le YAML
    build_reference,  # construit une "piece de reference" : mediane pixel a pixel d'images normales
    choose_parameters,  # essaie tous les couples (seuil, aire minimale) et garde le meilleur
    evaluate,  # mesure IoU, Dice, rappel et fausses alarmes pour un reglage donne
    normal_rows,  # garde seulement les lignes d'images sans defaut
    response_maps,  # calcule la carte de reponse de chaque image (valeur haute = suspect) et son masque de verite
    segment,  # seuil + tri des taches trop petites : carte de reponse -> masque predit
    split_for_baseline,  # decoupe le jeu en train / val / test
)
from vtb.core.config import load_config, save_config  # lit le YAML (avec verification) et en ecrit une copie
from vtb.core.runtime import seed_everything, setup_logging  # graines du hasard fixees ; journal console + fichier
from vtb.io.catalogs import build_catalog  # liste les images du jeu dans un tableau (le catalogue)
from vtb.io.images import to_uint8, write_image  # conversion en 8 bits (0-255) ; ecriture d'une image sur disque
from vtb.preprocess import Pipeline  # enchaine les operations de pretraitement decrites dans le YAML
from vtb.preprocess.ops.geometry import resize  # redimensionne une image
from vtb.split import check_no_leakage, save_split  # verifie qu'aucune image n'est dans deux parties ; enregistre split.csv

logger = logging.getLogger(__name__)  # "journal" propre a ce fichier : logger.info(...) ecrit un message de suivi


def overlay(response: np.ndarray, predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:  # renvoie une image RGB a regarder
    """Carte de reponse en gris, verite terrain en vert, prediction en rouge (jaune = recouvrement)."""
    # carte [0, 1] -> gris 0-255 copie sur 3 canaux (R, V, B), en flottants, assombrie (x 0,7) pour laisser voir les couleurs
    rgb = np.stack([to_uint8((response * 255).astype(np.uint8))] * 3, axis=-1).astype(np.float32) * 0.7
    rgb[..., 1] += 110 * truth  # canal vert (indice 1) renforce la ou est le vrai defaut
    rgb[..., 0] += 110 * predicted  # canal rouge (indice 0) renforce la ou le defaut est predit (vert + rouge = jaune)
    return np.clip(rgb, 0, 255).astype(np.uint8)  # borne a 0-255 puis repasse en entiers 8 bits (format d'image standard)


def main() -> None:  # point d'entree : decoupe, regle sur val, mesure sur test, ecrit les resultats ; ne renvoie rien
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # l'aide reprend la 1re ligne de la docstring
    parser.add_argument("--config", required=True, type=Path)  # fichier YAML de l'experience (obligatoire)
    args = parser.parse_args()  # lit la ligne de commande ; args.config contient le chemin
    config = load_config(args.config, ClassicalConfig)  # charge le YAML ; une cle inconnue ou mal typee arrete tout ici
    output = config.runtime.output_path  # dossier de sortie (objet Path)
    setup_logging(output / "baseline.log")  # journal sur la console et dans baseline.log (cree le dossier au besoin)
    save_config(config, output / "config_used.yaml")  # copie de la config utilisee, pour savoir plus tard ce qui a ete lance
    seed_everything(config.runtime.seed)  # fixe les graines du hasard : memes resultats a chaque lancement
    pipeline = Pipeline(config.preprocess)  # pretraitement decrit dans le YAML (le meme pour toutes les images)

    # catalogue des images puis decoupage train / val / test
    split = split_for_baseline(build_catalog(config.data), config, config.runtime.seed)
    check_no_leakage(split)  # erreur si une meme image (ou piece) se retrouve dans deux parties : fuite de donnees
    save_split(split, config.split.file)  # enregistre le decoupage (split.csv) pour pouvoir le relire
    reference = None  # pas de piece de reference par defaut (methode "response")
    if config.method == "reference":  # la reference ne voit que des images normales de train
        pool = normal_rows(split[split["split"] == "train"]).head(config.reference.n_images)  # les N premieres normales de train
        # mediane (ou moyenne) pixel a pixel de ces images pretraitees : la piece "parfaite" a laquelle on compare
        reference = build_reference(pool["path"].tolist(), pipeline, config.data.color, config.reference.statistic)
        write_image(output / "reference.png", reference)  # enregistre la reference pour pouvoir la regarder
        logger.info("reference : %s de %d images normales de train", config.reference.statistic, len(pool))  # resume

    # cartes de reponse des images de val, puis essai de tous les reglages : best = meilleur reglage, grid = tous les essais
    best, grid = choose_parameters(*response_maps(split[split["split"] == "val"], pipeline, config, reference), config)
    grid.to_csv(output / "grid_val.csv", index=False)  # garde la trace de tous les reglages essayes
    # affiche le reglage retenu : critere de choix, seuil sur la carte, aire minimale d'une tache (pixels)
    logger.info("reglage retenu sur val (critere %s) : seuil %.3f, aire min %.0f", config.select, best["threshold"], best["min_area"])

    test = split[split["split"] == "test"]  # les images de test (regardees une seule fois, avec le reglage deja fige)
    responses, masks = response_maps(test, pipeline, config, reference)  # cartes de reponse et masques de verite du test
    metrics = {  # mesures finales sur le test
        "method": config.method,  # methode utilisee ("response" ou "reference")
        "n_test": len(test),  # nombre d'images de test
        # ** deplie le dict renvoye : IoU et Dice (recouvrement des masques), rappel et fausses alarmes par image
        **evaluate(responses, masks, best["threshold"], best["min_area"], config.invert),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")  # ecrit metrics.json
    logger.info(  # affiche les mesures de test ; chaque %... est remplace par la valeur correspondante
        "test (%d images) : IoU %.3f | Dice %.3f | rappel image %.3f | fausses alarmes %.3f",  # modele du message
        len(test),  # nombre d'images
        metrics["iou"],  # IoU : surface commune / surface totale entre masque predit et vrai (1 = parfait)
        metrics["dice"],  # Dice : 2 x surface commune / somme des surfaces (proche de l'IoU, un peu plus indulgent)
        metrics["recall"],  # part des images defectueuses detectees
        metrics["false_alarm_rate"],  # part des images normales signalees a tort
    )

    # images a montrer : les 4 premieres avec defaut, puis les 2 premieres sans defaut
    picks = [i for i, m in enumerate(masks) if m.any()][:4] + [i for i, m in enumerate(masks) if not m.any()][:2]
    # pour chaque image choisie : superposition carte (gris) + verite (vert) + prediction (rouge)
    tiles = [overlay(responses[i], segment(responses[i], best["threshold"], best["min_area"], config.invert), masks[i]) for i in picks]
    height = min(t.shape[0] for t in tiles)  # hauteur commune : la plus petite des vignettes
    write_image(  # ecrit la planche d'exemples
        output / "examples.jpg",  # fichier de sortie
        # met toutes les vignettes a la meme hauteur (largeur proportionnelle) puis les colle cote a cote (axis=1)
        np.concatenate([resize(t, [height, round(t.shape[1] * height / t.shape[0])]) for t in tiles], axis=1),
    )
    if config.assets_dir:  # si un dossier de documentation est donne dans la config
        Path(config.assets_dir).mkdir(parents=True, exist_ok=True)  # le cree s'il n'existe pas
        for name in ("examples.jpg", "metrics.json", "baseline.log"):  # pour chacun des trois fichiers a publier
            # copie en prefixant par le nom de l'experience (copy2 garde aussi la date du fichier)
            shutil.copy2(output / name, Path(config.assets_dir) / f"{config.runtime.output_path.name}_{name}")


if __name__ == "__main__":  # vrai seulement quand on lance le fichier directement (pas quand on l'importe)
    main()  # lance le script
