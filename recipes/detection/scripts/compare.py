"""Tableau comparatif des detecteurs deja evalues : moteur, temps d'entrainement, mAP, point de fonctionnement, latence ONNX.

    python recipes/detection/scripts/compare.py --outputs outputs/detection --markdown docs/assets/detection/comparison.md

Les experiences ne sont comparables que si elles partagent le meme split.csv (meme test).
"""

import argparse  # argparse : lit les options passees en ligne de commande (--outputs, --markdown)
import json  # json : relit les fichiers de resultats .json ecrits par les autres scripts
import logging  # logging : affiche des messages de suivi (mieux que print dans un script)
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import pandas as pd  # pandas : tableaux de donnees (le tableau comparatif des detecteurs)
import yaml  # yaml : relit la configuration utilisee (config_used.yaml) pour connaitre le decoupage

from vtb.core.runtime import setup_logging  # configure l'affichage des messages de suivi sur la console

logger = logging.getLogger(__name__)  # "journal" propre a ce fichier : logger.info(...) ecrit un message de suivi


def collect(root: Path) -> pd.DataFrame:  # rassemble les resultats de chaque experience ; renvoie un tableau, une ligne par experience
    rows = []  # une ligne (dict) par experience trouvee
    for folder in sorted(  # pour chaque sous-dossier d'experience, par ordre alphabetique
        p  # l'element garde : le chemin du sous-dossier
        for p in root.iterdir()  # parcourt tout ce que contient le dossier racine
        if (p / "test_metrics.json").exists() and p.name != "notebook"  # seulement les experiences evaluees, sauf "notebook"
    ):  # le notebook refait une experience : on ne le compte pas dans le tableau
        test = json.loads((folder / "test_metrics.json").read_text())  # mesures sur le jeu de test (ecrites par evaluate.py)
        # infos du detecteur (ecrites par train.py : moteur, taille d'entree, duree), ou dict vide si le fichier manque
        info = json.loads((folder / "detector.json").read_text()) if (folder / "detector.json").exists() else {}
        # carte du modele (ecrite par export.py : nom, latences...), ou dict vide si le modele n'a pas ete exporte
        card = json.loads((folder / "model_card.json").read_text()) if (folder / "model_card.json").exists() else {}
        # configuration utilisee (copiee au lancement), ou dict vide ; safe_load lit le YAML sans executer de code
        used = yaml.safe_load((folder / "config_used.yaml").read_text()) if (folder / "config_used.yaml").exists() else {}
        # point de fonctionnement = precision et rappel au seuil de confiance utilise en production, toutes classes confondues
        op = test.get("operating_point", {}).get("overall", {})
        rows.append(  # ajoute la ligne de cette experience
            {  # une ligne = un dict {nom de colonne: valeur}
                "experience": folder.name,  # nom du dossier de l'experience
                "moteur": info.get("backend", ""),  # bibliotheque de detection (ultralytics, yolox...) ; "" si absent
                "modele": card.get("model_name", ""),  # nom du modele exporte
                "split": used.get("split", {}).get("file", ""),  # fichier de decoupage : a comparer, le test doit etre le meme
                "entree_px": info.get("input_size", ""),  # taille des images donnees au reseau (pixels)
                "temps_s": info.get("train_seconds", ""),  # duree d'entrainement en secondes
                # mAP50 : precision moyenne (aire sous la courbe precision-rappel) avec une boite juste si IoU >= 0,5
                "test_map50": round(test["map50"], 3),
                "test_map50_95": round(test["map50_95"], 3),  # meme mesure moyennee sur des seuils d'IoU de 0,5 a 0,95 (plus severe)
                "precision@conf": op.get("precision", ""),  # part des boites predites qui sont de vrais defauts
                "rappel@conf": op.get("recall", ""),  # part des vrais defauts retrouves
                # temps par image du modele exporte en ONNX (format standard de modele) avec onnxruntime sur CPU
                "onnx_ms": card.get("latency_ms", {}).get("onnxruntime_cpu_ms", ""),
            }
        )
    return pd.DataFrame(rows)  # transforme la liste de lignes en tableau pandas


def main() -> None:  # point d'entree du script : construit le tableau, l'affiche et l'ecrit en Markdown si demande
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # l'aide reprend la 1re ligne de la docstring
    parser.add_argument("--outputs", type=Path, default=Path("outputs/detection"))  # dossier des experiences
    parser.add_argument("--markdown", type=Path, default=None)  # optionnel : fichier .md ou ecrire le tableau
    args = parser.parse_args()  # lit la ligne de commande ; args.outputs et args.markdown contiennent les valeurs
    setup_logging()  # active l'affichage des messages de suivi
    table = collect(args.outputs)  # tableau comparatif de toutes les experiences
    if table.empty:  # si aucune experience evaluee n'a ete trouvee
        logger.warning("aucune experience evaluee sous %s", args.outputs)  # avertit l'utilisateur
        return  # arrete la fonction : rien a afficher ni a ecrire
    logger.info("\n%s", table.to_string(index=False))  # affiche le tableau en texte (sans numeros de ligne)
    if args.markdown:  # si un fichier Markdown a ete demande
        args.markdown.parent.mkdir(parents=True, exist_ok=True)  # cree son dossier s'il n'existe pas
        args.markdown.write_text(table.to_markdown(index=False) + "\n", encoding="utf-8")  # ecrit le tableau au format Markdown


if __name__ == "__main__":  # vrai seulement quand on lance le fichier directement (pas quand on l'importe)
    main()  # lance le script
