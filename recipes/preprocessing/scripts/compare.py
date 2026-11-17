"""Compare des pipelines de pretraitement a la METRIQUE (F1 macro en validation croisee, sonde lineaire).

    python recipes/preprocessing/scripts/compare.py --config recipes/preprocessing/configs/steel_gray.yaml

Pour chaque pipeline du YAML : caracteristiques d'un ResNet18 ImageNet gele, regression
logistique evaluee en validation croisee sur train + val (F1 macro moyen et ecart-type). Le test
n'est pas lu. Produit split.csv, comparison.csv et comparison.md dans output_dir.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[str], X | None) partout

import argparse  # argparse : lit les options passees en ligne de commande (--config)
import logging  # logging : affiche des messages de suivi (mieux que print dans un script)
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

from vtb.core.config import load_config, save_config  # lit le YAML (avec verification) et en ecrit une copie

# choix CPU / GPU ; graines du hasard ; journal console + fichier ; chronometre d'un bloc de code
from vtb.core.runtime import get_device, seed_everything, setup_logging, timer
from vtb.io.catalogs import build_catalog  # liste les images du jeu dans un tableau (le catalogue)

# config de la recette ; comparaison des pipelines ; tirage d'au plus N images par classe
from vtb.preprocess.benchmark import PreprocessingConfig, compare_pipelines, subsample_per_class

# fuite = meme image dans deux parties (test truque) ; make_split decoupe ; save_split enregistre split.csv
from vtb.split import check_no_leakage, make_split, save_split

logger = logging.getLogger(__name__)  # "journal" propre a ce fichier : logger.info(...) ecrit un message de suivi


def main() -> None:  # point d'entree : compare les pretraitements du YAML et ecrit le classement ; ne renvoie rien
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # l'aide reprend la 1re ligne de la docstring
    parser.add_argument("--config", required=True, type=Path)  # fichier YAML de l'experience (obligatoire)
    args = parser.parse_args()  # lit la ligne de commande ; args.config contient le chemin
    config = load_config(args.config, PreprocessingConfig)  # charge le YAML ; une cle inconnue ou mal typee arrete tout ici
    output = config.runtime.output_path  # dossier de sortie (objet Path)
    setup_logging(output / "compare.log")  # journal sur la console et dans compare.log (cree le dossier au besoin)
    save_config(config, output / "config_used.yaml")  # copie de la config utilisee, pour savoir plus tard ce qui a ete lance
    seed_everything(config.runtime.seed)  # fixe les graines du hasard : memes resultats a chaque lancement
    pipelines = config.build_pipelines()  # valide toutes les operations avant le moindre calcul

    # catalogue des images, reduit a au plus max_per_class images par classe (comparaison rapide et equilibree)
    catalog = subsample_per_class(build_catalog(config.data), config.compare.max_per_class, config.runtime.seed)
    split = make_split(catalog, config.split, config.runtime.seed)  # decoupage train / val / test
    check_no_leakage(split)  # erreur si une meme image (ou piece) se retrouve dans deux parties
    save_split(split, config.split.file)  # enregistre le decoupage (split.csv)
    dev = split[split["split"] != "test"].reset_index(drop=True)  # images de developpement = train + val ; le test reste cache
    # validation croisee : dev est coupe en n plis ; chaque pli sert une fois de validation, les autres d'entrainement
    logger.info("comparaison sur %d images (train + val, %d plis) ; test non lu", len(dev), config.compare.n_folds)
    if config.compare.simulate_vignetting > 0:  # si la config demande de simuler un eclairage inegal
        # vignettage = bords de l'image plus sombres que le centre (defaut d'optique ou d'eclairage)
        logger.info("eclairage inegal simule : vignettage aleatoire de force <= %.2f", config.compare.simulate_vignetting)

    with timer("comparaison des pipelines"):  # chronometre le bloc et affiche sa duree a la fin
        # pour chaque pipeline : vecteurs d'un ResNet18 fige, puis regression logistique (classifieur lineaire simple)
        # evaluee en validation croisee ; renvoie un tableau F1 macro moyen et ecart-type par pipeline
        results = compare_pipelines(
            dev,  # images de developpement (train + val)
            pipelines,  # les pretraitements a comparer (nom -> Pipeline)
            config.compare,  # reglages de la comparaison (nombre de plis...)
            config.data.color,  # lecture en "rgb" ou "gray"
            get_device(config.runtime.device),  # CPU ou GPU
            config.runtime.seed,  # graine : memes plis a chaque execution
        )
    results.to_csv(output / "comparison.csv", index=False)  # ecrit le classement en CSV
    # ecrit aussi le classement en tableau Markdown, nombres a 3 decimales
    (output / "comparison.md").write_text(results.to_markdown(index=False, floatfmt=".3f") + "\n", encoding="utf-8")
    logger.info("\n%s", results.to_string(index=False, float_format=lambda v: f"{v:.3f}"))  # affiche le classement


if __name__ == "__main__":  # vrai seulement quand on lance le fichier directement (pas quand on l'importe)
    main()  # lance le script
