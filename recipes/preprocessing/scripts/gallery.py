"""Planches avant / apres de chaque operation du catalogue, sur des images d'exemple.

    python recipes/preprocessing/scripts/gallery.py --config recipes/preprocessing/configs/steel_gray.yaml

Produit une planche JPEG par famille dans <output_dir>/gallery/ (et une copie dans gallery.assets_dir
pour les articles du site) ainsi que catalog.csv, la liste des operations et de leurs parametres.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[str], X | None) partout

import argparse  # argparse : lit les options passees en ligne de commande (--config)
import logging  # logging : affiche des messages de suivi (mieux que print dans un script)
import shutil  # shutil : copie de fichiers (planches vers le dossier de la documentation)
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

from vtb.core.config import load_config, save_config  # lit le YAML (avec verification) et en ecrit une copie
from vtb.core.runtime import setup_logging  # configure le journal (console + fichier)
from vtb.io.catalogs import build_catalog  # liste les images du jeu dans un tableau (le catalogue)
from vtb.io.images import read_image  # lit une image en tableau numpy (couleur ou niveaux de gris)
from vtb.preprocess import list_ops  # liste toutes les operations de pretraitement disponibles
from vtb.preprocess.benchmark import PreprocessingConfig  # le modele de configuration : les cles autorisees dans le YAML
from vtb.preprocess.gallery import save_gallery  # dessine et enregistre les planches avant / apres

logger = logging.getLogger(__name__)  # "journal" propre a ce fichier : logger.info(...) ecrit un message de suivi


def main() -> None:  # point d'entree : ecrit une planche avant / apres par famille d'operations ; ne renvoie rien
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # l'aide reprend la 1re ligne de la docstring
    parser.add_argument("--config", required=True, type=Path)  # fichier YAML de l'experience (obligatoire)
    args = parser.parse_args()  # lit la ligne de commande ; args.config contient le chemin
    config = load_config(args.config, PreprocessingConfig)  # charge le YAML ; une cle inconnue ou mal typee arrete tout ici
    output = config.runtime.output_path  # dossier de sortie (objet Path)
    setup_logging(output / "gallery.log")  # journal sur la console et dans gallery.log (cree le dossier au besoin)
    save_config(config, output / "config_used.yaml")  # copie de la config utilisee

    # Images d'exemple : celles du YAML, sinon les premieres du catalogue (une par classe si possible).
    paths = list(config.gallery.images)  # chemins donnes dans le YAML (liste vide s'il n'y en a pas)
    if not paths:  # si le YAML n'en donne aucun
        catalog = build_catalog(config.data)  # catalogue de toutes les images du jeu
        # 1re image de chaque classe, puis au plus n_images d'entre elles ; tolist() : colonne -> liste Python
        paths = catalog.groupby("label").head(1)["path"].head(config.gallery.n_images).tolist()
    images = [read_image(p, color=config.data.color) for p in paths]  # lit chaque image d'exemple
    logger.info("%d images d'exemple : %s", len(images), [Path(p).name for p in paths])  # nombre et noms des images

    # applique chaque operation aux images et ecrit une planche JPEG par famille ; renvoie la liste des fichiers ecrits
    sheets = save_gallery(images, output / "gallery", config.gallery.params, config.gallery.families, config.gallery.skip)
    table = list_ops()  # tableau de toutes les operations : famille, nom, resume, parametres par defaut
    table.to_csv(output / "gallery" / "catalog.csv", index=False)  # l'enregistre en CSV
    logger.info("%d operations au catalogue, %d planches", len(table), len(sheets))  # resume

    if config.gallery.assets_dir:  # si un dossier de documentation est donne dans le YAML
        assets = Path(config.gallery.assets_dir)  # chemin de ce dossier
        assets.mkdir(parents=True, exist_ok=True)  # le cree s'il n'existe pas
        for sheet in sheets:  # pour chaque planche ecrite
            shutil.copy2(sheet, assets / f"gallery_{sheet.name}")  # la copie avec le prefixe "gallery_"
        logger.info("planches copiees dans %s", assets)  # indique ou


if __name__ == "__main__":  # vrai seulement quand on lance le fichier directement (pas quand on l'importe)
    main()  # lance le script
