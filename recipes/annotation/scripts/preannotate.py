"""Exporte des predictions (vtb.json avec scores) vers un outil d'annotation, et l'ordre de relecture par incertitude.

    python recipes/annotation/scripts/preannotate.py --predictions outputs/yolo/predictions.vtb.json \
        --tool labelstudio --out outputs/annotation/preannotations.json --document-root data/raw

Outils : labelstudio (fichier de taches avec predictions), cvat (XML a charger par Upload
annotations), labelme ou xanylabeling (un JSON par image, a placer a cote des images).
Le CSV --order donne les images de la plus incertaine a la plus sure : corriger dans cet ordre.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[str], X | None) partout

import argparse  # argparse : lit les options passees en ligne de commande (--predictions, --tool, ...)
import logging  # logging : affiche des messages de suivi (mieux que print dans un script)
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import pandas as pd  # pandas : tableaux de donnees (ici pour ecrire le CSV d'ordre de relecture)

# export vers l'outil d'annotation et calcul de l'incertitude de chaque image (pour relire d'abord les cas douteux)
from vtb.annotate.preannotate import detection_uncertainty, export_predictions, order_by_uncertainty
from vtb.annotate.schema import load_annotations  # relit un fichier vtb.json (format pivot de la toolbox)
from vtb.core.runtime import setup_logging  # configure l'affichage des messages de suivi sur la console

logger = logging.getLogger(__name__)  # "journal" propre a ce fichier : logger.info(...) ecrit un message de suivi


def main() -> None:  # point d'entree du script : exporte les pre-annotations et, si demande, l'ordre de relecture
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # l'aide reprend la 1re ligne de la docstring
    # predictions d'un modele deja entraine : chaque objet a une boite, une classe et un score de confiance
    parser.add_argument("--predictions", required=True, type=Path, help="fichier vtb.json de predictions (objets avec score)")
    parser.add_argument("--tool", required=True, choices=["labelstudio", "cvat", "labelme", "xanylabeling"])  # outil de destination
    parser.add_argument("--out", required=True, type=Path)  # fichier ou dossier de sortie (selon l'outil)
    # Label Studio retrouve les images locales par un chemin relatif a ce dossier racine
    parser.add_argument("--document-root", type=Path, default=None, help="Label Studio : dossier racine des fichiers locaux")
    parser.add_argument("--images-dir", type=Path, default=None, help="CVAT : noms d'images relatifs a ce dossier")  # pour CVAT
    # les detections de score plus faible sont ignorees (trop douteuses pour aider l'annotateur)
    parser.add_argument("--min-score", type=float, default=0.25)
    # optionnel : CSV des images de la plus incertaine a la plus sure, pour corriger d'abord les cas douteux
    parser.add_argument("--order", type=Path, default=None, help="CSV d'ordre de relecture (incertitude decroissante)")
    args = parser.parse_args()  # lit la ligne de commande ; args.predictions, args.tool... contiennent les valeurs
    setup_logging()  # active l'affichage des messages de suivi

    predictions, class_names = load_annotations(args.predictions)  # liste d'annotations par image + noms des classes
    # ecrit les predictions dans le format de l'outil choisi ; renvoie le chemin ecrit
    out = export_predictions(predictions, args.tool, args.out, class_names, args.document_root, args.images_dir, args.min_score)
    logger.info("%d images pre-annotees -> %s", len(predictions), out)  # resume : combien d'images, et ou
    if args.order is not None:  # si l'on a demande le CSV d'ordre de relecture
        scores = detection_uncertainty(predictions)  # incertitude de chaque image : score le plus proche de 0,5 = le plus douteux
        order = order_by_uncertainty(predictions)  # indices des images, de la plus incertaine a la plus sure
        # tableau a deux colonnes (chemin, incertitude) dans l'ordre de relecture
        table = pd.DataFrame({"path": [predictions[i].image_path for i in order], "uncertainty": [scores[i] for i in order]})
        args.order.parent.mkdir(parents=True, exist_ok=True)  # cree le dossier du CSV s'il n'existe pas encore
        table.to_csv(args.order, index=False)  # ecrit le CSV (index=False : sans la colonne de numeros de ligne)
        logger.info("ordre de relecture -> %s", args.order)  # indique ou le CSV a ete ecrit


if __name__ == "__main__":  # vrai seulement quand on lance le fichier directement (pas quand on l'importe)
    main()  # lance le script
