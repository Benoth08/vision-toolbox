"""Annotateur de boites a la souris (fenetre OpenCV) : glisser = boite, 1-9 = classe, x = effacer, n / p = image, q = quitter.

    python recipes/annotation/scripts/annotate_boxes.py --images data/a_annoter --classes rayure,inclusion \
        --out outputs/annotation/boites.vtb.json --yolo-dir outputs/annotation/labels

Reprise automatique : les boites deja enregistrees dans --out sont rechargees.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[str], X | None) partout

import argparse  # argparse : lit les options passees en ligne de commande (--images, --classes, ...)
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

from vtb.annotate.tools.box_annotator import run_box_annotator  # la fenetre OpenCV ou l'on dessine les boites a la souris
from vtb.core.runtime import setup_logging  # configure l'affichage des messages de suivi sur la console
from vtb.io.images import list_images  # liste les images d'un dossier, triees par nom


def main() -> None:  # point d'entree du script : lit les options puis ouvre l'annotateur ; ne renvoie rien
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # l'aide reprend la 1re ligne de la docstring
    parser.add_argument("--images", required=True, type=Path)  # dossier des images a annoter (obligatoire)
    parser.add_argument("--classes", required=True, help="a,b,c (9 au plus au clavier)")  # noms des classes, separes par des virgules
    parser.add_argument("--out", required=True, type=Path, help="fichier vtb.json")  # fichier ou les boites sont enregistrees
    # optionnel : dossier ou ecrire aussi un fichier texte au format YOLO par image (pour entrainer un detecteur YOLO)
    parser.add_argument("--yolo-dir", type=Path, default=None, help="ecrit aussi un .txt YOLO par image")
    # les grandes images sont reduites a l'affichage pour tenir a l'ecran (les boites restent en pixels d'origine)
    parser.add_argument("--max-display", type=int, default=1000, help="taille maximale de la fenetre (pixels)")
    args = parser.parse_args()  # lit la ligne de commande ; args.images, args.classes... contiennent les valeurs
    setup_logging()  # active l'affichage des messages de suivi
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]  # "a, b,c" -> ["a", "b", "c"] (espaces et vides retires)
    # ouvre la fenetre et attend les actions au clavier / a la souris jusqu'a ce que l'on quitte (touche q)
    run_box_annotator([str(p) for p in list_images(args.images)], classes, args.out, args.yolo_dir, args.max_display)


if __name__ == "__main__":  # vrai seulement quand on lance le fichier directement (pas quand on l'importe)
    main()  # lance le script
