"""Trieuse de classification au clavier (fenetre OpenCV) : 1-9 = classe, retour arriere = annuler, s = passer, q = quitter.

    python recipes/annotation/scripts/sort_images.py --images data/a_trier --classes ok,rayure,inclusion \
        --csv outputs/annotation/tri.csv --copy-to data/raw/mon_jeu_trie

Le CSV est enregistre a chaque decision : relancer la meme commande reprend la ou on s'etait
arrete. --copy-to produit directement un dossier "un sous-dossier par classe".
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[str], X | None) partout

import argparse  # argparse : lit les options passees en ligne de commande (--images, --classes, ...)
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

from vtb.annotate.tools.sorter import run_sorter  # la fenetre OpenCV ou l'on range chaque image dans une classe au clavier
from vtb.core.runtime import setup_logging  # configure l'affichage des messages de suivi sur la console
from vtb.io.images import list_images  # liste les images d'un dossier, triees par nom


def main() -> None:  # point d'entree du script : lit les options puis ouvre la trieuse ; ne renvoie rien
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # l'aide reprend la 1re ligne de la docstring
    parser.add_argument("--images", required=True, type=Path)  # dossier des images a trier (obligatoire)
    parser.add_argument("--classes", required=True, help="a,b,c (9 au plus)")  # noms des classes ; touche 1 = 1re classe, etc.
    parser.add_argument("--csv", required=True, type=Path)  # CSV des decisions (path, label), relu pour reprendre plus tard
    parser.add_argument("--copy-to", type=Path, default=None)  # optionnel : copie chaque image dans <dossier>/<classe>/
    args = parser.parse_args()  # lit la ligne de commande ; args.images, args.classes... contiennent les valeurs
    setup_logging()  # active l'affichage des messages de suivi
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]  # "a, b,c" -> ["a", "b", "c"] (espaces et vides retires)
    # ouvre la fenetre et attend les touches jusqu'a q (quitter) ; chaque decision est ecrite aussitot dans le CSV
    run_sorter([str(p) for p in list_images(args.images)], classes, args.csv, args.copy_to)


if __name__ == "__main__":  # vrai seulement quand on lance le fichier directement (pas quand on l'importe)
    main()  # lance le script
