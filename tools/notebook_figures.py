"""Extrait les figures d'un notebook execute vers docs/assets/<sujet>/ (pour les articles du site).

    python tools/notebook_figures.py recipes/preprocessing/notebooks/preprocessing.ipynb docs/assets/preprocessing --prefix nb

Chaque image de sortie devient <prefix>_<cellule>_<rang>.<png|jpg> ; la liste des fichiers ecrits
est affichee, a citer ensuite dans l'article avec {{img:...}}.
"""

from __future__ import annotations  # permet d'ecrire les types modernes sans les evaluer a l'execution

import argparse  # lecture des arguments de la ligne de commande
import base64  # les images d'un notebook sont stockees en texte base64 : il faut les decoder en octets
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import nbformat  # lecture et ecriture des fichiers notebook (.ipynb)


def main() -> None:  # extrait les images des sorties du notebook ; ne renvoie rien
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # aide = premiere ligne de la docstring
    parser.add_argument("notebook", type=Path)  # notebook .ipynb deja execute
    parser.add_argument("out_dir", type=Path)  # dossier ou ecrire les images
    parser.add_argument("--prefix", default="nb")  # debut du nom des fichiers
    args = parser.parse_args()  # lit les arguments tapes
    notebook = nbformat.read(args.notebook, as_version=4)  # charge le notebook (format version 4)
    args.out_dir.mkdir(parents=True, exist_ok=True)  # cree le dossier de sortie
    code_index = 0  # numero de la cellule de code courante (les cellules de texte ne comptent pas)
    for cell in notebook.cells:  # pour chaque cellule du notebook
        if cell.cell_type != "code":  # si ce n'est pas une cellule de code (texte markdown)
            continue  # elle n'a pas de sortie : on passe
        code_index += 1  # une cellule de code de plus
        rank = 0  # rang de l'image dans cette cellule (1re, 2e...)
        for output in cell.get("outputs", []):  # pour chaque sortie de la cellule
            data = output.get("data", {})  # contenus de la sortie, par type (texte, image...)
            for mime, suffix in (("image/jpeg", "jpg"), ("image/png", "png")):  # types image cherches, avec leur extension
                if mime in data:  # si la sortie contient une image de ce type
                    rank += 1  # une image de plus dans cette cellule
                    target = args.out_dir / f"{args.prefix}_{code_index:02d}_{rank}.{suffix}"  # nom : prefixe_cellule_rang.ext
                    target.write_bytes(base64.b64decode(data[mime]))  # decode le texte base64 et ecrit les octets de l'image
                    print(target)  # affiche le chemin ecrit (a citer dans l'article)
                    break  # une seule version par sortie (JPEG en priorite, sinon PNG)


if __name__ == "__main__":  # vrai seulement quand le script est lance directement
    main()  # lance l'extraction
