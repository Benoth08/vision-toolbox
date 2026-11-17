"""Recupere NEU-CLS et le range selon les trois organisations vues dans le cours.

Usage (depuis la racine du projet) :

    python tools/get_data/get_neu_cls.py            # telecharge puis prepare
    python tools/get_data/get_neu_cls.py --source chemin/vers/NEU-CLS-deja-telecharge

Produit dans data/raw :

    neu_cls_split/   cas A : train/ val/ test/ puis un sous-dossier par classe
    neu_cls_flat/    cas B : un sous-dossier par classe, sans decoupage
    neu_cls_csv/     cas C : images/ a plat + labels.csv (filename, label)

Le jeu officiel (Northeastern University) n'a pas de lien direct : on utilise
la copie GitHub d'un reconditionnement Kaggle, figee a un commit precis, en
clone partiel (seul le dossier d'images est telecharge, environ 130 Mo).
"""

from __future__ import annotations  # permet d'ecrire les types modernes sans les evaluer a l'execution

import argparse  # lecture des options de la ligne de commande
import csv  # ecriture du fichier labels.csv
import shutil  # copie de fichiers
import subprocess  # lance d'autres programmes (ici git) depuis Python
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

REPO_URL = "https://github.com/aviralchharia/Surface-Defect-Detection-in-Hot-Rolled-Steel-Strips.git"  # depot GitHub source
COMMIT = "2a1e01db3b04a6eb9502b73e30fc88a6bb242575"  # version figee : tout le monde recupere exactement les memes images
REMOTE_DIR = "NEU Metal Surface Defects Data"  # dossier du depot qui contient les images
# Noms d'origine (Kaggle) -> noms utilises dans le cours (minuscules, sans espace).
CLASS_RENAME = {
    "Crazing": "crazing",  # craquelures
    "Inclusion": "inclusion",  # inclusions (corps etrangers)
    "Patches": "patches",  # plaques
    "Pitted": "pitted",  # piqures
    "Rolled": "rolled",  # calamine incrustee au laminage
    "Scratches": "scratches",  # rayures
}
SUBSET_RENAME = {"train": "train", "valid": "val", "test": "test"}  # noms des sous-ensembles : d'origine -> cours


def download(target: Path) -> Path:  # telecharge le dossier d'images du depot ; renvoie son chemin
    """Clone partiel du depot au commit fige ; renvoie le dossier des images."""
    if (target / REMOTE_DIR).is_dir():  # si le dossier est deja la
        print(f"Deja telecharge : {target / REMOTE_DIR}")  # on le dit
        return target / REMOTE_DIR  # et on ne retelecharge pas
    target.mkdir(parents=True, exist_ok=True)  # cree le dossier de telechargement
    run = lambda *args: subprocess.run(["git", *args], cwd=target, check=True)  # noqa: E731 - lance "git <args>" ; erreur si git echoue
    run("init", "-q")  # cree un depot git vide
    run("remote", "add", "origin", REPO_URL)  # lui indique l'adresse du depot distant
    run("sparse-checkout", "set", "--no-cone", f"/{REMOTE_DIR}/")  # clone partiel : seulement le dossier d'images
    # recupere uniquement le commit fige (--depth 1 : sans l'historique ; blob:none : contenus telecharges a la demande)
    run("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", COMMIT)
    run("checkout", "-q", "FETCH_HEAD")  # ecrit les fichiers de ce commit sur le disque
    return target / REMOTE_DIR  # renvoie le dossier des images


def layout_a(source: Path, dest: Path) -> None:  # cas A : dossiers train/val/test puis classe ; ne renvoie rien
    """Cas A : on garde le decoupage fourni (train / val / test)."""
    for subset_src, subset_dst in SUBSET_RENAME.items():  # pour chaque sous-ensemble (nom d'origine, nom du cours)
        for class_src, class_dst in CLASS_RENAME.items():  # pour chaque classe (nom d'origine, nom du cours)
            out = dest / subset_dst / class_dst  # dossier de destination, ex. neu_cls_split/train/crazing
            out.mkdir(parents=True, exist_ok=True)  # le cree s'il n'existe pas
            for image in sorted((source / subset_src / class_src).glob("*.bmp")):  # pour chaque image .bmp de ce dossier
                shutil.copy2(image, out / image.name)  # copie l'image


def layout_b(split_root: Path, dest: Path) -> None:  # cas B : un dossier par classe, sans decoupage ; ne renvoie rien
    """Cas B : toutes les images d'une classe dans un seul dossier."""
    for subset in SUBSET_RENAME.values():  # pour chaque sous-ensemble du cas A (train, val, test)
        for class_dir in sorted((split_root / subset).iterdir()):  # pour chaque dossier de classe
            out = dest / class_dir.name  # dossier de la classe, sans sous-ensemble
            out.mkdir(parents=True, exist_ok=True)  # le cree s'il n'existe pas
            for image in sorted(class_dir.glob("*.bmp")):  # pour chaque image de la classe
                shutil.copy2(image, out / image.name)  # copie l'image (train, val et test sont melanges)


def layout_c(flat_root: Path, dest: Path) -> None:  # cas C : images a plat + labels.csv ; ne renvoie rien
    """Cas C : un dossier d'images a plat et un CSV (filename, label)."""
    images_dir = dest / "images"  # toutes les images dans un seul dossier
    images_dir.mkdir(parents=True, exist_ok=True)  # le cree s'il n'existe pas
    rows = []  # lignes du CSV : nom de fichier et classe
    for class_dir in sorted(flat_root.iterdir()):  # pour chaque dossier de classe du cas B
        for image in sorted(class_dir.glob("*.bmp")):  # pour chaque image de la classe
            shutil.copy2(image, images_dir / image.name)  # copie l'image dans le dossier unique
            rows.append({"filename": image.name, "label": class_dir.name})  # note sa classe (le nom du dossier)
    with open(dest / "labels.csv", "w", newline="", encoding="utf-8") as handle:  # ouvre le CSV en ecriture, ferme a la fin du bloc
        writer = csv.DictWriter(handle, fieldnames=["filename", "label"])  # ecrivain de CSV a partir de dictionnaires
        writer.writeheader()  # premiere ligne : les noms de colonnes
        writer.writerows(rows)  # une ligne par image


def main() -> None:  # telecharge (si besoin) puis produit les trois organisations ; ne renvoie rien
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # aide = premiere ligne de la docstring
    parser.add_argument("--source", type=Path, default=None, help="dossier NEU-CLS deja telecharge")  # copie locale existante
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))  # dossier des donnees brutes
    args = parser.parse_args()  # lit les options tapees

    source = args.source or download(args.raw_dir / "_download_neu_cls")  # dossier donne, sinon telechargement
    split_root, flat_root, csv_root = (  # les trois dossiers de sortie
        args.raw_dir / "neu_cls_split",  # cas A
        args.raw_dir / "neu_cls_flat",  # cas B
        args.raw_dir / "neu_cls_csv",  # cas C
    )
    layout_a(source, split_root)  # cas A a partir du telechargement
    layout_b(split_root, flat_root)  # cas B a partir du cas A
    layout_c(flat_root, csv_root)  # cas C a partir du cas B
    for root in (split_root, flat_root, csv_root):  # pour chaque organisation produite
        print(f"{root} : {sum(1 for _ in root.rglob('*.bmp'))} images")  # compte ses images (verification)


if __name__ == "__main__":  # vrai seulement quand le script est lance directement
    main()  # lance la recuperation
