"""Recupere le jeu Magnetic Tile Defect (segmentation, module 21).

Usage (depuis la racine du projet) :

    python tools/get_data/get_magnetic_tile.py
    python tools/get_data/get_magnetic_tile.py --source chemin/vers/depot-deja-clone

Produit data/raw/magnetic_tile/ :

    MT_Blowhole/Imgs/   115 images + 115 masques   (trou)
    MT_Break/Imgs/       85 + 85                    (cassure)
    MT_Crack/Imgs/       57 + 57                    (fissure)
    MT_Fray/Imgs/        32 + 32                    (effilochage)
    MT_Uneven/Imgs/     103 + 103                   (surface irreguliere)
    MT_Free/Imgs/       952 + 952                   (sans defaut ; masque vide)

Chaque image `x.jpg` (niveaux de gris, tailles variables autour de 250 x 300) a
son masque `x.png` : blanc (255) sur le defaut, noir ailleurs. Source : Huang
et al., "Surface defect saliency of magnetic tile" (2018), depot GitHub de
l'auteur, fige a un commit (environ 110 Mo).
"""

from __future__ import annotations  # permet d'ecrire les types modernes sans les evaluer a l'execution

import argparse  # lecture des options de la ligne de commande
import shutil  # copie de fichiers
import subprocess  # lance d'autres programmes (ici git) depuis Python
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

REPO_URL = "https://github.com/abin24/Magnetic-tile-defect-datasets..git"  # depot GitHub (son nom finit lui-meme par un point)
COMMIT = "cc4eb530b57b9cb7225bf76023873043ffd50655"  # version figee du depot : tout le monde recupere exactement les memes fichiers
CLASSES = ("MT_Blowhole", "MT_Break", "MT_Crack", "MT_Fray", "MT_Uneven", "MT_Free")  # les 6 dossiers de classes a recuperer


def download(target: Path) -> Path:  # telecharge les dossiers utiles du depot ; renvoie le dossier
    """Clone partiel (les 6 dossiers de classes) au commit fige ; renvoie le dossier."""
    if all((target / name / "Imgs").is_dir() for name in CLASSES):  # si les 6 dossiers sont deja la
        print(f"Deja telecharge : {target}")  # on le dit
        return target  # et on ne retelecharge pas
    target.mkdir(parents=True, exist_ok=True)  # cree le dossier de telechargement
    run = lambda *args: subprocess.run(["git", *args], cwd=target, check=True)  # noqa: E731 - lance "git <args>" ; erreur si git echoue
    run("init", "-q")  # cree un depot git vide dans le dossier
    run("remote", "add", "origin", REPO_URL)  # lui indique l'adresse du depot distant
    run("sparse-checkout", "set", "--no-cone", *[f"/{name}/" for name in CLASSES])  # clone partiel : seulement ces 6 dossiers
    # recupere uniquement le commit fige (--depth 1 : sans l'historique ; blob:none : contenus telecharges a la demande)
    run("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", COMMIT)
    run("checkout", "-q", "FETCH_HEAD")  # ecrit les fichiers de ce commit sur le disque
    return target  # renvoie le dossier telecharge


def main() -> None:  # telecharge (si besoin) puis copie les images et masques dans data/raw/magnetic_tile ; ne renvoie rien
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # aide = premiere ligne de la docstring
    parser.add_argument("--source", type=Path, default=None, help="depot deja clone")  # copie locale existante (evite le telechargement)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))  # dossier des donnees brutes
    args = parser.parse_args()  # lit les options tapees

    source = args.source or download(args.raw_dir / "_download_magnetic_tile")  # depot donne, sinon telechargement
    dest = args.raw_dir / "magnetic_tile"  # dossier final
    for name in CLASSES:  # pour chaque classe
        out = dest / name / "Imgs"  # dossier de destination de la classe
        out.mkdir(parents=True, exist_ok=True)  # le cree s'il n'existe pas
        files = sorted((source / name / "Imgs").iterdir())  # tous les fichiers du dossier source, tries par nom
        for path in files:  # pour chaque fichier
            if path.suffix.lower() in {".jpg", ".png"}:  # si c'est une image (.jpg) ou un masque (.png)
                shutil.copy2(path, out / path.name)  # on le copie
        n_images = len(list(out.glob("*.jpg")))  # nombre d'images copiees
        print(f"  {name:12s} {n_images:4d} images, {len(list(out.glob('*.png')))} masques")  # bilan de la classe
    print(f"{dest} pret")  # fin


if __name__ == "__main__":  # vrai seulement quand le script est lance directement
    main()  # lance la recuperation
