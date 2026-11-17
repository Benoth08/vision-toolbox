"""Recupere NEU-DET (detection : images + annotations Pascal VOC) pour le module 14.

Usage (depuis la racine du projet) :

    python tools/get_data/get_neu_det.py

Produit data/raw/neu_det/IMAGES/*.jpg et data/raw/neu_det/ANNOTATIONS/*.xml.
Source officielle sans lien direct : on utilise une copie GitHub figee a un
commit, en clone partiel (seuls les dossiers utiles sont telecharges).
"""

from __future__ import annotations  # permet d'ecrire les types modernes sans les evaluer a l'execution

import subprocess  # lance d'autres programmes (ici git) depuis Python
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

REPO_URL = "https://github.com/siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection.git"  # depot GitHub source
COMMIT = "351bee21b56bbf97a1e1d18262bceb202eb7ef41"  # version figee : tout le monde recupere exactement les memes fichiers
SUBDIRS = ["IMAGES", "ANNOTATIONS"]  # dossiers utiles : images et boites annotees (XML au format Pascal VOC)


def download(target: Path) -> None:  # telecharge les deux dossiers dans target ; ne renvoie rien
    if all((target / sub).is_dir() for sub in SUBDIRS):  # si les deux dossiers sont deja la
        print(f"Deja telecharge : {target}")  # on le dit
        return  # et on ne retelecharge pas
    target.mkdir(parents=True, exist_ok=True)  # cree le dossier de destination
    run = lambda *args: subprocess.run(["git", *args], cwd=target, check=True)  # noqa: E731 - lance "git <args>" ; erreur si git echoue
    if not (target / ".git").is_dir():  # si le dossier n'est pas encore un depot git (premier essai)
        run("init", "-q")  # cree un depot git vide
        run("remote", "add", "origin", REPO_URL)  # lui indique l'adresse du depot distant
    run("sparse-checkout", "set", "--no-cone", *[f"/{sub}/" for sub in SUBDIRS])  # clone partiel : seulement ces deux dossiers
    # recupere uniquement le commit fige (--depth 1 : sans l'historique ; blob:none : contenus telecharges a la demande)
    run("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", COMMIT)
    run("checkout", "-q", "FETCH_HEAD")  # ecrit les fichiers de ce commit sur le disque
    print(f"{target} : {sum(1 for _ in (target / 'IMAGES').glob('*.jpg'))} images")  # bilan : nombre d'images


if __name__ == "__main__":  # vrai seulement quand le script est lance directement
    download(Path("data/raw/neu_det"))  # telecharge dans data/raw/neu_det (chemin relatif : lancer depuis la racine du projet)
