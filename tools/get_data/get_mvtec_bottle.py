"""Recupere la categorie `bottle` de MVTec AD (detection d'anomalies, module 19).

Usage (depuis la racine du projet) :

    python tools/get_data/get_mvtec_bottle.py
    python tools/get_data/get_mvtec_bottle.py --source chemin/vers/bottle   # deja telecharge (site officiel)

Produit data/raw/mvtec_bottle/ avec l'arborescence officielle de MVTec AD :

    train/good/*.png                  209 images normales (les seules utilisees pour apprendre)
    test/good/*.png                    20 images normales de test
    test/<type_de_defaut>/*.png        63 images defectueuses (broken_large, broken_small, contamination)
    ground_truth/<type>/*_mask.png     le masque pixel de chaque defaut (pour l'AUROC pixel)

Le site officiel (mvtec.com) exige un formulaire : on utilise une copie GitHub de
la categorie, figee a un commit, en clone partiel (environ 300 Mo). Licence
MVTec AD : CC BY-NC-SA 4.0 (usage non commercial). Toute autre categorie
(carpet, screw, ...) a la meme arborescence : le code du cours ne change pas.
"""

from __future__ import annotations  # permet d'ecrire les types modernes sans les evaluer a l'execution

import argparse  # lecture des options de la ligne de commande
import shutil  # copie de fichiers
import subprocess  # lance d'autres programmes (ici git) depuis Python
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

REPO_URL = "https://github.com/Manjeet908/C_25.git"  # depot GitHub qui contient une copie de la categorie bottle
COMMIT = "850495af6de6551c59208de26f815212ff1e6824"  # version figee : tout le monde recupere exactement les memes images
REMOTE_DIR = "dataset/bottle"  # dossier du depot qui contient la categorie


def download(target: Path) -> Path:  # telecharge le dossier bottle du depot ; renvoie son chemin
    """Clone partiel du depot au commit fige ; renvoie le dossier bottle."""
    if (target / REMOTE_DIR).is_dir():  # si le dossier est deja la
        print(f"Deja telecharge : {target / REMOTE_DIR}")  # on le dit
        return target / REMOTE_DIR  # et on ne retelecharge pas
    target.mkdir(parents=True, exist_ok=True)  # cree le dossier de telechargement
    run = lambda *args: subprocess.run(["git", *args], cwd=target, check=True)  # noqa: E731 - lance "git <args>" ; erreur si git echoue
    run("init", "-q")  # cree un depot git vide
    run("remote", "add", "origin", REPO_URL)  # lui indique l'adresse du depot distant
    run("sparse-checkout", "set", "--no-cone", f"/{REMOTE_DIR}/")  # clone partiel : seulement le dossier bottle
    # recupere uniquement le commit fige (--depth 1 : sans l'historique ; blob:none : contenus telecharges a la demande)
    run("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", COMMIT)
    run("checkout", "-q", "FETCH_HEAD")  # ecrit les fichiers de ce commit sur le disque
    return target / REMOTE_DIR  # renvoie le dossier bottle


def copy_tree(source: Path, dest: Path) -> int:  # copie les images et masques en gardant l'arborescence ; renvoie leur nombre
    """Copie train/, test/ et ground_truth/ ; renvoie le nombre d'images copiees."""
    count = 0  # nombre de fichiers copies
    for subset in ("train", "test", "ground_truth"):  # pour chaque partie du jeu (ground_truth = masques des defauts)
        for image in sorted((source / subset).rglob("*.png")):  # pour chaque PNG, sous-dossiers compris
            out = dest / image.relative_to(source)  # meme chemin relatif sous le dossier final
            out.parent.mkdir(parents=True, exist_ok=True)  # cree le dossier si besoin
            shutil.copy2(image, out)  # copie le fichier
            count += 1  # un fichier de plus
    return count  # renvoie le nombre de fichiers copies


def main() -> None:  # telecharge (si besoin) puis copie dans data/raw/mvtec_bottle ; ne renvoie rien
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # aide = premiere ligne de la docstring
    parser.add_argument("--source", type=Path, default=None, help="dossier bottle deja telecharge")  # copie locale existante
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))  # dossier des donnees brutes
    args = parser.parse_args()  # lit les options tapees

    source = args.source or download(args.raw_dir / "_download_mvtec")  # dossier donne, sinon telechargement
    dest = args.raw_dir / "mvtec_bottle"  # dossier final
    count = copy_tree(source, dest)  # copie tout
    print(f"{dest} : {count} fichiers PNG (images + masques)")  # bilan global
    # pour chaque sous-dossier attendu : nombre d'images (verification)
    for subset in ("train/good", "test/good", "test/broken_large", "test/broken_small", "test/contamination"):
        print(f"  {subset:22s} {len(list((dest / subset).glob('*.png')))} images")  # nom aligne sur 22 caracteres, puis le compte


if __name__ == "__main__":  # vrai seulement quand le script est lance directement
    main()  # lance la recuperation
