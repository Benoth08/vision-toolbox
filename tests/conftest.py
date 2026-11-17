"""Jeux de donnees synthetiques : les tests tournent sans rien telecharger, en quelques secondes."""

# conftest.py est lu automatiquement par pytest : les "fixtures" definies ici sont disponibles dans
# tous les fichiers de tests. Une fixture est une fonction qui prepare des donnees ; un test la recoit
# simplement en mettant son nom comme parametre (pytest l'appelle pour lui avant le test).

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[int], X | None) partout

from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)
import pytest  # pytest : l'outil qui trouve et lance les tests (fonctions test_...) et fournit les fixtures

from vtb.io.images import write_image  # ecrit un tableau numpy en fichier image (cree les dossiers au besoin)


@pytest.fixture  # declare la fonction suivante comme fixture : pytest l'appelle pour chaque test qui la demande
def rng() -> np.random.Generator:  # fournit un generateur de nombres aleatoires et le renvoie
    return np.random.default_rng(0)  # graine fixe 0 : memes "hasards" a chaque lancement, donc tests reproductibles


@pytest.fixture  # fixture : fabrique un jeu de donnees "un dossier par classe" pour les tests de classification
def class_folder(tmp_path: Path, rng: np.random.Generator) -> Path:  # tmp_path : dossier temporaire neuf fourni par pytest
    """Rangement en dossiers de classes : 3 classes x 12 images grises 48 x 48, nommees <piece>_<vue>.png."""
    root = tmp_path / "classes"  # dossier racine du jeu de donnees, dans le dossier temporaire du test
    for k, name in enumerate(("alpha", "beta", "gamma")):  # pour chaque classe (k = son numero 0, 1, 2)
        for i in range(12):  # 12 images par classe
            # bruit gaussien centre sur 80, 140 puis 200 selon la classe : les classes se distinguent par leur luminosite ;
            # clip borne a 0-255 et astype(uint8) donne une vraie image 8 bits
            image = np.clip(rng.normal(80 + 60 * k, 20, (48, 48)), 0, 255).astype(np.uint8)
            # nom <piece>_<vue> : i // 3 = numero de piece, i % 3 = numero de vue ; sert a tester le decoupage par groupe
            write_image(root / name / f"P{k}{i // 3:02d}_v{i % 3}.png", image)  # 4 pieces x 3 vues par classe
    return root  # renvoie le dossier racine au test


@pytest.fixture  # fixture : fabrique un mini jeu de donnees d'anomalies au format MVTec AD
def mvtec_folder(tmp_path: Path, rng: np.random.Generator) -> Path:  # renvoie le dossier racine de l'arborescence MVTec
    """Arborescence MVTec AD minimale : 10 normales en train, 3 normales + 3 defauts en test, avec masques.

    Les normales ont chacune leur grain (comme de vraies images) : des images identiques donneraient
    des distances nulles, un cas degenere qui ne teste rien.
    """
    root = tmp_path / "mvtec"  # dossier racine dans le dossier temporaire du test

    def normal() -> np.ndarray:  # fabrique une image "normale" (sans defaut) couleur 32 x 32 et la renvoie
        return np.clip(rng.normal(120, 12, (32, 32, 3)), 0, 255).astype(np.uint8)  # gris moyen 120 avec un leger grain aleatoire

    for i in range(10):  # 10 images normales pour l'entrainement (MVTec n'a que des normales en train)
        write_image(root / "train" / "good" / f"{i:03d}.png", normal())  # train/good/000.png, 001.png...
    for i in range(3):  # 3 normales + 3 defauts pour le test
        write_image(root / "test" / "good" / f"{i:03d}.png", normal())  # image de test sans defaut
        defect = normal()  # on part d'une image normale...
        defect[10:16, 10:16] = 250  # ...et on y peint un carre 6 x 6 presque blanc : c'est le "defaut"
        write_image(root / "test" / "crack" / f"{i:03d}.png", defect)  # rangee dans la categorie de defaut "crack"
        mask = np.zeros((32, 32), np.uint8)  # masque = image noire de meme taille qui dira OU est le defaut
        mask[10:16, 10:16] = 255  # blanc exactement la ou le carre a ete peint (verite terrain pixel par pixel)
        write_image(root / "ground_truth" / "crack" / f"{i:03d}_mask.png", mask)  # nom attendu par MVTec : <image>_mask.png
    return root  # renvoie le dossier racine au test


# Modele de fichier XML Pascal VOC (format d'annotation de detection) : image 200 x 100, deux objets.
# La boite "hole" va jusqu'a xmax=205 > 200 : elle deborde de l'image, pour verifier qu'elle est bien recadree.
VOC_TEMPLATE = """<annotation><filename>{name}</filename><size><width>200</width><height>100</height><depth>3</depth></size>
<object><name>crack</name><bndbox><xmin>10</xmin><ymin>20</ymin><xmax>50</xmax><ymax>60</ymax></bndbox></object>
<object><name>hole</name><bndbox><xmin>150</xmin><ymin>0</ymin><xmax>205</xmax><ymax>40</ymax></bndbox></object>
</annotation>"""


@pytest.fixture  # fixture : fabrique un petit jeu de detection (images + annotations VOC)
def voc_folder(tmp_path: Path) -> Path:  # renvoie le dossier racine qui contient images/ et annotations/
    """Detection au format VOC : 4 images 200 x 100, deux objets chacune (une boite deborde de l'image)."""
    root = tmp_path / "voc"  # dossier racine dans le dossier temporaire du test
    for i in range(4):  # 4 images
        name = f"img_{i}.jpg"  # nom du fichier image, repris dans le XML
        write_image(root / "images" / name, np.zeros((100, 200, 3), np.uint8))  # image noire hauteur 100 x largeur 200
        (root / "annotations").mkdir(parents=True, exist_ok=True)  # cree le dossier des XML s'il n'existe pas encore
        # ecrit le XML de l'image : le modele avec {name} remplace par le nom du fichier image
        (root / "annotations" / f"img_{i}.xml").write_text(VOC_TEMPLATE.format(name=name), encoding="utf-8")
    return root  # renvoie le dossier racine au test


@pytest.fixture  # fixture : fournit une image 16 bits pour les tests de pretraitement
def image16() -> np.ndarray:  # renvoie un tableau 64 x 64 d'entiers 16 bits
    """Image 16 bits (thermique, radiographie) qui n'occupe qu'une partie de la dynamique."""
    yy, xx = np.mgrid[0:64, 0:64]  # deux grilles 64 x 64 : numero de ligne (yy) et de colonne (xx) de chaque pixel
    # degrade de 20000 a environ 25000 sur 0-65535 possibles : seule une petite partie de la plage 16 bits est utilisee
    return (20000 + 50 * xx + 30 * yy).astype(np.uint16)
