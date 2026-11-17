"""Lecture et ecriture d'images avec OpenCV, robustes aux chemins Windows et aux images 16 bits.

Convention de toute la toolbox : une image en memoire est un `numpy.ndarray`
- en niveaux de gris : forme (H, W) ;
- en couleur : forme (H, W, 3), canaux dans l'ordre **RGB** (OpenCV lit en BGR : on convertit ici,
  une seule fois, pour ne plus jamais y penser) ;
- de type uint8 (0-255) ou uint16 (0-65535), conserve tel quel : les images 16 bits (thermique,
  radiographie, certaines cameras industrielles) ne sont jamais ecrasees en 8 bits sans le dire.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (ex. str | Path)

from collections.abc import Sequence  # type "suite d'elements" (liste, tuple...) pour annoter les parametres
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux
from typing import Literal  # Literal : liste fermee de valeurs permises pour un parametre

import cv2  # OpenCV : lecture, ecriture et traitement d'images
import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)

from vtb.core.config import IMAGE_EXTENSIONS  # extensions de fichier reconnues comme images (.png, .jpg...)

ColorMode = Literal["rgb", "gray", "unchanged"]  # les 3 facons de lire une image : couleur, gris, telle que stockee


def read_image(path: str | Path, color: ColorMode = "rgb") -> np.ndarray:  # lit une image et renvoie un tableau numpy
    """Lit une image ; color = rgb (H, W, 3), gray (H, W) ou unchanged (tel que stocke, BGR remis en RGB).

    `cv2.imread` echoue en silence sur les chemins contenant des accents sous Windows : on lit les
    octets avec numpy puis on decode, ce qui marche partout.
    """
    data = np.fromfile(str(path), dtype=np.uint8)  # lit les octets bruts du fichier (accepte les chemins avec accents)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)  # decode les octets en image, sans changer couleurs ni profondeur (8/16 bits)
    if image is None:  # None = le fichier n'est pas une image lisible
        raise ValueError(f"image illisible ou corrompue : {path}")  # on arrete tout de suite avec un message clair
    if image.ndim == 3 and image.shape[2] == 4:  # 4 canaux = couleur + canal alpha (transparence) : on l'ignore
        image = image[:, :, :3]  # garde seulement les 3 premiers canaux (couleur)
    if color == "unchanged":  # si on veut l'image telle que stockee
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image  # remet seulement la couleur en ordre RGB
    if color == "gray":  # si on veut une image en niveaux de gris
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image  # convertit la couleur en gris (deja gris : inchange)
    if image.ndim == 2:  # ici on veut du RGB ; 2 dimensions = image grise sur le disque
        return np.stack([image] * 3, axis=-1)  # gris -> 3 canaux identiques (reseaux pre-entraines en RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # image couleur : OpenCV lit en BGR, on remet en RGB


def write_image(path: str | Path, image: np.ndarray) -> None:  # ecrit une image sur le disque (rien a renvoyer)
    """Ecrit une image RGB ou grise (uint8 ou uint16) ; le format suit l'extension du chemin."""
    path = Path(path)  # convertit en objet chemin
    path.parent.mkdir(parents=True, exist_ok=True)  # cree le dossier de destination s'il n'existe pas
    to_write = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else image  # OpenCV ecrit en BGR : on reconvertit la couleur
    ok, buffer = cv2.imencode(path.suffix, to_write)  # encode l'image en octets au format de l'extension (.png, .jpg...)
    if not ok:  # si l'encodage a echoue (extension inconnue, type non supporte)
        raise ValueError(f"encodage impossible vers {path.suffix} pour {path}")  # erreur claire
    buffer.tofile(str(path))  # ecrit les octets dans le fichier (marche aussi avec les accents sous Windows)


def read_mask(path: str | Path, threshold: int = 127) -> np.ndarray:  # lit un masque et renvoie un tableau de 0 et de 1
    """Masque binaire (H, W) uint8 en {0, 1}. Le seuil a 127 ignore les bords anti-creneles des PNG."""
    mask = read_image(path, color="gray")  # lit le masque en niveaux de gris
    return (mask > threshold).astype(np.uint8)  # pixel au-dessus du seuil -> 1 (defaut), sinon 0 (fond)


# liste les images d'un dossier (ou renvoie le fichier seul) ; renvoie une liste de chemins triee
def list_images(root: str | Path, extensions: Sequence[str] = IMAGE_EXTENSIONS, recursive: bool = True) -> list[Path]:
    """Images d'un dossier, triees (l'ordre du disque differe entre Windows et Linux), extension en minuscules."""
    root = Path(root)  # convertit en objet chemin
    if root.is_file():  # si on a donne un fichier et non un dossier
        return [root]  # la liste contient seulement ce fichier
    allowed = {e.lower() for e in extensions}  # ensemble des extensions acceptees, en minuscules (.PNG = .png)
    pattern = root.rglob("*") if recursive else root.glob("*")  # rglob = tous les sous-dossiers ; glob = ce dossier seulement
    return sorted(p for p in pattern if p.is_file() and p.suffix.lower() in allowed)  # garde les fichiers image, tries par chemin


def max_value(image: np.ndarray) -> float:  # renvoie la plus grande valeur possible d'un pixel selon son type
    """Valeur maximale du type : 255 en 8 bits, 65535 en 16 bits, 1 pour un flottant suppose dans [0, 1]."""
    if image.dtype == np.uint8:  # entier 8 bits (0 a 255)
        return 255.0  # valeur maximale en 8 bits
    if image.dtype == np.uint16:  # entier 16 bits (0 a 65535)
        return 65535.0  # valeur maximale en 16 bits
    return 1.0  # sinon : nombre a virgule, suppose deja entre 0 et 1


# convertit une image en 8 bits (0-255) et renvoie le tableau converti
def to_uint8(image: np.ndarray, low_percentile: float | None = None, high_percentile: float | None = None) -> np.ndarray:
    """Ramene une image en 8 bits pour l'affichage ou une bibliotheque qui l'exige.

    Sans percentiles : division par la valeur maximale du type (aucune information de contraste
    n'est inventee). Avec percentiles (ex. 1 et 99) : etirement robuste, utile pour voir quelque
    chose dans une image 16 bits qui n'occupe qu'une petite partie de la dynamique.
    """
    if image.dtype == np.uint8 and low_percentile is None:  # deja en 8 bits et pas d'etirement demande
        return image  # rien a faire
    data = image.astype(np.float32)  # passe en nombres a virgule pour calculer sans debordement
    if low_percentile is None or high_percentile is None:  # sans percentiles : on garde toute la plage du type
        low, high = 0.0, max_value(image)  # bornes = 0 et la valeur maximale du type
    else:  # avec percentiles : etirement robuste
        low, high = np.percentile(data, [low_percentile, high_percentile])  # valeurs sous lesquelles tombent 1 % et 99 % des pixels (ex.)
    scaled = (data - low) / max(float(high - low), 1e-6)  # ramene [low, high] vers [0, 1] ; 1e-6 evite une division par zero
    return (np.clip(scaled, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)  # coupe hors [0, 1], passe en 0-255, arrondit, convertit en 8 bits


def image_size(path: str | Path) -> tuple[int, int]:  # renvoie (largeur, hauteur) d'une image
    """(largeur, hauteur) sans decoder toute l'image quand c'est possible (lecture d'en-tete par PIL)."""
    from PIL import Image  # PIL (Pillow) : lit la taille dans l'en-tete du fichier, sans decoder les pixels

    with Image.open(path) as handle:  # ouvre l'image (refermee automatiquement a la sortie du bloc)
        return handle.size  # taille (largeur, hauteur) lue dans l'en-tete
