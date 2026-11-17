"""Seuillage : passer d'une image (ou d'une carte de reponse) a un masque binaire.

Toutes les fonctions rendent un masque uint8 {0, 1} de forme (H, W). Les methodes automatiques
(Otsu, triangle) choisissent le seuil image par image : pratique, mais le seuil n'est alors plus
le meme d'une image a l'autre ; en production, un seuil FIXE regle sur la validation est plus
previsible (method="fixed").
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[str], X | None) partout

import cv2  # OpenCV : fonctions de seuillage (threshold, adaptiveThreshold)
import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)

from vtb.io.images import to_uint8  # ramene une image en 8 bits (valeurs 0-255)

METHODS = ("otsu", "triangle", "adaptive", "fixed")  # methodes de seuillage acceptees


def to_gray8(image: np.ndarray) -> np.ndarray:  # renvoie l'image en niveaux de gris 8 bits (tableau 2D, 0-255)
    """Niveaux de gris 8 bits (les seuils automatiques d'OpenCV ne travaillent qu'en 8 bits)."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # couleur (3 canaux) -> gris ; gris inchange
    return to_uint8(gray)  # convertit en octets 0-255 si l'image etait en 16 bits ou en flottants


# seuille une image ; renvoie un masque 0/1 (1 = objet)
def threshold(
    image: np.ndarray,  # image a seuiller (gris ou couleur)
    method: str = "otsu",  # "otsu", "triangle", "adaptive" ou "fixed"
    value: float = 128.0,  # seuil (0-255) pour la methode "fixed"
    block_size: int = 51,  # taille du voisinage (en pixels) pour la methode "adaptive"
    c: float = 5.0,  # pour "adaptive" : on retire c a la moyenne locale pour obtenir le seuil
    invert: bool = False,  # True : l'objet est plus sombre que le fond
) -> np.ndarray:  # renvoie le masque (H, W) en uint8
    """Masque binaire : otsu (deux populations), triangle (un pic + une queue, ex. petits defauts sur fond), adaptive (seuil local), fixed.

    value : seuil en niveaux 0-255 pour fixed ; block_size et c pour adaptive (seuil = moyenne
    locale - c). invert : l'objet est plus SOMBRE que le fond (fissures, piqures).
    OpenCvSharp : Cv2.Threshold(src, dst, 0, 255, ThresholdTypes.Binary | ThresholdTypes.Otsu), Cv2.AdaptiveThreshold.
    """
    if method not in METHODS:  # methode inconnue
        raise ValueError(f"methode de seuillage inconnue : {method} ({METHODS})")  # arret, rappelle les methodes possibles
    gray = to_gray8(image)  # les seuils d'OpenCV veulent une image grise 8 bits
    kind = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY  # BINARY : clair -> 1 ; BINARY_INV : sombre -> 1
    if method == "adaptive":  # seuil local : chaque pixel est compare a la moyenne de son voisinage
        block = int(block_size) | 1  # impair (| 1 force le dernier bit a 1) : OpenCV exige un voisinage centre
        mask = cv2.adaptiveThreshold(gray, 1, cv2.ADAPTIVE_THRESH_MEAN_C, kind, block, float(c))  # valeur 1 pour l'objet
    elif method == "fixed":  # seuil fixe choisi a la main
        _, mask = cv2.threshold(gray, float(value), 1, kind)  # pixel > value -> 1 ; renvoie (seuil, masque)
    else:  # otsu ou triangle : seuil calcule automatiquement a partir de l'histogramme
        flag = cv2.THRESH_OTSU if method == "otsu" else cv2.THRESH_TRIANGLE  # choix de l'algorithme automatique
        _, mask = cv2.threshold(gray, 0, 1, kind | flag)  # le 0 est ignore : OpenCV calcule lui-meme le seuil
    return mask.astype(np.uint8)  # masque 0/1 en octets


def threshold_value(image: np.ndarray, method: str = "otsu") -> float:  # renvoie le seuil automatique (0-255) sans seuiller
    """Le seuil qu'Otsu ou triangle choisiraient (pour l'afficher ou le figer ensuite en seuil fixe)."""
    flag = cv2.THRESH_OTSU if method == "otsu" else cv2.THRESH_TRIANGLE  # choix de l'algorithme automatique
    value, _ = cv2.threshold(to_gray8(image), 0, 255, cv2.THRESH_BINARY | flag)  # on garde le seuil calcule, pas le masque
    return float(value)  # seuil en niveaux de gris 0-255
