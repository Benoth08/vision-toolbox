"""Nettete : redonner du pique a des images legerement floues (mise au point, flou de bouge).

Toute accentuation amplifie aussi le bruit : a combiner avec un leger debruitage en amont.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes

import cv2  # OpenCV : lecture, ecriture et traitement d'images
import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)

from vtb.preprocess.dtypes import as_float, from_float  # aller-retour entre type entier et flottant [0, 1]
from vtb.preprocess.registry import register  # decorateur qui inscrit l'operation dans le registre


@register("unsharp_mask", family="nettete")  # inscrit l'operation "unsharp_mask", famille nettete
def unsharp_mask(image: np.ndarray, sigma: float = 2.0, amount: float = 1.0) -> np.ndarray:  # accentue les contours ; renvoie l'image
    """Masque flou : image + amount x (image - flou), l'accentuation classique des contours.

    OpenCvSharp : Cv2.GaussianBlur puis Cv2.AddWeighted(src, 1 + amount, blur, -amount, 0, dst).
    """
    x, _ = as_float(image)  # image en flottants [0, 1]
    blurred = cv2.GaussianBlur(x, (0, 0), sigmaX=float(sigma), borderType=cv2.BORDER_REFLECT)  # version floue de l'image
    # (1 + amount) x image - amount x flou = image + amount x (image - flou) : les details sont renforces
    return from_float(cv2.addWeighted(x, 1.0 + amount, blurred, -amount, 0.0), image)


@register("laplacian_sharpen", family="nettete")  # inscrit l'operation "laplacian_sharpen"
def laplacian_sharpen(image: np.ndarray, amount: float = 0.5) -> np.ndarray:  # accentue avec le laplacien ; renvoie l'image
    """Accentuation par laplacien : image - amount x laplacien ; plus agressive que le masque flou sur les details fins.

    OpenCvSharp : Cv2.Laplacian(src, lap, MatType.CV_32F) puis Cv2.AddWeighted.
    """
    x, _ = as_float(image)  # image en flottants [0, 1]
    lap = cv2.Laplacian(x, cv2.CV_32F, ksize=3, borderType=cv2.BORDER_REFLECT)  # laplacien = courbure locale de l'intensite
    return from_float(x - amount * lap, image)  # retirer le laplacien creuse les transitions : bords plus nets ; retour au type d'origine
