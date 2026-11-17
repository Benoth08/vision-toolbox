"""Contours : intensite du gradient, pour les approches classiques ou comme canal supplementaire.

Les sorties sont mises a une echelle FIXE (parametre gain), jamais normalisees image par image :
une meme arete doit donner la meme valeur d'une image a l'autre, sinon un seuil ne veut plus
rien dire. Toutes ces operations sont sensibles au bruit : lisser avant (gaussian_blur).
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes

import cv2  # OpenCV : lecture, ecriture et traitement d'images
import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)

from vtb.io.images import max_value  # valeur maximale d'un pixel selon le type (255, 65535 ou 1)
from vtb.preprocess.dtypes import as_float, from_float, per_channel  # aller-retour vers les flottants ; application par canal
from vtb.preprocess.registry import register  # decorateur qui inscrit l'operation dans le registre


def _gradient_magnitude(x: np.ndarray, kind: str) -> np.ndarray:  # renvoie la force du changement de niveau en chaque pixel
    # gradient = variation de l'intensite entre pixels voisins (fort sur un contour)
    if kind == "scharr":  # filtre de Scharr
        gx, gy = cv2.Scharr(x, cv2.CV_32F, 1, 0), cv2.Scharr(x, cv2.CV_32F, 0, 1)  # variation horizontale et verticale
    else:  # filtre de Sobel
        gx, gy = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)  # variations, noyau 3x3
    return np.sqrt(gx**2 + gy**2)  # norme : racine(gx^2 + gy^2), force du contour quelle que soit sa direction


@register("sobel", family="contours")  # inscrit l'operation "sobel", famille contours
def sobel(image: np.ndarray, gain: float = 0.25) -> np.ndarray:  # renvoie la carte des contours de Sobel (meme type que l'image)
    """Norme du gradient de Sobel 3x3 ; gain 0.25 ramene un saut noir/blanc franc vers le haut de la dynamique.

    OpenCvSharp : Cv2.Sobel(src, gx, MatType.CV_32F, 1, 0) et (0, 1), puis Cv2.Magnitude.
    """
    # pour chaque canal : passage en [0, 1], gradient, multiplie par le gain fixe ; puis retour au type d'origine
    return from_float(per_channel(lambda c: gain * _gradient_magnitude(as_float(c)[0], "sobel"), image), image)


@register("scharr", family="contours")  # inscrit l'operation "scharr"
def scharr(image: np.ndarray, gain: float = 0.0625) -> np.ndarray:  # renvoie la carte des contours de Scharr
    """Norme du gradient de Scharr : plus precise que Sobel sur les orientations obliques (gain plus faible, noyau plus fort).

    OpenCvSharp : Cv2.Scharr(src, gx, MatType.CV_32F, 1, 0) et (0, 1), puis Cv2.Magnitude.
    """
    # meme principe que sobel, avec le filtre de Scharr
    return from_float(per_channel(lambda c: gain * _gradient_magnitude(as_float(c)[0], "scharr"), image), image)


@register("laplacian_edges", family="contours")  # inscrit l'operation "laplacian_edges"
def laplacian_edges(image: np.ndarray, gain: float = 0.5) -> np.ndarray:  # renvoie la carte |laplacien| (points et lignes fines)
    """Valeur absolue du laplacien : repond aux points et aux lignes fines dans toutes les directions.

    OpenCvSharp : Cv2.Laplacian(src, dst, MatType.CV_32F, 3) puis Cv2.Abs.
    """
    # laplacien = derivee seconde (courbure de l'intensite) ; valeur absolue, gain fixe, par canal, retour au type d'origine
    return from_float(per_channel(lambda c: gain * np.abs(cv2.Laplacian(as_float(c)[0], cv2.CV_32F, ksize=3)), image), image)


@register("canny", family="contours")  # inscrit l'operation "canny"
# detecteur de Canny ; renvoie une image binaire des aretes (0 ou max du type)
def canny(image: np.ndarray, low: float = 50.0, high: float = 150.0, blur_sigma: float = 1.0) -> np.ndarray:
    """Detecteur de Canny : aretes fines et binaires (0 ou valeur max du type) ; seuils en niveaux sur 0-255.

    Calcule sur la luminance ; en couleur, le resultat est recopie sur les 3 canaux pour garder la
    forme de l'image. OpenCvSharp : Cv2.Canny(src, dst, low, high).
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # passe en gris si l'image est en couleur
    gray8 = gray if gray.dtype == np.uint8 else (as_float(gray)[0] * 255.0 + 0.5).astype(np.uint8)  # Canny d'OpenCV exige du 8 bits
    if blur_sigma > 0:  # si un lissage prealable est demande
        gray8 = cv2.GaussianBlur(gray8, (0, 0), blur_sigma)  # flou gaussien pour ne pas detecter le bruit comme des aretes
    # Canny : low/high = seuils bas et haut du gradient ; aretes -> 1.0 puis multipliees par la valeur max du type
    edges = (cv2.Canny(gray8, float(low), float(high)) > 0).astype(np.float64) * max_value(image)
    edges = edges.astype(image.dtype)  # remet dans le type de l'image d'entree
    return edges if image.ndim == 2 else np.stack([edges] * 3, axis=-1)  # en couleur : meme carte recopiee sur les 3 canaux
