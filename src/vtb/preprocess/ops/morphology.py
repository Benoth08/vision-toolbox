"""Morphologie mathematique : nettoyer, isoler les petits objets, mesurer des formes.

La taille de l'element structurant fixe ce qui disparait : une ouverture de taille 7 efface tout
objet clair plus fin que 7 pixels. C'est le parametre a regler en fonction de la taille des
defauts. OpenCV travaille directement en 8 et 16 bits, en gris comme en couleur.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes

import cv2  # OpenCV : lecture, ecriture et traitement d'images
import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)

from vtb.preprocess.registry import register  # decorateur qui inscrit l'operation dans le registre

SHAPES = {"ellipse": cv2.MORPH_ELLIPSE, "rect": cv2.MORPH_RECT, "cross": cv2.MORPH_CROSS}  # formes possibles de l'element structurant


def structuring_element(kernel: int, shape: str = "ellipse") -> np.ndarray:  # renvoie le petit masque qui sonde l'image
    """Element structurant carre de cote `kernel` (OpenCvSharp : Cv2.GetStructuringElement)."""
    # element structurant = petit motif (disque, carre, croix) que l'on promene sur l'image
    if shape not in SHAPES:  # forme inconnue
        raise ValueError(f"forme inconnue : {shape} ({sorted(SHAPES)})")  # erreur avec la liste des formes valides
    return cv2.getStructuringElement(SHAPES[shape], (int(kernel), int(kernel)))  # masque kernel x kernel de la forme demandee


# applique une operation morphologique d'OpenCV ; renvoie l'image transformee
def _morph(image: np.ndarray, operation: int, kernel: int, shape: str, iterations: int) -> np.ndarray:
    # iterations = nombre de repetitions ; BORDER_REFLECT = bords completes par effet miroir
    return cv2.morphologyEx(image, operation, structuring_element(kernel, shape), iterations=int(iterations), borderType=cv2.BORDER_REFLECT)


@register("morph_open", family="morphologie")  # inscrit l'operation "morph_open", famille morphologie
# ouverture ; renvoie l'image sans les petits objets clairs
def morph_open(image: np.ndarray, kernel: int = 5, shape: str = "ellipse", iterations: int = 1) -> np.ndarray:
    """Ouverture (erosion puis dilatation) : efface les petits objets clairs et les pointes, garde les grands.

    OpenCvSharp : Cv2.MorphologyEx(src, dst, MorphTypes.Open, element).
    """
    return _morph(image, cv2.MORPH_OPEN, kernel, shape, iterations)  # erosion (retrecit le clair) puis dilatation (le regrossit)


@register("morph_close", family="morphologie")  # inscrit l'operation "morph_close"
# fermeture ; renvoie l'image avec les petits trous sombres bouches
def morph_close(image: np.ndarray, kernel: int = 5, shape: str = "ellipse", iterations: int = 1) -> np.ndarray:
    """Fermeture (dilatation puis erosion) : bouche les petits trous et les fissures sombres fines.

    OpenCvSharp : Cv2.MorphologyEx(src, dst, MorphTypes.Close, element).
    """
    return _morph(image, cv2.MORPH_CLOSE, kernel, shape, iterations)  # dilatation puis erosion


@register("morph_gradient", family="morphologie")  # inscrit l'operation "morph_gradient"
def morph_gradient(image: np.ndarray, kernel: int = 3, shape: str = "ellipse") -> np.ndarray:  # renvoie les contours epais
    """Gradient morphologique (dilatation - erosion) : contour epais des objets, robuste au bruit.

    OpenCvSharp : Cv2.MorphologyEx(src, dst, MorphTypes.Gradient, element).
    """
    return _morph(image, cv2.MORPH_GRADIENT, kernel, shape, 1)  # dilatation - erosion, une seule fois


@register("tophat", family="morphologie")  # inscrit l'operation "tophat"
def tophat(image: np.ndarray, kernel: int = 15, shape: str = "ellipse") -> np.ndarray:  # renvoie les petits objets clairs
    """Top-hat (image - ouverture) : ne garde que les petits objets CLAIRS (plus petits que kernel), sur fond noir.

    OpenCvSharp : Cv2.MorphologyEx(src, dst, MorphTypes.TopHat, element).
    """
    return _morph(image, cv2.MORPH_TOPHAT, kernel, shape, 1)  # image - ouverture, une seule fois


@register("blackhat", family="morphologie")  # inscrit l'operation "blackhat"
def blackhat(image: np.ndarray, kernel: int = 15, shape: str = "ellipse") -> np.ndarray:  # renvoie les petits objets sombres, en clair
    """Black-hat (fermeture - image) : ne garde que les petits objets SOMBRES (piqures, fissures), rendus en clair.

    OpenCvSharp : Cv2.MorphologyEx(src, dst, MorphTypes.BlackHat, element).
    """
    return _morph(image, cv2.MORPH_BLACKHAT, kernel, shape, 1)  # fermeture - image, une seule fois
