"""Echelle et orientation : faire ressortir les structures d'une taille ou d'une direction donnees.

Ces operations produisent une image de "reponse" (ou est la structure cherchee ?) plutot
qu'une image corrigee. Elles servent de canal supplementaire, de base de comparaison classique
(seuillage de la reponse) ou de pretraitement quand les defauts ont une echelle connue.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes

import cv2  # OpenCV : lecture, ecriture et traitement d'images
import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)

from vtb.preprocess.dtypes import as_float, from_float, per_channel  # aller-retour vers les flottants ; application par canal
from vtb.preprocess.registry import register  # decorateur qui inscrit l'operation dans le registre


@register("difference_of_gaussians", family="echelle")  # inscrit l'operation "difference_of_gaussians", famille echelle
def difference_of_gaussians(  # difference de deux flous ; renvoie l'image de reponse
    image: np.ndarray,  # image d'entree
    sigma_small: float = 1.0,  # petit flou : retire seulement le bruit tres fin
    sigma_large: float = 4.0,  # grand flou : retire aussi les structures plus petites que lui
    mode: str = "signed",  # "signed" (clair / sombre distingues) ou "abs" (tout en clair)
    gain: float = 4.0,  # multiplicateur fixe de la reponse (pour la rendre visible)
) -> np.ndarray:  # renvoie une image du meme type que l'entree
    """Difference de gaussiennes (passe-bande) : garde les structures de taille entre sigma_small et sigma_large.

    mode signed : 0.5 + gain x DoG (gris moyen = rien, clair = plus clair que l'entourage, sombre
    = plus sombre) ; mode abs : gain x |DoG| (toute structure claire ou sombre ressort en clair).
    OpenCvSharp : deux Cv2.GaussianBlur puis Cv2.Subtract sur des Mat flottantes.
    """
    if sigma_large <= sigma_small:  # les deux flous doivent etre dans le bon ordre
        raise ValueError("sigma_large doit etre plus grand que sigma_small")  # erreur claire

    def one(channel: np.ndarray) -> np.ndarray:  # calcule la reponse sur un canal ; renvoie des flottants
        x, _ = as_float(channel)  # canal en flottants [0, 1]
        dog = cv2.GaussianBlur(x, (0, 0), sigma_small) - cv2.GaussianBlur(x, (0, 0), sigma_large)  # petit flou - grand flou
        return 0.5 + gain * dog if mode == "signed" else gain * np.abs(dog)  # signed : centre sur gris moyen ; abs : valeur absolue

    if mode not in ("signed", "abs"):  # mode inconnu
        raise ValueError("mode : signed ou abs")  # erreur claire
    return from_float(per_channel(one, image), image)  # applique a chaque canal, retour au type d'origine


# fabrique les filtres de Gabor (un par orientation) ; renvoie la liste des noyaux
def gabor_kernels(wavelength: float, sigma: float, orientations: int, aspect: float) -> list[np.ndarray]:
    """Banc de noyaux de Gabor (partie reelle), un par orientation, de moyenne nulle et de norme L1 unite."""
    # filtre de Gabor = onde (rayures) dans une enveloppe gaussienne : repond aux motifs periodiques d'une direction donnee
    size = int(2 * np.ceil(3 * sigma) + 1)  # taille du noyau : couvre 3 sigma de chaque cote, impaire
    kernels = []  # liste des noyaux
    for k in range(orientations):  # pour chaque orientation
        theta = np.pi * k / orientations  # angle de l'orientation k (entre 0 et 180 degres)
        # noyau de Gabor : taille, largeur sigma, angle, periode, allongement, phase 0, en flottants 32 bits
        kernel = cv2.getGaborKernel((size, size), sigma, theta, wavelength, aspect, 0, ktype=cv2.CV_32F)
        kernel -= kernel.mean()  # moyenne nulle : une zone uniforme ne repond pas
        kernels.append(kernel / max(float(np.abs(kernel).sum()), 1e-6))  # divise par la somme des valeurs absolues (norme L1 = 1)
    return kernels  # renvoie les noyaux


@register("gabor", family="echelle")  # inscrit l'operation "gabor"
def gabor(  # banc de filtres de Gabor ; renvoie l'image de reponse
    image: np.ndarray,  # image d'entree
    wavelength: float = 8.0,  # periode des rayures cherchees, en pixels
    sigma: float = 4.0,  # largeur de l'enveloppe gaussienne
    orientations: int = 4,  # nombre de directions testees
    aspect: float = 0.5,  # allongement de l'enveloppe (1 = ronde)
    gain: float = 4.0,  # multiplicateur fixe de la reponse
) -> np.ndarray:  # renvoie une image du meme type que l'entree
    """Banc de Gabor : reponse maximale sur plusieurs orientations ; fait ressortir textures et rayures d'une periode donnee.

    wavelength = periode en pixels de la structure cherchee ; orientations = nombre de directions
    testees (4 = tous les 45 degres). OpenCvSharp : Cv2.GetGaborKernel puis Cv2.Filter2D.
    """
    kernels = gabor_kernels(wavelength, sigma, int(orientations), aspect)  # noyaux calcules une seule fois

    def one(channel: np.ndarray) -> np.ndarray:  # reponse de Gabor sur un canal ; renvoie des flottants
        x, _ = as_float(channel)  # canal en flottants [0, 1]
        # filtre le canal avec chaque noyau (filter2D = convolution) et garde la valeur absolue de la reponse
        responses = [np.abs(cv2.filter2D(x, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT)) for k in kernels]
        return gain * np.max(responses, axis=0)  # garde, pour chaque pixel, la plus forte reponse toutes orientations confondues

    return from_float(per_channel(one, image), image)  # applique a chaque canal, retour au type d'origine
