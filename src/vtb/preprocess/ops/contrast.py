"""Contraste : rendre visibles des defauts peu contrastes.

En couleur, le contraste est travaille sur la luminance (canal L de l'espace LAB) et pas sur
R, G, B separement : egaliser chaque canal a part changerait les couleurs.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes

import cv2  # OpenCV : lecture, ecriture et traitement d'images
import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)

from vtb.preprocess.dtypes import as_float, from_float  # aller-retour entre type entier et flottant [0, 1]
from vtb.preprocess.registry import register  # decorateur qui inscrit l'operation dans le registre


def _on_luminance(image: np.ndarray, func) -> np.ndarray:  # applique func a la luminosite seule ; renvoie l'image
    """Applique func (canal gris du meme type que l'image) sur la luminance L de LAB, puis revient en RGB."""
    if image.ndim == 2:  # image grise : c'est deja la luminance
        return func(image)  # on applique directement
    if image.dtype == np.uint8:  # couleur 8 bits
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)  # L sur 0-255 en 8 bits (LAB : L = luminosite, A et B = couleur)
        lab[:, :, 0] = func(lab[:, :, 0])  # transforme seulement le canal L
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)  # revient en RGB : couleurs inchangees, contraste modifie
    # 16 bits : LAB en flottant (L dans 0-100), L ramene en uint16 pour l'operation puis retour.
    x, _ = as_float(image)  # image en flottants [0, 1]
    lab = cv2.cvtColor(x, cv2.COLOR_RGB2LAB)  # conversion en LAB (L entre 0 et 100)
    luminance = np.clip(lab[:, :, 0] / 100.0 * 65535.0 + 0.5, 0, 65535).astype(np.uint16)  # L ramene sur 0-65535 en entier 16 bits
    lab[:, :, 0] = func(luminance).astype(np.float32) / 65535.0 * 100.0  # applique func puis remet L entre 0 et 100
    return from_float(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), image)  # revient en RGB puis au type d'origine


def _equalize_channel(channel: np.ndarray) -> np.ndarray:  # egalise l'histogramme d'un canal (8 ou 16 bits) ; renvoie le canal
    if channel.dtype == np.uint8:  # 8 bits : OpenCV sait faire
        return cv2.equalizeHist(channel)  # egalisation d'histogramme d'OpenCV
    # OpenCV n'egalise que le 8 bits : en 16 bits on applique la fonction de repartition directement.
    hist = np.bincount(channel.ravel(), minlength=65536).astype(np.float64)  # histogramme : nombre de pixels pour chaque niveau
    cdf = hist.cumsum()  # fonction de repartition : nombre de pixels de niveau <= v
    cdf_min = cdf[np.nonzero(hist)[0][0]]  # valeur de la repartition au premier niveau present dans l'image
    table = np.round((cdf - cdf_min) / max(cdf[-1] - cdf_min, 1.0) * 65535.0)  # table niveau -> nouveau niveau etale sur 0-65535
    return np.clip(table, 0, 65535).astype(np.uint16)[channel]  # applique la table a chaque pixel (indexation par le niveau)


@register("equalize_hist", family="contraste")  # inscrit l'operation "equalize_hist", famille contraste
def equalize_hist(image: np.ndarray) -> np.ndarray:  # egalisation globale ; renvoie l'image contrastee
    """Egalisation d'histogramme globale : repartit les niveaux de gris uniformement (fort, parfois excessif).

    OpenCvSharp : Cv2.EqualizeHist(src, dst) (8 bits seulement, comme OpenCV).
    """
    if image.dtype not in (np.uint8, np.uint16):  # seuls les entiers 8 et 16 bits sont geres
        raise ValueError(f"equalize_hist attend uint8 ou uint16, recu {image.dtype}")  # erreur claire
    return _on_luminance(image, _equalize_channel)  # egalise la luminance seulement


@register("clahe", family="contraste")  # inscrit l'operation "clahe"
def clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:  # contraste local (CLAHE) ; renvoie l'image
    """Egalisation adaptative limitee (CLAHE) : contraste local par tuiles, sans exploser le bruit.

    clip_limit borne l'amplification (2 a 3 en general ; plus haut, le bruit et la texture du
    metal ressortent autant que les defauts) ; tile_grid = nombre de tuiles par cote.
    OpenCvSharp : Cv2.CreateCLAHE(clipLimit, new Size(tileGrid, tileGrid)).Apply(src, dst).
    """
    if image.dtype not in (np.uint8, np.uint16):  # seuls les entiers 8 et 16 bits sont geres
        raise ValueError(f"clahe attend uint8 ou uint16, recu {image.dtype}")  # erreur claire
    # cree l'operateur CLAHE : limite d'amplification et grille tile_grid x tile_grid de zones egalisees separement
    operator = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid), int(tile_grid)))
    return _on_luminance(image, operator.apply)  # applique CLAHE a la luminance seulement
