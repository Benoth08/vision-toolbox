"""Intensite : ramener des images d'expositions differentes a la meme echelle de gris.

Piege commun a toute cette famille : une operation calculee image par image (percentiles,
min-max) change l'echelle d'une image a l'autre. C'est voulu (on uniformise), mais il faut alors
l'appliquer exactement pareil en production, ce que garantit le Pipeline.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes

import cv2  # OpenCV : lecture, ecriture et traitement d'images
import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)

from vtb.io.images import max_value  # valeur maximale d'un pixel selon le type (255, 65535 ou 1)
from vtb.preprocess.dtypes import as_float, from_float  # aller-retour entre type entier et flottant [0, 1]
from vtb.preprocess.registry import register  # decorateur qui inscrit l'operation dans le registre


@register("percentile_stretch", family="intensite")  # inscrit l'operation "percentile_stretch", famille intensite
def percentile_stretch(image: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:  # etirement robuste ; renvoie l'image
    """Etire l'intervalle [percentile bas, percentile haut] sur toute la dynamique : robuste aux pixels aberrants.

    Les percentiles sont calcules sur tous les canaux a la fois, pour ne pas changer la teinte
    d'une image couleur. OpenCvSharp : pas d'equivalent direct, calculer les percentiles sur
    l'histogramme puis Cv2.ConvertScaleAbs / Mat.ConvertTo(alpha, beta).
    """
    x, _ = as_float(image)  # image en flottants [0, 1]
    lo, hi = np.percentile(x, [low, high])  # niveaux sous lesquels tombent low % et high % des pixels
    return from_float((x - lo) / max(float(hi - lo), 1e-6), image)  # [lo, hi] etire vers [0, 1] (le reste est ecrete), retour au type


@register("minmax_normalize", family="intensite")  # inscrit l'operation "minmax_normalize"
def minmax_normalize(image: np.ndarray) -> np.ndarray:  # etire [min, max] sur toute la plage ; renvoie l'image
    """Etire [min, max] de l'image sur toute la dynamique du type (sensible au moindre pixel aberrant).

    OpenCvSharp : Cv2.Normalize(src, dst, 0, 255, NormTypes.MinMax).
    """
    peak = max_value(image)  # valeur maximale du type (255 ou 65535)
    return cv2.normalize(image, None, 0, peak, cv2.NORM_MINMAX).astype(image.dtype)  # min -> 0, max -> peak ; garde le type


def _lut_apply(image: np.ndarray, table: np.ndarray) -> np.ndarray:  # remplace chaque niveau par sa valeur dans la table
    """Applique une table de correspondance a toute la dynamique du type (256 ou 65536 entrees)."""
    # table de correspondance (LUT) = tableau "ancien niveau -> nouveau niveau"
    if image.dtype == np.uint8:  # 8 bits : OpenCV a une fonction rapide
        return cv2.LUT(image, table.astype(np.uint8))  # applique la table de 256 valeurs
    return table.astype(image.dtype)[image]  # 16 bits : indexation numpy (chaque pixel sert d'indice dans la table)


@register("gamma", family="intensite")  # inscrit l'operation "gamma"
def gamma(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:  # correction gamma ; renvoie l'image corrigee
    """Correction gamma par table : gamma < 1 eclaircit les zones sombres, gamma > 1 les assombrit.

    OpenCvSharp : Cv2.LUT(src, table, dst) avec la meme table de 256 valeurs.
    """
    if gamma <= 0:  # gamma nul ou negatif n'a pas de sens
        raise ValueError("gamma doit etre strictement positif")  # erreur claire
    peak = max_value(image)  # valeur maximale du type
    if image.dtype not in (np.uint8, np.uint16):  # seuls les entiers 8 et 16 bits sont geres (table de correspondance)
        raise ValueError(f"gamma attend une image uint8 ou uint16, recu {image.dtype}")  # erreur claire
    levels = np.arange(int(peak) + 1, dtype=np.float64) / peak  # tous les niveaux possibles, ramenes dans [0, 1]
    return _lut_apply(image, np.round(levels**gamma * peak))  # table : niveau^gamma remis a l'echelle, appliquee a l'image


@register("log_transform", family="intensite")  # inscrit l'operation "log_transform"
def log_transform(image: np.ndarray, strength: float = 100.0) -> np.ndarray:  # compression logarithmique ; renvoie l'image
    """Compression logarithmique log(1 + c x) / log(1 + c) : remonte les details sombres d'une grande dynamique (thermique).

    OpenCvSharp : Cv2.Log sur une Mat flottante (ajouter 1 avant), puis remise a l'echelle.
    """
    x, _ = as_float(image)  # image en flottants [0, 1]
    return from_float(np.log1p(strength * x) / np.log1p(strength), image)  # log1p(v) = log(1 + v) ; resultat dans [0, 1], retour au type


@register("window_level", family="intensite")  # inscrit l'operation "window_level"
# fenetre / niveau : etale une plage de valeurs choisie ; renvoie l'image (8 bits par defaut)
def window_level(image: np.ndarray, center: float | None = None, width: float | None = None, out_dtype: str = "uint8") -> np.ndarray:
    """Fenetre / niveau (radiologie, thermique) : seules les valeurs dans [centre - largeur/2, centre + largeur/2] sont etalees.

    center et width sont en unites brutes de l'image (ex. niveaux 16 bits). Sans valeurs, la
    fenetre couvre les percentiles 1 a 99. out_dtype : uint8 (le cas courant : passer de 16 a
    8 bits pour un reseau) ou same (garder le type d'entree).
    OpenCvSharp : Mat.ConvertTo(dst, MatType.CV_8U, alpha, beta) avec alpha = 255 / largeur.
    """
    data = image.astype(np.float32)  # valeurs brutes en flottants (sans les ramener dans [0, 1])
    if center is None or width is None:  # fenetre non donnee : on la calcule
        lo, hi = np.percentile(data, [1.0, 99.0])  # niveaux a 1 % et 99 % des pixels
        center, width = (lo + hi) / 2, max(hi - lo, 1.0)  # centre et largeur de la fenetre (largeur au moins 1)
    scaled = np.clip((data - (center - width / 2)) / max(width, 1e-6), 0.0, 1.0)  # bas de fenetre -> 0, haut -> 1, le reste ecrete
    if out_dtype == "same":  # garder le type d'entree
        return from_float(scaled, image)  # retour au type d'origine
    if out_dtype != "uint8":  # valeur non prevue
        raise ValueError("out_dtype : uint8 ou same")  # erreur claire
    return (scaled * 255.0 + 0.5).astype(np.uint8)  # sortie 8 bits (0-255, arrondie)


@register("to_uint8", family="intensite")  # inscrit l'operation "to_uint8"
def to_uint8(image: np.ndarray, low: float = -1.0, high: float = -1.0) -> np.ndarray:  # convertit en 8 bits ; renvoie l'image uint8
    """Convertit en 8 bits (sortie uint8) : division par la valeur max du type, ou etirement entre deux percentiles si low et high >= 0.

    OpenCvSharp : Mat.ConvertTo(dst, MatType.CV_8U, 255.0 / 65535.0).
    """
    from vtb.io.images import to_uint8 as convert  # la vraie fonction de conversion (renommee : meme nom que cette operation)

    if low >= 0 and high >= 0:  # percentiles fournis (les valeurs -1 par defaut veulent dire "pas d'etirement")
        return convert(image, low, high)  # etirement entre les deux percentiles
    return convert(image)  # simple division par la valeur max du type


@register("invert", family="intensite")  # inscrit l'operation "invert"
def invert(image: np.ndarray) -> np.ndarray:  # negatif de l'image ; renvoie l'image inversee
    """Negatif (max - x) : rend clairs des defauts sombres, pour les operations qui cherchent des objets clairs.

    OpenCvSharp : Cv2.BitwiseNot(src, dst) en 8 bits.
    """
    return (max_value(image) - image.astype(np.float64)).astype(image.dtype)  # max - valeur pour chaque pixel, garde le type
