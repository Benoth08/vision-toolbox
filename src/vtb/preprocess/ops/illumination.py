"""Correction d'illumination (shading) : rendre le fond uniforme avant toute analyse.

Une image se modelise souvent comme I = R x E + B : la reflectance de l'objet R (ce qui nous
interesse), multipliee par l'eclairage E (champ de microscope, vignettage d'objectif, eclairage
rasant inegal), plus un fond additif B (lumiere parasite, offset du capteur). Les variantes
ci-dessous estiment E ou B et les retirent :

- shading_division         I / E    (E estime par un flou tres large)   -> eclairage multiplicatif
- shading_subtraction      I - B    (B estime par un flou tres large)   -> fond additif
- morphological_background I - ouverture (ou fermeture)                  -> petits objets sur fond variable
- flat_field               (I - noir) / (blanc - noir)                   -> la version rigoureuse, avec images de reference
- column_profile           I / profil median par colonne                 -> cameras lineaires (bande d'acier)
- polynomial_background    I / surface polynomiale ajustee               -> gros objets qu'un flou "verrait"

Le flou large est calcule sur l'image reduite (facteur `downscale`) puis agrandi : pour un
eclairage lisse le resultat est le meme, et le calcul est des dizaines de fois plus rapide.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes

from functools import lru_cache  # lru_cache : memorise le resultat d'une fonction (references lues une seule fois)
from typing import Literal  # Literal : liste fermee de valeurs permises

import cv2  # OpenCV : lecture, ecriture et traitement d'images
import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)
from scipy.ndimage import median_filter  # filtre median de scipy

from vtb.preprocess.dtypes import as_float, from_float, per_channel  # aller-retour vers les flottants ; application par canal
from vtb.preprocess.registry import register  # decorateur qui inscrit l'operation dans le registre


def estimate_background(  # estime le fond lisse (eclairage) d'un canal ; renvoie un tableau de meme taille
    channel: np.ndarray,  # canal en flottants
    sigma: float,  # taille du flou, en pixels de l'image d'origine (bien plus grand que les objets)
    method: Literal["gaussian", "median"] = "gaussian",  # flou gaussien ou median
    downscale: int = 4,  # facteur de reduction avant le flou (plus rapide)
) -> np.ndarray:  # renvoie le fond estime (float32)
    """Fond lisse d'un canal flottant : flou gaussien (ou median, plus robuste aux objets brillants) a tres grande echelle.

    sigma est en pixels de l'image d'origine et doit etre nettement plus grand que les objets
    (sinon le "fond" contient les objets et la division les efface). Regle de depart : 1/10 a
    1/5 de la taille de l'image.
    """
    height, width = channel.shape  # dimensions du canal
    factor = max(1, int(downscale))  # facteur de reduction (au moins 1 = pas de reduction)
    small = (  # image reduite pour aller plus vite
        # reduction (INTER_AREA = moyenne des pixels regroupes), ou le canal tel quel si factor = 1
        cv2.resize(channel, (max(1, width // factor), max(1, height // factor)), interpolation=cv2.INTER_AREA) if factor > 1 else channel
    )
    small_sigma = max(sigma / factor, 0.5)  # sigma ramene a l'echelle reduite (au moins 0.5)
    if method == "gaussian":  # flou gaussien
        blurred = cv2.GaussianBlur(small, (0, 0), sigmaX=small_sigma, borderType=cv2.BORDER_REFLECT)  # bords en miroir
    elif method == "median":  # flou median (ignore mieux les objets brillants)
        size = int(2 * round(small_sigma) + 1)  # fenetre carree de cote ~2 sigma
        blurred = median_filter(small, size=size, mode="reflect")  # mediane sur la fenetre, bords en miroir
    else:  # methode inconnue
        raise ValueError(f"methode de fond inconnue : {method}")  # erreur claire
    if factor > 1:  # si on avait reduit l'image
        blurred = cv2.resize(blurred, (width, height), interpolation=cv2.INTER_LINEAR)  # on remet le fond a la taille d'origine
    return blurred.astype(np.float32)  # renvoie le fond en flottants 32 bits


@register("shading_division", family="illumination")  # inscrit l'operation "shading_division", famille illumination
def shading_division(  # divise l'image par son fond flou ; renvoie l'image corrigee
    image: np.ndarray,  # image d'entree
    sigma: float = 50.0,  # taille du flou qui estime l'eclairage
    method: str = "gaussian",  # flou gaussien ou median
    downscale: int = 4,  # reduction avant le flou (vitesse)
    keep_mean: bool = True,  # True : garde la luminosite moyenne d'origine
) -> np.ndarray:  # renvoie une image du meme type que l'entree
    """Divise l'image par son fond flou : corrige un eclairage multiplicatif (champ de microscope, vignettage)."""

    def correct(channel: np.ndarray) -> np.ndarray:  # corrige un canal ; renvoie des flottants
        x, _ = as_float(channel)  # canal en flottants [0, 1]
        # estime l'eclairage ("type: ignore" : method est un str ici, le verificateur de types attend un Literal)
        background = estimate_background(x, sigma, method, downscale)  # type: ignore[arg-type]
        corrected = x / np.maximum(background, 1e-6)  # divise par l'eclairage (jamais par 0)
        # Sans ce facteur l'image tournerait autour de 1.0 (blanc) : on la ramene a la luminosite moyenne d'origine.
        return corrected * float(background.mean()) if keep_mean else corrected * 0.5  # sinon centree sur 0.5 (gris moyen)

    return from_float(per_channel(correct, image), image)  # applique a chaque canal, retour au type d'origine


@register("shading_subtraction", family="illumination")  # inscrit l'operation "shading_subtraction"
# soustrait le fond flou ; renvoie l'image corrigee
def shading_subtraction(image: np.ndarray, sigma: float = 50.0, method: str = "gaussian", downscale: int = 4) -> np.ndarray:
    """Soustrait le fond flou puis rajoute sa moyenne : corrige un fond additif (lumiere parasite, offset)."""

    def correct(channel: np.ndarray) -> np.ndarray:  # corrige un canal ; renvoie des flottants
        x, _ = as_float(channel)  # canal en flottants [0, 1]
        # estime le fond additif ("type: ignore" : method est un str, le verificateur attend un Literal)
        background = estimate_background(x, sigma, method, downscale)  # type: ignore[arg-type]
        return x - background + float(background.mean())  # retire le fond puis rajoute son niveau moyen

    return from_float(per_channel(correct, image), image)  # applique a chaque canal, retour au type d'origine


@register("morphological_background", family="illumination")  # inscrit l'operation "morphological_background"
# retire un fond estime par morphologie ; renvoie l'image corrigee
def morphological_background(image: np.ndarray, kernel: int = 51, objects: str = "bright", keep_mean: bool = True) -> np.ndarray:
    """Retire un fond estime par ouverture (objets clairs) ou fermeture (objets sombres), a la maniere du "rolling ball".

    L'element structurant (disque de diametre `kernel`) doit etre plus grand que les objets : tout
    ce qui est plus petit disparait de l'estimation du fond et reste donc dans l'image corrigee.
    """
    shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))  # disque de diametre kernel (element structurant)
    operation = {"bright": cv2.MORPH_OPEN, "dark": cv2.MORPH_CLOSE}[objects]  # objets clairs -> ouverture ; sombres -> fermeture

    def correct(channel: np.ndarray) -> np.ndarray:  # corrige un canal ; renvoie des flottants
        x, _ = as_float(channel)  # canal en flottants [0, 1]
        background = cv2.morphologyEx(x, operation, shape)  # fond = image sans les objets plus petits que le disque
        offset = float(background.mean()) if keep_mean else 0.0  # niveau moyen a rajouter (ou 0)
        return x - background + offset  # retire le fond : il ne reste que les objets (+ niveau moyen)

    return from_float(per_channel(correct, image), image)  # applique a chaque canal, retour au type d'origine


@lru_cache(maxsize=8)  # garde en memoire les 8 dernieres references lues
def _reference(path: str, gray: bool) -> np.ndarray:  # lit une image de reference ; renvoie des flottants [0, 1]
    """Image de reference lue une seule fois (flat ou noir), en flottant."""
    from vtb.io.images import read_image  # lecture d'image (importee ici, seulement si besoin)

    ref = read_image(path, color="gray" if gray else "rgb")  # lue en gris ou en couleur comme l'image a corriger
    return as_float(ref)[0]  # convertie en flottants [0, 1]


@register("flat_field", family="illumination")  # inscrit l'operation "flat_field"
# correction par images de reference blanche et noire ; renvoie l'image corrigee
def flat_field(image: np.ndarray, flat_path: str = "", dark_path: str = "") -> np.ndarray:
    """Correction par images de reference : (I - noir) / (blanc - noir) x moyenne(blanc - noir).

    flat_path : image sans objet sous le meme eclairage (lame vide, plaque de reference blanche) ;
    dark_path : image capteur obture (facultative). C'est la methode exacte quand on peut acquerir
    ces references ; elles doivent avoir la meme taille que les images.
    """
    if not flat_path:  # reference blanche obligatoire
        raise ValueError("flat_field : flat_path est obligatoire")  # erreur claire
    gray = image.ndim == 2  # True si l'image est en niveaux de gris
    flat = _reference(flat_path, gray)  # reference blanche (eclairage seul)
    dark = _reference(dark_path, gray) if dark_path else np.zeros_like(flat)  # reference noire, ou zero si absente
    if flat.shape != image.shape[: flat.ndim] or flat.shape != image.shape:  # tailles differentes
        raise ValueError(f"flat_field : reference {flat.shape} et image {image.shape} de tailles differentes")  # erreur claire
    x, _ = as_float(image)  # image en flottants [0, 1]
    gain = np.maximum(flat - dark, 1e-6)  # reponse de l'eclairage en chaque pixel (jamais 0)
    return from_float((x - dark) / gain * float(gain.mean()), image)  # corrige puis garde la luminosite moyenne ; retour au type


@register("column_profile", family="illumination")  # inscrit l'operation "column_profile"
# divise chaque colonne (ou ligne) par son profil ; renvoie l'image corrigee
def column_profile(image: np.ndarray, axis: str = "columns", statistic: str = "median", keep_mean: bool = True) -> np.ndarray:
    """Divise chaque colonne (ou ligne) par son profil median : l'inhomogeneite d'une camera lineaire est dans la largeur."""
    reduce = {"median": np.median, "mean": np.mean}[statistic]  # fonction de resume choisie : mediane ou moyenne
    along = 0 if axis == "columns" else 1  # colonnes : statistique sur les lignes (axe 0)

    def correct(channel: np.ndarray) -> np.ndarray:  # corrige un canal ; renvoie des flottants
        x, _ = as_float(channel)  # canal en flottants [0, 1]
        profile = np.maximum(reduce(x, axis=along, keepdims=True), 1e-6)  # profil : une valeur par colonne (ou ligne), jamais 0
        corrected = x / profile  # divise chaque colonne par sa valeur de profil
        return corrected * float(profile.mean()) if keep_mean else corrected * 0.5  # luminosite moyenne d'origine, ou centree sur 0.5

    return from_float(per_channel(correct, image), image)  # applique a chaque canal, retour au type d'origine


def _polynomial_terms(xx: np.ndarray, yy: np.ndarray, order: int) -> np.ndarray:  # renvoie les termes du polynome (une colonne par terme)
    """Monomes x^i y^j avec i + j <= order, empiles en colonnes."""
    return np.stack([xx**i * yy**j for i in range(order + 1) for j in range(order + 1 - i)], axis=-1)  # ex. ordre 2 : 1, y, y2, x, xy, x2


@register("polynomial_background", family="illumination")  # inscrit l'operation "polynomial_background"
def polynomial_background(  # retire une surface polynomiale ajustee sur le fond ; renvoie l'image corrigee
    image: np.ndarray,  # image d'entree
    order: int = 2,  # degre du polynome (2 ou 3)
    downscale: int = 8,  # reduction avant l'ajustement (vitesse)
    mode: str = "divide",  # "divide" (eclairage multiplicatif) ou autre valeur = soustraction
    outlier_sigma: float = 2.5,  # seuil d'exclusion des pixels trop loin de la surface (les objets)
) -> np.ndarray:  # renvoie une image du meme type que l'entree
    """Ajuste une surface polynomiale (ordre 2 ou 3) sur le fond et la retire ; deux passes pour ignorer les objets."""

    def correct(channel: np.ndarray) -> np.ndarray:  # corrige un canal ; renvoie des flottants
        x, _ = as_float(channel)  # canal en flottants [0, 1]
        height, width = x.shape  # dimensions du canal
        factor = max(1, int(downscale))  # facteur de reduction (au moins 1)
        small = cv2.resize(x, (max(2, width // factor), max(2, height // factor)), interpolation=cv2.INTER_AREA)  # image reduite
        sy, sx = np.mgrid[0 : small.shape[0], 0 : small.shape[1]]  # coordonnees ligne et colonne de chaque pixel reduit
        terms = _polynomial_terms(sx.ravel() / small.shape[1], sy.ravel() / small.shape[0], order)  # termes du polynome (x, y dans [0, 1])
        values = small.ravel()  # valeurs des pixels, a plat
        keep = np.ones_like(values, dtype=bool)  # pixels utilises pour l'ajustement : tous au depart
        for _ in range(2):  # 2e passe : sans les pixels tres eloignes de la surface (les objets)
            coeffs, *_ = np.linalg.lstsq(terms[keep], values[keep], rcond=None)  # moindres carres : coefficients du polynome
            residual = values - terms @ coeffs  # ecart entre chaque pixel et la surface ajustee
            keep = np.abs(residual) < outlier_sigma * residual[keep].std() + 1e-9  # garde les pixels proches de la surface
        fy, fx = np.mgrid[0:height, 0:width]  # coordonnees de chaque pixel a pleine taille
        surface = (_polynomial_terms(fx / width, fy / height, order) @ coeffs).astype(np.float32)  # surface evaluee a pleine taille
        if mode == "divide":  # correction multiplicative
            return x / np.maximum(surface, 1e-6) * float(surface.mean())  # divise par la surface, garde la luminosite moyenne
        return x - surface + float(surface.mean())  # sinon correction additive : soustrait la surface

    return from_float(per_channel(correct, image), image)  # applique a chaque canal, retour au type d'origine


# assombrit artificiellement les bords ; renvoie l'image vignettee
def apply_vignetting(image: np.ndarray, strength: float = 0.6, center: tuple[float, float] = (0.5, 0.5)) -> np.ndarray:
    """Ajoute un vignettage radial artificiel (assombrissement des bords) : pour tester et montrer la correction."""
    height, width = image.shape[:2]  # dimensions de l'image
    yy, xx = np.mgrid[0:height, 0:width]  # coordonnees ligne et colonne de chaque pixel
    r = np.sqrt(((xx / width - center[0]) ** 2 + (yy / height - center[1]) ** 2) / 0.5)  # distance normalisee au centre
    gain = (1.0 - strength * np.clip(r, 0, 1) ** 2).astype(np.float32)  # 1 au centre, diminue vers les bords
    x, _ = as_float(image)  # image en flottants [0, 1]
    return from_float(x * (gain[..., None] if image.ndim == 3 else gain), image)  # multiplie par le gain (sur les 3 canaux si couleur)
