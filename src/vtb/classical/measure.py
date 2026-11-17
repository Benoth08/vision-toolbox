"""Mesure : position de bords au sous-pixel, largeur d'une bande, calibration pixels -> millimetres.

Une mesure fiable demande trois choses, dans cet ordre : une optique corrigee (calibration et
preprocess `undistort`), une echelle connue (mm par pixel), et une localisation des bords plus
fine que le pixel (interpolation du maximum de gradient). Sans la premiere, les bords de
l'image mesurent faux ; sans la derniere, la precision est limitee a +/- 0,5 pixel.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[str], X | None) partout

import json  # ecriture de fichiers JSON (texte structure en dictionnaires et listes)
from collections.abc import Sequence  # Sequence : type "liste ou tuple", pour annoter les parametres
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import cv2  # OpenCV : conversion en gris, detection du damier, calibration de camera
import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)
import pandas as pd  # pandas : tableaux a lignes et colonnes nommees (DataFrame)
from scipy.ndimage import gaussian_filter1d  # lissage gaussien d'un signal 1D (un profil d'intensite)


def _parabolic_peak(values: np.ndarray, index: int) -> float:  # affine la position d'un maximum ; renvoie une position decimale
    """Sommet de la parabole passant par les trois points autour de index : la position au sous-pixel."""
    if index <= 0 or index >= len(values) - 1:  # maximum au bord : pas de voisin des deux cotes
        return float(index)  # on garde la position entiere
    left, center, right = values[index - 1], values[index], values[index + 1]  # le point et ses deux voisins
    denominator = left - 2 * center + right  # courbure de la parabole qui passe par ces trois points
    # sommet de la parabole = index + decalage entre -0,5 et +0,5 ; courbure nulle (plateau) : on garde index
    return float(index) if abs(denominator) < 1e-12 else float(index + 0.5 * (left - right) / denominator)


# position du bord le plus net d'un profil (ligne de pixels) ; renvoie une position en pixels, au sous-pixel
def strongest_edge(profile: Sequence[float], direction: str = "any", smooth_sigma: float = 1.0) -> float:
    """Position (en pixels, sous-pixel) du bord le plus net d'un profil d'intensite.

    direction : rising (sombre -> clair), falling (clair -> sombre) ou any. Le bord est au maximum
    de la derivee du profil lisse ; sa position est affinee par interpolation parabolique.
    """
    values = (  # profil en flottants, lisse si demande (le lissage evite qu'un pixel de bruit passe pour un bord)
        gaussian_filter1d(np.asarray(profile, dtype=np.float64), smooth_sigma)  # profil lisse par un flou gaussien
        if smooth_sigma > 0  # seulement si une force de lissage est donnee
        else np.asarray(profile, dtype=np.float64)  # sinon le profil brut
    )
    gradient = np.gradient(values)  # gradient = variation d'un pixel au suivant (derivee) : fort la ou il y a un bord
    if direction == "rising":  # bord sombre -> clair
        score = gradient  # on cherche la plus forte montee
    elif direction == "falling":  # bord clair -> sombre
        score = -gradient  # on cherche la plus forte descente
    elif direction == "any":  # n'importe quel bord
        score = np.abs(gradient)  # on cherche la plus forte variation, dans un sens ou l'autre
    else:  # direction inconnue
        raise ValueError("direction : rising, falling ou any")  # arret avec message clair
    return _parabolic_peak(score, int(np.argmax(score)))  # position du maximum, affinee au sous-pixel


# largeur d'une bande claire (ou sombre) ligne par ligne ; renvoie un DataFrame (row, left, right, width)
def strip_width(image: np.ndarray, bright_strip: bool = True, smooth_sigma: float = 1.0, step: int = 1) -> pd.DataFrame:
    """Largeur d'une bande (tole vue de dessus) ligne par ligne : bord gauche, bord droit, largeur en pixels.

    bright_strip : la bande est plus claire que le fond (sinon les sens des bords s'inversent).
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # couleur -> gris ; gris inchange
    data = gray.astype(np.float64)  # flottants : les derivees peuvent etre negatives
    width = data.shape[1]  # largeur de l'image en pixels
    rows = []  # une mesure par ligne analysee
    for y in range(0, data.shape[0], max(1, int(step))):  # une ligne sur step (step >= 1)
        profile = data[y]  # profil d'intensite de la ligne y
        half = width // 2  # milieu de l'image : bord gauche cherche a gauche, bord droit a droite
        left_dir, right_dir = ("rising", "falling") if bright_strip else ("falling", "rising")  # sens attendu des deux bords
        left = strongest_edge(profile[:half], left_dir, smooth_sigma)  # bord gauche, dans la moitie gauche
        right = half + strongest_edge(profile[half:], right_dir, smooth_sigma)  # bord droit (position remise dans toute la ligne)
        rows.append({"row": y, "left": left, "right": right, "width": right - left})  # largeur = droite - gauche
    return pd.DataFrame(rows)  # tableau des mesures


def calibrate_from_distance(pixels: float, millimeters: float) -> float:  # renvoie l'echelle en millimetres par pixel
    """Echelle (mm par pixel) a partir d'une distance connue mesuree dans l'image (cale etalon, mire)."""
    if pixels <= 0 or millimeters <= 0:  # une distance nulle ou negative n'a pas de sens
        raise ValueError("distances strictement positives attendues")  # arret avec message clair
    return float(millimeters) / float(pixels)  # echelle = mm / pixels


# calibration complete d'une camera avec des photos de damier ; renvoie un dict (matrice, distorsion, erreur, echelle...)
def calibrate_chessboard(images: Sequence[np.ndarray], pattern: tuple[int, int] = (9, 6), square_mm: float = 10.0) -> dict:
    """Calibration de camera sur des photos d'un damier : matrice camera, distorsion, erreur et echelle.

    pattern : nombre de coins INTERIEURS (colonnes, lignes). Il faut une dizaine de vues du damier
    dans des positions et inclinaisons variees. mm_per_pixel est l'echelle moyenne mesuree dans le
    plan du damier de la premiere vue (valable si les pieces sont a la meme distance).
    OpenCvSharp : Cv2.FindChessboardCorners, Cv2.CornerSubPix, Cv2.CalibrateCamera.
    """
    template = np.zeros((pattern[0] * pattern[1], 3), np.float32)  # position 3D reelle de chaque coin (z = 0 : damier plat)
    template[:, :2] = np.mgrid[0 : pattern[0], 0 : pattern[1]].T.reshape(-1, 2) * square_mm  # grille (x, y) en millimetres
    object_points, image_points, size = [], [], None  # coins reels, coins vus dans les images, taille d'image
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)  # arret de l'affinage : 30 essais ou precision 1e-3
    for image in images:  # pour chaque photo du damier
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # couleur -> gris
        # la detection du damier veut du 8 bits : remise a l'echelle 0-255 si besoin
        gray = gray if gray.dtype == np.uint8 else cv2.convertScaleAbs(gray, alpha=255.0 / max(float(gray.max()), 1.0))
        size = gray.shape[::-1]  # (largeur, hauteur) : ordre attendu par OpenCV (shape donne hauteur, largeur)
        found, corners = cv2.findChessboardCorners(gray, pattern, None)  # cherche les coins interieurs du damier
        if found:  # si le damier complet a ete trouve
            corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # affine chaque coin au sous-pixel
            object_points.append(template)  # positions reelles des coins
            image_points.append(corners)  # positions vues dans l'image
    if not image_points:  # aucune image utilisable
        raise ValueError(f"damier {pattern} introuvable dans les {len(images)} images")  # arret avec message clair
    # calcule la matrice de la camera (focale, centre) et la distorsion de l'objectif ; rms = erreur moyenne en pixels
    rms, matrix, coeffs, _, _ = cv2.calibrateCamera(object_points, image_points, size, None, None)
    first = image_points[0].reshape(pattern[1], pattern[0], 2)  # coins de la premiere vue ranges en grille lignes x colonnes
    step_px = float(np.mean(np.linalg.norm(np.diff(first, axis=1), axis=-1)))  # ecart moyen en pixels entre coins voisins
    return {  # resultat sous forme de dictionnaire (pret pour JSON)
        "camera_matrix": matrix.tolist(),  # matrice 3 x 3 de la camera (tolist : tableau numpy -> listes Python)
        "dist_coeffs": coeffs.ravel().tolist(),  # coefficients de distorsion, a plat
        "rms_px": float(rms),  # erreur de reprojection en pixels (bon si < 0,5)
        "mm_per_pixel": float(square_mm) / step_px,  # echelle : taille d'une case en mm / taille en pixels
        "n_views": len(image_points),  # nombre de vues utilisees
        "image_size": list(size),  # (largeur, hauteur) des images
    }


def save_calibration(calibration: dict, path: str | Path) -> None:  # enregistre la calibration en JSON ; ne renvoie rien
    """Enregistre la calibration en JSON (relue par le pretraitement `undistort`)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)  # cree le dossier de destination s'il n'existe pas
    Path(path).write_text(json.dumps(calibration, indent=2), encoding="utf-8")  # ecrit le JSON indente
