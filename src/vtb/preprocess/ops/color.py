"""Couleur : separer luminosite et teinte, choisir le canal qui contient l'information.

Rappel : dans la toolbox les images couleur sont en RGB (la conversion depuis le BGR d'OpenCV est
faite a la lecture). Un programme C# qui lit l'image avec OpenCvSharp recoit du BGR : il doit
convertir en RGB avant d'appliquer ces operations (Cv2.CvtColor(src, dst, ColorConversionCodes.BGR2RGB)).
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes

import cv2  # OpenCV : lecture, ecriture et traitement d'images
import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)

from vtb.preprocess.dtypes import as_float, from_float  # aller-retour entre type entier et flottant [0, 1]
from vtb.preprocess.registry import register  # decorateur qui inscrit l'operation dans le registre

# codes de conversion OpenCV depuis RGB : HSV (teinte, saturation, valeur) et LAB (luminance + 2 axes de couleur)
SPACES = {"hsv": cv2.COLOR_RGB2HSV, "lab": cv2.COLOR_RGB2LAB}


@register("to_gray", family="couleur")  # inscrit l'operation "to_gray", famille couleur
def to_gray(image: np.ndarray) -> np.ndarray:  # convertit en niveaux de gris ; renvoie une image (H, W)
    """Couleur -> niveaux de gris (forme (H, W)), ponderation de luminance 0.299 R + 0.587 G + 0.114 B.

    Une image deja grise est rendue telle quelle. OpenCvSharp : Cv2.CvtColor(src, dst, ColorConversionCodes.RGB2GRAY).
    """
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # deja grise : inchangee ; sinon conversion


@register("extract_channel", family="couleur")  # inscrit l'operation "extract_channel"
def extract_channel(image: np.ndarray, space: str = "hsv", channel: int = 2) -> np.ndarray:  # renvoie un seul canal (H, W)
    """Un seul canal d'un espace de couleur (rgb, hsv, lab), remis sur toute la dynamique du type (forme (H, W)).

    HSV : 0 = teinte, 1 = saturation, 2 = valeur ; LAB : 0 = luminance, 1 = vert-rouge, 2 = bleu-jaune.
    Le calcul se fait en flottant (teinte 0-360, L 0-100, a et b centres sur 0) puis remis dans
    [0, 1] : le resultat ne depend pas des conventions 8 bits d'OpenCV (teinte sur 0-179).
    OpenCvSharp : Cv2.CvtColor sur une Mat CV_32FC3, puis Cv2.ExtractChannel.
    """
    if image.ndim == 2:  # une image grise n'a pas de canaux de couleur
        raise ValueError("extract_channel attend une image couleur (H, W, 3)")  # erreur claire
    if not 0 <= channel <= 2:  # numero de canal hors de 0..2
        raise ValueError("channel : 0, 1 ou 2")  # erreur claire
    x, _ = as_float(image)  # image en flottants [0, 1]
    if space == "rgb":  # espace RGB
        values = x[:, :, channel]  # le canal demande, deja dans [0, 1]
    elif space == "hsv":  # espace HSV
        hsv = cv2.cvtColor(x, SPACES["hsv"])  # conversion en HSV (en flottant : teinte 0-360, S et V dans 0-1)
        values = hsv[:, :, 0] / 360.0 if channel == 0 else hsv[:, :, channel]  # teinte ramenee dans [0, 1] ; S et V tels quels
    elif space == "lab":  # espace LAB
        lab = cv2.cvtColor(x, SPACES["lab"])  # conversion en LAB (L 0-100, a et b environ -128..127)
        values = lab[:, :, 0] / 100.0 if channel == 0 else (lab[:, :, channel] + 128.0) / 255.0  # chaque canal ramene dans [0, 1]
    else:  # autre espace : non gere
        raise ValueError(f"espace inconnu : {space} (rgb, hsv, lab)")  # erreur claire
    return from_float(values, image)  # retour au type de l'image d'entree (8 ou 16 bits)


@register("gray_world", family="couleur")  # inscrit l'operation "gray_world"
def gray_world(image: np.ndarray) -> np.ndarray:  # corrige une dominante de couleur ; renvoie l'image corrigee
    """Balance des blancs "monde gris" : chaque canal est remis a la moyenne commune (corrige une dominante d'eclairage).

    Hypothese : en moyenne, la scene est grise. Fausse pour une scene a dominante reelle (rouille,
    metal chaud) : a valider sur des images representatives. Une image grise est rendue telle quelle.
    OpenCvSharp : Cv2.Mean par canal puis multiplication canal par canal.
    """
    if image.ndim == 2:  # image grise : rien a equilibrer
        return image  # inchangee
    x, _ = as_float(image)  # image en flottants [0, 1]
    means = x.reshape(-1, 3).mean(axis=0)  # moyenne de chaque canal R, G, B sur toute l'image
    # chaque canal multiplie par (moyenne commune / sa moyenne) : les 3 canaux finissent avec la meme moyenne
    return from_float(x * (means.mean() / np.maximum(means, 1e-6)), image)
