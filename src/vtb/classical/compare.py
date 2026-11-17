"""Comparaison a une piece de reference : aligner, soustraire, seuiller, mesurer les ecarts.

La methode la plus simple de controle d'aspect quand les pieces sont identiques et bien
positionnees (circuit imprime, etiquette, piece usinee) : tout ecart significatif a la reference
est un defaut. Elle echoue des que la piece varie naturellement (textures, coulee, laminage) :
c'est la que les methodes apprises (anomalies) prennent le relais.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[str], X | None) partout

import cv2  # OpenCV : conversion en gris, flou gaussien
import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)
import pandas as pd  # pandas : tableaux a lignes et colonnes nommees (DataFrame)

from vtb.classical.blobs import detect_blobs  # detection et mesure des taches d'un masque
from vtb.preprocess.dtypes import as_float  # convertit une image en flottants entre 0 et 1
from vtb.preprocess.ops.alignment import align_to  # recale une image sur une reference (meme position, meme taille)


# compare une image a la reference ; renvoie un dict : image alignee, carte de difference, masque, tableau des ecarts
def compare_to_reference(
    image: np.ndarray,  # image de la piece a controler
    reference: np.ndarray,  # image d'une piece bonne, prise dans les memes conditions
    align: str = "ecc",  # methode d'alignement : "ecc" (petits decalages), "orb" (points d'interet) ou "none"
    blur_sigma: float = 1.0,  # force du flou applique avant soustraction (0 = pas de flou)
    threshold: float = 0.15,  # ecart minimal (fraction de 0 a 1 de la dynamique) pour compter comme defaut
    min_area: float = 10,  # taille minimale d'un ecart, en pixels
) -> dict[str, np.ndarray | pd.DataFrame]:  # renvoie un dictionnaire de tableaux
    """Image vs reference -> carte de difference (float [0, 1]), masque des ecarts, tableau des blobs.

    threshold : ecart minimal en fraction de la dynamique ; blur_sigma : lissage des deux images
    avant soustraction (absorbe le bruit et un residu d'alignement sub-pixel).
    """
    if image.dtype != reference.dtype or image.ndim != reference.ndim:  # types (8/16 bits) ou nombre de canaux differents
        raise ValueError("image et reference doivent avoir le meme type et le meme nombre de canaux")  # arret avec message clair
    aligned = align_to(image, reference, align)  # superpose l'image sur la reference (sinon tout decalage serait un "defaut")
    # passe en gris si besoin, puis en flottants 0-1 ; le second resultat (valeur max du type) ne sert pas : "_"
    a, _ = as_float(aligned if aligned.ndim == 2 else cv2.cvtColor(aligned, cv2.COLOR_RGB2GRAY))
    b, _ = as_float(reference if reference.ndim == 2 else cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY))  # meme chose pour la reference
    if blur_sigma > 0:  # si un lissage est demande
        # flou gaussien des deux images ; (0, 0) = taille de noyau deduite de sigma
        a, b = cv2.GaussianBlur(a, (0, 0), blur_sigma), cv2.GaussianBlur(b, (0, 0), blur_sigma)
    diff = np.abs(a - b)  # carte de difference : ecart absolu pixel par pixel, entre 0 et 1
    # masque des ecarts (1 si diff > seuil), puis detection des taches et de leurs mesures (intensite lue dans diff)
    table, mask = detect_blobs((diff > threshold).astype(np.uint8), diff, min_area=min_area)
    return {"aligned": aligned, "diff": diff, "mask": mask, "blobs": table}  # tous les resultats, pour afficher et decider
