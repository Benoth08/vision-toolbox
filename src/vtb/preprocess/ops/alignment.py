"""Alignement sur une image de reference : la base de toute comparaison a une piece "bonne".

Comparer une image a une reference (absdiff) n'a de sens que si les deux sont superposees au
pixel pres : un decalage d'un pixel fait apparaitre de faux defauts sur tous les bords. Deux
methodes :
- ECC (maximisation de correlation) : precise, pour de petits decalages et des images proches ;
- ORB + homographie : points caracteristiques, pour des deplacements ou rotations plus grands.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes

from functools import lru_cache  # lru_cache : memorise le resultat d'une fonction (la reference n'est lue qu'une fois)

import cv2  # OpenCV : lecture, ecriture et traitement d'images
import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)

from vtb.io.images import read_image, to_uint8  # lecture d'image ; conversion en 8 bits
from vtb.preprocess.dtypes import as_float  # conversion en flottants [0, 1]
from vtb.preprocess.registry import register  # decorateur qui inscrit l'operation dans le registre

MOTIONS = {  # types de mouvement que ECC sait corriger, du plus simple au plus libre
    "translation": cv2.MOTION_TRANSLATION,  # simple decalage x, y
    "euclidean": cv2.MOTION_EUCLIDEAN,  # decalage + rotation
    "affine": cv2.MOTION_AFFINE,  # + changement d'echelle et cisaillement
    "homography": cv2.MOTION_HOMOGRAPHY,  # + effet de perspective
}


@lru_cache(maxsize=8)  # garde en memoire les 8 dernieres references lues
def _reference(path: str, gray: bool) -> np.ndarray:  # lit l'image de reference ; renvoie le tableau
    """Image de reference lue une seule fois (meme convention que les images : gris ou RGB, type d'origine)."""
    return read_image(path, color="gray" if gray else "rgb")  # lue en gris ou en couleur selon l'image a aligner


def _gray_float(image: np.ndarray) -> np.ndarray:  # renvoie l'image en gris et en flottants [0, 1] (format attendu par ECC)
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # passe en gris si besoin
    return as_float(gray)[0]  # flottants [0, 1] (on ignore la valeur max renvoyee)


def _warp(image: np.ndarray, matrix: np.ndarray, shape: tuple[int, int]) -> np.ndarray:  # deforme l'image ; renvoie l'image recalee
    height, width = shape  # taille de sortie (celle de la reference)
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP  # interpolation lineaire ; matrice lue comme "sortie -> entree"
    if matrix.shape == (3, 3):  # matrice 3x3 = homographie (perspective)
        # BORDER_REPLICATE : les zones hors image sont remplies en recopiant le bord
        return cv2.warpPerspective(image, matrix, (width, height), flags=flags, borderMode=cv2.BORDER_REPLICATE)
    return cv2.warpAffine(image, matrix, (width, height), flags=flags, borderMode=cv2.BORDER_REPLICATE)  # matrice 2x3 = affine


def ecc_transform(  # estime la transformation qui aligne image sur reference ; renvoie la matrice
    image: np.ndarray,  # image a aligner
    reference: np.ndarray,  # image de reference
    motion: str = "euclidean",  # type de mouvement autorise
    iterations: int = 100,  # nombre maximal d'iterations de l'optimisation
    eps: float = 1e-5,  # arret quand l'amelioration devient plus petite que eps
) -> np.ndarray:  # renvoie une matrice 2x3 (ou 3x3 pour une homographie)
    """Matrice qui superpose image sur reference (convention WARP_INVERSE_MAP d'OpenCV), estimee par ECC."""
    if motion not in MOTIONS:  # type de mouvement inconnu
        raise ValueError(f"mouvement inconnu : {motion} ({sorted(MOTIONS)})")  # erreur avec la liste des choix
    matrix = np.eye(3, dtype=np.float32) if motion == "homography" else np.eye(2, 3, dtype=np.float32)  # depart : identite
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(iterations), float(eps))  # arret : eps OU nb d'iterations
    # ECC cherche la matrice qui maximise la correlation entre reference et image deformee ; 5 = flou gaussien avant calcul
    _, matrix = cv2.findTransformECC(_gray_float(reference), _gray_float(image), matrix, MOTIONS[motion], criteria, None, 5)
    return matrix  # renvoie la matrice trouvee


# homographie par points caracteristiques ORB ; renvoie une matrice 3x3
def orb_homography(image: np.ndarray, reference: np.ndarray, n_features: int = 2000, ransac_threshold: float = 3.0) -> np.ndarray:
    """Homographie image -> reference par points ORB et RANSAC (convention WARP_INVERSE_MAP, comme ecc_transform)."""
    orb = cv2.ORB_create(nfeatures=int(n_features))  # detecteur ORB : trouve des coins remarquables et les decrit
    gray_ref = to_uint8(reference if reference.ndim == 2 else cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY))  # reference en gris 8 bits
    gray_img = to_uint8(image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))  # image en gris 8 bits
    kp_ref, des_ref = orb.detectAndCompute(gray_ref, None)  # points (kp) et descripteurs (des) de la reference
    kp_img, des_img = orb.detectAndCompute(gray_img, None)  # points et descripteurs de l'image
    if des_ref is None or des_img is None or len(kp_ref) < 4 or len(kp_img) < 4:  # il faut au moins 4 points pour une homographie
        raise ValueError("orb : trop peu de points caracteristiques (image trop uniforme ?)")  # erreur claire
    # associe chaque point de la reference a son plus proche dans l'image (distance de Hamming, verification croisee), tri par distance
    matches = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des_ref, des_img), key=lambda m: m.distance)
    if len(matches) < 4:  # pas assez de correspondances
        raise ValueError("orb : moins de 4 correspondances")  # erreur claire
    src = np.float32([kp_ref[m.queryIdx].pt for m in matches])  # coordonnees des points dans la reference
    dst = np.float32([kp_img[m.trainIdx].pt for m in matches])  # coordonnees des points correspondants dans l'image
    # RANSAC : ignore les fausses correspondances ; ransac_threshold = ecart tolere en pixels ; reference -> image = carte inverse
    matrix, _ = cv2.findHomography(src, dst, cv2.RANSAC, float(ransac_threshold))
    if matrix is None:  # aucune homographie trouvee
        raise ValueError("orb : homographie introuvable")  # erreur claire
    return matrix  # renvoie la matrice 3x3


# aligne image sur reference avec la methode choisie ; renvoie l'image alignee
def align_to(image: np.ndarray, reference: np.ndarray, method: str = "ecc", motion: str = "euclidean") -> np.ndarray:
    """Superpose image sur reference (meme taille que la reference en sortie)."""
    if method == "none":  # pas d'alignement demande
        return image  # image inchangee
    matrix = ecc_transform(image, reference, motion) if method == "ecc" else orb_homography(image, reference)  # estime la matrice
    return _warp(image, matrix, reference.shape[:2])  # deforme l'image a la taille de la reference


@register("align_ecc", family="alignement")  # inscrit l'operation "align_ecc", famille alignement
# recale l'image sur une reference par ECC ; renvoie l'image recalee
def align_ecc(image: np.ndarray, reference_path: str = "", motion: str = "euclidean", iterations: int = 100) -> np.ndarray:
    """Recale l'image sur une image de reference par ECC (petits decalages, rotations faibles) ; sortie a la taille de la reference.

    OpenCvSharp : Cv2.FindTransformECC puis Cv2.WarpAffine (ou WarpPerspective) avec WarpInverseMap.
    """
    if not reference_path:  # chemin de reference vide
        raise ValueError("align_ecc : reference_path est obligatoire")  # erreur claire
    reference = _reference(reference_path, image.ndim == 2)  # reference lue (une seule fois grace au cache)
    return _warp(image, ecc_transform(image, reference, motion, iterations), reference.shape[:2])  # estime puis applique la matrice


@register("align_orb", family="alignement")  # inscrit l'operation "align_orb"
# recale l'image par points ORB + homographie ; renvoie l'image recalee
def align_orb(image: np.ndarray, reference_path: str = "", n_features: int = 2000) -> np.ndarray:
    """Recale l'image sur une reference par points ORB + homographie RANSAC (deplacements plus grands, perspective).

    OpenCvSharp : ORB.Create, BFMatcher, Cv2.FindHomography, Cv2.WarpPerspective.
    """
    if not reference_path:  # chemin de reference vide
        raise ValueError("align_orb : reference_path est obligatoire")  # erreur claire
    reference = _reference(reference_path, image.ndim == 2)  # reference lue (cache)
    return _warp(image, orb_homography(image, reference, n_features), reference.shape[:2])  # estime puis applique l'homographie


@register("absdiff_reference", family="alignement")  # inscrit l'operation "absdiff_reference"
# difference absolue avec une piece de reference ; renvoie l'image des ecarts
def absdiff_reference(image: np.ndarray, reference_path: str = "", align: str = "ecc") -> np.ndarray:
    """Difference absolue avec une piece de reference (alignement ecc, orb ou none au prealable) : les ecarts ressortent en clair.

    OpenCvSharp : Cv2.Absdiff(src, reference, dst).
    """
    if not reference_path:  # chemin de reference vide
        raise ValueError("absdiff_reference : reference_path est obligatoire")  # erreur claire
    reference = _reference(reference_path, image.ndim == 2)  # reference lue (cache)
    if align not in ("ecc", "orb", "none"):  # methode d'alignement inconnue
        raise ValueError("align : ecc, orb ou none")  # erreur claire
    if reference.dtype != image.dtype:  # types differents (ex. 8 bits contre 16 bits) : difference sans sens
        raise ValueError(f"absdiff_reference : image {image.dtype} et reference {reference.dtype} de types differents")  # erreur
    aligned = align_to(image, reference, align)  # aligne l'image sur la reference
    if aligned.shape != reference.shape:  # tailles differentes apres alignement (cas align="none")
        raise ValueError(f"absdiff_reference : image {aligned.shape} et reference {reference.shape} de tailles differentes")  # erreur
    return cv2.absdiff(aligned, reference)  # |image - reference| pixel par pixel : les defauts ressortent en clair
