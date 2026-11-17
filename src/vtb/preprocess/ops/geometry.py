"""Operations geometriques : taille, region d'interet, redressement, correction d'optique.

Regle de production : la meme echelle a l'entrainement et en inference. Un modele entraine sur
des images reduites a 256 px ne reconnait pas un defaut vu a la resolution native.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes

import json  # lecture de fichiers JSON (ici la calibration de la camera)
from functools import lru_cache  # lru_cache : memorise le resultat d'une fonction (fichier lu une seule fois)
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import cv2  # OpenCV : lecture, ecriture et traitement d'images
import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)

from vtb.io.images import to_uint8  # conversion en 8 bits
from vtb.preprocess.registry import register  # decorateur qui inscrit l'operation dans le registre

# methodes d'interpolation (calcul des nouveaux pixels quand on change la taille) : plus proche voisin, lineaire, cubique, moyenne
INTERPOLATIONS = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC, "area": cv2.INTER_AREA}


def _size_hw(size: int | list[int] | tuple[int, int]) -> tuple[int, int]:  # renvoie (hauteur, largeur) a partir d'un entier ou d'une liste
    if isinstance(size, int):  # un seul entier = un carre
        return size, size  # hauteur = largeur = size
    return int(size[0]), int(size[1])  # sinon [hauteur, largeur]


@register("resize", family="geometrie")  # inscrit l'operation "resize", famille geometrie
def resize(image: np.ndarray, size: int | list[int] = 256, interpolation: str = "auto") -> np.ndarray:  # redimensionne ; renvoie l'image
    """Redimensionne a size = [hauteur, largeur] (ou un entier pour un carre) ; auto = area pour reduire, linear pour agrandir.

    INTER_AREA moyenne les pixels quand on reduit : c'est la seule interpolation qui ne cree pas
    de crenelage (moirage) sur les textures fines, frequentes sur les surfaces metalliques.
    """
    height, width = _size_hw(size)  # taille voulue
    if interpolation == "auto":  # choix automatique de l'interpolation
        shrinking = height * width < image.shape[0] * image.shape[1]  # True si on reduit l'image
        flag = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR  # reduction : moyenne ; agrandissement : lineaire
    else:  # interpolation imposee par la configuration
        flag = INTERPOLATIONS[interpolation]  # code OpenCV correspondant
    return cv2.resize(image, (width, height), interpolation=flag)  # attention : OpenCV attend (largeur, hauteur)


@register("crop", family="geometrie")  # inscrit l'operation "crop"
def crop(image: np.ndarray, x: int = 0, y: int = 0, width: int = 0, height: int = 0) -> np.ndarray:  # recadre ; renvoie la zone
    """Region d'interet fixe (x, y = coin haut gauche) ; width / height a 0 = jusqu'au bord."""
    h, w = image.shape[:2]  # hauteur et largeur de l'image
    x2 = w if width <= 0 else min(w, x + width)  # bord droit de la zone (sans depasser l'image)
    y2 = h if height <= 0 else min(h, y + height)  # bord bas de la zone
    if x >= x2 or y >= y2:  # zone vide
        raise ValueError(f"crop vide : x={x}, y={y}, width={width}, height={height} pour une image {w}x{h}")  # erreur claire
    return image[y:y2, x:x2].copy()  # copie de la zone (lignes y..y2, colonnes x..x2)


BORDERS = {"constant": cv2.BORDER_CONSTANT, "reflect": cv2.BORDER_REFLECT, "replicate": cv2.BORDER_REPLICATE}  # remplissages de bordure


@register("pad", family="geometrie")  # inscrit l'operation "pad"
def pad(  # ajoute une bordure ; renvoie l'image agrandie
    image: np.ndarray,  # image d'entree
    top: int = 0,  # pixels ajoutes en haut
    bottom: int = 0,  # pixels ajoutes en bas
    left: int = 0,  # pixels ajoutes a gauche
    right: int = 0,  # pixels ajoutes a droite
    mode: str = "constant",  # remplissage : constant, reflect (miroir) ou replicate (repete le bord)
    value: int = 0,  # valeur de remplissage en mode constant
) -> np.ndarray:  # renvoie l'image avec sa bordure
    """Ajoute une bordure (constante, miroir ou repetition du bord), par exemple pour atteindre un multiple de 32.

    OpenCvSharp : Cv2.CopyMakeBorder(src, dst, top, bottom, left, right, BorderTypes.Constant, value).
    """
    if mode not in BORDERS:  # mode inconnu
        raise ValueError(f"mode de bordure inconnu : {mode} ({sorted(BORDERS)})")  # erreur avec la liste des modes
    fill = (value,) * 3 if image.ndim == 3 else value  # en couleur : la meme valeur pour les 3 canaux
    return cv2.copyMakeBorder(image, int(top), int(bottom), int(left), int(right), BORDERS[mode], value=fill)  # ajoute la bordure


# calcule les parametres du letterbox ; renvoie (ratio, largeur, hauteur, decalage x, decalage y)
def letterbox_params(height: int, width: int, size: int) -> tuple[float, int, int, int, int]:
    """Parametres du letterbox : (ratio, nouvelle largeur, nouvelle hauteur, decalage x, decalage y).

    Sert a ramener des boites predites dans l'image d'origine : x_orig = (x - pad_x) / ratio.
    """
    # letterbox = reduire l'image sans la deformer pour qu'elle tienne dans un carre, puis combler le vide avec du gris
    ratio = min(size / height, size / width)  # facteur d'echelle : le cote le plus long tient juste dans le carre
    new_w, new_h = round(width * ratio), round(height * ratio)  # taille de l'image redimensionnee
    return ratio, new_w, new_h, (size - new_w) // 2, (size - new_h) // 2  # decalages = moitie de l'espace vide (image centree)


@register("letterbox", family="geometrie")  # inscrit l'operation "letterbox"
def letterbox(image: np.ndarray, size: int = 640, pad_value: int = 114) -> np.ndarray:  # renvoie une image carree size x size
    """Redimensionne sans deformer dans un carre de cote size, centre, bordure grise (convention YOLO : 114).

    OpenCvSharp : Cv2.Resize puis Cv2.CopyMakeBorder (ou copie dans une Mat remplie de pad_value).
    """
    height, width = image.shape[:2]  # taille d'origine
    _, new_w, new_h, pad_x, pad_y = letterbox_params(height, width, int(size))  # taille reduite et decalages
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)  # redimensionne sans deformer
    canvas = np.full((size, size, *image.shape[2:]), pad_value, dtype=image.dtype)  # carre rempli de gris (114), memes canaux
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized  # colle l'image au centre du carre
    return canvas  # renvoie le carre


def _foreground_mask(image: np.ndarray, threshold: float, invert: bool) -> np.ndarray:  # renvoie le masque 0/255 de l'objet
    """Masque binaire 8 bits de l'objet : seuil d'Otsu si threshold < 0, sinon seuil fixe en niveaux 0-255."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # passe en gris si besoin
    gray8 = to_uint8(gray)  # en 8 bits (le seuillage d'Otsu d'OpenCV l'exige)
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY  # invert : l'objet est plus sombre que le fond
    if threshold < 0:  # seuil automatique
        _, mask = cv2.threshold(gray8, 0, 255, flag | cv2.THRESH_OTSU)  # Otsu : choisit le seuil qui separe le mieux deux populations
    else:  # seuil fixe donne
        _, mask = cv2.threshold(gray8, float(threshold), 255, flag)  # pixels au-dessus du seuil -> 255, les autres -> 0
    return mask  # renvoie le masque


@register("auto_roi", family="geometrie")  # inscrit l'operation "auto_roi"
def auto_roi(  # region d'interet automatique ; renvoie la zone recadree
    image: np.ndarray,  # image d'entree
    mode: str = "largest_contour",  # "largest_contour" (plus grand objet) ou "columns" (bande verticale)
    threshold: float = -1.0,  # seuil de l'objet ; < 0 = seuil d'Otsu automatique
    margin: int = 8,  # marge en pixels ajoutee autour de la zone trouvee
    invert: bool = False,  # True : objet plus sombre que le fond
) -> np.ndarray:  # renvoie l'image recadree
    """Region d'interet automatique : boite du plus grand objet (largest_contour) ou bords gauche / droit d'une bande (columns).

    threshold < 0 : seuil d'Otsu ; invert : objet plus sombre que le fond. Le mode columns garde
    toute la hauteur et recadre la largeur entre les bords detectes de la bande (camera lineaire
    au-dessus d'une tole : le fond de part et d'autre n'a pas d'interet). Si rien n'est trouve,
    l'image est rendue entiere plutot que vide.
    OpenCvSharp : Cv2.Threshold, Cv2.FindContours, Cv2.BoundingRect.
    """
    mask = _foreground_mask(image, threshold, invert)  # masque de l'objet
    height, width = mask.shape  # dimensions
    if mode == "columns":  # mode bande verticale
        occupied = np.nonzero(mask.mean(axis=0) > 127)[0]  # colonnes dont plus de la moitie des pixels appartiennent a l'objet
        if occupied.size == 0:  # aucune colonne occupee
            return image.copy()  # image entiere plutot que vide
        x1, x2 = max(0, occupied[0] - margin), min(width, occupied[-1] + 1 + margin)  # premiere et derniere colonne, + marge
        return image[:, x1:x2].copy()  # toute la hauteur, largeur recadree
    if mode != "largest_contour":  # mode inconnu
        raise ValueError("mode : largest_contour ou columns")  # erreur claire
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # contours exterieurs des objets du masque
    if not contours:  # aucun objet
        return image.copy()  # image entiere
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))  # boite englobante du plus grand contour (par surface)
    x1, y1 = max(0, x - margin), max(0, y - margin)  # coin haut gauche avec marge (sans sortir de l'image)
    return image[y1 : min(height, y + h + margin), x1 : min(width, x + w + margin)].copy()  # zone recadree avec marge


@register("deskew", family="geometrie")  # inscrit l'operation "deskew"
# redresse l'objet a l'horizontale ; renvoie l'image tournee
def deskew(image: np.ndarray, threshold: float = -1.0, invert: bool = False, max_angle: float = 45.0) -> np.ndarray:
    """Redresse l'image : rotation qui remet l'axe principal de l'objet (cv2.minAreaRect) a l'horizontale.

    Angles au-dela de max_angle ignores (ambiguite d'un quart de tour). Les coins decouverts sont
    remplis par repetition du bord. OpenCvSharp : Cv2.MinAreaRect, Cv2.GetRotationMatrix2D, Cv2.WarpAffine.
    """
    points = cv2.findNonZero(_foreground_mask(image, threshold, invert))  # coordonnees de tous les pixels de l'objet
    if points is None:  # aucun pixel d'objet
        return image.copy()  # image inchangee
    (_, _), (w, h), angle = cv2.minAreaRect(points)  # plus petit rectangle tourne qui contient l'objet : centre, taille, angle
    if w < h:  # OpenCV renvoie l'angle du cote "largeur" : on se ramene a l'axe le plus long
        angle -= 90.0  # tourne d'un quart de tour
    angle = (angle + 90.0) % 180.0 - 90.0  # ramene l'angle entre -90 et 90 degres
    if abs(angle) > max_angle or abs(angle) < 1e-3:  # angle trop grand (ambigu) ou quasi nul
        return image.copy()  # pas de rotation
    height, width = image.shape[:2]  # dimensions
    rotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)  # matrice de rotation autour du centre, sans zoom
    # applique la rotation ; coins decouverts remplis en repetant le bord
    return cv2.warpAffine(image, rotation, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


@register("perspective", family="geometrie")  # inscrit l'operation "perspective"
# redresse une vue en biais ; renvoie un rectangle vu de face
def perspective(image: np.ndarray, points: list[list[float]] | None = None, width: int = 0, height: int = 0) -> np.ndarray:
    """Redresse une vue en biais : les 4 points (haut gauche, haut droit, bas droit, bas gauche) deviennent un rectangle width x height.

    Les points se mesurent une fois pour une camera fixe (cliquer les coins d'une plaque de
    reference). OpenCvSharp : Cv2.GetPerspectiveTransform puis Cv2.WarpPerspective.
    """
    if not points or len(points) != 4:  # il faut exactement 4 coins
        # erreur claire qui rappelle l'ordre attendu des coins
        raise ValueError("perspective : points doit contenir 4 coins [x, y] (haut gauche, haut droit, bas droit, bas gauche)")
    src = np.array(points, dtype=np.float32)  # coins dans l'image, en flottants
    # largeur de sortie : donnee, sinon la plus longue des aretes haute et basse
    out_w = int(width) or int(round(max(np.linalg.norm(src[1] - src[0]), np.linalg.norm(src[2] - src[3]))))
    # hauteur de sortie : donnee, sinon la plus longue des aretes gauche et droite
    out_h = int(height) or int(round(max(np.linalg.norm(src[3] - src[0]), np.linalg.norm(src[2] - src[1]))))
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)  # coins du rectangle d'arrivee
    matrix = cv2.getPerspectiveTransform(src, dst)  # matrice 3x3 qui envoie les 4 coins sur le rectangle
    # applique la deformation ; zones hors image remplies en repetant le bord
    return cv2.warpPerspective(image, matrix, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


@lru_cache(maxsize=4)  # garde en memoire les 4 dernieres calibrations lues
def load_calibration(path: str) -> tuple[np.ndarray, np.ndarray]:  # lit une calibration ; renvoie (matrice camera, distorsion)
    """Lit une calibration (JSON ecrit par vtb.classical.measure.save_calibration) : matrice camera et distorsion."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))  # lit et decode le JSON
    # matrice camera (focale, centre optique) et coefficients de distorsion de l'objectif, en flottants 64 bits
    return np.array(data["camera_matrix"], dtype=np.float64), np.array(data["dist_coeffs"], dtype=np.float64)


@register("undistort", family="geometrie")  # inscrit l'operation "undistort"
def undistort(  # corrige la distorsion de l'objectif ; renvoie l'image corrigee
    image: np.ndarray,  # image d'entree
    calibration_path: str = "",  # fichier JSON de calibration
    camera_matrix: list[list[float]] | None = None,  # ou bien : matrice camera 3x3 donnee directement
    dist_coeffs: list[float] | None = None,  # et coefficients de distorsion donnes directement
) -> np.ndarray:  # renvoie l'image sans distorsion
    """Corrige la distorsion de l'objectif (lignes droites qui se courbent sur les bords) a partir d'une calibration.

    calibration_path : JSON produit par vtb.classical.measure.calibrate_chessboard ; ou bien
    camera_matrix et dist_coeffs donnes directement. Indispensable avant une mesure en millimetres.
    OpenCvSharp : Cv2.Undistort(src, dst, cameraMatrix, distCoeffs).
    """
    if calibration_path:  # fichier de calibration donne
        matrix, coeffs = load_calibration(calibration_path)  # lu depuis le fichier (une seule fois grace au cache)
    elif camera_matrix is not None and dist_coeffs is not None:  # valeurs donnees directement
        matrix, coeffs = np.array(camera_matrix, dtype=np.float64), np.array(dist_coeffs, dtype=np.float64)  # converties en tableaux
    else:  # rien de fourni
        raise ValueError("undistort : donner calibration_path, ou camera_matrix et dist_coeffs")  # erreur claire
    return cv2.undistort(image, matrix, coeffs)  # redresse les lignes courbees par l'objectif
