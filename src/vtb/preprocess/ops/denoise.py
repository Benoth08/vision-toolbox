"""Debruitage : reduire le bruit de capteur sans effacer les petits defauts.

Piege de toute la famille : un debruitage fort efface aussi les defauts fins (piqures, micro-
fissures). Le reglage se valide sur la metrique de validation, pas a l'oeil.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes

import cv2  # OpenCV : lecture, ecriture et traitement d'images
import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)
from scipy.ndimage import median_filter  # filtre median de scipy (accepte toutes les tailles en 16 bits)

from vtb.preprocess.dtypes import as_float, from_float, per_channel  # aller-retour vers les flottants ; application par canal
from vtb.preprocess.registry import register  # decorateur qui inscrit l'operation dans le registre


def _odd(value: int) -> int:  # renvoie l'entier impair le plus proche par le haut (taille de fenetre valide)
    value = int(value)  # convertit en entier (le YAML peut donner 3.0)
    return value if value % 2 == 1 else value + 1  # deja impair : inchange ; pair : +1 (une fenetre a besoin d'un centre)


@register("gaussian_blur", family="debruitage")  # inscrit l'operation sous le nom "gaussian_blur", famille debruitage
def gaussian_blur(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:  # floute l'image ; renvoie l'image floutee
    """Flou gaussien : bruit fin et uniforme ; adoucit aussi les bords.

    OpenCvSharp : Cv2.GaussianBlur(src, dst, new Size(0, 0), sigma).
    """
    # taille de noyau (0, 0) = deduite de sigma (largeur du flou) ; BORDER_REFLECT = bords completes par effet miroir
    return cv2.GaussianBlur(image, (0, 0), sigmaX=float(sigma), borderType=cv2.BORDER_REFLECT)


@register("median_blur", family="debruitage")  # inscrit l'operation "median_blur"
def median_blur(image: np.ndarray, ksize: int = 3) -> np.ndarray:  # filtre median ; renvoie l'image filtree
    """Filtre median : retire les pixels isoles (poussieres, sel et poivre) en gardant les bords nets.

    OpenCV n'accepte les grandes fenetres qu'en 8 bits ; en 16 bits au-dela de 5 on passe par
    scipy (meme resultat, plus lent). OpenCvSharp : Cv2.MedianBlur(src, dst, ksize).
    """
    ksize = _odd(ksize)  # taille de fenetre impaire obligatoire
    if image.dtype == np.uint8 or ksize <= 5:  # cas gere par OpenCV (8 bits, ou petite fenetre en 16 bits)
        return cv2.medianBlur(image, ksize)  # chaque pixel remplace par la mediane de son voisinage
    return per_channel(lambda c: median_filter(c, size=ksize, mode="reflect"), image)  # sinon scipy, canal par canal


@register("bilateral", family="debruitage")  # inscrit l'operation "bilateral"
# filtre bilateral ; renvoie l'image lissee en gardant les contours
def bilateral(image: np.ndarray, diameter: int = 7, sigma_color: float = 25.0, sigma_space: float = 5.0) -> np.ndarray:
    """Filtre bilateral : lisse les zones homogenes et preserve les contours (sigma_color en niveaux sur 0-255).

    OpenCV le fait en 8 bits ou en flottant : en 16 bits on calcule en flottant dans [0, 1] avec
    sigma_color ramene a la meme echelle. OpenCvSharp : Cv2.BilateralFilter(src, dst, d, sigmaColor, sigmaSpace).
    """
    if image.dtype == np.uint8:  # cas 8 bits : OpenCV direct
        # diameter = taille du voisinage ; sigma_color = ecart de niveau tolere ; sigma_space = portee spatiale
        return cv2.bilateralFilter(image, int(diameter), float(sigma_color), float(sigma_space))
    x, _ = as_float(image)  # autres types : passage en flottant dans [0, 1]
    # sigma_color divise par 255 pour rester a la meme echelle que l'image dans [0, 1], puis retour au type d'origine
    return from_float(cv2.bilateralFilter(x, int(diameter), float(sigma_color) / 255.0, float(sigma_space)), image)


@register("nl_means", family="debruitage")  # inscrit l'operation "nl_means"
# moyennes non locales ; renvoie l'image debruitee
def nl_means(image: np.ndarray, strength: float = 7.0, template_window: int = 7, search_window: int = 21) -> np.ndarray:
    """Moyennes non locales : le debruitage le plus fort de la liste, et le plus lent (a reserver hors temps reel).

    strength (h) est en niveaux sur 0-255 ; il est remis a l'echelle en 16 bits.
    OpenCvSharp : Cv2.FastNlMeansDenoising / Cv2.FastNlMeansDenoisingColored.
    """
    tw, sw = _odd(template_window), _odd(search_window)  # tailles impaires : petit motif compare, zone de recherche
    if image.dtype == np.uint8:  # cas 8 bits
        if image.ndim == 2:  # image grise
            return cv2.fastNlMeansDenoising(image, None, float(strength), tw, sw)  # debruitage version gris
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # la version couleur passe en LAB en supposant du BGR
        # debruite (force identique pour luminance et couleur), puis remet en RGB
        return cv2.cvtColor(cv2.fastNlMeansDenoisingColored(bgr, None, float(strength), float(strength), tw, sw), cv2.COLOR_BGR2RGB)
    if image.dtype != np.uint16:  # ni 8 ni 16 bits : non gere
        raise ValueError(f"nl_means attend uint8 ou uint16, recu {image.dtype}")  # erreur claire
    h16 = [float(strength) * 257.0]  # force remise a l'echelle 16 bits : 65535 / 255 = 257

    def one(channel: np.ndarray) -> np.ndarray:  # debruite un seul canal 16 bits ; renvoie le canal debruite
        # en 16 bits OpenCV exige la norme L1 (somme des ecarts absolus) pour comparer les motifs
        return cv2.fastNlMeansDenoising(channel, h=h16, templateWindowSize=tw, searchWindowSize=sw, normType=cv2.NORM_L1)

    return per_channel(one, image)  # applique a chaque canal
