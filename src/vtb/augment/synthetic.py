"""Defauts synthetiques : fabriquer des images defectueuses (et leur masque) a partir d'images saines.

A quoi ca sert, quand les vrais defauts sont rares :
- verifier de bout en bout une chaine de detection d'anomalies avant d'avoir des defauts reels ;
- entrainer des methodes qui s'appuient sur des anomalies simulees (DRAEM, CutPaste) ;
- enrichir un jeu de segmentation (avec prudence : un defaut synthetique trop facile apprend au
  modele a reconnaitre la synthese, pas le defaut).
Chaque fonction renvoie (image, masque) : image du meme type que l'entree, masque uint8 {0, 1}.
On ne mesure JAMAIS une performance finale sur des defauts synthetiques : le test reste reel.
"""

from __future__ import annotations  # permet d'ecrire les types (ex. "int | None") sans qu'ils soient evalues au chargement

import cv2  # OpenCV : traitement d'images (flou, rotation, dessin de courbes)
import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)

# noms exportes par "from vtb.augment.synthetic import *" (les fonctions publiques du module)
__all__ = ["apply_vignetting", "cutpaste", "perlin_anomaly", "perlin_noise", "synthetic_scratch"]


# Le module est autonome (copiable sans le reste de la toolbox) : ces trois aides reprennent
# vtb.preprocess.dtypes et vtb.preprocess.ops.illumination.apply_vignetting a l'identique.
def _peak(image: np.ndarray) -> float:  # renvoie la plus grande valeur possible d'un pixel selon le type de l'image
    """255 en 8 bits, 65535 en 16 bits, 1 pour un flottant suppose dans [0, 1]."""
    return {np.dtype(np.uint8): 255.0, np.dtype(np.uint16): 65535.0}.get(image.dtype, 1.0)  # type inconnu du dict : 1.0


def as_float(image: np.ndarray) -> tuple[np.ndarray, float]:  # renvoie l'image en decimaux dans [0, 1] et la valeur max d'origine
    """Image -> (float32 dans [0, 1], valeur maximale du type d'origine)."""
    peak = _peak(image)  # 255, 65535 ou 1
    return image.astype(np.float32) / peak, peak  # division par le maximum : tous les types sont ramenes a la meme echelle


def from_float(values: np.ndarray, like: np.ndarray) -> np.ndarray:  # renvoie l'image reconvertie dans le type de `like`
    """float32 dans [0, 1] -> meme type que `like` (ecrete en 8 et 16 bits)."""
    if like.dtype in (np.uint8, np.uint16):  # si l'image d'origine est en entiers 8 ou 16 bits
        # ecrete dans [0, 1], remet a l'echelle du type, arrondit (+ 0.5) puis convertit en entiers
        return (np.clip(values, 0.0, 1.0) * _peak(like) + 0.5).astype(like.dtype)
    return values.astype(like.dtype)  # sinon : image en decimaux, simple changement de type


# renvoie l'image avec des bords assombris (vignettage) ; strength : force, center : centre (x, y) en fraction
def apply_vignetting(image: np.ndarray, strength: float = 0.6, center: tuple[float, float] = (0.5, 0.5)) -> np.ndarray:
    """Ajoute un vignettage radial artificiel (assombrissement des bords) : pour tester et montrer la correction."""
    height, width = image.shape[:2]  # hauteur et largeur (les deux premieres dimensions)
    yy, xx = np.mgrid[0:height, 0:width]  # numero de ligne et de colonne de chaque pixel
    r = np.sqrt(((xx / width - center[0]) ** 2 + (yy / height - center[1]) ** 2) / 0.5)  # distance au centre, ~1 dans les coins
    gain = (1.0 - strength * np.clip(r, 0, 1) ** 2).astype(np.float32)  # gain 1 au centre, 1 - strength dans les coins
    x, _ = as_float(image)  # image en decimaux dans [0, 1]
    # applique le gain (repete sur les canaux si couleur) puis revient au type d'origine
    return from_float(x * (gain[..., None] if image.ndim == 3 else gain), image)


def _rng(seed: np.random.Generator | int | None) -> np.random.Generator:  # renvoie un generateur aleatoire numpy
    # generateur deja fourni : utilise tel quel ; sinon nouveau generateur (graine entiere = reproductible, None = au hasard)
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


def _expand(mask: np.ndarray, image: np.ndarray) -> np.ndarray:  # renvoie le masque mis a la forme de l'image
    """Masque (H, W) -> forme de l'image (ajoute l'axe des canaux en couleur)."""
    return mask[..., None] if image.ndim == 3 else mask  # (H, W, 1) pour une image couleur : il s'applique aux 3 canaux


# renvoie un tableau (hauteur, largeur) de bruit lisse ; cells : nombre de cases par cote (petit = grosses taches)
def perlin_noise(shape: tuple[int, int], cells: int = 4, rng: np.random.Generator | int | None = None) -> np.ndarray:
    """Bruit de Perlin 2D (gradients aleatoires sur une grille de `cells` cases), valeurs a peu pres dans [-1, 1].

    C'est un bruit lisse et organique : seuille, il donne des taches aux contours irreguliers,
    plus credibles qu'un rectangle pour simuler une salissure ou une corrosion.
    """
    generator = _rng(rng)  # generateur aleatoire
    height, width = shape  # taille voulue
    grid_h = int(np.ceil(height / cells)) * cells  # taille multiple du nombre de cases, recadree a la fin
    grid_w = int(np.ceil(width / cells)) * cells  # idem pour la largeur
    angles = generator.uniform(0, 2 * np.pi, (cells + 1, cells + 1))  # un angle au hasard a chaque noeud de la grille
    gradients = np.stack([np.cos(angles), np.sin(angles)], axis=-1)  # vecteur de longueur 1 dans cette direction, par noeud
    yy, xx = np.mgrid[0:grid_h, 0:grid_w].astype(np.float64)  # coordonnees de chaque pixel
    gy, gx = yy / (grid_h / cells), xx / (grid_w / cells)  # coordonnees exprimees en cases (0 a cells)
    y0, x0 = np.floor(gy).astype(int), np.floor(gx).astype(int)  # case dans laquelle tombe chaque pixel (coin haut-gauche)
    fy, fx = gy - y0, gx - x0  # position du pixel a l'interieur de sa case (entre 0 et 1)

    def dot(oy: int, ox: int) -> np.ndarray:  # contribution du coin (oy, ox) de la case, pour chaque pixel
        g = gradients[y0 + oy, x0 + ox]  # vecteur aleatoire de ce coin
        return g[..., 0] * (fx - ox) + g[..., 1] * (fy - oy)  # produit scalaire avec le vecteur coin -> pixel

    fade = lambda t: t * t * t * (t * (t * 6 - 15) + 10)  # noqa: E731  (lissage de Perlin : derivees nulles aux noeuds)
    u, v = fade(fx), fade(fy)  # poids de melange horizontal et vertical, lisses
    top = dot(0, 0) * (1 - u) + dot(0, 1) * u  # melange des deux coins du haut
    bottom = dot(1, 0) * (1 - u) + dot(1, 1) * u  # melange des deux coins du bas
    noise = (top * (1 - v) + bottom * v) * np.sqrt(2)  # melange haut / bas ; x racine de 2 pour s'etaler sur ~[-1, 1]
    return noise[:height, :width].astype(np.float32)  # recadre a la taille demandee


def perlin_anomaly(  # ajoute une tache de forme organique ; renvoie (image modifiee, masque de la tache)
    image: np.ndarray,  # image saine (H, W) ou (H, W, C)
    rng: np.random.Generator | int | None = None,  # generateur aleatoire ou graine
    texture: np.ndarray | None = None,  # texture etrangere a incruster ; None = bruit lisse aleatoire
    threshold: float = 0.4,  # seuil sur le bruit de Perlin : plus haut = taches plus petites
    opacity: tuple[float, float] = (0.4, 0.9),  # intervalle d'opacite de la tache (0 = invisible, 1 = opaque)
    cells: tuple[int, int] = (2, 6),  # intervalle du nombre de cases du bruit (petit = grosses taches)
) -> tuple[np.ndarray, np.ndarray]:  # renvoie l'image et le masque
    """Anomalie facon DRAEM : une tache de forme Perlin ou l'image est melangee a une texture etrangere.

    texture : image de meme taille et de meme type (une autre piece, une texture de la base DTD) ;
    sans texture, un bruit lisse aleatoire est utilise. opacity : intervalle de transparence.
    """
    generator = _rng(rng)  # generateur aleatoire
    height, width = image.shape[:2]  # taille de l'image
    mask = np.zeros((height, width), np.uint8)  # masque vide au depart
    for _ in range(10):  # un seuil trop haut peut ne rien laisser : on retire le bruit
        # bruit de Perlin avec un nombre de cases tire dans l'intervalle
        noise = perlin_noise((height, width), int(generator.integers(cells[0], cells[1] + 1)), generator)
        mask = (noise > threshold).astype(np.uint8)  # 1 la ou le bruit depasse le seuil : la forme de la tache
        if mask.any():  # si la tache n'est pas vide
            break  # on la garde
    else:  # ce "else" de boucle s'execute si aucun des 10 essais n'a donne de tache
        mask[height // 3 : height // 2, width // 3 : width // 2] = 1  # tache de secours : un rectangle pres du centre
    x, _ = as_float(image)  # image en decimaux dans [0, 1]
    if texture is None:  # si aucune texture n'est fournie
        # bruit aleatoire floute : une "salissure" lisse
        stain = cv2.GaussianBlur(generator.uniform(0, 1, x.shape).astype(np.float32), (0, 0), 3)
        # contraste renforce (x 6) autour d'un niveau moyen tire au hasard, garde dans [0, 1]
        foreign = np.clip((stain - stain.mean()) * 6 + generator.uniform(0.1, 0.9), 0, 1)
    else:  # sinon on utilise la texture fournie
        foreign, _ = as_float(texture)  # texture en decimaux dans [0, 1]
        if foreign.shape != x.shape:  # si la texture n'a pas la meme forme que l'image
            raise ValueError(f"texture {foreign.shape} et image {x.shape} de formes differentes")  # erreur claire
    beta = float(generator.uniform(*opacity))  # opacite tiree
    m = _expand(mask.astype(np.float32), image)  # masque en decimaux, a la forme de l'image
    # dans la tache : melange image / texture selon beta ; hors tache (m = 0) : image inchangee
    return from_float(x * (1 - beta * m) + beta * m * foreign, image), mask


def cutpaste(  # deplace un morceau de l'image ; renvoie (image modifiee, masque du morceau colle)
    image: np.ndarray,  # image saine
    rng: np.random.Generator | int | None = None,  # generateur aleatoire ou graine
    area: tuple[float, float] = (0.02, 0.15),  # surface du morceau en fraction de l'image
    aspect: tuple[float, float] = (0.3, 3.3),  # rapport largeur / hauteur du morceau
    rotate: bool = True,  # True : le morceau est tourne avant d'etre colle
) -> tuple[np.ndarray, np.ndarray]:  # renvoie l'image et le masque
    """CutPaste : decoupe un morceau de l'image, le tourne eventuellement et le colle ailleurs (une "piece" deplacee).

    Le defaut a la texture de la piece mais une position et une orientation incoherentes : c'est
    l'anomalie la plus difficile a voir, utile pour eprouver un detecteur.
    """
    generator = _rng(rng)  # generateur aleatoire
    height, width = image.shape[:2]  # taille de l'image
    patch_area = generator.uniform(*area) * height * width  # surface du morceau en pixels
    ratio = np.exp(generator.uniform(np.log(aspect[0]), np.log(aspect[1])))  # rapport largeur / hauteur tire en echelle log
    ph = int(np.clip(np.sqrt(patch_area / ratio), 4, height - 2))  # hauteur du morceau, entre 4 et hauteur - 2
    pw = int(np.clip(np.sqrt(patch_area * ratio), 4, width - 2))  # largeur du morceau, entre 4 et largeur - 2
    # coin haut-gauche de la zone source, tire pour que le morceau tienne dans l'image
    sy, sx = int(generator.integers(0, height - ph + 1)), int(generator.integers(0, width - pw + 1))
    patch = image[sy : sy + ph, sx : sx + pw].copy()  # copie du morceau decoupe
    patch_mask = np.ones((ph, pw), np.uint8)  # masque du morceau : tout a 1 au depart
    if rotate:  # si on tourne le morceau
        angle = float(generator.uniform(-45, 45))  # angle tire entre -45 et 45 degres
        center = (pw / 2, ph / 2)  # centre de rotation : milieu du morceau
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)  # matrice de rotation (echelle 1)
        # tourne le morceau ; les coins vides sont remplis par effet miroir
        patch = cv2.warpAffine(patch, matrix, (pw, ph), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        # tourne aussi son masque ; coins vides a 0 : seule la partie tournee sera collee
        patch_mask = cv2.warpAffine(patch_mask, matrix, (pw, ph), flags=cv2.INTER_NEAREST, borderValue=0)
    ty, tx = int(generator.integers(0, height - ph + 1)), int(generator.integers(0, width - pw + 1))  # position de collage
    out = image.copy()  # copie de l'image : l'originale reste intacte
    region = out[ty : ty + ph, tx : tx + pw]  # zone de destination
    keep = _expand(patch_mask.astype(bool), image)  # masque du morceau en vrai / faux, a la forme de l'image
    out[ty : ty + ph, tx : tx + pw] = np.where(keep, patch, region)  # colle le morceau la ou le masque vaut 1
    mask = np.zeros((height, width), np.uint8)  # masque complet, vide
    mask[ty : ty + ph, tx : tx + pw] = patch_mask  # place le masque du morceau a l'endroit du collage
    return out, mask  # renvoie l'image modifiee et le masque du defaut


def synthetic_scratch(  # dessine une rayure ; renvoie (image modifiee, masque de la rayure)
    image: np.ndarray,  # image saine
    rng: np.random.Generator | int | None = None,  # generateur aleatoire ou graine
    length: tuple[float, float] = (0.2, 0.6),  # longueur en fraction de la diagonale de l'image
    thickness: tuple[int, int] = (1, 3),  # epaisseur en pixels
    contrast: tuple[float, float] = (0.15, 0.4),  # ecart de luminosite avec la surface
    curvature: float = 0.15,  # courbure maximale (0 = rayure droite)
) -> tuple[np.ndarray, np.ndarray]:  # renvoie l'image et le masque
    """Rayure synthetique : une courbe fine, legerement arquee, plus claire ou plus sombre que la surface.

    length en fraction de la diagonale, thickness en pixels, contrast en fraction de la dynamique
    (le signe, clair ou sombre, est tire au hasard).
    """
    generator = _rng(rng)  # generateur aleatoire
    height, width = image.shape[:2]  # taille de l'image
    diagonal = float(np.hypot(height, width))  # longueur de la diagonale (Pythagore)
    span = generator.uniform(*length) * diagonal  # longueur de la rayure en pixels
    angle = generator.uniform(0, np.pi)  # direction de la rayure (0 a 180 degres, en radians)
    # centre de la rayure, pas trop pres des bords (entre 20 % et 80 % de la largeur et de la hauteur)
    center = np.array([generator.uniform(0.2, 0.8) * width, generator.uniform(0.2, 0.8) * height])
    direction = np.array([np.cos(angle), np.sin(angle)])  # vecteur unitaire le long de la rayure
    normal = np.array([-direction[1], direction[0]])  # vecteur perpendiculaire (pour courber la rayure)
    start, end = center - direction * span / 2, center + direction * span / 2  # extremites de la rayure
    bend = normal * generator.uniform(-curvature, curvature) * span  # decalage du milieu : donne la courbure
    t = np.linspace(0, 1, 50)[:, None]  # 50 positions de 0 a 1 le long de la courbe
    curve = (1 - t) ** 2 * start + 2 * (1 - t) * t * (center + bend) + t**2 * end  # courbe de Bezier quadratique
    canvas = np.zeros((height, width), np.uint8)  # toile noire pour dessiner la rayure
    cv2.polylines(  # trace la courbe en blanc (255) sur la toile
        canvas,  # image ou dessiner (modifiee sur place)
        [np.round(curve).astype(np.int32)],  # liste de courbes : ici une seule, points arrondis en pixels entiers
        False,  # courbe ouverte : on ne relie pas le dernier point au premier
        255,  # couleur du trait : blanc
        int(generator.integers(thickness[0], thickness[1] + 1)),  # epaisseur du trait, tiree au hasard
        cv2.LINE_AA,  # trait anti-crenele (bords adoucis)
    )
    soft = cv2.GaussianBlur(canvas.astype(np.float32) / 255.0, (0, 0), 0.7)  # rayure dans [0, 1], legerement adoucie
    sign = 1.0 if generator.random() < 0.5 else -1.0  # une chance sur deux : rayure claire ou sombre
    amount = sign * generator.uniform(*contrast)  # ecart de luminosite signe
    x, _ = as_float(image)  # image en decimaux dans [0, 1]
    out = x + amount * _expand(soft, image)  # ajoute la rayure a l'image
    return from_float(out, image), (soft > 0.25).astype(np.uint8)  # image au type d'origine ; masque = rayure assez marquee
