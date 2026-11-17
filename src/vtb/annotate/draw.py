"""Dessin des annotations sur une image : la premiere verification de toute conversion.

Toujours regarder quelques images annotees apres une lecture ou une conversion : un format mal
lu (coordonnees normalisees prises pour des pixels, x et y inverses) se voit en une seconde.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[str], X | None) partout

import zlib  # crc32 : somme de controle d'un texte, stable d'une execution a l'autre (couleur des classes)

import cv2  # OpenCV : lecture, ecriture, dessin et traitement d'images
import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)

from vtb.annotate.schema import ImageAnnotation  # format pivot : toutes les etiquettes d'une image

PALETTE = [  # 9 couleurs bien distinctes (rouge, vert, bleu de 0 a 255), une par classe, reutilisees en boucle
    (230, 25, 75),  # rouge
    (60, 180, 75),  # vert
    (255, 225, 25),  # jaune
    (0, 130, 200),  # bleu
    (245, 130, 48),  # orange
    (145, 30, 180),  # violet
    (70, 240, 240),  # cyan
    (240, 50, 230),  # magenta
    (210, 245, 60),  # vert citron
]


def color_of(label: str, class_names: list[str]) -> tuple[int, int, int]:  # renvoie la couleur (R, G, B) d'une classe
    """Couleur RGB stable par classe (meme couleur d'une image a l'autre)."""
    # position de la classe dans la liste ; classe inconnue : nombre tire de son nom par crc32
    # (somme de controle stable d'une execution a l'autre, contrairement a hash())
    index = class_names.index(label) if label in class_names else zlib.crc32(str(label).encode())
    return PALETTE[index % len(PALETTE)]  # % (modulo) : apres la 9e classe, on recommence au debut de la palette


# superpose une carte de chaleur coloree a une image ; renvoie la nouvelle image RGB
def overlay_heatmap(
    image: np.ndarray,  # image de fond (niveaux de gris ou couleur)
    heat: np.ndarray,  # carte de valeurs (score d'anomalie, Grad-CAM...), de n'importe quelle taille
    alpha: float = 0.45,  # transparence de la carte : 0 = invisible, 1 = cache l'image
    vmin: float | None = None,  # valeur qui donnera le bleu ; None = minimum de la carte
    vmax: float | None = None,  # valeur qui donnera le rouge ; None = maximum de la carte
) -> np.ndarray:  # renvoie un tableau hauteur x largeur x 3 (RGB, 0-255)
    """Superpose une carte (anomalie, Grad-CAM) en fausses couleurs ; vmin / vmax fixes = meme echelle pour toutes les images."""
    rgb = image if image.ndim == 3 else np.stack([image] * 3, axis=-1)  # gris (2D) -> copie sur 3 canaux pour etre en couleur
    # si l'image n'est pas en octets 0-255 (ex. flottants), on la ramene a cette echelle
    rgb = rgb if rgb.dtype == np.uint8 else (rgb.astype(np.float32) / max(float(rgb.max()), 1.0) * 255).astype(np.uint8)
    # agrandit (ou reduit) la carte a la taille exacte de l'image ; INTER_LINEAR = interpolation lissee
    heat = cv2.resize(heat.astype(np.float32), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    low = float(heat.min()) if vmin is None else vmin  # bas de l'echelle : impose ou pris dans la carte
    high = float(heat.max()) if vmax is None else vmax  # haut de l'echelle : impose ou pris dans la carte
    scaled = np.clip((heat - low) / max(high - low, 1e-9), 0, 1)  # ramene la carte entre 0 et 1 (1e-9 evite /0)
    # applique la palette JET (bleu -> rouge) ; OpenCV donne du BGR, on repasse en RGB
    colored = cv2.cvtColor(cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb, 1.0 - alpha, colored, alpha, 0.0)  # melange : (1 - alpha) x image + alpha x carte


# dessine toutes les annotations d'une image ; renvoie une copie RGB dessinee (l'image d'origine n'est pas modifiee)
def draw_annotation(
    image: np.ndarray,  # image a annoter (niveaux de gris ou couleur)
    ann: ImageAnnotation,  # annotations de cette image
    class_names: list[str] | None = None,  # liste des classes pour fixer les couleurs ; None = classes de l'image
    thickness: int = 2,  # epaisseur des traits en pixels
    show_labels: bool = True,  # True : ecrit le nom de la classe (et le score) au-dessus de chaque objet
) -> np.ndarray:  # renvoie l'image dessinee, tableau hauteur x largeur x 3
    """Copie RGB de l'image avec boites, boites orientees et polygones dessines (scores affiches s'ils existent)."""
    canvas = image.copy() if image.ndim == 3 else np.stack([image] * 3, axis=-1).copy()  # copie en couleur : on dessine dessus
    if canvas.dtype != np.uint8:  # si l'image n'est pas en octets 0-255
        canvas = (canvas.astype(np.float32) / max(float(canvas.max()), 1.0) * 255).astype(np.uint8)  # ramene a 0-255
    names = class_names or sorted(set(ann.object_labels()))  # classes donnees, sinon celles presentes dans l'image

    # ecrit le texte d'un objet ; fonction interne qui voit canvas et show_labels ; ne renvoie rien
    def caption(label: str, score: float | None, x: float, y: float, color: tuple[int, int, int]) -> None:
        if show_labels:  # seulement si l'affichage des noms est demande
            text = label if score is None else f"{label} {score:.2f}"  # nom seul, ou nom + score a 2 decimales
            # ecrit le texte 4 px au-dessus de l'objet (au moins a y = 10 pour rester dans l'image) ; LINE_AA = lisse
            cv2.putText(canvas, text, (int(x), max(int(y) - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    for box in ann.boxes:  # pour chaque boite alignee
        color = color_of(box.label, names)  # couleur de sa classe
        # trace le rectangle entre les deux coins (OpenCV veut des pixels entiers)
        cv2.rectangle(canvas, (int(round(box.x1)), int(round(box.y1))), (int(round(box.x2)), int(round(box.y2))), color, thickness)
        caption(box.label, box.score, box.x1, box.y1, color)  # ecrit le nom au-dessus du coin haut gauche
    for rbox in ann.rotated:  # pour chaque boite orientee
        color = color_of(rbox.label, names)  # couleur de sa classe
        corners = np.round(np.array(rbox.corners())).astype(np.int32)  # 4 coins arrondis en entiers (tableau 4 x 2)
        cv2.polylines(canvas, [corners], True, color, thickness, cv2.LINE_AA)  # trace le contour ferme (True = on referme)
        caption(rbox.label, rbox.score, float(corners[:, 0].min()), float(corners[:, 1].min()), color)  # nom en haut a gauche
    for poly in ann.polygons:  # pour chaque polygone
        color = color_of(poly.label, names)  # couleur de sa classe
        points = np.round(np.array(poly.points)).astype(np.int32)  # sommets arrondis en entiers
        cv2.polylines(canvas, [points], True, color, thickness, cv2.LINE_AA)  # trace le contour ferme
        caption(poly.label, poly.score, float(points[:, 0].min()), float(points[:, 1].min()), color)  # nom en haut a gauche
    return canvas  # renvoie l'image dessinee
