"""Geometrie commune aux formats : boites orientees, IoU, aires.

Les formats ne decrivent pas une boite orientee de la meme facon :
- toolbox (OpenCV) : centre, largeur, hauteur, angle en degres ;
- Label Studio     : coin "haut gauche" de la boite non tournee, taille, rotation horaire autour de ce coin ;
- CVAT             : boite non tournee (xtl, ytl, xbr, ybr) et rotation horaire autour du centre ;
- labelme / X-AnyLabeling : les quatre coins.
Toutes ces conventions passent par le meme repere : un coin d'origine p0, un vecteur largeur u
et un vecteur hauteur v (axes image : x vers la droite, y vers le bas, donc un angle positif
tourne dans le sens horaire a l'ecran). Les conversions sont exactes, sans minAreaRect.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[str], X | None) partout

import math  # fonctions mathematiques de base : cos, sin, atan2, hypot, conversion degres/radians

import numpy as np  # numpy : calcul sur des tableaux de nombres

from vtb.annotate.schema import Box, RotatedBox  # Box : boite alignee sur les axes ; RotatedBox : boite tournee d'un angle


# repere d'une boite orientee ; renvoie (coin d'origine p0, vecteur largeur u, vecteur hauteur v)
def rotated_frame(rbox: RotatedBox) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """(p0, u, v) : coin d'origine, vecteur largeur (longueur w), vecteur hauteur (longueur h)."""
    theta = math.radians(rbox.angle)  # cos et sin attendent des radians : on convertit l'angle donne en degres
    u = (rbox.w * math.cos(theta), rbox.w * math.sin(theta))  # vecteur le long de la largeur, tourne de l'angle
    v = (-rbox.h * math.sin(theta), rbox.h * math.cos(theta))  # vecteur le long de la hauteur, perpendiculaire a u
    p0 = (rbox.cx - u[0] / 2 - v[0] / 2, rbox.cy - u[1] / 2 - v[1] / 2)  # coin d'origine : on recule du centre d'un demi u et v
    return p0, u, v  # renvoie le repere complet de la boite


def ordered_corners(rbox: RotatedBox) -> list[tuple[float, float]]:  # renvoie les 4 coins (x, y) dans le sens horaire
    """Les quatre coins dans l'ordre p0, p0 + u, p0 + u + v, p0 + v (sens horaire a l'ecran)."""
    p0, u, v = rotated_frame(rbox)  # recupere le coin d'origine et les deux vecteurs cotes
    # coins : p0, puis p0 + largeur, puis p0 + largeur + hauteur, puis p0 + hauteur
    return [p0, (p0[0] + u[0], p0[1] + u[1]), (p0[0] + u[0] + v[0], p0[1] + u[1] + v[1]), (p0[0] + v[0], p0[1] + v[1])]


# cree une boite orientee a partir du coin d'origine (convention Label Studio) ; renvoie une RotatedBox centree
def rotated_from_origin(label: str, x0: float, y0: float, w: float, h: float, angle: float, score: float | None = None) -> RotatedBox:
    """Boite orientee a partir du coin d'origine (convention Label Studio)."""
    theta = math.radians(angle)  # conversion degres -> radians pour cos et sin
    cx = x0 + (w * math.cos(theta) - h * math.sin(theta)) / 2  # centre x = coin + moitie de (u + v), composante x
    cy = y0 + (w * math.sin(theta) + h * math.cos(theta)) / 2  # centre y = coin + moitie de (u + v), composante y
    return RotatedBox(label, cx, cy, w, h, angle % 360.0, score)  # angle ramene entre 0 et 360 degres


# cree une boite orientee a partir de ses quatre coins ordonnes ; renvoie une RotatedBox centree
def rotated_from_corners(label: str, corners: list[tuple[float, float]], score: float | None = None) -> RotatedBox:
    """Boite orientee a partir de quatre coins ordonnes p0, p1, p2, p3 (largeur = p0p1, hauteur = p0p3)."""
    (x0, y0), (x1, y1), _, (x3, y3) = corners[:4]  # garde p0, p1 et p3 ; p2 (oppose a p0) n'est pas necessaire
    w, h = math.hypot(x1 - x0, y1 - y0), math.hypot(x3 - x0, y3 - y0)  # hypot = longueur d'un segment (Pythagore)
    angle = math.degrees(math.atan2(y1 - y0, x1 - x0))  # angle du cote largeur par rapport a l'horizontale, en degres
    cx, cy = float(np.mean([p[0] for p in corners[:4]])), float(np.mean([p[1] for p in corners[:4]]))  # centre = moyenne des coins
    return RotatedBox(label, cx, cy, w, h, angle % 360.0, score)  # angle ramene entre 0 et 360 degres


def box_iou(a: Box, b: Box) -> float:  # IoU de deux boites alignees : 0 = disjointes, 1 = identiques
    """Intersection sur union de deux boites alignees."""
    iw = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))  # largeur de la zone commune (0 si pas de chevauchement)
    ih = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))  # hauteur de la zone commune (0 si pas de chevauchement)
    inter = iw * ih  # aire de l'intersection
    union = a.width * a.height + b.width * b.height - inter  # aire de l'union : on retire la partie comptee deux fois
    return inter / union if union > 0 else 0.0  # IoU ; deux boites d'aire nulle donnent 0


def polygon_area(points: list[tuple[float, float]]) -> float:  # aire d'un polygone donne par ses sommets, en pixels carres
    """Aire d'un polygone simple (formule du lacet)."""
    if len(points) < 3:  # moins de 3 points : pas une surface
        return 0.0  # aire nulle
    xs, ys = np.array([p[0] for p in points]), np.array([p[1] for p in points])  # separe les x et les y en deux tableaux
    # formule du lacet : demi-valeur absolue de la somme des produits croises des sommets voisins (roll = decalage d'un cran)
    return float(0.5 * abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))))


def same_corners(a: RotatedBox, b: RotatedBox, tol: float = 1e-3) -> bool:  # meme rectangle a tol pixel pres ? renvoie True/False
    """Deux boites orientees decrivent-elles le meme rectangle (quel que soit le coin d'origine choisi) ?"""
    ca, cb = np.array(ordered_corners(a)), np.array(ordered_corners(b))  # coins des deux boites, tableaux 4 x 2
    return all(np.min(np.linalg.norm(cb - p, axis=1)) < tol for p in ca)  # chaque coin de a a un coin de b tout proche
