"""Verifier une conversion : deux listes pivot decrivent-elles les memes objets ?

A utiliser apres chaque conversion de format sur un vrai jeu : un format mal lu ne leve pas
d'erreur, il donne des boites decalees ou vides, et l'entrainement "marche" quand meme.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[str], X | None) partout

import math  # fonctions mathematiques de base (ici math.dist : distance entre deux points)
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux
from typing import Literal  # Literal : type qui n'accepte que quelques valeurs texte precises

from vtb.annotate.geometry import same_corners  # teste si deux boites orientees decrivent le meme rectangle
from vtb.annotate.schema import ImageAnnotation  # format pivot : toutes les etiquettes d'une image

PathMode = Literal["exact", "name", "resolved"]  # trois facons de comparer deux chemins d'image (voir _same_path)


def _same_path(a: str, b: str, mode: PathMode) -> bool:  # dit si deux chemins designent la meme image ; renvoie True/False
    if mode == "exact":  # mode "exact" : les deux textes doivent etre identiques
        return a == b  # compare les chemins caractere par caractere
    if mode == "name":  # mode "name" : seul le nom du fichier compte (dossiers ignores)
        return Path(a).name == Path(b).name  # compare "img_001.jpg" a "img_001.jpg", quel que soit le dossier
    return Path(a).resolve() == Path(b).resolve()  # mode "resolved" : compare les chemins absolus complets


# liste les differences entre deux annotations d'une meme image ; renvoie une liste de phrases (vide = identiques)
def annotation_differences(a: ImageAnnotation, b: ImageAnnotation, tol: float = 1e-3, path_mode: PathMode = "name") -> list[str]:
    """Liste lisible des differences (vide = identiques a tol pixel pres) ; l'ordre des objets d'une meme sorte compte."""
    issues = []  # liste des differences trouvees, remplie au fur et a mesure
    if not _same_path(a.image_path, b.image_path, path_mode):  # si les deux annotations ne visent pas la meme image
        issues.append(f"image {a.image_path} != {b.image_path}")  # note la difference de chemin
    if (a.width, a.height) != (b.width, b.height):  # si la taille de l'image differe (largeur, hauteur)
        issues.append(f"taille {a.width}x{a.height} != {b.width}x{b.height}")  # note la difference de taille
    if sorted(a.labels) != sorted(b.labels):  # si les etiquettes d'image entiere different (ordre ignore grace au tri)
        issues.append(f"etiquettes {a.labels} != {b.labels}")  # note la difference d'etiquettes
    for kind in ("boxes", "rotated", "polygons"):  # pour chaque sorte d'objet : boites, boites orientees, polygones
        items_a, items_b = getattr(a, kind), getattr(b, kind)  # getattr lit l'attribut par son nom (ex. a.boxes)
        if len(items_a) != len(items_b):  # si le nombre d'objets differe, inutile de comparer un par un
            issues.append(f"{kind} : {len(items_a)} != {len(items_b)}")  # note la difference de nombre
            continue  # passe a la sorte d'objet suivante
        for i, (x, y) in enumerate(zip(items_a, items_b)):  # pour chaque paire d'objets de meme rang (i = numero)
            if x.label != y.label:  # si les classes different
                issues.append(f"{kind}[{i}] classe {x.label} != {y.label}")  # note la difference de classe
            # boite alignee : un coin qui bouge de plus de tol pixel = coordonnees differentes
            elif kind == "boxes" and max(abs(x.x1 - y.x1), abs(x.y1 - y.y1), abs(x.x2 - y.x2), abs(x.y2 - y.y2)) > tol:
                issues.append(f"boxes[{i}] coordonnees differentes")  # note la difference de position
            elif kind == "rotated" and not same_corners(x, y, tol):  # boite orientee : compare les quatre coins
                issues.append(f"rotated[{i}] coins differents")  # note la difference de coins
            # polygone : nombre de points different, ou un point eloigne de plus de tol pixel
            elif kind == "polygons" and (len(x.points) != len(y.points) or any(math.dist(p, q) > tol for p, q in zip(x.points, y.points))):
                issues.append(f"polygons[{i}] points differents")  # note la difference de contour
            # score de confiance (pre-annotation) : compare seulement si les deux en ont un
            if x.score is not None and y.score is not None and abs(x.score - y.score) > 1e-3:
                issues.append(f"{kind}[{i}] score {x.score} != {y.score}")  # note la difference de score
    return issues  # renvoie toutes les differences trouvees (liste vide = identiques)


# compare deux jeux complets image par image ; renvoie la liste de toutes les differences
def same_annotations(a: list[ImageAnnotation], b: list[ImageAnnotation], tol: float = 1e-3, path_mode: PathMode = "name") -> list[str]:
    """Compare deux jeux image par image (apparies par nom de fichier) ; renvoie toutes les differences."""
    index_b = {Path(x.image_path).name: x for x in b}  # dictionnaire nom de fichier -> annotation du second jeu
    issues = []  # liste des differences trouvees
    if len(a) != len(b):  # si les deux jeux n'ont pas le meme nombre d'images
        issues.append(f"{len(a)} images != {len(b)} images")  # note la difference de taille de jeu
    for x in a:  # pour chaque image du premier jeu
        y = index_b.get(Path(x.image_path).name)  # cherche la meme image (meme nom) dans le second jeu ; None si absente
        if y is None:  # si l'image n'existe pas dans le second jeu
            issues.append(f"{Path(x.image_path).name} absente du second jeu")  # note l'absence
            continue  # passe a l'image suivante
        # ajoute les differences de cette image, prefixees par son nom de fichier
        issues.extend(f"{Path(x.image_path).name} : {d}" for d in annotation_differences(x, y, tol, path_mode))
    return issues  # renvoie toutes les differences (liste vide = jeux identiques)
