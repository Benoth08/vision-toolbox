"""Labelme et X-AnyLabeling : un fichier JSON par image, de meme nom que l'image.

X-AnyLabeling lit et ecrit ce meme format, avec deux ajouts que l'on gere : le type de forme
"rotation" (quatre coins d'une boite orientee, plus "direction" en radians) et le champ "score"
des pre-annotations. Formes lues : rectangle (deux coins), polygon, rotation ; les autres
(point, ligne, cercle) sont ignorees avec un avertissement. Les etiquettes d'image
(classification) sont les "flags" a vrai.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[str], X | None) partout

import json  # lecture et ecriture de fichiers JSON (texte structure en dictionnaires et listes)
import logging  # messages d'information et d'avertissement, plus souples que print
import math  # fonctions mathematiques de base ; ici conversion degres -> radians
import os  # fonctions du systeme ; ici os.path pour calculer et nettoyer des chemins
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

from vtb.annotate.geometry import ordered_corners, rotated_from_corners  # boite orientee <-> ses quatre coins
from vtb.annotate.schema import Box, ImageAnnotation, Polygon  # format pivot de la toolbox
from vtb.io.images import image_size  # lit la largeur et la hauteur d'une image sans la charger entierement

logger = logging.getLogger(__name__)  # journal propre a ce module


# convertit une annotation pivot en dictionnaire labelme (pret a ecrire en JSON) ; renvoie ce dictionnaire
def annotation_to_labelme(ann: ImageAnnotation, json_dir: Path, image_flags: list[str] | None = None) -> dict:
    """ImageAnnotation -> dictionnaire labelme ; imagePath relatif au dossier du JSON (comme le fait labelme)."""
    shapes = []  # liste des formes dessinees (labelme appelle "shape" chaque objet)
    for box in ann.boxes:  # pour chaque boite alignee
        shapes.append(_shape(box.label, [(box.x1, box.y1), (box.x2, box.y2)], "rectangle", box.score))  # rectangle = 2 coins
    for rbox in ann.rotated:  # pour chaque boite orientee
        shape = _shape(rbox.label, ordered_corners(rbox), "rotation", rbox.score)  # forme "rotation" = 4 coins (X-AnyLabeling)
        shape["direction"] = math.radians(rbox.angle)  # X-AnyLabeling attend aussi l'angle, en radians
        shapes.append(shape)  # ajoute la forme a la liste
    for poly in ann.polygons:  # pour chaque polygone
        shapes.append(_shape(poly.label, list(poly.points), "polygon", poly.score))  # forme "polygon" = liste de sommets
    try:  # le calcul du chemin relatif peut echouer sous Windows
        image_rel = Path(os.path.relpath(ann.image_path, json_dir)).as_posix()  # chemin de l'image vu depuis le dossier du JSON
    except ValueError:  # Windows : image sur un autre lecteur
        image_rel = str(ann.image_path)  # on garde alors le chemin complet
    flags = {name: name in ann.labels for name in (image_flags or ann.labels)}  # cases a cocher d'image : cochee si etiquette presente
    return {  # dictionnaire au format labelme
        "version": "5.4.1",  # version de labelme imitee (l'outil la verifie a l'ouverture)
        "flags": flags,  # etiquettes d'image (classification)
        "shapes": shapes,  # objets dessines
        "imagePath": image_rel,  # chemin de l'image relatif au JSON
        "imageData": None,  # pas d'image en base64 dans le JSON : fichiers legers
        "imageHeight": ann.height,  # hauteur de l'image en pixels
        "imageWidth": ann.width,  # largeur de l'image en pixels
    }


# construit le dictionnaire d'une forme labelme ; renvoie ce dictionnaire
def _shape(label: str, points: list[tuple[float, float]], shape_type: str, score: float | None) -> dict:
    shape = {  # champs attendus par labelme pour chaque forme
        "label": label,  # classe de l'objet
        "points": [[float(x), float(y)] for x, y in points],  # points en listes [x, y] (JSON n'a pas de tuples)
        "group_id": None,  # regroupement de formes (non utilise)
        "description": "",  # texte libre (vide)
        "shape_type": shape_type,  # "rectangle", "rotation" ou "polygon"
        "flags": {},  # cases a cocher propres a la forme (aucune)
    }
    if score is not None:  # si c'est une pre-annotation avec une confiance
        shape["score"] = score  # champ "score" lu par X-AnyLabeling
    return shape  # renvoie la forme


# ecrit un JSON labelme par image dans out_dir ; ne renvoie rien
def write_labelme_folder(annotations: list[ImageAnnotation], out_dir: str | Path, image_flags: list[str] | None = None) -> None:
    """Un JSON par image dans out_dir (nom de l'image, extension .json)."""
    out_dir = Path(out_dir)  # convertit en Path (accepte aussi un simple texte)
    out_dir.mkdir(parents=True, exist_ok=True)  # cree le dossier s'il n'existe pas
    for ann in annotations:  # pour chaque image
        payload = annotation_to_labelme(ann, out_dir, image_flags)  # construit son dictionnaire labelme
        # ecrit "<nom de l'image sans extension>.json" ; stem = nom sans extension
        (out_dir / f"{Path(ann.image_path).stem}.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")


# convertit un dictionnaire labelme (lu dans json_path) en annotation pivot ; renvoie l'ImageAnnotation
def labelme_to_annotation(payload: dict, json_path: Path, images_dir: Path | None = None) -> ImageAnnotation:
    """Dictionnaire labelme -> ImageAnnotation ; l'image est cherchee a cote du JSON puis dans images_dir."""
    # normpath et non resolve : on garde les chemins tels qu'ecrits (liens symboliques compris).
    candidate = Path(os.path.normpath(json_path.parent / payload.get("imagePath", "")))  # image indiquee, vue depuis le JSON
    if not candidate.is_file() and images_dir is not None:  # si elle n'y est pas et qu'un dossier d'images est donne
        candidate = images_dir / Path(payload.get("imagePath", json_path.stem)).name  # on la cherche par son nom dans ce dossier
    width, height = payload.get("imageWidth"), payload.get("imageHeight")  # taille notee dans le JSON (peut manquer)
    if not width or not height:  # si la taille manque
        width, height = image_size(candidate)  # on la lit dans le fichier image
    # annotation vide : chemin, taille, et etiquettes d'image = flags coches (valeur vraie)
    ann = ImageAnnotation(str(candidate), int(width), int(height), labels=[k for k, v in (payload.get("flags") or {}).items() if v])
    ignored = 0  # compteur des formes non gerees
    for shape in payload.get("shapes", []):  # pour chaque forme dessinee
        points = [(float(x), float(y)) for x, y in shape["points"]]  # points en tuples de flottants
        kind, label, score = shape.get("shape_type", "polygon"), shape["label"], shape.get("score")  # type, classe, score
        if kind == "rectangle" and len(points) == 2:  # rectangle : deux coins opposes
            (xa, ya), (xb, yb) = points  # labelme ne garantit pas l'ordre des deux coins
            ann.boxes.append(Box(label, min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb), score))  # remet haut gauche / bas droit
        elif kind == "rotation" and len(points) == 4:  # boite orientee X-AnyLabeling : quatre coins
            ann.rotated.append(rotated_from_corners(label, points, score))  # reconstruit centre, taille et angle
        elif kind == "polygon" and len(points) >= 3:  # polygone : au moins trois sommets
            ann.polygons.append(Polygon(label, points, score))  # garde les sommets tels quels
        else:  # autre forme (point, ligne, cercle) ou forme incomplete
            ignored += 1  # on la compte comme ignoree
    if ignored:  # si des formes ont ete ignorees
        logger.warning("%s : %d formes non gerees (point, ligne, cercle...) ignorees", json_path.name, ignored)  # previent
    return ann  # renvoie l'annotation de l'image


def read_labelme_folder(images_dir: str | Path | None, json_dir: str | Path) -> list[ImageAnnotation]:  # lit tous les JSON d'un dossier
    """Tous les JSON d'un dossier, tries par nom."""
    images = Path(images_dir) if images_dir else None  # dossier d'images de secours, ou None
    # pour chaque .json du dossier (trie par nom) : lit le texte, decode le JSON, convertit en annotation pivot
    return [labelme_to_annotation(json.loads(p.read_text(encoding="utf-8")), p, images) for p in sorted(Path(json_dir).glob("*.json"))]
