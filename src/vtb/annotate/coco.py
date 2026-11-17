"""COCO : un seul fichier JSON pour tout le jeu (images, categories, annotations).

Ce que COCO represente nativement : boites `bbox = [x, y, largeur, hauteur]` en pixels et
polygones `segmentation = [[x1, y1, x2, y2, ...]]`. Deux extensions, ignorees sans dommage par
les autres outils, rendent l'aller-retour sans perte :
- une boite orientee est ecrite comme le polygone de ses quatre coins (ce que tout outil COCO
  sait lire) et ses parametres exacts sont gardes dans `attributes.rotated_box` ;
- les etiquettes d'image (classification) vont dans le champ `labels` de l'image.
Les masques RLE (`iscrowd = 1`) ne sont pas lus : ils sont ignores avec un avertissement.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[str], X | None) partout

import json  # lecture et ecriture de fichiers JSON (texte structure en dictionnaires et listes)
import logging  # messages d'information et d'avertissement, plus souples que print
import os  # fonctions du systeme ; ici os.path.relpath pour calculer un chemin relatif
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

from vtb.annotate.geometry import ordered_corners, polygon_area  # coins d'une boite orientee ; aire d'un polygone
from vtb.annotate.schema import Box, ImageAnnotation, Polygon, RotatedBox, class_names_of  # format pivot de la toolbox

logger = logging.getLogger(__name__)  # journal propre a ce module (son nom apparait devant chaque message)


def _relative_name(image_path: str, images_dir: Path | None) -> str:  # renvoie le nom d'image a ecrire dans le champ file_name
    """file_name COCO : chemin relatif au dossier d'images s'il est donne, sinon le nom du fichier."""
    if images_dir is not None:  # si on connait le dossier racine des images
        try:  # le calcul du chemin relatif peut echouer sous Windows
            return Path(os.path.relpath(image_path, images_dir)).as_posix()  # chemin relatif avec des "/" (portable)
        except ValueError:  # Windows : image et dossier sur des lecteurs differents (C: et D:)
            pass  # on retombe sur le simple nom de fichier ci-dessous
    return Path(image_path).name  # nom du fichier seul (ex. "img_001.jpg")


# ecrit la liste pivot dans un fichier COCO ; ne renvoie rien
def write_coco(
    annotations: list[ImageAnnotation],  # le jeu a ecrire (format pivot)
    out_json: str | Path,  # chemin du fichier COCO a creer
    class_names: list[str] | None = None,  # ordre des classes impose ; None = deduit des annotations
    images_dir: str | Path | None = None,  # dossier racine des images, pour ecrire des chemins relatifs
) -> None:  # None : la fonction ne renvoie rien, elle ecrit un fichier
    """Liste pivot -> fichier COCO. Les ids de categories commencent a 1 (0 est reserve au fond par convention)."""
    names = class_names or class_names_of(annotations)  # classes donnees, sinon deduites et triees
    category_id = {name: i + 1 for i, name in enumerate(names)}  # nom de classe -> identifiant COCO (1, 2, 3...)
    images, objects = [], []  # entrees "images" et "annotations" du fichier, remplies dans la boucle
    for image_id, ann in enumerate(annotations, start=1):  # pour chaque image, avec un identifiant qui commence a 1
        entry = {  # description de l'image au format COCO
            "id": image_id,  # identifiant unique de l'image
            "file_name": _relative_name(ann.image_path, Path(images_dir) if images_dir else None),  # chemin de l'image
            "width": ann.width,  # largeur en pixels
            "height": ann.height,  # hauteur en pixels
        }
        if ann.labels:  # si l'image a des etiquettes de classification
            entry["labels"] = list(ann.labels)  # extension maison : COCO n'a pas de champ prevu pour cela
        if ann.mask_path:  # si l'image a un masque de segmentation
            entry["mask_path"] = ann.mask_path  # extension maison : garde le chemin du masque
        images.append(entry)  # ajoute l'image a la liste
        for box in ann.boxes:  # pour chaque boite alignee de l'image
            obj = {  # objet COCO correspondant
                "image_id": image_id,  # lien vers l'image
                "category_id": category_id[box.label],  # classe, sous forme d'identifiant
                "bbox": [box.x1, box.y1, box.width, box.height],  # COCO : coin haut gauche + largeur + hauteur
                "area": box.width * box.height,  # aire en pixels carres (sert a classer petits / grands objets)
                "iscrowd": 0,  # 0 = objet isole (1 = foule decrite en RLE, non gere ici)
                "segmentation": [],  # pas de contour pour une simple boite
            }
            if box.score is not None:  # si c'est une prediction avec une confiance
                obj["score"] = box.score  # garde la confiance
            objects.append(obj)  # ajoute l'objet a la liste
        for rbox in ann.rotated:  # pour chaque boite orientee de l'image
            corners = ordered_corners(rbox)  # ses quatre coins (x, y)
            xs, ys = [p[0] for p in corners], [p[1] for p in corners]  # separe les x et les y des coins
            obj = {  # objet COCO correspondant
                "image_id": image_id,  # lien vers l'image
                "category_id": category_id[rbox.label],  # classe, sous forme d'identifiant
                "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],  # boite alignee qui englobe les coins
                "area": rbox.w * rbox.h,  # aire reelle de la boite tournee
                "iscrowd": 0,  # objet isole
                "segmentation": [[v for p in corners for v in p]],  # polygone des 4 coins aplati : [x0, y0, x1, y1, ...]
                "attributes": {"rotated_box": [rbox.cx, rbox.cy, rbox.w, rbox.h, rbox.angle]},  # parametres exacts (extension)
            }
            if rbox.score is not None:  # si c'est une prediction avec une confiance
                obj["score"] = rbox.score  # garde la confiance
            objects.append(obj)  # ajoute l'objet a la liste
        for poly in ann.polygons:  # pour chaque polygone de l'image
            bbox = poly.bounding_box()  # boite alignee qui l'englobe (champ bbox obligatoire en COCO)
            obj = {  # objet COCO correspondant
                "image_id": image_id,  # lien vers l'image
                "category_id": category_id[poly.label],  # classe, sous forme d'identifiant
                "bbox": [bbox.x1, bbox.y1, bbox.width, bbox.height],  # boite englobante au format COCO
                "area": polygon_area(poly.points),  # aire du contour
                "iscrowd": 0,  # objet isole
                "segmentation": [[v for p in poly.points for v in p]],  # points aplatis : [x0, y0, x1, y1, ...]
            }
            if poly.score is not None:  # si c'est une prediction avec une confiance
                obj["score"] = poly.score  # garde la confiance
            objects.append(obj)  # ajoute l'objet a la liste
    for annotation_id, obj in enumerate(objects, start=1):  # pour chaque objet, un numero unique a partir de 1
        obj["id"] = annotation_id  # COCO exige un identifiant par annotation
    payload = {  # contenu complet du fichier COCO
        "images": images,  # liste des images
        "annotations": objects,  # liste des objets
        "categories": [{"id": category_id[n], "name": n, "supercategory": "defect"} for n in names],  # liste des classes
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)  # cree le dossier de destination s'il n'existe pas
    Path(out_json).write_text(json.dumps(payload, indent=1), encoding="utf-8")  # ecrit le JSON indente


def read_coco(images_dir: str | Path, coco_json: str | Path) -> list[ImageAnnotation]:  # lit un fichier COCO ; renvoie la liste pivot
    """Fichier COCO -> liste pivot, dans l'ordre des images du fichier.

    Une annotation avec bbox et sans segmentation devient une Box ; avec segmentation, un
    Polygon (un objet en plusieurs morceaux donne plusieurs polygones) ; avec
    attributes.rotated_box, une RotatedBox exacte.
    """
    payload = json.loads(Path(coco_json).read_text(encoding="utf-8"))  # lit et decode tout le fichier JSON
    names = {c["id"]: c["name"] for c in payload.get("categories", [])}  # identifiant de classe -> nom de classe
    by_image: dict[int, ImageAnnotation] = {}  # identifiant d'image -> son annotation pivot (remplie ensuite)
    for image in payload["images"]:  # pour chaque image declaree dans le fichier
        by_image[image["id"]] = ImageAnnotation(  # cree une annotation vide (sans objet) pour cette image
            image_path=str(Path(images_dir) / image["file_name"]),  # chemin complet = dossier d'images + file_name
            width=int(image["width"]),  # largeur en pixels
            height=int(image["height"]),  # hauteur en pixels
            labels=list(image.get("labels", [])),  # etiquettes d'image (extension maison) ; vide si absentes
            mask_path=image.get("mask_path"),  # chemin du masque (extension maison) ; None si absent
        )
    skipped = 0  # compteur des objets ignores (masques RLE)
    for obj in payload.get("annotations", []):  # pour chaque objet du fichier
        ann, label, score = by_image[obj["image_id"]], names[obj["category_id"]], obj.get("score")  # son image, sa classe, son score
        # RLE = masque compresse en "longueurs de plages" ; iscrowd=1 ou segmentation en dict signale ce cas
        if obj.get("iscrowd", 0) == 1 or isinstance(obj.get("segmentation"), dict):
            skipped += 1  # on compte l'objet ignore
            continue  # et on passe au suivant
        rotated = obj.get("attributes", {}).get("rotated_box")  # parametres exacts de boite orientee, s'ils existent
        if rotated is not None:  # si c'est une boite orientee ecrite par la toolbox
            ann.rotated.append(RotatedBox(label, *[float(v) for v in rotated], score=score))  # recree la boite (cx, cy, w, h, angle)
        elif obj.get("segmentation"):  # sinon, s'il y a un contour : c'est un polygone
            for flat in obj["segmentation"]:  # pour chaque morceau de l'objet (liste plate x, y, x, y...)
                # regroupe les valeurs par paires (x, y) : indices 0-1, 2-3, 4-5...
                ann.polygons.append(Polygon(label, [(float(flat[i]), float(flat[i + 1])) for i in range(0, len(flat), 2)], score))
        else:  # sinon, c'est une simple boite
            x, y, w, h = (float(v) for v in obj["bbox"])  # COCO : coin haut gauche + largeur + hauteur
            ann.boxes.append(Box(label, x, y, x + w, y + h, score))  # convertit en deux coins (x1, y1, x2, y2)
    if skipped:  # si des objets ont ete ignores
        # previent l'utilisateur (warning = avertissement, le programme continue)
        logger.warning("%d annotations COCO en RLE (iscrowd) ignorees : convertir les masques avec vtb.annotate.masks", skipped)
    return list(by_image.values())  # les annotations, dans l'ordre des images du fichier
