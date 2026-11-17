"""CVAT for images 1.1 : un fichier XML pour toute la tache.

Formes gerees : box (avec l'attribut rotation : degres, sens horaire, autour du centre),
polygon (points "x1,y1;x2,y2;..."), tag (etiquette d'image). C'est le format d'export
"CVAT for images 1.1" et aussi un format d'import : on s'en sert pour envoyer des
pre-annotations dans CVAT (Actions > Upload annotations).
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[str], X | None) partout

import os  # fonctions du systeme ; ici os.path.relpath pour calculer un chemin relatif
import xml.etree.ElementTree as ET  # ElementTree : lire et ecrire du XML (arbre de balises <image>, <box>...)
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

from vtb.annotate.schema import Box, ImageAnnotation, Polygon, RotatedBox, class_names_of  # format pivot de la toolbox


def _points_text(points: list[tuple[float, float]]) -> str:  # renvoie les points au format texte CVAT "x1,y1;x2,y2;..."
    return ";".join(f"{x:.2f},{y:.2f}" for x, y in points)  # chaque point en "x,y" (2 decimales), separes par ";"


# ecrit la liste pivot dans un fichier XML CVAT 1.1 ; ne renvoie rien
def write_cvat_xml(
    annotations: list[ImageAnnotation],  # le jeu a ecrire (format pivot)
    out_xml: str | Path,  # chemin du fichier XML a creer
    class_names: list[str] | None = None,  # ordre des classes impose ; None = deduit des annotations
    images_dir: str | Path | None = None,  # dossier racine des images, pour ecrire des chemins relatifs
) -> None:  # None : la fonction ne renvoie rien, elle ecrit un fichier
    """Liste pivot -> XML CVAT 1.1. name = chemin relatif a images_dir (ou nom du fichier)."""
    root = ET.Element("annotations")  # balise racine <annotations> du document
    ET.SubElement(root, "version").text = "1.1"  # ajoute <version>1.1</version> : CVAT verifie la version du format
    labels = ET.SubElement(ET.SubElement(ET.SubElement(root, "meta"), "task"), "labels")  # cree <meta><task><labels>
    for name in class_names or class_names_of(annotations):  # pour chaque classe du jeu
        ET.SubElement(ET.SubElement(labels, "label"), "name").text = name  # ajoute <label><name>classe</name></label>
    for image_id, ann in enumerate(annotations):  # pour chaque image, avec un numero a partir de 0
        # nom de l'image : chemin relatif au dossier d'images s'il est donne, sinon simple nom de fichier
        name = Path(os.path.relpath(ann.image_path, images_dir)).as_posix() if images_dir else Path(ann.image_path).name
        # balise <image> avec son numero, son nom et sa taille (XML : toutes les valeurs sont du texte)
        node = ET.SubElement(root, "image", id=str(image_id), name=name, width=str(ann.width), height=str(ann.height))
        for label in ann.labels:  # pour chaque etiquette d'image (classification)
            ET.SubElement(node, "tag", label=label, source="manual")  # CVAT la represente par une balise <tag>
        for box in ann.boxes:  # pour chaque boite alignee
            ET.SubElement(node, "box", _box_attributes(box.label, box.x1, box.y1, box.x2, box.y2, 0.0, box.score))  # rotation 0
        for rbox in ann.rotated:  # pour chaque boite orientee
            angle = rbox.angle % 360  # angle ramene entre 0 et 360 degres
            attrs = _box_attributes(  # CVAT : boite non tournee autour du meme centre + angle de rotation
                rbox.label,  # classe de l'objet
                rbox.cx - rbox.w / 2,  # x1 : centre moins une demi-largeur (boite avant rotation)
                rbox.cy - rbox.h / 2,  # y1 : centre moins une demi-hauteur
                rbox.cx + rbox.w / 2,  # x2 : centre plus une demi-largeur
                rbox.cy + rbox.h / 2,  # y2 : centre plus une demi-hauteur
                angle,  # rotation autour du centre, en degres
                rbox.score,  # confiance (None pour une verite terrain)
            )
            ET.SubElement(node, "box", attrs)  # ajoute la balise <box> avec ces attributs
        for poly in ann.polygons:  # pour chaque polygone
            attrs = {  # attributs de la balise <polygon>
                "label": poly.label,  # classe de l'objet
                "source": "auto" if poly.score is not None else "manual",  # "auto" = pre-annotation par un modele
                "occluded": "0",  # objet non cache (champ attendu par CVAT)
                "points": _points_text(poly.points),  # contour au format "x1,y1;x2,y2;..."
                "z_order": "0",  # ordre d'empilement a l'affichage (champ attendu par CVAT)
            }
            element = ET.SubElement(node, "polygon", attrs)  # ajoute la balise <polygon>
            if poly.score is not None:  # si c'est une prediction avec une confiance
                ET.SubElement(element, "attribute", name="score").text = f"{poly.score:.4f}"  # garde le score en sous-balise
    ET.indent(root)  # indente le XML pour qu'il soit lisible par un humain
    Path(out_xml).parent.mkdir(parents=True, exist_ok=True)  # cree le dossier de destination s'il n'existe pas
    ET.ElementTree(root).write(out_xml, encoding="utf-8", xml_declaration=True)  # ecrit le fichier avec l'entete <?xml ...?>


# attributs texte d'une balise <box> CVAT ; renvoie un dictionnaire nom -> valeur
def _box_attributes(label: str, x1: float, y1: float, x2: float, y2: float, rotation: float, score: float | None) -> dict[str, str]:
    attrs = {  # attributs communs a toutes les boites
        "label": label,  # classe de l'objet
        "source": "auto" if score is not None else "manual",  # "auto" = pre-annotation par un modele
        "occluded": "0",  # objet non cache
        "xtl": f"{x1:.4f}",  # x du coin haut gauche (top left)
        "ytl": f"{y1:.4f}",  # y du coin haut gauche
        "xbr": f"{x2:.4f}",  # x du coin bas droit (bottom right)
        "ybr": f"{y2:.4f}",  # y du coin bas droit
        "z_order": "0",  # ordre d'empilement a l'affichage
    }
    if rotation:  # si la boite est tournee (angle non nul)
        attrs["rotation"] = f"{rotation:.4f}"  # ajoute l'angle en degres, sens horaire autour du centre
    if score is not None:  # si c'est une prediction avec une confiance
        attrs["score"] = f"{score:.4f}"  # attribut ignore par CVAT a l'import, garde pour la tracabilite
    return attrs  # renvoie le dictionnaire d'attributs


def read_cvat_xml(images_dir: str | Path, cvat_xml: str | Path) -> list[ImageAnnotation]:  # lit un XML CVAT ; renvoie la liste pivot
    """XML CVAT 1.1 -> liste pivot ; une box avec rotation non nulle devient une RotatedBox."""
    root = ET.parse(cvat_xml).getroot()  # lit le fichier XML et recupere la balise racine
    annotations = []  # liste des annotations pivot, une par image
    for node in root.iter("image"):  # pour chaque balise <image> du document
        # annotation vide : chemin complet (dossier + nom), largeur, hauteur
        ann = ImageAnnotation(str(Path(images_dir) / node.get("name")), int(node.get("width")), int(node.get("height")))
        for tag in node.iter("tag"):  # pour chaque etiquette d'image <tag>
            ann.labels.append(tag.get("label"))  # ajoute la classe aux etiquettes d'image
        for box in node.iter("box"):  # pour chaque boite <box>
            x1, y1, x2, y2 = (float(box.get(k)) for k in ("xtl", "ytl", "xbr", "ybr"))  # lit les deux coins en nombres
            score = float(box.get("score")) if box.get("score") else None  # score s'il existe, sinon None (verite terrain)
            rotation = float(box.get("rotation", 0) or 0)  # angle ; 0 si l'attribut manque ou est vide
            if abs(rotation % 360) < 1e-9:  # si l'angle vaut 0 (a l'arrondi pres) : boite alignee
                ann.boxes.append(Box(box.get("label"), x1, y1, x2, y2, score))  # garde les coins tels quels
            else:  # sinon : boite orientee
                # centre = milieu des coins, taille = ecart des coins, angle ramene entre 0 et 360
                ann.rotated.append(RotatedBox(box.get("label"), (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, rotation % 360, score))
        for poly in node.iter("polygon"):  # pour chaque polygone <polygon>
            # decoupe "x1,y1;x2,y2" en paires puis chaque paire en deux nombres
            points = [tuple(float(v) for v in pair.split(",")) for pair in poly.get("points").split(";")]
            score_node = poly.find("attribute[@name='score']")  # cherche la sous-balise du score ; None si absente
            score = float(score_node.text) if score_node is not None and score_node.text else None  # score ou None
            ann.polygons.append(Polygon(poly.get("label"), [(p[0], p[1]) for p in points], score))  # ajoute le polygone
        annotations.append(ann)  # ajoute l'image terminee a la liste
    return annotations  # renvoie toutes les annotations, dans l'ordre du fichier
