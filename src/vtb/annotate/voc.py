"""Pascal VOC : un fichier XML par image, coins (xmin, ymin, xmax, ymax) en pixels."""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[str], X | None) partout

import xml.etree.ElementTree as ET  # ElementTree : lire et ecrire du XML (arbre de balises)
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

from vtb.annotate.schema import Box, ImageAnnotation  # format pivot de la toolbox
from vtb.io.images import list_images  # liste les fichiers images d'un dossier, triee


# lit un fichier XML VOC ; renvoie l'ImageAnnotation correspondante
def read_voc_file(xml_path: str | Path, image_path: str | Path | None = None) -> ImageAnnotation:
    """Un XML VOC -> ImageAnnotation. image_path : chemin reel de l'image (le champ <path> du XML est souvent faux)."""
    root = ET.parse(xml_path).getroot()  # lit le XML et recupere la balise racine <annotation>
    # taille de l'image (texte -> float -> int, car certains outils ecrivent "640.0")
    width, height = int(float(root.findtext("size/width"))), int(float(root.findtext("size/height")))
    boxes = []  # boites lues dans le fichier
    for obj in root.iter("object"):  # pour chaque objet <object> du fichier
        box = obj.find("bndbox")  # sa boite englobante <bndbox>
        coords = [float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]  # les deux coins, en nombres
        boxes.append(Box(obj.findtext("name").strip(), *coords).clipped(width, height))  # cree la boite, ramenee dans l'image
    # chemin de l'image : celui donne, sinon le <filename> du XML place a cote du XML
    path = str(image_path) if image_path is not None else str(Path(xml_path).with_name(root.findtext("filename") or ""))
    return ImageAnnotation(image_path=path, width=width, height=height, boxes=boxes)  # renvoie l'annotation pivot


# lit un dossier d'images et le dossier de leurs XML VOC ; renvoie la liste pivot
def read_voc_folder(images_dir: str | Path, annotations_dir: str | Path) -> list[ImageAnnotation]:
    """Toutes les images d'un dossier et leur XML de meme nom ; une image sans XML est une erreur explicite."""
    annotations = []  # liste des annotations, une par image
    for image_path in list_images(images_dir):  # pour chaque image du dossier
        xml_path = Path(annotations_dir) / f"{image_path.stem}.xml"  # XML attendu : meme nom, extension .xml
        if not xml_path.exists():  # si le XML manque
            raise FileNotFoundError(f"annotation VOC absente pour {image_path.name} : {xml_path}")  # arret avec message clair
        annotations.append(read_voc_file(xml_path, image_path))  # lit le XML et ajoute l'annotation
    return annotations  # renvoie toutes les annotations


def write_voc_file(annotation: ImageAnnotation, xml_path: str | Path) -> None:  # ecrit une annotation en XML VOC ; ne renvoie rien
    """ImageAnnotation -> XML VOC (boites seulement ; VOC ne sait pas representer polygones et boites orientees)."""
    root = ET.Element("annotation")  # balise racine <annotation>
    ET.SubElement(root, "filename").text = Path(annotation.image_path).name  # <filename> : nom du fichier image
    size = ET.SubElement(root, "size")  # balise <size> qui regroupe les dimensions
    ET.SubElement(size, "width").text = str(annotation.width)  # largeur en pixels
    ET.SubElement(size, "height").text = str(annotation.height)  # hauteur en pixels
    ET.SubElement(size, "depth").text = "3"  # nombre de canaux (3 = couleur RGB)
    for box in annotation.boxes:  # pour chaque boite alignee
        obj = ET.SubElement(root, "object")  # balise <object>
        ET.SubElement(obj, "name").text = box.label  # <name> : classe de l'objet
        ET.SubElement(obj, "difficult").text = "0"  # 0 = objet facile (champ attendu par les outils VOC)
        bnd = ET.SubElement(obj, "bndbox")  # balise <bndbox> qui contient les coins
        for key, value in zip(("xmin", "ymin", "xmax", "ymax"), (box.x1, box.y1, box.x2, box.y2)):  # chaque coordonnee avec son nom
            ET.SubElement(bnd, key).text = f"{value:.1f}"  # ecrite avec une decimale
    Path(xml_path).parent.mkdir(parents=True, exist_ok=True)  # cree le dossier de destination s'il n'existe pas
    ET.ElementTree(root).write(xml_path, encoding="utf-8", xml_declaration=True)  # ecrit le fichier avec l'entete <?xml ...?>


def write_voc_folder(annotations: list[ImageAnnotation], annotations_dir: str | Path) -> None:  # un XML VOC par image ; ne renvoie rien
    for ann in annotations:  # pour chaque image
        write_voc_file(ann, Path(annotations_dir) / f"{Path(ann.image_path).stem}.xml")  # XML de meme nom que l'image
