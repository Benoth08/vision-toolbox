"""Format YOLO (Ultralytics) : un .txt par image, coordonnees NORMALISEES dans [0, 1].

Trois variantes selon la tache, une ligne par objet :
- detect  : `classe cx cy w h`
- obb     : `classe x1 y1 x2 y2 x3 y3 x4 y4` (les quatre coins d'une boite orientee)
- segment : `classe x1 y1 x2 y2 ... xn yn` (un polygone)

Arborescence attendue par Ultralytics : images/<split>/a.jpg <-> labels/<split>/a.txt, et un
data.yaml qui donne les chemins et les noms de classes dans l'ordre des indices.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[str], X | None) partout

import shutil  # copie de fichiers (shutil.copy2 garde aussi la date du fichier)
from collections.abc import Callable, Mapping  # Callable : type "fonction" ; Mapping : type "dictionnaire"
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux
from typing import Literal  # Literal : type qui n'accepte que quelques valeurs texte precises

import numpy as np  # numpy : calcul sur des tableaux de nombres
import yaml  # PyYAML : lit et ecrit les fichiers .yaml (ici data.yaml)

from vtb.annotate.schema import Box, ImageAnnotation, Polygon, RotatedBox  # format pivot de la toolbox
from vtb.io.images import image_size, list_images, read_image, write_image  # taille, liste, lecture et ecriture d'images

YoloTask = Literal["detect", "obb", "segment"]  # les trois variantes du format YOLO


def _clip01(values: list[float]) -> list[float]:  # ramene chaque valeur entre 0 et 1 ; renvoie la nouvelle liste
    return [min(max(v, 0.0), 1.0) for v in values]  # max(v, 0) puis min(..., 1) : une coordonnee normalisee reste dans l'image


# convertit une annotation pivot en lignes de texte YOLO ; renvoie la liste des lignes (une par objet)
def annotation_to_yolo_lines(annotation: ImageAnnotation, class_to_idx: Mapping[str, int], task: YoloTask = "detect") -> list[str]:
    """Une ImageAnnotation -> lignes du .txt YOLO pour la tache demandee."""
    w, h = float(annotation.width), float(annotation.height)  # taille de l'image, pour normaliser (diviser par la taille)
    lines: list[str] = []  # lignes du fichier .txt
    if task == "detect":  # detection : boites alignees
        for box in annotation.boxes:  # pour chaque boite
            b = box.clipped(annotation.width, annotation.height)  # ramene la boite dans l'image
            # centre et taille, divises par la taille de l'image (normalisation entre 0 et 1)
            cx, cy, bw, bh = _clip01([(b.x1 + b.x2) / 2 / w, (b.y1 + b.y2) / 2 / h, b.width / w, b.height / h])
            lines.append(f"{class_to_idx[b.label]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")  # "classe cx cy w h", 6 decimales
    elif task == "obb":  # detection orientee : boites tournees
        for rbox in annotation.rotated:  # pour chaque boite orientee
            coords = _clip01([v for x, y in rbox.corners() for v in (x / w, y / h)])  # 4 coins normalises, a plat : x1 y1 x2 y2...
            lines.append(f"{class_to_idx[rbox.label]} " + " ".join(f"{v:.6f}" for v in coords))  # "classe x1 y1 ... x4 y4"
    elif task == "segment":  # segmentation d'instance : polygones
        for poly in annotation.polygons:  # pour chaque polygone
            coords = _clip01([v for x, y in poly.points for v in (x / w, y / h)])  # sommets normalises, a plat
            lines.append(f"{class_to_idx[poly.label]} " + " ".join(f"{v:.6f}" for v in coords))  # "classe x1 y1 ... xn yn"
    else:  # tache inconnue
        raise ValueError(f"tache YOLO inconnue : {task}")  # arret avec message clair
    return lines  # renvoie les lignes du fichier


# convertit des lignes YOLO en objets en pixels ; renvoie (boites, boites orientees, polygones)
def parse_yolo_lines(
    lines: list[str],  # lignes du fichier .txt
    width: int,  # largeur de l'image en pixels (pour "denormaliser")
    height: int,  # hauteur de l'image en pixels
    class_names: list[str],  # noms des classes : l'indice ecrit dans le fichier pointe dans cette liste
    task: YoloTask | None = None,  # variante du format ; None = devinee d'apres le nombre de valeurs
) -> tuple[list[Box], list[RotatedBox], list[Polygon]]:  # renvoie trois listes d'objets
    """Lignes YOLO -> objets en pixels.

    task fixe l'interpretation. Sans task, elle est deduite du nombre de valeurs (4 = boite,
    8 = boite orientee, sinon polygone) : c'est ambigu pour un polygone a 4 sommets, qui sera lu
    comme une boite orientee ; passer task="segment" pour un jeu de segmentation.
    """
    import cv2  # OpenCV importe ici seulement (sert au cas "obb")

    boxes, rotated, polygons = [], [], []  # objets lus, ranges par sorte
    for line in lines:  # pour chaque ligne du fichier
        values = line.split()  # decoupe la ligne aux espaces
        if not values:  # ligne vide
            continue  # on la saute
        label, coords = class_names[int(values[0])], [float(v) for v in values[1:]]  # 1re valeur = classe, le reste = nombres
        # variante imposee, sinon devinee : 4 nombres = boite, 8 = boite orientee, autre = polygone
        kind = task or ("detect" if len(coords) == 4 else "obb" if len(coords) == 8 else "segment")
        if kind == "detect":  # boite alignee
            cx, cy, bw, bh = coords[0] * width, coords[1] * height, coords[2] * width, coords[3] * height  # normalise -> pixels
            boxes.append(Box(label, cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))  # centre + taille -> deux coins
        elif kind == "obb":  # boite orientee
            # 4 coins en pixels, tableau 4 x 2 de flottants 32 bits (format attendu par OpenCV)
            pts = np.array([[coords[i] * width, coords[i + 1] * height] for i in range(0, 8, 2)], dtype=np.float32)
            (cx, cy), (rw, rh), angle = cv2.minAreaRect(pts)  # rectangle tourne passant par ces coins : centre, taille, angle
            rotated.append(RotatedBox(label, float(cx), float(cy), float(rw), float(rh), float(angle)))  # boite pivot
        else:  # polygone
            # regroupe les nombres par paires (x, y) et repasse en pixels
            polygons.append(Polygon(label, [(coords[i] * width, coords[i + 1] * height) for i in range(0, len(coords), 2)]))
    return boxes, rotated, polygons  # renvoie les trois listes


# lit un dossier d'images et le dossier de leurs .txt YOLO ; renvoie la liste pivot
def read_yolo_folder(
    images_dir: str | Path,  # dossier des images
    labels_dir: str | Path,  # dossier des .txt
    class_names: list[str],  # noms des classes, dans l'ordre des indices
    task: YoloTask | None = None,  # variante du format ; None = devinee
) -> list[ImageAnnotation]:  # renvoie une ImageAnnotation par image
    """Images + .txt de meme nom. Une image sans .txt est une image sans objet (negatif), pas une erreur."""
    annotations = []  # une annotation par image
    for image_path in list_images(images_dir):  # pour chaque image du dossier
        width, height = image_size(image_path)  # sa taille (YOLO ne la stocke pas : il faut la lire)
        txt = Path(labels_dir) / f"{image_path.stem}.txt"  # fichier d'etiquettes attendu, de meme nom
        lines = txt.read_text(encoding="utf-8").splitlines() if txt.exists() else []  # ses lignes ; pas de fichier = aucun objet
        boxes, rotated, polygons = parse_yolo_lines(lines, width, height, class_names, task)  # convertit en objets pivot
        # assemble l'annotation de l'image
        annotations.append(ImageAnnotation(str(image_path), width, height, boxes=boxes, rotated=rotated, polygons=polygons))
    return annotations  # renvoie toutes les annotations


# ecrit un jeu complet au format Ultralytics (images, etiquettes, data.yaml) ; renvoie le chemin de data.yaml
def write_yolo_dataset(
    annotations: list[ImageAnnotation],  # le jeu a ecrire
    splits: Mapping[str, str],  # chemin de chaque image -> "train", "val" ou "test"
    class_names: list[str],  # noms des classes : leur position donne l'indice ecrit dans les .txt
    out_dir: str | Path,  # dossier racine du jeu YOLO
    task: YoloTask = "detect",  # variante : "detect", "obb" ou "segment"
    transform: Callable[[np.ndarray], np.ndarray] | None = None,  # pretraitement a appliquer aux images ; None = copie
) -> Path:  # renvoie le chemin du data.yaml
    """Ecrit images/<split>, labels/<split> et data.yaml ; renvoie le chemin du data.yaml.

    splits : chemin d'image -> train / val / test. transform : pretraitement applique aux images
    copiees (la meme fonction qu'en production) ; None = copie a l'identique, plus rapide.
    """
    out_dir = Path(out_dir)  # convertit en Path
    class_to_idx = {name: i for i, name in enumerate(class_names)}  # nom de classe -> indice (0, 1, 2...)
    for ann in annotations:  # pour chaque image
        split = splits[ann.image_path]  # sous-ensemble de cette image
        source = Path(ann.image_path)  # fichier image d'origine
        image_dst = out_dir / "images" / split / source.name  # destination de l'image : images/<split>/<nom>
        label_dst = out_dir / "labels" / split / f"{source.stem}.txt"  # destination des etiquettes : labels/<split>/<nom>.txt
        image_dst.parent.mkdir(parents=True, exist_ok=True)  # cree le dossier images/<split> si besoin
        label_dst.parent.mkdir(parents=True, exist_ok=True)  # cree le dossier labels/<split> si besoin
        if transform is None:  # pas de pretraitement
            shutil.copy2(source, image_dst)  # simple copie du fichier (rapide)
        else:  # pretraitement demande
            write_image(image_dst, transform(read_image(source)))  # lit, transforme, ecrit
        lines = annotation_to_yolo_lines(ann, class_to_idx, task)  # lignes YOLO de cette image
        label_dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")  # ecrit le .txt (vide si aucun objet)
    present = sorted({s for s in splits.values()})  # sous-ensembles utilises (train, val, test), sans doublon
    # contenu de data.yaml : dossier racine, un chemin d'images par sous-ensemble, et les noms {0: "a", 1: "b"...}
    data_yaml = {"path": str(out_dir.resolve()), **{s: f"images/{s}" for s in present}, "names": dict(enumerate(class_names))}
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")  # ecrit data.yaml
    return out_dir / "data.yaml"  # renvoie son chemin (c'est lui qu'on donne a Ultralytics pour entrainer)
