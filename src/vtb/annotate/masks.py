"""Masques PNG <-> objets : polygones (YOLO-seg, COCO), boites orientees (YOLO-OBB), boites.

Convention des masques de classes : 0 = fond, k = classe class_names[k - 1]. Un masque binaire
(0 / 255, le cas de Magnetic Tile, MVTec, Severstal apres decodage) se lit avec une seule
classe : tout pixel non nul est un defaut.

Les contours sont pris avec cv2.findContours en RETR_EXTERNAL : les trous d'un objet ne sont
pas representes (un polygone YOLO-seg n'en a pas). `epsilon` (en pixels) simplifie le contour
avec approxPolyDP : 0 garde tous les points, 1 a 2 px divise souvent leur nombre par 5 sans
changer l'aire de facon visible.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[str], X | None) partout

from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import cv2  # OpenCV : contours, composantes connexes, dessin de formes pleines
import numpy as np  # numpy : calcul sur des tableaux de nombres (un masque = un tableau 2D)

from vtb.annotate.schema import Box, ImageAnnotation, Polygon, RotatedBox  # format pivot de la toolbox
from vtb.io.images import list_images, read_image, write_image  # lister, lire et ecrire des images


# generateur (yield) : produit, une a une, les paires (nom de classe, plan binaire 0/1) des classes presentes
def _planes(mask: np.ndarray, class_names: list[str]):
    """(nom, plan binaire uint8) pour chaque classe presente.

    Une seule classe : masque binaire, seuil a 127 (les PNG exportes ont souvent des bords
    anti-creneles a 1..254 qu'il ne faut pas compter comme des objets separes). Plusieurs classes :
    valeur de pixel k = class_names[k - 1].
    """
    if len(class_names) == 1:  # cas masque binaire (une seule classe)
        plane = (mask > 127).astype(np.uint8)  # 1 la ou le pixel est clair (objet), 0 ailleurs
        if plane.any():  # seulement s'il y a au moins un pixel objet
            yield class_names[0], plane  # yield : rend cette paire a l'appelant puis reprend ici
        return  # fin du generateur pour le cas binaire
    for value in (int(v) for v in np.unique(mask) if v != 0):  # pour chaque valeur de pixel presente, sauf 0 (fond)
        if 1 <= value <= len(class_names):  # seulement les valeurs qui correspondent a une classe connue
            yield class_names[value - 1], (mask == value).astype(np.uint8)  # plan 0/1 des pixels de cette classe


# transforme un masque en polygones (un par objet) ; renvoie la liste de Polygon
def mask_to_polygons(mask: np.ndarray, class_names: list[str], min_area: float = 10.0, epsilon: float = 1.0) -> list[Polygon]:
    """Un polygone par composante connexe et par classe ; les composantes plus petites que min_area pixels sont ignorees."""
    polygons = []  # polygones trouves
    for name, binary in _planes(mask, class_names):  # pour chaque classe presente et son plan binaire
        # contours exterieurs de chaque tache blanche ; CHAIN_APPROX_SIMPLE garde seulement les coins des segments droits
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:  # pour chaque objet (contour)
            if cv2.contourArea(contour) < min_area:  # si l'objet est trop petit (bruit, poussiere)
                continue  # on l'ignore
            approx = cv2.approxPolyDP(contour, epsilon, True) if epsilon > 0 else contour  # simplifie le contour a epsilon px pres
            if len(approx) >= 3:  # un polygone a besoin d'au moins 3 sommets
                polygons.append(Polygon(name, [(float(x), float(y)) for x, y in approx.reshape(-1, 2)]))  # (N,1,2) -> (N,2)
    return polygons  # renvoie tous les polygones


# generateur : pour chaque objet assez grand, produit (nom de classe, coordonnees de tous ses pixels)
def _components(mask: np.ndarray, class_names: list[str], min_area: float):
    """Composantes connexes (8-connexite) : (nom, points (N, 2) en x, y) pour chaque objet assez grand."""
    for name, binary in _planes(mask, class_names):  # pour chaque classe presente
        # composante connexe = groupe de pixels qui se touchent (8 voisins, diagonales comprises) ; labels = numero par pixel
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for k in range(1, count):  # pour chaque composante (la 0 est le fond)
            if stats[k, cv2.CC_STAT_AREA] >= min_area:  # si elle a assez de pixels
                ys, xs = np.nonzero(labels == k)  # lignes (y) et colonnes (x) de ses pixels
                yield name, np.stack([xs, ys], axis=1).astype(np.float32)  # tableau N x 2 de points (x, y)


# transforme un masque en boites orientees minimales (une par objet) ; renvoie la liste de RotatedBox
def mask_to_rotated_boxes(mask: np.ndarray, class_names: list[str], min_area: float = 10.0) -> list[RotatedBox]:
    """Une boite orientee minimale (cv2.minAreaRect) par composante connexe : la forme naturelle d'une rayure ou d'une fissure."""
    boxes = []  # boites trouvees
    for name, points in _components(mask, class_names, min_area):  # pour chaque objet et ses pixels
        # un pixel (x, y) couvre le carre [x, x + 1] : on prend ses 4 coins ; sinon un objet d'un pixel de large aurait une boite plate.
        corners = np.concatenate([points, points + (1, 0), points + (0, 1), points + (1, 1)]).astype(np.float32)
        (cx, cy), (w, h), angle = cv2.minAreaRect(corners)  # plus petit rectangle tourne qui contient tous ces points
        boxes.append(RotatedBox(name, float(cx), float(cy), float(w), float(h), float(angle)))  # converti en boite pivot
    return boxes  # renvoie toutes les boites orientees


# transforme un masque en boites alignees (une par objet) ; renvoie la liste de Box
def mask_to_boxes(mask: np.ndarray, class_names: list[str], min_area: float = 10.0) -> list[Box]:
    """Une boite alignee par composante connexe (coins en bord de pixel : x2 = dernier x + 1)."""
    return [  # liste construite en une seule expression
        # coins = plus petits x et y des pixels, plus grands x et y + 1 (bord droit / bas du dernier pixel)
        Box(name, float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max() + 1), float(p[:, 1].max() + 1))
        for name, p in _components(mask, class_names, min_area)  # pour chaque objet assez grand
    ]


def objects_to_mask(ann: ImageAnnotation, class_names: list[str]) -> np.ndarray:  # dessine les objets ; renvoie un masque de classes
    """Polygones, boites orientees et boites -> masque de classes uint8 (valeur k + 1 pour class_names[k])."""
    mask = np.zeros((ann.height, ann.width), np.uint8)  # masque noir (0 = fond), de la taille de l'image
    index = {name: i + 1 for i, name in enumerate(class_names)}  # classe -> valeur de pixel (1, 2, 3...)
    for box in ann.boxes:  # pour chaque boite alignee
        cv2.rectangle(  # dessine un rectangle plein
            mask,  # sur le masque
            (int(round(box.x1)), int(round(box.y1))),  # coin haut gauche, en pixels entiers
            (int(round(box.x2)) - 1, int(round(box.y2)) - 1),  # coin bas droit : -1 car OpenCV inclut ce dernier pixel
            index[box.label],  # "couleur" = valeur de la classe
            -1,  # epaisseur -1 = remplir l'interieur
        )
    for rbox in ann.rotated:  # pour chaque boite orientee
        cv2.fillPoly(mask, [np.round(np.array(rbox.corners())).astype(np.int32)], index[rbox.label])  # remplit ses 4 coins
    for poly in ann.polygons:  # pour chaque polygone
        cv2.fillPoly(mask, [np.round(np.array(poly.points)).astype(np.int32)], index[poly.label])  # remplit le contour
    return mask  # renvoie le masque de classes


# lit un dossier d'images et leurs masques ; renvoie la liste pivot (objets deduits des masques)
def read_mask_folder(
    images_dir: str | Path,  # dossier des images
    masks_dir: str | Path,  # dossier des masques (meme arborescence que les images)
    class_names: list[str],  # classes : valeur de pixel k = class_names[k - 1]
    kind: str = "polygons",  # sorte d'objets a deduire : "polygons", "rotated", "boxes" ou "mask"
    mask_suffix: str = "",  # suffixe du nom des masques (ex. "_mask" pour img_001_mask.png)
    extensions: list[str] | None = None,  # extensions d'images a lire ; None = les plus courantes
    min_area: float = 10.0,  # taille minimale d'un objet, en pixels
    epsilon: float = 1.0,  # simplification des contours, en pixels
) -> list[ImageAnnotation]:  # renvoie une ImageAnnotation par image
    """Images + masques de meme nom (masks_dir/<sous-dossier>/<nom><suffixe>.png) -> objets.

    kind : "polygons" (segmentation d'instance), "rotated" (OBB), "boxes" (detection) ou
    "mask" (aucun objet, seulement mask_path). Quand les masques sont a cote des images (meme
    dossier), passer extensions=[".jpg"] pour ne pas lire les masques comme des images.
    """
    images_dir, masks_dir = Path(images_dir), Path(masks_dir)  # convertit les deux chemins en Path
    annotations = []  # une annotation par image
    for image_path in list_images(images_dir, extensions or [".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"]):  # chaque image
        # masque attendu : meme sous-dossier relatif, meme nom + suffixe, extension .png
        mask_path = (masks_dir / image_path.relative_to(images_dir)).parent / f"{image_path.stem}{mask_suffix}.png"
        if not mask_path.exists():  # si le masque manque
            raise FileNotFoundError(f"masque absent pour {image_path} : {mask_path}")  # arret avec message clair
        mask = read_image(mask_path, color="gray")  # lit le masque en niveaux de gris (un seul canal)
        height, width = mask.shape  # taille du masque = taille de l'image
        ann = ImageAnnotation(str(image_path), width, height, mask_path=str(mask_path))  # annotation qui pointe vers le masque
        if kind == "polygons":  # segmentation d'instance
            ann.polygons = mask_to_polygons(mask, class_names, min_area, epsilon)  # un polygone par objet
        elif kind == "rotated":  # detection orientee (OBB)
            ann.rotated = mask_to_rotated_boxes(mask, class_names, min_area)  # une boite tournee par objet
        elif kind == "boxes":  # detection classique
            ann.boxes = mask_to_boxes(mask, class_names, min_area)  # une boite alignee par objet
        elif kind != "mask":  # "mask" = on garde seulement le chemin du masque ; toute autre valeur est une erreur
            raise ValueError(f"kind inconnu : {kind}")  # arret avec message clair
        annotations.append(ann)  # ajoute l'image terminee
    return annotations  # renvoie toutes les annotations


# ecrit un masque PNG par image dans out_dir ; ne renvoie rien
def write_mask_folder(annotations: list[ImageAnnotation], out_dir: str | Path, class_names: list[str], binary: bool = False) -> None:
    """Un PNG de classes par image (<nom>.png) dans out_dir ; binary=True ecrit 0 / 255 (une seule classe)."""
    for ann in annotations:  # pour chaque image
        mask = objects_to_mask(ann, class_names)  # dessine ses objets dans un masque de classes
        if binary:  # si un masque binaire est demande
            mask = (mask > 0).astype(np.uint8) * 255  # tout objet -> 255 (blanc), fond -> 0 (noir)
        write_image(Path(out_dir) / f"{Path(ann.image_path).stem}.png", mask)  # ecrit "<nom de l'image>.png"
