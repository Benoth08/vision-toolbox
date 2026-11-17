"""Le format pivot des annotations : toute la toolbox ne manipule que ces classes.

Chaque format externe (VOC, YOLO, COCO, Labelme, Label Studio, CVAT, masques PNG) a un lecteur
qui produit une liste d'`ImageAnnotation` et un ecrivain qui la consomme. Convertir A vers B,
c'est lire A puis ecrire B : n formats demandent 2n fonctions, pas n x n convertisseurs.

Coordonnees : toujours en **pixels de l'image d'origine**, origine en haut a gauche, x vers la
droite, y vers le bas. Les formats normalises (YOLO) sont convertis a la lecture et a l'ecriture.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[str], X | None) partout

import json  # lecture et ecriture de fichiers JSON (texte structure en dictionnaires et listes)
from collections import Counter  # Counter : compte combien de fois chaque valeur apparait
from dataclasses import asdict, dataclass, field  # dataclass : classe "fiche de donnees" ecrite en quelques lignes
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux


@dataclass  # genere automatiquement le constructeur, l'affichage et la comparaison a partir des champs ci-dessous
class Box:  # boite rectangulaire alignee sur les axes de l'image, decrite par deux coins
    """Boite alignee sur les axes, coins en pixels."""

    label: str  # nom de la classe de l'objet (ex. "rayure")
    x1: float  # abscisse du coin haut gauche, en pixels
    y1: float  # ordonnee du coin haut gauche (y vers le bas)
    x2: float  # abscisse du coin bas droit
    y2: float  # ordonnee du coin bas droit
    score: float | None = None  # confiance : remplie pour une prediction, None pour une verite terrain

    @property  # se lit comme un attribut (box.width), sans parentheses, mais calcule a chaque lecture
    def width(self) -> float:  # largeur de la boite en pixels
        return self.x2 - self.x1  # droite moins gauche

    @property  # se lit comme un attribut (box.height)
    def height(self) -> float:  # hauteur de la boite en pixels
        return self.y2 - self.y1  # bas moins haut

    def clipped(self, width: int, height: int) -> Box:  # renvoie une copie de la boite qui ne deborde plus de l'image
        """La meme boite ramenee dans l'image (certaines annotations debordent d'un pixel)."""
        # coins ramenes entre 0 et la taille de l'image ; classe et score inchanges
        return Box(self.label, max(0.0, self.x1), max(0.0, self.y1), min(float(width), self.x2), min(float(height), self.y2), self.score)


@dataclass  # genere le constructeur et l'affichage a partir des champs
class RotatedBox:  # boite tournee d'un angle : utile pour les objets longs et obliques
    """Boite orientee (rayures longues et obliques) : centre, taille, angle en degres (sens trigonometrique d'OpenCV)."""

    label: str  # nom de la classe de l'objet
    cx: float  # abscisse du centre, en pixels
    cy: float  # ordonnee du centre
    w: float  # largeur (cote "le long" de l'angle)
    h: float  # hauteur (cote perpendiculaire)
    angle: float  # rotation en degres
    score: float | None = None  # confiance de la prediction ; None pour une verite terrain

    def corners(self) -> list[tuple[float, float]]:  # renvoie les quatre coins (x, y) de la boite
        """Les quatre coins, dans l'ordre d'OpenCV (cv2.boxPoints)."""
        import cv2  # OpenCV importe ici seulement : le module reste utilisable sans OpenCV tant qu'on n'appelle pas corners()

        # boxPoints calcule les coins a partir de ((centre), (taille), angle) ; on convertit en float Python
        return [(float(x), float(y)) for x, y in cv2.boxPoints(((self.cx, self.cy), (self.w, self.h), self.angle))]


@dataclass  # genere le constructeur et l'affichage a partir des champs
class Polygon:  # contour d'un objet, liste de points : sert a la segmentation d'instance (un masque par objet)
    """Contour d'un objet (segmentation d'instance), points en pixels."""

    label: str  # nom de la classe de l'objet
    points: list[tuple[float, float]]  # sommets du contour, (x, y) en pixels, dans l'ordre du trace
    score: float | None = None  # confiance de la prediction ; None pour une verite terrain

    def bounding_box(self) -> Box:  # renvoie la plus petite boite alignee qui contient tout le polygone
        xs, ys = [p[0] for p in self.points], [p[1] for p in self.points]  # separe les x et les y des sommets
        return Box(self.label, min(xs), min(ys), max(xs), max(ys), self.score)  # coins = extremes des x et des y


@dataclass  # genere le constructeur et l'affichage a partir des champs
class ImageAnnotation:  # fiche complete d'une image : chemin, taille et toutes ses etiquettes
    """Tout ce qu'on sait d'une image : etiquettes d'image, boites, boites orientees, polygones, masque."""

    image_path: str  # chemin du fichier image
    width: int  # largeur de l'image en pixels
    height: int  # hauteur de l'image en pixels
    # field(default_factory=list) : chaque image recoit sa propre liste vide (jamais une liste partagee)
    labels: list[str] = field(default_factory=list)  # classification (une ou plusieurs etiquettes)
    boxes: list[Box] = field(default_factory=list)  # boites alignees (detection)
    rotated: list[RotatedBox] = field(default_factory=list)  # boites orientees (detection orientee)
    polygons: list[Polygon] = field(default_factory=list)  # contours (segmentation d'instance)
    mask_path: str | None = None  # masque PNG de classes (0 = fond) pour la segmentation semantique ; None si absent

    def object_labels(self) -> list[str]:  # renvoie la classe de chaque objet (boites, boites orientees, polygones)
        return [o.label for o in (*self.boxes, *self.rotated, *self.polygons)]  # * deplie les trois listes en une seule suite

    def main_label(self, empty: str = "background") -> str:  # renvoie une seule etiquette representative de l'image
        """Etiquette d'image si elle existe, sinon la classe la plus frequente des objets : sert a stratifier."""
        if self.labels:  # si l'image a une etiquette de classification
            return self.labels[0]  # on prend la premiere
        counts = Counter(self.object_labels())  # sinon : compte les objets par classe
        return counts.most_common(1)[0][0] if counts else empty  # classe la plus frequente ; image sans objet = "background"

    def to_dict(self) -> dict:  # renvoie l'annotation sous forme de dictionnaire (pret a ecrire en JSON)
        return asdict(self)  # asdict convertit la dataclass et ses sous-objets en dictionnaires imbriques

    @staticmethod  # methode qui n'a pas besoin d'un objet existant : on l'appelle sur la classe
    def from_dict(data: dict) -> ImageAnnotation:  # reconstruit une ImageAnnotation a partir d'un dictionnaire lu en JSON
        return ImageAnnotation(  # cree l'objet champ par champ
            image_path=data["image_path"],  # chemin de l'image
            width=int(data["width"]),  # largeur, forcee en entier
            height=int(data["height"]),  # hauteur, forcee en entier
            labels=list(data.get("labels", [])),  # etiquettes d'image ; liste vide si la cle manque
            boxes=[Box(**b) for b in data.get("boxes", [])],  # ** passe chaque cle du dict comme argument nomme
            rotated=[RotatedBox(**r) for r in data.get("rotated", [])],  # meme principe pour les boites orientees
            # JSON stocke les points en listes [x, y] : on les remet en tuples (x, y)
            polygons=[Polygon(p["label"], [tuple(pt) for pt in p["points"]], p.get("score")) for p in data.get("polygons", [])],
            mask_path=data.get("mask_path"),  # chemin du masque ; None si absent
        )


def class_names_of(annotations: list[ImageAnnotation]) -> list[str]:  # renvoie la liste triee de toutes les classes du jeu
    """Toutes les classes presentes (etiquettes d'image et objets), triees : l'ordre fixe les indices."""
    names: set[str] = set()  # ensemble : chaque nom n'y figure qu'une fois
    for ann in annotations:  # pour chaque image du jeu
        names.update(ann.labels)  # ajoute ses etiquettes d'image
        names.update(ann.object_labels())  # ajoute les classes de ses objets
    return sorted(names)  # tri alphabetique : la classe numero 0, 1, 2... est toujours la meme


# ecrit le jeu dans le format natif vtb.json ; ne renvoie rien
def save_annotations(annotations: list[ImageAnnotation], path: str | Path, class_names: list[str] | None = None) -> None:
    """Format natif de la toolbox (vtb.json) : un seul fichier JSON lisible, sans perte."""
    payload = {  # contenu complet du fichier
        "format": "vtb",  # marque qui permet de reconnaitre le fichier a la relecture
        "version": 1,  # version du format, pour pouvoir le faire evoluer plus tard
        "class_names": class_names or class_names_of(annotations),  # classes donnees, sinon deduites des annotations
        "images": [a.to_dict() for a in annotations],  # une entree par image
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)  # cree le dossier de destination s'il n'existe pas
    Path(path).write_text(json.dumps(payload, indent=1), encoding="utf-8")  # ecrit le JSON indente (lisible par un humain)


def load_annotations(path: str | Path) -> tuple[list[ImageAnnotation], list[str]]:  # relit vtb.json ; renvoie (annotations, classes)
    """Relit un fichier vtb.json : (annotations, noms de classes)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))  # lit le fichier texte et le decode en dictionnaire
    if payload.get("format") != "vtb":  # si la marque "vtb" manque, ce n'est pas notre format
        raise ValueError(f"{path} n'est pas un fichier vtb.json")  # arret avec un message clair
    return [ImageAnnotation.from_dict(d) for d in payload["images"]], list(payload["class_names"])  # objets + noms de classes
