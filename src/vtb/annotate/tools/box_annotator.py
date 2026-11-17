"""Annotateur de boites a la souris, sans serveur : pour annoter seul une centaine d'images.

    glisser (clic gauche)      : dessiner une boite de la classe courante
    1..9                       : choisir la classe courante
    Suppr, x ou retour arriere : effacer la derniere boite       c : tout effacer sur l'image
    n, Entree ou fleche droite : image suivante (enregistre)    p ou fleche gauche : precedente
    q ou Echap                 : quitter (enregistre)

Enregistrement : un fichier vtb.json pour tout le dossier (repris au lancement suivant) et, si
demande, un .txt YOLO par image. Pour annoter a plusieurs, des polygones ou des milliers
d'images : Label Studio, CVAT ou X-AnyLabeling (voir recipes/annotation/README.md).
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[str], X | None) partout

import logging  # messages d'information et d'avertissement, plus souples que print
from dataclasses import dataclass, field  # dataclass : classe "fiche de donnees" ecrite en quelques lignes
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

from vtb.annotate.schema import Box, ImageAnnotation  # format pivot : boite alignee, annotation d'une image

logger = logging.getLogger(__name__)  # journal propre a ce module


@dataclass  # genere le constructeur a partir des champs ci-dessous
class BoxAnnotatorState:  # etat de l'annotation d'une image, sans affichage : testable sans ecran
    """Etat d'une image en cours d'annotation ; les coordonnees souris sont celles de l'affichage (reduit d'un facteur scale)."""

    annotation: ImageAnnotation  # annotation de l'image, completee au fil des boites dessinees
    class_names: list[str]  # classes possibles (touche 1 = premiere classe)
    scale: float = 1.0  # facteur d'affichage : image affichee = image reelle x scale
    current_class: int = 0  # indice de la classe utilisee pour les prochaines boites
    min_size: float = 3.0  # une boite plus petite (en pixels image) est un clic involontaire
    drag_start: tuple[float, float] | None = None  # point ou le clic a commence ; None si on ne dessine pas
    preview: tuple[float, float, float, float] | None = None  # boite en cours de trace (x1, y1, x2, y2), pour l'affichage
    history: list[Box] = field(default_factory=list)  # boites effacees (gardees en memoire)

    def _to_image(self, x: float, y: float) -> tuple[float, float]:  # renvoie (x, y) en pixels de l'image reelle
        """Coordonnees d'affichage -> pixels de l'image, bornees a l'image."""
        # divise par scale (affichage -> image), puis borne entre 0 et la largeur / hauteur
        return min(max(x / self.scale, 0.0), float(self.annotation.width)), min(max(y / self.scale, 0.0), float(self.annotation.height))

    def on_mouse(self, event: str, x: float, y: float) -> bool:  # traite un evenement souris ; renvoie True s'il faut redessiner
        """event : down, move ou up. Renvoie True si l'affichage doit etre redessine."""
        px, py = self._to_image(x, y)  # position de la souris en pixels de l'image
        if event == "down":  # bouton presse : debut d'une boite
            self.drag_start, self.preview = (px, py), (px, py, px, py)  # memorise le point de depart ; boite de taille nulle
            return True  # redessiner
        if event == "move" and self.drag_start is not None:  # souris deplacee pendant le trace
            self.preview = (*self.drag_start, px, py)  # boite du point de depart jusqu'a la souris
            return True  # redessiner
        if event == "up" and self.drag_start is not None:  # bouton relache : fin du trace
            (x0, y0), self.drag_start, self.preview = self.drag_start, None, None  # recupere le depart, puis remet a zero
            x1, x2, y1, y2 = min(x0, px), max(x0, px), min(y0, py), max(y0, py)  # remet les coins dans l'ordre (trace dans tout sens)
            if x2 - x1 >= self.min_size and y2 - y1 >= self.min_size:  # boite assez grande (sinon simple clic)
                self.annotation.boxes.append(Box(self.class_names[self.current_class], x1, y1, x2, y2))  # ajoute la boite
            return True  # redessiner
        return False  # autre evenement : rien a redessiner

    def handle_key(self, key: str) -> str:  # applique une touche ; renvoie le nom de l'action
        """Renvoie l'action : class, delete, clear, next, prev, quit ou ignored."""
        if key.isdigit() and 1 <= int(key) <= min(9, len(self.class_names)):  # chiffre qui correspond a une classe
            self.current_class = int(key) - 1  # touche 1 -> classe d'indice 0
            return "class"  # changement de classe
        if key in ("delete", "x", "backspace"):  # effacer la derniere boite
            if self.annotation.boxes:  # s'il y a au moins une boite
                self.history.append(self.annotation.boxes.pop())  # retire la derniere et la garde dans l'historique
            return "delete"  # effacement
        if key == "c":  # tout effacer sur l'image
            self.history.extend(self.annotation.boxes)  # garde toutes les boites dans l'historique
            self.annotation.boxes = []  # vide la liste des boites
            return "clear"  # effacement complet
        if key in ("n", "enter", "right", "space"):  # image suivante
            return "next"  # passage a la suivante
        if key in ("p", "left"):  # image precedente
            return "prev"  # retour a la precedente
        if key in ("q", "esc"):  # quitter
            return "quit"  # arret
        return "ignored"  # toute autre touche est sans effet


# enregistre toutes les annotations (vtb.json, et YOLO si demande) ; ne renvoie rien
def save_session(annotations: list[ImageAnnotation], class_names: list[str], out_json: Path, yolo_dir: Path | None) -> None:
    """vtb.json de toutes les images (+ un .txt YOLO par image annotee si yolo_dir)."""
    from vtb.annotate.schema import save_annotations  # ecrivain du format natif vtb.json
    from vtb.annotate.yolo import annotation_to_yolo_lines  # conversion d'une annotation en lignes YOLO

    save_annotations(annotations, out_json, class_names)  # ecrit le vtb.json complet
    if yolo_dir is not None:  # si un export YOLO est demande
        yolo_dir.mkdir(parents=True, exist_ok=True)  # cree le dossier si besoin
        index = {n: i for i, n in enumerate(class_names)}  # nom de classe -> indice YOLO
        (yolo_dir / "classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")  # une classe par ligne
        for ann in annotations:  # pour chaque image
            lines = annotation_to_yolo_lines(ann, index, "detect")  # ses boites en lignes YOLO
            # ecrit "<nom de l'image>.txt" (vide si aucune boite : image negative)
            (yolo_dir / f"{Path(ann.image_path).stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# prepare les annotations de la session ; renvoie une ImageAnnotation par image (reprise ou vide)
def start_session(image_paths: list[str], out_json: Path) -> list[ImageAnnotation]:
    """Annotations a completer : reprises du vtb.json existant, ou vides pour les nouvelles images."""
    from vtb.annotate.schema import load_annotations  # lecteur du format natif vtb.json
    from vtb.io.images import image_size  # lit la taille d'une image

    existing = {}  # chemin -> annotation deja faite
    if out_json.exists():  # si une session precedente a ete enregistree
        existing = {a.image_path: a for a in load_annotations(out_json)[0]}  # [0] = les annotations (sans les classes)
    annotations = []  # annotations de la session
    for path in image_paths:  # pour chaque image a annoter
        if path in existing:  # deja commencee
            annotations.append(existing[path])  # on reprend son annotation
        else:  # nouvelle image
            width, height = image_size(path)  # lit sa taille
            annotations.append(ImageAnnotation(path, width, height))  # annotation vide
    return annotations  # renvoie la liste complete


# lance la fenetre d'annotation et la boucle souris / clavier ; renvoie les annotations finales
def run_box_annotator(
    image_paths: list[str],  # images a annoter
    class_names: list[str],  # classes possibles
    out_json: Path,  # fichier vtb.json de sauvegarde
    yolo_dir: Path | None = None,  # dossier d'export YOLO ; None = pas d'export
    max_display: int = 1000,  # taille maximale d'affichage en pixels
) -> list[ImageAnnotation]:  # renvoie toutes les annotations
    """Boucle d'affichage OpenCV (fenetre ; non testable sans ecran)."""
    import cv2  # OpenCV : fenetre, souris, clavier, dessin

    from vtb.annotate.draw import color_of, draw_annotation  # couleur d'une classe ; dessin des annotations
    from vtb.annotate.tools.keys import key_name  # code clavier -> nom de touche
    from vtb.io.images import read_image, to_uint8  # lecture d'image ; conversion en octets 0-255

    annotations = start_session(image_paths, out_json)  # annotations reprises ou vides
    window = "vtb - boites (glisser ; 1-9 classe ; x effacer ; n/p image ; q quitter)"  # titre = aide-memoire
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)  # cree une fenetre redimensionnable
    position = 0  # numero de l'image affichee
    events = {cv2.EVENT_LBUTTONDOWN: "down", cv2.EVENT_MOUSEMOVE: "move", cv2.EVENT_LBUTTONUP: "up"}  # code OpenCV -> nom
    current: dict[str, BoxAnnotatorState] = {}  # la fonction de rappel souris lit toujours l'etat de l'image affichee

    # fonction de rappel : OpenCV l'appelle a chaque evenement souris dans la fenetre ; ne renvoie rien
    def on_mouse(event: int, x: int, y: int, *_: object) -> None:
        if event in events and "state" in current:  # evenement utile et une image est affichee
            current["state"].on_mouse(events[event], x, y)  # transmet a l'etat de l'image courante

    cv2.setMouseCallback(window, on_mouse)  # branche la fonction de rappel sur la fenetre
    while 0 <= position < len(annotations):  # tant que la position designe une image
        ann = annotations[position]  # annotation de l'image courante
        image = to_uint8(read_image(ann.image_path, color="rgb"))  # lit l'image en couleur, octets 0-255
        scale = min(1.0, max_display / max(image.shape[:2]))  # facteur de reduction pour l'affichage
        state = BoxAnnotatorState(ann, class_names, scale=scale)  # nouvel etat pour cette image
        current["state"] = state  # le rappel souris travaille desormais sur cette image
        action = "ignored"  # action de depart (pour entrer dans la boucle)
        while action not in ("next", "prev", "quit"):  # tant qu'on reste sur cette image
            canvas = draw_annotation(image, ann, class_names)  # image avec les boites deja tracees
            if state.preview is not None:  # une boite est en cours de trace
                x1, y1, x2, y2 = state.preview  # ses coins (pixels image)
                # trace la boite provisoire, trait fin, dans la couleur de la classe courante
                cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color_of(class_names[state.current_class], class_names), 1)
            # reduit pour l'affichage si besoin (INTER_AREA : adapte a la reduction)
            view = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else canvas
            view = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)  # OpenCV affiche en BGR : on convertit
            # texte d'etat : position et classe courante
            text = f"{position + 1}/{len(annotations)}  classe : {state.current_class + 1}:{class_names[state.current_class]}"
            cv2.putText(view, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)  # ecrit en jaune
            cv2.imshow(window, view)  # affiche
            action = state.handle_key(key_name(cv2.waitKeyEx(30)))  # attend 30 ms une touche (pour suivre la souris en direct)
        save_session(annotations, class_names, out_json, yolo_dir)  # enregistre a chaque changement d'image
        if action == "quit":  # q ou Echap
            break  # sort de la boucle
        position += 1 if action == "next" else -1  # image suivante ou precedente
        position = max(position, 0)  # ne descend pas sous la premiere image
    cv2.destroyAllWindows()  # ferme la fenetre
    # message de bilan : nombre d'images et nombre total de boites
    logger.info("%d images, %d boites -> %s", len(annotations), sum(len(a.boxes) for a in annotations), out_json)
    return annotations  # renvoie les annotations finales
