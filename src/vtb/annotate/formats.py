"""Point d'entree unique pour lire et ecrire les annotations, quel que soit le format.

Ajouter un format = ecrire un lecteur et un ecrivain, puis les inscrire dans READERS / WRITERS.

| format      | annotations_path / out          | boites | orientees | polygones | etiquettes d'image |
|-------------|---------------------------------|--------|-----------|-----------|--------------------|
| voc         | dossier de .xml                 | oui    | non       | non       | non                |
| yolo        | dossier de .txt (+ classes)     | oui    | oui       | oui       | non                |
| coco        | fichier .json                   | oui    | oui (1)   | oui       | oui (1)            |
| labelme     | dossier de .json (X-AnyLabeling)| oui    | oui       | oui       | oui (flags)        |
| labelstudio | fichier .json (export / import) | oui    | oui       | oui       | oui (choices)      |
| cvat        | fichier .xml (CVAT images 1.1)  | oui    | oui       | oui       | oui (tags)         |
| masks       | dossier de .png                 | (2)    | (2)       | (2)       | non                |
| vtb         | fichier .vtb.json (natif)       | oui    | oui       | oui       | oui                |

(1) extension de la toolbox, ignoree par les autres outils ; (2) objets deduits des masques
(kind = polygons / rotated / boxes) a la lecture, masques dessines a l'ecriture.

Parametres supplementaires (mots-cles) : class_names (yolo, masks, ecritures), images_dir (coco,
cvat en ecriture : chemins relatifs), document_root et as_predictions (labelstudio), kind,
mask_suffix, extensions, min_area, epsilon (masks), task (yolo : detect / obb / segment).
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[str], X | None) partout

from collections.abc import Callable  # Callable : type "fonction", pour annoter les tables de lecteurs / ecrivains
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import yaml  # PyYAML : lit les fichiers .yaml (ici data.yaml, ou YOLO range ses noms de classes)

# un lecteur et un ecrivain par format ; chacun convertit vers / depuis le format pivot ImageAnnotation
from vtb.annotate.coco import read_coco, write_coco  # COCO : un fichier JSON pour tout le jeu
from vtb.annotate.cvat import read_cvat_xml, write_cvat_xml  # CVAT : un fichier XML pour toute la tache
from vtb.annotate.labelme import read_labelme_folder, write_labelme_folder  # labelme : un JSON par image
from vtb.annotate.labelstudio import read_labelstudio, write_labelstudio_tasks  # Label Studio : un JSON de taches
from vtb.annotate.masks import read_mask_folder, write_mask_folder  # masques PNG : une image de classes par image

# format pivot et format natif vtb.json de la toolbox
from vtb.annotate.schema import ImageAnnotation, class_names_of, load_annotations, save_annotations
from vtb.annotate.voc import read_voc_folder, write_voc_folder  # Pascal VOC : un XML par image
from vtb.annotate.yolo import read_yolo_folder, write_yolo_dataset  # YOLO : un .txt par image, coordonnees normalisees


def _yolo_class_names(labels_dir: Path) -> list[str]:  # trouve les noms de classes d'un jeu YOLO ; renvoie la liste ordonnee
    """Noms de classes YOLO : data.yaml (parent ou grand-parent) ou classes.txt a cote des .txt."""
    for candidate in (labels_dir / "classes.txt", labels_dir.parent / "classes.txt"):  # cherche classes.txt ici puis au-dessus
        if candidate.exists():  # si ce fichier existe
            # une classe par ligne non vide, espaces de debut et de fin retires
            return [line.strip() for line in candidate.read_text(encoding="utf-8").splitlines() if line.strip()]
    for candidate in (labels_dir.parent / "data.yaml", labels_dir.parent.parent / "data.yaml"):  # sinon cherche data.yaml
        if candidate.exists():  # si ce fichier existe
            names = yaml.safe_load(candidate.read_text(encoding="utf-8"))["names"]  # lit la cle "names" du YAML
            # "names" peut etre un dict {0: "a", 1: "b"} (trie par numero) ou une simple liste
            return [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    raise FileNotFoundError(f"ni classes.txt ni data.yaml pres de {labels_dir}")  # rien trouve : arret avec message clair


# noms de classes d'un dossier de masques ; renvoie la liste ordonnee (indice = valeur de pixel)
def _mask_class_names(masks_dir: Path, class_names: list[str] | None) -> list[str]:
    """classes.txt dans le dossier des masques, sinon une seule classe "defect" (masque binaire)."""
    if class_names:  # si l'appelant a donne les classes
        return list(class_names)  # on les utilise telles quelles
    candidate = masks_dir / "classes.txt"  # sinon on cherche classes.txt dans le dossier des masques
    if candidate.exists():  # si ce fichier existe
        return [line.strip() for line in candidate.read_text(encoding="utf-8").splitlines() if line.strip()]  # une classe par ligne
    return ["defect"]  # par defaut : une seule classe (masque binaire objet / fond)


# lecteur du format natif vtb.json ; **_ avale les options inutiles ici ; renvoie la liste pivot
def _read_vtb(images_dir: Path, annotations_path: Path, **_) -> list[ImageAnnotation]:
    annotations, _names = load_annotations(annotations_path)  # relit le fichier (les noms de classes ne servent pas ici)
    return annotations  # renvoie seulement les annotations


# lecteur des masques PNG : filtre les options utiles puis appelle read_mask_folder ; renvoie la liste pivot
def _read_masks(images_dir: Path, masks_dir: Path, class_names: list[str] | None = None, **kwargs) -> list[ImageAnnotation]:
    # garde seulement les options que read_mask_folder connait, et seulement celles qui ont une valeur
    accepted = {k: v for k, v in kwargs.items() if k in {"kind", "mask_suffix", "extensions", "min_area", "epsilon"} and v is not None}
    return read_mask_folder(images_dir, masks_dir, _mask_class_names(masks_dir, class_names), **accepted)  # ** passe le dict en options


# ecrivain YOLO simplifie (tout le jeu dans un seul sous-ensemble) ; ne renvoie rien
def _write_yolo(
    annotations: list[ImageAnnotation],  # le jeu a ecrire
    out: Path,  # dossier de sortie du jeu YOLO
    class_names: list[str] | None = None,  # ordre des classes ; None = deduit des annotations
    task: str = "detect",  # "detect" (boites), "obb" (boites orientees) ou "segment" (polygones)
    split: str = "train",  # sous-ensemble ou ranger toutes les images
    **_,  # options inutiles pour YOLO, ignorees
) -> None:  # None : ne renvoie rien, ecrit des fichiers
    """Ecriture YOLO simple : tout dans un sous-ensemble (split) ; pour un vrai jeu decoupe, appeler write_yolo_dataset."""
    names = class_names or class_names_of(annotations)  # classes donnees, sinon deduites et triees
    # chaque image est affectee au meme sous-ensemble ; "type: ignore" fait taire un avertissement du verificateur de types
    write_yolo_dataset(annotations, {a.image_path: split for a in annotations}, names, out, task=task)  # type: ignore[arg-type]


# table des lecteurs : nom du format -> fonction (images, annotations, options) qui renvoie la liste pivot
READERS: dict[str, Callable[..., list[ImageAnnotation]]] = {
    "voc": lambda images, ann, **_: read_voc_folder(images, ann),  # lambda = petite fonction ecrite sur une ligne
    "yolo": lambda images, ann, class_names=None, task=None, **_: read_yolo_folder(  # YOLO a besoin des noms de classes
        images,  # dossier des images
        ann,  # dossier des .txt
        class_names or _yolo_class_names(Path(ann)),  # classes donnees, sinon lues dans classes.txt / data.yaml
        task,  # type de taches (detect / obb / segment) ; None = devine
    ),
    "coco": lambda images, ann, **_: read_coco(images, ann),  # COCO : fichier JSON unique
    "labelme": lambda images, ann, **_: read_labelme_folder(images, ann),  # labelme : dossier de JSON
    # Label Studio : document_root = racine des chemins d'image ; source = lire les annotations ou les predictions
    "labelstudio": lambda images, ann, document_root=None, source="annotations", **_: read_labelstudio(images, ann, document_root, source),
    "cvat": lambda images, ann, **_: read_cvat_xml(images, ann),  # CVAT : fichier XML unique
    "masks": _read_masks,  # masques PNG : fonction definie plus haut
    "vtb": _read_vtb,  # format natif de la toolbox
}

# table des ecrivains : nom du format -> fonction (annotations, sortie, options) qui ecrit les fichiers
WRITERS: dict[str, Callable[..., None]] = {
    "voc": lambda annotations, out, **_: write_voc_folder(annotations, out),  # un XML par image
    "yolo": _write_yolo,  # YOLO : fonction definie plus haut
    # COCO : class_names fixe l'ordre des categories ; images_dir sert a ecrire des chemins relatifs
    "coco": lambda annotations, out, class_names=None, images_dir=None, **_: write_coco(annotations, out, class_names, images_dir),
    # labelme : image_flags = cases a cocher d'image a proposer dans l'outil
    "labelme": lambda annotations, out, image_flags=None, **_: write_labelme_folder(annotations, out, image_flags),
    # Label Studio : as_predictions=True ecrit des pre-annotations (a corriger) plutot que des annotations
    "labelstudio": lambda annotations, out, document_root=None, as_predictions=False, **_: write_labelstudio_tasks(
        annotations,  # le jeu a ecrire
        out,  # fichier JSON de taches
        document_root,  # racine des chemins d'image vue par Label Studio
        as_predictions,  # True : pre-annotations ; False : annotations validees
    ),
    # CVAT : memes options que COCO
    "cvat": lambda annotations, out, class_names=None, images_dir=None, **_: write_cvat_xml(annotations, out, class_names, images_dir),
    # masques PNG : binary=True ecrit 0 / 255 (objet ou fond) au lieu d'un numero de classe par pixel
    "masks": lambda annotations, out, class_names=None, binary=False, **_: write_mask_folder(
        annotations,  # le jeu a ecrire
        out,  # dossier des masques
        class_names or class_names_of(annotations),  # ordre des classes : valeur de pixel = indice + 1
        binary,  # masque binaire ou masque de classes
    ),
    "vtb": lambda annotations, out, class_names=None, **_: save_annotations(annotations, out, class_names),  # format natif
}


# lit un jeu annote dans le format demande ; renvoie la liste pivot
def read_annotations(fmt: str, images_dir: str | Path, annotations_path: str | Path, **kwargs) -> list[ImageAnnotation]:
    """Lit un jeu annote : images_dir (les images) et annotations_path (dossier ou fichier selon le format)."""
    if fmt not in READERS:  # si le format n'est pas dans la table des lecteurs
        raise ValueError(f"format d'annotation inconnu : {fmt} (disponibles : {sorted(READERS)})")  # arret, liste les formats connus
    return READERS[fmt](Path(images_dir), Path(annotations_path), **kwargs)  # appelle le bon lecteur avec les options


def write_annotations(fmt: str, annotations: list[ImageAnnotation], out: str | Path, **kwargs) -> None:  # ecrit dans le format demande
    """Ecrit un jeu annote dans le format demande (dossier ou fichier selon le format)."""
    if fmt not in WRITERS:  # si le format n'est pas dans la table des ecrivains
        raise ValueError(f"ecriture non disponible pour : {fmt} (disponibles : {sorted(WRITERS)})")  # arret avec message clair
    WRITERS[fmt](annotations, Path(out), **kwargs)  # appelle le bon ecrivain avec les options


# convertit un format en un autre (lecture puis ecriture) ; renvoie les annotations lues
def convert(src_fmt: str, src_images: str | Path, src: str | Path, dst_fmt: str, dst: str | Path, **kwargs) -> list[ImageAnnotation]:
    """Conversion A -> B en passant par le pivot ; renvoie les annotations lues (pour controle).

    Les mots-cles servent a la lecture (class_names, task, document_root, kind...) et a
    l'ecriture ; images_dir (chemins relatifs dans COCO / CVAT) vaut src_images par defaut.
    """
    # options qui concernent la lecture (les autres ne sont passees qu'a l'ecriture)
    read_keys = {"class_names", "task", "document_root", "source", "kind", "mask_suffix", "extensions", "min_area", "epsilon"}
    # lit le jeu source en ne transmettant que les options de lecture
    annotations = read_annotations(src_fmt, src_images, src, **{k: v for k, v in kwargs.items() if k in read_keys})
    kwargs.setdefault("images_dir", src_images)  # si images_dir n'est pas donne, on prend le dossier des images source
    write_annotations(dst_fmt, annotations, dst, **kwargs)  # ecrit le jeu dans le format cible
    return annotations  # renvoie ce qui a ete lu, pour pouvoir le verifier
