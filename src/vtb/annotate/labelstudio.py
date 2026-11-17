"""Label Studio : export JSON (liste de taches) en lecture, taches a importer en ecriture.

Coordonnees Label Studio : en POURCENTAGE de la taille de l'image. Un rectangle est donne par le
coin haut gauche de la boite non tournee (x, y), sa taille, et une rotation horaire en degres
autour de ce coin ; un polygone par ses points en pourcentage.

Types de resultats geres : rectanglelabels (avec ou sans rotation), polygonlabels, choices
(etiquette d'image). Le pinceau (brushlabels, encode en RLE propre a Label Studio) n'est PAS
gere : exporter ces projets au format "COCO" ou en masques PNG depuis Label Studio, puis les
lire avec vtb.annotate.coco ou vtb.annotate.masks. Seul l'export "JSON" standard est lu (pas le
"JSON_MIN").

Images : pour un serveur local, Label Studio sert les fichiers d'un dossier racine
(LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT) sous l'URL /data/local-files/?d=<chemin relatif>.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[str], X | None) partout

import json  # lecture et ecriture de fichiers JSON (texte structure en dictionnaires et listes)
import logging  # messages d'information et d'avertissement, plus souples que print
import os  # fonctions du systeme ; ici os.path pour calculer et nettoyer des chemins
import re  # expressions regulieres : rechercher / remplacer des motifs dans du texte
import uuid  # identifiants uniques aleatoires (Label Studio en veut un par resultat)
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux
from typing import Literal  # Literal : type qui n'accepte que quelques valeurs texte precises
from urllib.parse import parse_qs, unquote, urlparse  # decoupage d'URL : partie chemin, parametres ?d=..., %20 -> espace

from vtb.annotate.geometry import rotated_frame, rotated_from_origin  # conversions boite orientee <-> coin d'origine
from vtb.annotate.schema import Box, ImageAnnotation, Polygon  # format pivot de la toolbox
from vtb.io.images import image_size  # lit la largeur et la hauteur d'une image

logger = logging.getLogger(__name__)  # journal propre a ce module

LOCAL_FILES_PREFIX = "/data/local-files/?d="  # debut d'URL sous lequel Label Studio sert les fichiers locaux


def image_url(image_path: str, document_root: str | Path | None) -> str:  # renvoie la reference d'image a mettre dans la tache
    """Chemin d'image -> valeur de data.image : URL local-files si document_root est donne, sinon le chemin tel quel."""
    if document_root is None:  # pas de dossier racine : Label Studio ne servira pas l'image lui-meme
        return Path(image_path).as_posix()  # chemin tel quel, avec des "/"
    relative = Path(os.path.relpath(image_path, document_root)).as_posix()  # chemin de l'image relatif au dossier racine
    return LOCAL_FILES_PREFIX + relative  # URL que Label Studio sait servir


# retrouve le fichier local designe par une reference d'image Label Studio ; renvoie un Path
def resolve_image(reference: str, images_dir: Path | None, document_root: Path | None) -> Path:
    """data.image -> chemin local. Gere local-files, les fichiers televerses (prefixe aleatoire) et les URL."""
    if reference.startswith("/data/local-files/"):  # cas 1 : fichier servi depuis le dossier racine local
        relative = unquote(parse_qs(urlparse(reference).query).get("d", [""])[0])  # lit le parametre ?d= et decode les %xx
        if document_root is not None:  # si on connait le dossier racine
            return Path(os.path.normpath(Path(document_root) / relative))  # chemin complet = racine + chemin relatif
        return (images_dir or Path(".")) / Path(relative).name  # sinon : on cherche le nom du fichier dans images_dir
    name = Path(unquote(urlparse(reference).path)).name  # cas 2 : URL ou chemin ; on garde le nom du fichier
    if reference.startswith("/data/upload/"):  # fichier televerse dans Label Studio
        name = re.sub(r"^[0-9a-f]{8}-", "", name)  # Label Studio prefixe les fichiers televerses : on retire "1a2b3c4d-"
    if images_dir is not None and (images_dir / name).exists():  # si l'image existe dans le dossier d'images donne
        return images_dir / name  # on la prend la
    return Path(reference) if Path(reference).exists() else (images_dir or Path(".")) / name  # sinon la reference brute, ou au mieux


# convertit les objets d'une image en elements "result" Label Studio ; renvoie la liste de ces elements
def _result_items(ann: ImageAnnotation, from_name: str, to_name: str, choice_name: str) -> list[dict]:
    """Objets pivot -> elements "result" Label Studio (en pourcentage)."""
    w, h = float(ann.width), float(ann.height)  # taille de l'image, pour passer des pixels aux pourcentages
    # champs communs a tous les elements : noms des outils de la config, taille d'origine, image non tournee
    common = {"from_name": from_name, "to_name": to_name, "original_width": ann.width, "original_height": ann.height, "image_rotation": 0}
    items: list[dict] = []  # elements "result" a renvoyer
    for box in ann.boxes:  # pour chaque boite alignee
        # coin haut gauche et taille, en pourcentage de l'image ; rotation nulle
        value = {"x": box.x1 / w * 100, "y": box.y1 / h * 100, "width": box.width / w * 100, "height": box.height / h * 100, "rotation": 0}
        items.append(_item("rectanglelabels", {**value, "rectanglelabels": [box.label]}, common, box.score))  # ajoute la classe
    for rbox in ann.rotated:  # pour chaque boite orientee
        (x0, y0), _, _ = rotated_frame(rbox)  # coin d'origine : Label Studio tourne la boite autour de ce coin
        # coin d'origine et taille en pourcentage, rotation en degres entre 0 et 360
        value = {"x": x0 / w * 100, "y": y0 / h * 100, "width": rbox.w / w * 100, "height": rbox.h / h * 100, "rotation": rbox.angle % 360}
        items.append(_item("rectanglelabels", {**value, "rectanglelabels": [rbox.label]}, common, rbox.score))  # ajoute la classe
    for poly in ann.polygons:  # pour chaque polygone
        # sommets en pourcentage, contour ferme, classe
        value = {"points": [[x / w * 100, y / h * 100] for x, y in poly.points], "closed": True, "polygonlabels": [poly.label]}
        items.append(_item("polygonlabels", value, common, poly.score))  # ajoute l'element polygone
    if ann.labels:  # si l'image a des etiquettes de classification
        items.append(_item("choices", {"choices": list(ann.labels)}, {**common, "from_name": choice_name}, None))  # outil "choices"
    return items  # renvoie tous les elements


def _item(kind: str, value: dict, common: dict, score: float | None) -> dict:  # assemble un element "result" ; renvoie le dict
    item = {"id": uuid.uuid4().hex[:10], "type": kind, "value": value, **common}  # identifiant aleatoire de 10 caracteres
    if score is not None:  # si c'est une prediction avec une confiance
        item["score"] = float(score)  # garde la confiance
    return item  # renvoie l'element


# ecrit le fichier de taches a importer dans Label Studio ; ne renvoie rien
def write_labelstudio_tasks(
    annotations: list[ImageAnnotation],  # le jeu a ecrire
    out_json: str | Path,  # fichier JSON de taches a creer
    document_root: str | Path | None = None,  # dossier racine servi par Label Studio ; None = chemins bruts
    as_predictions: bool = False,  # True : pre-annotations a corriger ; False : annotations validees
    model_version: str = "vtb",  # nom de version affiche pour les predictions
    from_name: str = "label",  # nom de l'outil d'etiquetage dans la configuration Label Studio
    to_name: str = "image",  # nom de l'image dans la configuration Label Studio
    choice_name: str = "choice",  # nom de l'outil de classification dans la configuration
) -> None:  # None : ne renvoie rien, ecrit un fichier
    """Fichier de taches a importer dans Label Studio.

    as_predictions=True : les objets sont des PRE-ANNOTATIONS (champ predictions, avec score) que
    l'annotateur corrige ; sinon ce sont des annotations validees. from_name / to_name doivent
    correspondre aux noms de la configuration d'etiquetage (voir templates/labelstudio_*.xml).
    """
    tasks = []  # une tache par image
    for ann in annotations:  # pour chaque image
        result = _result_items(ann, from_name, to_name, choice_name)  # ses objets au format Label Studio
        task: dict = {"data": {to_name: image_url(ann.image_path, document_root)}}  # la tache pointe vers l'image
        if as_predictions:  # mode pre-annotation
            scores = [r["score"] for r in result if "score" in r]  # scores de tous les objets predits
            task["predictions"] = [  # liste des predictions de la tache (une seule ici)
                # score global de la prediction = moyenne des scores des objets (0 s'il n'y en a pas)
                {"model_version": model_version, "score": float(sum(scores) / len(scores)) if scores else 0.0, "result": result}
            ]
        else:  # mode annotations validees
            task["annotations"] = [{"result": result}]  # une annotation qui contient tous les objets
        tasks.append(task)  # ajoute la tache a la liste
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)  # cree le dossier de destination s'il n'existe pas
    Path(out_json).write_text(json.dumps(tasks, indent=1), encoding="utf-8")  # ecrit le JSON indente


# lit un export JSON Label Studio ; renvoie la liste pivot
def read_labelstudio(
    images_dir: str | Path | None,  # dossier ou chercher les images (secours)
    export_json: str | Path,  # fichier JSON exporte depuis Label Studio
    document_root: str | Path | None = None,  # dossier racine local-files, pour retrouver les images
    source: Literal["annotations", "predictions"] = "annotations",  # lire les annotations humaines ou les predictions
    to_name: str = "image",  # nom de l'image dans la configuration Label Studio
) -> list[ImageAnnotation]:  # renvoie une ImageAnnotation par tache
    """Export JSON Label Studio -> liste pivot. On garde la derniere annotation non annulee de chaque tache."""
    tasks = json.loads(Path(export_json).read_text(encoding="utf-8"))  # lit et decode tout le fichier
    images = Path(images_dir) if images_dir else None  # dossier d'images en Path, ou None
    root = Path(document_root) if document_root else None  # dossier racine en Path, ou None
    annotations, unsupported = [], 0  # resultats, et compteur des elements non geres
    for task in tasks:  # pour chaque tache (= une image)
        # reference de l'image : champ to_name, sinon la premiere valeur de "data" ; puis traduite en fichier local
        path = resolve_image(str(task["data"].get(to_name) or next(iter(task["data"].values()))), images, root)
        entries = [a for a in task.get(source, []) if not a.get("was_cancelled", False)]  # ignore les annotations annulees
        result = entries[-1]["result"] if entries else []  # garde la derniere (la plus recente) ; aucune = image vide
        sizes = [(r["original_width"], r["original_height"]) for r in result if "original_width" in r]  # tailles notees
        width, height = sizes[0] if sizes else image_size(path)  # taille notee, sinon lue dans l'image
        ann = ImageAnnotation(str(path), int(width), int(height))  # annotation vide pour cette image
        for item in result:  # pour chaque element dessine
            value, kind, score = item.get("value", {}), item.get("type"), item.get("score")  # valeurs, type, score
            if kind == "rectanglelabels":  # rectangle (tourne ou non)
                x, y = value["x"] / 100 * width, value["y"] / 100 * height  # coin d'origine : pourcentage -> pixels
                bw, bh = value["width"] / 100 * width, value["height"] / 100 * height  # taille : pourcentage -> pixels
                label, rotation = value["rectanglelabels"][0], float(value.get("rotation", 0) or 0)  # classe et angle (0 si absent)
                if abs(rotation % 360) < 1e-9:  # angle nul : boite alignee
                    ann.boxes.append(Box(label, x, y, x + bw, y + bh, score))  # coin + taille -> deux coins
                else:  # angle non nul : boite orientee
                    ann.rotated.append(rotated_from_origin(label, x, y, bw, bh, rotation, score))  # convertit vers centre + angle
            elif kind == "polygonlabels":  # polygone
                points = [(p[0] / 100 * width, p[1] / 100 * height) for p in value["points"]]  # sommets : pourcentage -> pixels
                ann.polygons.append(Polygon(value["polygonlabels"][0], points, score))  # ajoute le polygone
            elif kind == "choices":  # etiquette d'image (classification)
                ann.labels.extend(value.get("choices", []))  # ajoute les classes cochees
            else:  # autre type (pinceau, points cles...)
                unsupported += 1  # compte comme non gere
        annotations.append(ann)  # ajoute l'image terminee
    if unsupported:  # si des elements ont ete ignores
        logger.warning("%d resultats Label Studio non geres (brushlabels, keypoints...) ignores", unsupported)  # previent
    return annotations  # renvoie toutes les annotations


# fabrique la configuration d'etiquetage XML a coller dans Label Studio ; renvoie le texte XML
def labelstudio_config(kind: Literal["classification", "boxes", "rotated", "polygons"], class_names: list[str]) -> str:
    """Configuration d'etiquetage (XML) a coller dans Settings > Labeling Interface du projet."""
    if kind == "classification":  # classification d'image : une liste de choix
        choices = "\n".join(f'    <Choice value="{name}"/>' for name in class_names)  # une ligne <Choice> par classe
        # bloc <Choices> : choix unique, affiche en ligne
        body = f'  <Choices name="choice" toName="image" choice="single" showInline="true">\n{choices}\n  </Choices>'
    else:  # detection ou segmentation : un outil de dessin
        tag = {"boxes": "RectangleLabels", "rotated": "RectangleLabels", "polygons": "PolygonLabels"}[kind]  # outil selon la tache
        extra = ' canRotate="true"' if kind == "rotated" else (' canRotate="false"' if kind == "boxes" else "")  # rotation permise ?
        labels = "\n".join(f'    <Label value="{name}"/>' for name in class_names)  # une ligne <Label> par classe
        body = f'  <{tag} name="label" toName="image"{extra}>\n{labels}\n  </{tag}>'  # bloc de l'outil avec ses classes
    return f'<View>\n  <Image name="image" value="$image" zoom="true"/>\n{body}\n</View>\n'  # page complete : image + outil
