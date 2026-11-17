"""Briques communes aux moteurs de detection : annotations par sous-ensemble, NMS, inference par tuiles, conversions."""

from __future__ import annotations  # les annotations de type restent du texte : syntaxe moderne "X | None" partout

import json  # lecture de fichiers JSON (le fichier d'annotations pivot)
from collections.abc import Callable  # type "fonction appelable" pour les annotations
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import numpy as np  # numpy : calcul sur des tableaux de nombres (ici des tableaux de boites)
import pandas as pd  # pandas : tableaux de donnees ; ici le fichier split.csv
import torch  # PyTorch : tenseurs (tableaux de nombres, sur CPU ou GPU)

# NMS (non-maximum suppression) : supprime les boites en double ; "batched" = separement pour chaque classe
from torchvision.ops import batched_nms

# format pivot des annotations : Box = boite droite ; ImageAnnotation = toutes les etiquettes d'une image ; lecture du fichier
from vtb.annotate.schema import Box, ImageAnnotation, load_annotations
from vtb.preprocess.tiles import offset_boxes, tile_grid  # decoupe en tuiles ; ramene des boites de tuile dans l'image entiere


# annotations d'un sous-ensemble (train, val ou test) ; renvoie (liste d'annotations, noms des classes)
def annotations_for(split_file: str | Path, subset: str) -> tuple[list[ImageAnnotation], list[str]]:
    """Annotations pivot d'un sous-ensemble, dans l'ordre du split.csv ; le pivot est ecrit par prepare a cote du split."""
    pivot = Path(split_file).with_suffix(".vtb.json")  # split.csv -> split.vtb.json : le fichier pivot a cote
    annotations, class_names = load_annotations(pivot)  # lit toutes les annotations et la liste des classes
    by_path = {a.image_path: a for a in annotations}  # index chemin d'image -> annotation (recherche rapide)
    frame = pd.read_csv(split_file, keep_default_na=False)  # lit split.csv ; keep_default_na=False : "" reste "" (pas NaN)
    # garde les images du sous-ensemble demande, dans l'ordre de split.csv, et renvoie aussi les classes
    return [by_path[p] for p in frame.loc[frame["split"] == subset, "path"]], class_names


def class_names_for(split_file: str | Path) -> list[str]:  # renvoie seulement la liste des classes du fichier pivot
    return json.loads(Path(split_file).with_suffix(".vtb.json").read_text(encoding="utf-8"))["class_names"]  # lit le JSON pivot


def boxes_from_array(rows: np.ndarray, class_names: list[str]) -> list[Box]:  # tableau de nombres -> liste de Box
    """(n, 6) [x1, y1, x2, y2, score, classe] -> boites du format pivot."""
    # pour chaque ligne : nom de classe (colonne 5), coins (x1, y1) et (x2, y2), score de confiance
    return [Box(class_names[int(r[5])], float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4])) for r in rows]


def boxes_to_array(boxes: list[Box], class_names: list[str]) -> np.ndarray:  # liste de Box -> tableau (n, 6) ; l'inverse
    index = {n: i for i, n in enumerate(class_names)}  # nom de classe -> numero
    if not boxes:  # si aucune boite
        return np.zeros((0, 6), np.float32)  # tableau vide mais avec 6 colonnes (le reste du code peut le traiter pareil)
    # une ligne par boite : coins, score (1.0 si c'est une verite terrain sans score), numero de classe
    return np.array([[b.x1, b.y1, b.x2, b.y2, b.score if b.score is not None else 1.0, index[b.label]] for b in boxes], np.float32)


def nms_per_class(rows: np.ndarray, iou: float) -> np.ndarray:  # supprime les doublons ; renvoie les lignes gardees
    """NMS par classe sur un tableau (n, 6) : deux boites de classes differentes ne s'eliminent jamais."""
    if len(rows) == 0:  # si aucune boite
        return rows  # rien a faire
    keep = batched_nms(  # renvoie les indices des boites gardees, de la plus sure a la moins sure
        torch.from_numpy(rows[:, :4]).float(),  # coins (n, 4) en tenseur float
        torch.from_numpy(rows[:, 4]).float(),  # scores de confiance (n,)
        torch.from_numpy(rows[:, 5]).long(),  # numeros de classe (n,) : la NMS est faite classe par classe
        iou,  # seuil IoU (intersection / union, de 0 a 1) : au-dela, la boite moins sure est un doublon
    )
    return rows[keep.numpy()]  # garde seulement ces lignes


def predict_tiled(  # predit sur une grande image tuile par tuile ; renvoie toutes les boites (n, 6) en coordonnees de l'image
    predict: Callable[[np.ndarray], np.ndarray],  # fonction qui predit sur une tuile et renvoie ses boites (n, 6)
    image: np.ndarray,  # la grande image complete
    tile: int,  # cote d'une tuile en pixels
    overlap: float,  # recouvrement entre tuiles voisines (fraction)
    iou_merge: float,  # seuil IoU pour fusionner les boites en double entre tuiles
) -> np.ndarray:  # renvoie un tableau (n, 6)
    """Inference sur une grande image par tuiles avec recouvrement, boites ramenees dans l'image puis fusionnees par NMS.

    predict(tuile) -> (n, 6) en coordonnees de la tuile. Regle de production : entrainer a la
    meme echelle que l'inference (sur des tuiles si l'on infere par tuiles).
    """
    rows = []  # boites trouvees dans chaque tuile
    for t in tile_grid(image.shape[0], image.shape[1], tile, overlap):  # pour chaque tuile de la grille couvrant l'image
        found = predict(image[t.slice()])  # predit sur le morceau d'image de la tuile
        if len(found):  # si la tuile contient au moins une boite
            rows.append(offset_boxes(found, t))  # decale les boites : coordonnees de tuile -> coordonnees de l'image
    merged = np.concatenate(rows) if rows else np.zeros((0, 6), np.float32)  # colle toutes les boites (ou tableau vide)
    return nms_per_class(merged, iou_merge)  # supprime les doublons des zones de recouvrement


def box_iou(a: np.ndarray, b: np.ndarray) -> float:  # recouvrement de deux boites ; renvoie un nombre entre 0 et 1
    """IoU de deux boites [x1, y1, x2, y2]."""
    # aire de l'intersection : largeur commune x hauteur commune (0 si les boites ne se touchent pas)
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter  # aire A + aire B - partie commune
    return float(inter / union) if union > 0 else 0.0  # IoU = intersection / union ; 0 si boites d'aire nulle


def mean_best_iou(mine: list[np.ndarray], reference: list[np.ndarray]) -> dict:  # compare deux listes de detections ; renvoie un dict
    """Compare deux chaines de detection image par image (verification d'export).

    Pour chaque boite de `mine`, IoU avec la boite la plus proche de `reference` de la meme image ;
    on rapporte aussi le nombre de boites de chaque cote (un ecart de nombre est un bug).
    """
    ious, n_mine, n_ref = [], 0, 0  # IoU trouves, nombre de boites de chaque cote
    for rows, ref in zip(mine, reference):  # pour chaque image : nos boites et celles de reference
        n_mine, n_ref = n_mine + len(rows), n_ref + len(ref)  # compte les boites de chaque cote
        for row in rows:  # pour chacune de nos boites
            if len(ref):  # boite identique = accord parfait, meme d'aire nulle (ecrasee sur le bord)
                # meilleur accord avec une boite de reference : 1 si coins egaux a 0.5 pixel pres, sinon l'IoU
                ious.append(max(1.0 if np.allclose(row[:4], r[:4], atol=0.5) else box_iou(row[:4], r[:4]) for r in ref))
    return {"boxes": n_mine, "reference_boxes": n_ref, "mean_iou": float(np.mean(ious)) if ious else None}  # resume chiffre


# ramene des predictions de l'image pretraitee vers l'image d'origine ; renvoie une nouvelle ImageAnnotation
def scale_annotation(ann: ImageAnnotation, sx: float, sy: float, image_path: str, width: int, height: int) -> ImageAnnotation:
    """Ramene des predictions faites sur l'image pretraitee dans l'image d'origine (facteurs sx, sy).

    Exact pour les boites et les polygones ; pour une boite orientee, exact si sx == sy (redimensionnement
    sans deformation), approche sinon.
    """
    from vtb.annotate.schema import Polygon, RotatedBox  # contour d'objet ; boite tournee (import local, seulement ici)

    out = ImageAnnotation(image_path, width, height)  # annotation vide pour l'image d'origine
    out.boxes = [Box(b.label, b.x1 * sx, b.y1 * sy, b.x2 * sx, b.y2 * sy, b.score) for b in ann.boxes]  # x par sx, y par sy
    out.polygons = [Polygon(p.label, [(x * sx, y * sy) for x, y in p.points], p.score) for p in ann.polygons]  # chaque point
    # centre et taille mis a l'echelle ; l'angle ne change pas (exact seulement si sx == sy)
    out.rotated = [RotatedBox(r.label, r.cx * sx, r.cy * sy, r.w * sx, r.h * sy, r.angle, r.score) for r in ann.rotated]
    return out  # renvoie l'annotation mise a l'echelle


def ensure_boxes(annotations: list[ImageAnnotation]) -> list[ImageAnnotation]:  # complete les boites droites ; renvoie la liste
    """YOLOX et Faster R-CNN n'apprennent que des boites droites : si une image n'a que des polygones
    ou des boites orientees (jeu cree depuis des masques), on prend leurs boites englobantes."""
    for ann in annotations:  # pour chaque image annotee
        if not ann.boxes:  # si elle n'a aucune boite droite
            # boite englobante de chaque polygone et de chaque boite orientee
            ann.boxes = [p.bounding_box() for p in ann.polygons] + [rotated_to_box(r) for r in ann.rotated]
    return annotations  # renvoie la meme liste (modifiee sur place)


def rotated_to_box(rbox) -> Box:  # boite orientee -> plus petite boite droite qui la contient
    """Boite droite englobant une boite orientee."""
    corners = np.array(rbox.corners())  # les 4 coins (x, y) de la boite tournee, en tableau (4, 2)
    return Box(  # construit la boite droite
        rbox.label,  # meme classe
        float(corners[:, 0].min()),  # x1 = plus petit x des coins
        float(corners[:, 1].min()),  # y1 = plus petit y des coins
        float(corners[:, 0].max()),  # x2 = plus grand x des coins
        float(corners[:, 1].max()),  # y2 = plus grand y des coins
        rbox.score,  # meme score de confiance
    )
