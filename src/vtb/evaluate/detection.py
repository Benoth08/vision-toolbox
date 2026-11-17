"""Evaluation de la detection au protocole COCO (pycocotools), commune a tous les moteurs.

mAP@0.5 : une detection est juste si son IoU avec une vraie boite de la meme classe atteint 0,5.
mAP@0.5:0.95 : moyenne sur dix seuils d'IoU de 0,50 a 0,95 ; recompense les boites precises.
Toujours regarder l'AP par classe : une moyenne correcte peut cacher une classe ratee.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes

import contextlib  # outils pour les blocs "with" (ici : rediriger l'affichage)
import io  # io : fichiers en memoire (ici pour avaler les messages de pycocotools)
from collections.abc import Sequence  # type "suite d'elements" (liste, tuple) pour les annotations

import numpy as np  # numpy : calcul sur des tableaux de nombres

from vtb.annotate.schema import Box, ImageAnnotation  # Box = boite (classe, x1, y1, x2, y2, score) ; annotations d'une image


# convertit les vraies annotations au format COCO ; renvoie le dict "verite terrain"
def _ground_truth(annotations: Sequence[ImageAnnotation], class_names: Sequence[str]) -> dict:
    index = {name: i for i, name in enumerate(class_names)}  # nom de classe -> numero
    images, objects = [], []  # liste des images et liste des objets au format COCO
    for image_id, ann in enumerate(annotations):  # pour chaque image annotee (image_id = son numero)
        images.append({"id": image_id, "file_name": ann.image_path, "width": ann.width, "height": ann.height})  # fiche de l'image
        # boites droites ; a defaut, boites englobantes des polygones ; a defaut, des boites tournees
        boxes = list(ann.boxes) or [p.bounding_box() for p in ann.polygons] or [_rotated_to_box(r) for r in ann.rotated]
        for box in boxes:  # pour chaque vraie boite
            w, h = box.x2 - box.x1, box.y2 - box.y1  # largeur et hauteur
            objects.append(  # ajoute l'objet
                {  # fiche COCO de l'objet
                    "id": len(objects) + 1,  # identifiant unique a partir de 1 (pycocotools reserve 0)
                    "image_id": image_id,  # image a laquelle il appartient
                    "category_id": index[box.label],  # numero de sa classe
                    "bbox": [box.x1, box.y1, w, h],  # boite au format COCO : x, y du coin haut gauche, largeur, hauteur
                    "area": w * h,  # surface (sert a classer petits / moyens / grands objets)
                    "iscrowd": 0,  # 0 = objet individuel (pas une foule regroupee)
                }
            )
    # dict COCO complet : images, objets, et liste des categories (numero, nom)
    return {"images": images, "annotations": objects, "categories": [{"id": i, "name": n} for i, n in enumerate(class_names)]}


def _rotated_to_box(rbox) -> Box:  # convertit une boite tournee en boite droite qui l'englobe
    corners = np.array(rbox.corners())  # les 4 coins de la boite tournee, tableau (4, 2)
    return Box(  # boite droite englobante
        rbox.label,  # meme classe
        float(corners[:, 0].min()),  # x1 = plus petit x des coins
        float(corners[:, 1].min()),  # y1 = plus petit y
        float(corners[:, 0].max()),  # x2 = plus grand x
        float(corners[:, 1].max()),  # y2 = plus grand y
        rbox.score,  # meme score
    )


# lance l'evaluation COCO ; renvoie l'objet evaluateur (ou None s'il n'y a aucune prediction)
def _evaluator(ground_truth: dict, predictions: Sequence[Sequence[Box]], class_names: Sequence[str]):
    from pycocotools.coco import COCO  # COCO : charge un jeu d'annotations au format COCO
    from pycocotools.cocoeval import COCOeval  # COCOeval : calcule les mAP selon le protocole officiel

    index = {name: i for i, name in enumerate(class_names)}  # nom de classe -> numero
    results = [  # predictions au format COCO (liste de dicts)
        {  # une prediction
            "image_id": i,  # numero de l'image
            "category_id": index[b.label],  # numero de la classe predite
            "bbox": [b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1],  # boite : x, y, largeur, hauteur
            "score": float(b.score if b.score is not None else 1.0),  # confiance (1.0 si absente)
        }
        for i, boxes in enumerate(predictions)  # pour chaque image et ses boites predites
        for b in boxes  # pour chaque boite predite
    ]
    if not results:  # aucune prediction du tout
        return None  # rien a evaluer
    with contextlib.redirect_stdout(io.StringIO()):  # pycocotools ecrit beaucoup sur la sortie standard
        coco_gt = COCO()  # jeu COCO vide
        coco_gt.dataset = ground_truth  # on lui donne notre verite terrain
        coco_gt.createIndex()  # construit ses tables de recherche internes
        evaluator = COCOeval(coco_gt, coco_gt.loadRes(results), iouType="bbox")  # evaluateur verite vs predictions, sur des boites
        evaluator.evaluate()  # apparie predictions et verites pour chaque image et classe
        evaluator.accumulate()  # construit les courbes precision / rappel
        evaluator.summarize()  # calcule les chiffres resumes (stats : mAP...)
    return evaluator  # renvoie l'evaluateur (resultats dans .eval et .stats)


# calcule mAP@0.5, mAP@0.5:0.95 et AP par classe ; renvoie un dict
def coco_metrics(annotations: Sequence[ImageAnnotation], predictions: Sequence[Sequence[Box]], class_names: Sequence[str]) -> dict:
    """mAP@0.5, mAP@0.5:0.95 et AP par classe ; predictions[i] = boites predites de annotations[i] (avec score)."""
    # IoU (intersection sur union) = surface commune / surface totale de deux boites ; AP = aire sous la courbe precision-rappel
    ground_truth = _ground_truth(annotations, class_names)  # verite terrain au format COCO
    evaluator = _evaluator(ground_truth, predictions, class_names)  # evaluation complete
    if evaluator is None:  # aucune prediction
        return {"map50": 0.0, "map50_95": 0.0, "per_class": {n: {"ap50": 0.0, "ap50_95": 0.0} for n in class_names}}  # tout a zero
    precision = evaluator.eval["precision"][:, :, :, 0, -1]  # (seuils IoU, rappel, classes) ; toutes tailles, 100 detections
    per_class = {}  # AP par classe
    for k, class_id in enumerate(evaluator.params.catIds):  # pour chaque classe evaluee (k = sa position dans le tableau)
        values = precision[:, :, k]  # precisions de cette classe (seuils IoU x niveaux de rappel)
        at50, valid = values[0][values[0] > -1], values[values > -1]  # au seuil 0.5 seulement / a tous les seuils ; -1 = non defini
        per_class[class_names[class_id]] = {  # resultats de la classe
            "ap50": round(float(at50.mean()), 4) if at50.size else 0.0,  # AP a IoU 0.5
            "ap50_95": round(float(valid.mean()), 4) if valid.size else 0.0,  # AP moyenne sur IoU 0.5 a 0.95
        }
    # stats[1] = mAP@0.5 ; stats[0] = mAP@0.5:0.95 (ordre fixe de pycocotools)
    return {"map50": float(evaluator.stats[1]), "map50_95": float(evaluator.stats[0]), "per_class": per_class}


def operating_point(  # mesure ce que voit l'operateur a un seuil de confiance donne ; renvoie un dict de comptages et taux
    annotations: Sequence[ImageAnnotation],  # vraies annotations, une par image
    predictions: Sequence[Sequence[Box]],  # boites predites, une liste par image
    class_names: Sequence[str],  # noms des classes
    conf: float,  # seuil de confiance de production : les boites en dessous sont ignorees
    iou: float = 0.5,  # IoU minimale pour qu'une prediction compte comme juste
) -> dict:  # renvoie les resultats
    """Ce que verra l'operateur au seuil de production `conf` : le mAP resume toute la courbe, pas ce point-la.

    Appariement glouton par image et par classe (scores decroissants, IoU >= iou) ; renvoie TP, FP, FN,
    precision et rappel par classe et au total, plus deux chiffres "image" : la part des images
    avec defaut ou au moins un defaut est signale, et la part des images saines signalees a tort.
    """
    from vtb.tasks.detection.common import box_iou  # calcule l'IoU de deux boites

    # TP = vrais positifs (bonnes detections), FP = fausses alarmes, FN = defauts manques ; un compteur par classe
    counts = {n: {"tp": 0, "fp": 0, "fn": 0} for n in class_names}
    flagged_defect = flagged_clean = n_defect = n_clean = 0  # compteurs au niveau image
    for ann, preds in zip(annotations, predictions):  # pour chaque image : verite et predictions
        truth = list(ann.boxes) or [p.bounding_box() for p in ann.polygons] or [_rotated_to_box(r) for r in ann.rotated]  # vraies boites
        kept = [b for b in preds if (b.score if b.score is not None else 1.0) >= conf]  # predictions au-dessus du seuil
        if truth:  # image avec au moins un defaut
            n_defect += 1  # une image defectueuse de plus
            flagged_defect += bool(kept)  # +1 si au moins une boite a ete signalee
        else:  # image saine
            n_clean += 1  # une image saine de plus
            flagged_clean += bool(kept)  # +1 si elle a ete signalee a tort
        for name in class_names:  # pour chaque classe
            gt = [np.array([b.x1, b.y1, b.x2, b.y2]) for b in truth if b.label == name]  # vraies boites de cette classe
            pr = sorted((b for b in kept if b.label == name), key=lambda b: -(b.score or 0.0))  # meilleur score d'abord
            used = [False] * len(gt)  # chaque vraie boite ne peut etre trouvee qu'une fois
            for b in pr:  # pour chaque prediction, de la plus sure a la moins sure
                # IoU avec chaque vraie boite encore libre (-1 pour celles deja prises)
                scores = [box_iou(np.array([b.x1, b.y1, b.x2, b.y2]), g) if not used[k] else -1.0 for k, g in enumerate(gt)]
                best = int(np.argmax(scores)) if scores else -1  # vraie boite la plus recouverte (-1 s'il n'y en a aucune)
                if best >= 0 and scores[best] >= iou:  # recouvrement suffisant
                    used[best] = True  # cette vraie boite est trouvee
                    counts[name]["tp"] += 1  # bonne detection
                else:  # pas de vraie boite assez proche
                    counts[name]["fp"] += 1  # fausse alarme
            counts[name]["fn"] += used.count(False)  # vraies boites jamais trouvees = defauts manques

    def rates(c: dict) -> dict:  # ajoute precision et rappel a un dict de comptages ; renvoie le dict complete
        precision = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0  # precision = part des alarmes justes
        recall = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0  # rappel = part des defauts trouves
        return {**c, "precision": round(precision, 4), "recall": round(recall, 4)}  # comptages + taux arrondis

    total = {k: sum(c[k] for c in counts.values()) for k in ("tp", "fp", "fn")}  # totaux toutes classes confondues
    return {  # resultats
        "conf": conf,  # seuil de confiance utilise
        "iou": iou,  # seuil d'IoU utilise
        "overall": rates(total),  # precision et rappel globaux
        "per_class": {n: rates(c) for n, c in counts.items()},  # precision et rappel par classe
        "image_recall": round(flagged_defect / n_defect, 4) if n_defect else None,  # part des images defectueuses signalees
        "image_false_alarm_rate": round(flagged_clean / n_clean, 4) if n_clean else None,  # part des images saines signalees a tort
    }
