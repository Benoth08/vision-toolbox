"""Metriques de segmentation : matrice de confusion des pixels -> IoU et Dice par classe, plus la decision par image.

IoU = TP / (TP + FP + FN) ; Dice = 2 TP / (2 TP + FP + FN). Le fond domine toujours les pixels :
on rapporte la moyenne sur les classes de DEFAUT (mIoU defauts) en plus de la moyenne globale.
La decision qui compte en production est souvent par image : "cette image contient-elle un defaut
d'au moins N pixels ?" -> rappel image et taux de fausse alarme.
"""

from __future__ import annotations  # permet les annotations de type modernes (float | None)

import numpy as np  # numpy : calcul sur des tableaux de nombres


# matrice de confusion : compte, pour chaque couple (classe vraie, classe predite), le nombre de pixels ; renvoie un tableau (K, K)
def confusion(pred: np.ndarray, truth: np.ndarray, num_classes: int) -> np.ndarray:
    """Matrice (K, K) : lignes = verite, colonnes = prediction, en nombre de pixels."""
    # un numero unique par couple (verite, prediction) : verite * K + prediction ; ravel met les pixels a plat
    index = truth.astype(np.int64).ravel() * num_classes + pred.astype(np.int64).ravel()
    # bincount compte chaque numero (K*K cases au total), puis on range les comptes en matrice K x K
    return np.bincount(index, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def scores_from_confusion(matrix: np.ndarray, class_names: list[str]) -> dict:  # IoU, Dice et precision pixel ; renvoie un dictionnaire
    tp = np.diag(matrix).astype(np.float64)  # TP (vrais positifs) : la diagonale = pixels bien classes, par classe
    fp, fn = matrix.sum(0) - tp, matrix.sum(1) - tp  # FP : predits a tort dans la classe ; FN : pixels de la classe rates
    with np.errstate(invalid="ignore", divide="ignore"):  # classe absente partout : 0/0 donne nan sans message d'avertissement
        iou = tp / (tp + fp + fn)  # IoU (intersection sur union) : recouvrement entre zone predite et zone vraie, de 0 a 1
        dice = 2 * tp / (2 * tp + fp + fn)  # Dice : autre mesure de recouvrement, toujours >= IoU
    # pour chaque classe (k = indice, n = nom) : IoU, Dice arrondis et nombre de pixels vrais de cette classe
    per_class = {n: {"iou": _r(iou[k]), "dice": _r(dice[k]), "pixels": int(matrix[k].sum())} for k, n in enumerate(class_names)}
    defects = [k for k in range(1, len(class_names)) if matrix[k].sum() > 0]  # classes de defaut (pas le fond 0) presentes dans la verite
    return {  # le dictionnaire des scores
        "pixel_accuracy": _r(tp.sum() / max(matrix.sum(), 1)),  # part des pixels bien classes (gonflee par le fond)
        "miou": _r(np.nanmean(iou)),  # moyenne des IoU de toutes les classes, fond compris (nanmean ignore les nan)
        "miou_defects": _r(np.nanmean(iou[defects])) if defects else None,  # moyenne des IoU des seuls defauts (le chiffre utile)
        "mdice_defects": _r(np.nanmean(dice[defects])) if defects else None,  # moyenne des Dice des seuls defauts
        "per_class": per_class,  # detail par classe
    }


def image_decisions(pred_areas: list[int], has_defect: list[bool], min_area: int) -> dict:  # bilan par image ; renvoie un dictionnaire
    """Une image est signalee si la surface predite de defaut atteint min_area pixels."""
    flagged = np.array(pred_areas) >= min_area  # vrai pour chaque image dont la surface de defaut predite est assez grande
    truth = np.array(has_defect, bool)  # vrai pour chaque image qui contient vraiment un defaut
    recall = flagged[truth].mean() if truth.any() else None  # rappel image : part des images defectueuses signalees
    false_alarm = flagged[~truth].mean() if (~truth).any() else None  # fausses alarmes : part des images saines signalees (~ = non)
    return {  # le dictionnaire du bilan
        "min_defect_area": min_area,  # surface minimale utilisee
        "image_recall": _r(recall),  # rappel image arrondi
        "image_false_alarm_rate": _r(false_alarm),  # taux de fausses alarmes arrondi
        "n_defect_images": int(truth.sum()),  # nombre d'images avec defaut
        "n_clean_images": int((~truth).sum()),  # nombre d'images saines
    }


def _r(value) -> float | None:  # arrondit a 4 decimales ; renvoie None si la valeur manque ou n'est pas un nombre fini
    return None if value is None or not np.isfinite(value) else round(float(value), 4)  # None aussi pour nan / infini (JSON propre)
