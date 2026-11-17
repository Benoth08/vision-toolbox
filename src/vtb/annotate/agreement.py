"""Accord entre annotateurs : mesurer la qualite des etiquettes avant d'accuser le modele.

Pratique recommandee : faire annoter deux fois 5 a 10 % du jeu par deux personnes, sans qu'elles
se voient. Si l'accord entre humains est de 0,80, aucun modele ne fera durablement mieux que
0,80 sur ces etiquettes : c'est le plafond, et c'est souvent le guide d'annotation qu'il faut
corriger, pas le reseau.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[str], X | None) partout

from collections.abc import Sequence  # Sequence : type "liste ou tuple", pour annoter les parametres

import numpy as np  # numpy : calcul sur des tableaux de nombres
import pandas as pd  # pandas : tableaux a lignes et colonnes nommees (DataFrame), pratiques a afficher
from scipy.optimize import linear_sum_assignment  # algorithme hongrois : meilleur appariement un-a-un entre deux listes

# kappa de Cohen (accord corrige du hasard) et matrice de confusion (qui a dit quoi contre qui)
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from vtb.annotate.geometry import box_iou  # IoU (intersection sur union) : recouvrement de deux boites, entre 0 et 1
from vtb.annotate.schema import Box, ImageAnnotation  # Box : boite alignee ; ImageAnnotation : toutes les etiquettes d'une image


# accord de classification entre deux annotateurs ; renvoie un dict (kappa, accord brut, nb d'images, matrice)
def classification_agreement(labels_a: Sequence[str], labels_b: Sequence[str]) -> dict:
    """Kappa de Cohen (accord corrige du hasard), accord brut, matrice de confusion a x b.

    Lecture du kappa : < 0,4 faible, 0,4 a 0,6 moyen, 0,6 a 0,8 bon, > 0,8 tres bon.
    """
    a, b = [str(x) for x in labels_a], [str(x) for x in labels_b]  # convertit tout en texte pour comparer sans surprise
    if len(a) != len(b):  # les deux listes doivent decrire les memes images, une etiquette par image
        raise ValueError("les deux annotateurs doivent avoir etiquete les memes images")  # arret avec un message clair
    classes = sorted(set(a) | set(b))  # toutes les classes vues par l'un ou l'autre, triees (ordre stable)
    matrix = confusion_matrix(a, b, labels=classes)  # case [i, j] = nb d'images classe i pour a et classe j pour b
    return {  # renvoie le bilan sous forme de dictionnaire
        # kappa (1 = accord parfait, 0 = pas mieux que le hasard) ; une seule classe en tout : accord parfait par convention
        "kappa": float(cohen_kappa_score(a, b)) if len(set(a) | set(b)) > 1 else 1.0,
        "raw_agreement": float(np.mean(np.array(a) == np.array(b))),  # part des images ou les deux disent la meme chose
        "n_images": len(a),  # nombre d'images comparees
        # matrice de confusion en tableau lisible : lignes = classes de a, colonnes = classes de b
        "confusion": pd.DataFrame(matrix, index=[f"a:{c}" for c in classes], columns=[f"b:{c}" for c in classes]),
    }


# apparie les boites de deux annotateurs ; renvoie (paires, boites de a sans partenaire, boites de b sans partenaire)
def match_boxes(
    boxes_a: Sequence[Box],  # boites du premier annotateur
    boxes_b: Sequence[Box],  # boites du second annotateur
    iou_threshold: float = 0.5,  # recouvrement minimal pour dire "c'est le meme objet"
    same_label: bool = True,  # True : on n'apparie que deux boites de meme classe
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:  # type de retour : paires (i, j, iou), restes de a, restes de b
    """Appariement optimal (algorithme hongrois) des boites de a et b ; une paire compte si IoU >= iou_threshold.

    same_label=True interdit d'apparier deux classes differentes ; False permet de mesurer
    ensuite le desaccord de classe sur des boites bien placees.
    Renvoie (paires (i, j, iou), indices non apparies de a, indices non apparies de b).
    """
    if not boxes_a or not boxes_b:  # si l'une des deux listes est vide, aucune paire possible
        return [], list(range(len(boxes_a))), list(range(len(boxes_b)))  # toutes les boites restent seules
    # tableau des IoU : ligne = boite de a, colonne = boite de b ; 0 si les classes different et qu'on l'interdit
    iou = np.array([[box_iou(x, y) if (not same_label or x.label == y.label) else 0.0 for y in boxes_b] for x in boxes_a])
    rows, cols = linear_sum_assignment(-iou)  # signe moins : l'algorithme minimise un cout, on veut maximiser l'IoU total
    # garde seulement les paires assez recouvrantes (IoU au moins egal au seuil)
    pairs = [(int(i), int(j), float(iou[i, j])) for i, j in zip(rows, cols) if iou[i, j] >= iou_threshold]
    matched_a, matched_b = {p[0] for p in pairs}, {p[1] for p in pairs}  # indices deja apparies de chaque cote
    # renvoie les paires, puis les indices de a et de b restes sans partenaire
    return pairs, [i for i in range(len(boxes_a)) if i not in matched_a], [j for j in range(len(boxes_b)) if j not in matched_b]


# accord de detection (boites) entre deux annotateurs ; renvoie precision, rappel, F1, IoU moyen, accord de classe
def box_agreement(reference: list[ImageAnnotation], other: list[ImageAnnotation], iou_threshold: float = 0.5) -> dict:
    """Accord de detection entre deux annotateurs (le premier sert de reference), images appariees par position.

    precision : part des boites de other retrouvees chez reference ; recall : l'inverse ;
    mean_iou : qualite du placement des boites appariees ; label_agreement : part des boites
    bien placees (meme sans condition de classe) qui portent la meme classe.
    """
    if len(reference) != len(other):  # les deux listes doivent avoir le meme nombre d'images
        raise ValueError("les deux jeux doivent contenir les memes images, dans le meme ordre")  # arret avec message clair
    tp = n_ref = n_other = 0  # tp = vrais positifs (boites retrouvees) ; compteurs de boites de chaque cote
    ious, label_pairs = [], []  # IoU des paires retenues ; True/False "meme classe" pour les boites bien placees
    for ref, oth in zip(reference, other):  # pour chaque image, la version de reference et l'autre version
        pairs, _, _ = match_boxes(ref.boxes, oth.boxes, iou_threshold, same_label=True)  # paires de meme classe
        tp += len(pairs)  # chaque paire = une boite retrouvee
        n_ref, n_other = n_ref + len(ref.boxes), n_other + len(oth.boxes)  # compte toutes les boites de chaque cote
        ious.extend(p[2] for p in pairs)  # garde l'IoU de chaque paire pour la moyenne
        loose, _, _ = match_boxes(ref.boxes, oth.boxes, iou_threshold, same_label=False)  # paires sans regarder la classe
        label_pairs.extend(ref.boxes[i].label == oth.boxes[j].label for i, j, _ in loose)  # meme classe sur ces paires ?
    precision = tp / n_other if n_other else 1.0  # precision : part des boites de other confirmees ; aucune boite = 1
    recall = tp / n_ref if n_ref else 1.0  # rappel : part des boites de reference retrouvees ; aucune boite = 1
    return {  # renvoie le bilan sous forme de dictionnaire
        "precision": precision,  # part des boites de other qui existent aussi chez reference
        "recall": recall,  # part des boites de reference que other a aussi tracees
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,  # F1 : moyenne harmonique des deux
        "mean_iou": float(np.mean(ious)) if ious else 0.0,  # qualite moyenne du placement des boites appariees
        "label_agreement": float(np.mean(label_pairs)) if label_pairs else 1.0,  # part des boites bien placees de meme classe
        "n_reference": n_ref,  # nombre total de boites chez reference
        "n_other": n_other,  # nombre total de boites chez other
    }


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:  # IoU de deux masques (pixels 0/1) ; renvoie un nombre entre 0 et 1
    """IoU de deux masques binaires (deux masques vides : accord parfait)."""
    a, b = a.astype(bool), b.astype(bool)  # convertit en vrai/faux : tout pixel non nul fait partie de l'objet
    union = np.logical_or(a, b).sum()  # nombre de pixels marques par au moins un des deux masques
    return float(np.logical_and(a, b).sum() / union) if union else 1.0  # pixels communs / union ; deux masques vides = 1


def mask_dice(a: np.ndarray, b: np.ndarray) -> float:  # Dice de deux masques (proche de l'IoU, plus indulgent) ; 0 a 1
    """Dice de deux masques binaires : 2 |A n B| / (|A| + |B|)."""
    a, b = a.astype(bool), b.astype(bool)  # convertit en vrai/faux : tout pixel non nul fait partie de l'objet
    total = a.sum() + b.sum()  # somme des tailles des deux masques
    return float(2 * np.logical_and(a, b).sum() / total) if total else 1.0  # 2 x pixels communs / total ; vides = 1


# accord de segmentation entre deux listes de masques ; renvoie IoU et Dice moyens, IoU global, nb d'images
def mask_agreement(masks_a: Sequence[np.ndarray], masks_b: Sequence[np.ndarray]) -> dict:
    """IoU et Dice moyens par image, et IoU cumule sur le jeu (moins sensible aux tres petits masques)."""
    ious = [mask_iou(a, b) for a, b in zip(masks_a, masks_b)]  # IoU de chaque paire de masques (une par image)
    dices = [mask_dice(a, b) for a, b in zip(masks_a, masks_b)]  # Dice de chaque paire de masques
    inter = sum(int(np.logical_and(a > 0, b > 0).sum()) for a, b in zip(masks_a, masks_b))  # pixels communs, tout le jeu
    union = sum(int(np.logical_or(a > 0, b > 0).sum()) for a, b in zip(masks_a, masks_b))  # pixels de l'union, tout le jeu
    return {  # renvoie le bilan sous forme de dictionnaire
        "mean_iou": float(np.mean(ious)),  # moyenne des IoU par image
        "mean_dice": float(np.mean(dices)),  # moyenne des Dice par image
        "dataset_iou": inter / union if union else 1.0,  # IoU calcule sur tous les pixels du jeu d'un coup
        "n_images": len(ious),  # nombre d'images comparees
    }
