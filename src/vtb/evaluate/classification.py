"""Metriques de classification : celles qui restent honnetes quand les classes sont desequilibrees."""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes

from collections.abc import Sequence  # type "suite d'elements" (liste, tuple) pour les annotations

import numpy as np  # numpy : calcul sur des tableaux de nombres

# scikit-learn : balanced_accuracy = moyenne des rappels par classe ; classification_report = precision / rappel / F1 par classe ;
# confusion_matrix = tableau verite x prediction ; f1_score = compromis precision / rappel ; log_loss = penalite des probabilites
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score, log_loss


# calcule les metriques de classification ; renvoie un dict pret a ecrire en JSON
def classification_metrics(targets: np.ndarray, preds: np.ndarray, probs: np.ndarray, class_names: Sequence[str]) -> dict:
    """accuracy, balanced accuracy, F1 macro (la metrique a suivre), log loss, et le detail par classe."""
    labels = list(range(len(class_names)))  # indices des classes : 0, 1, 2... (meme si une classe est absente des donnees)
    # rapport detaille par classe sous forme de dict ; zero_division=0 : 0 au lieu d'une alerte si une classe n'est jamais predite
    report = classification_report(targets, preds, labels=labels, target_names=list(class_names), output_dict=True, zero_division=0)
    return {  # dict des resultats
        "n_images": int(len(targets)),  # nombre d'images evaluees
        "accuracy": float((preds == targets).mean()),  # part de bonnes reponses (trompeuse si les classes sont desequilibrees)
        "balanced_accuracy": float(balanced_accuracy_score(targets, preds)),  # moyenne des rappels de chaque classe
        "f1_macro": float(f1_score(targets, preds, labels=labels, average="macro", zero_division=0)),  # F1 moyen, chaque classe pese pareil
        "log_loss": float(log_loss(targets, probs, labels=labels)),  # punit les probabilites sures d'elles mais fausses
        "per_class": {name: {k: round(float(v), 4) for k, v in report[name].items()} for name in class_names},  # detail par classe, arrondi
        "confusion_matrix": confusion_matrix(targets, preds, labels=labels).tolist(),  # matrice de confusion en listes (ecrivable en JSON)
    }


def report_text(targets: np.ndarray, preds: np.ndarray, class_names: Sequence[str]) -> str:  # renvoie le rapport par classe en texte
    return classification_report(  # rapport texte lisible (tableau precision / rappel / F1 / effectif)
        targets,  # vraies classes
        preds,  # classes predites
        labels=list(range(len(class_names))),  # toutes les classes, meme absentes
        target_names=list(class_names),  # noms affiches des classes
        digits=3,  # 3 chiffres apres la virgule
        zero_division=0,  # 0 au lieu d'une alerte quand une metrique n'est pas definie
    )


# cherche le seuil de score qui garantit une precision minimale ; renvoie ce seuil
def threshold_for_precision(scores: np.ndarray, positives: np.ndarray, min_precision: float) -> float:
    """Plus petit seuil (rappel maximal) qui garantit la precision demandee sur la validation : un choix metier."""
    order = np.argsort(-scores)  # ordre des exemples du score le plus haut au plus bas
    sorted_scores, sorted_pos = scores[order], positives[order].astype(bool)  # scores tries et vraies etiquettes dans le meme ordre
    tp = np.cumsum(sorted_pos)  # vrais positifs cumules si on coupe apres chaque exemple
    precision = tp / np.arange(1, len(sorted_pos) + 1)  # precision = vrais positifs / nombre d'exemples retenus
    ok = np.where(precision >= min_precision)[0]  # positions ou la precision demandee est atteinte
    # seuil le plus bas qui respecte la precision (rappel maximal) ; sinon un seuil au-dessus du plus haut score (rien n'est retenu)
    return float(sorted_scores[ok.max()]) if len(ok) else float(sorted_scores[0] + 1e-6)
