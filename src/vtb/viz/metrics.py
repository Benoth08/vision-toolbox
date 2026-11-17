"""Afficher des metriques : matrice de confusion, courbes ROC / PR, seuil, calibration, erreurs.

    from vtb import viz
    viz.confusion_matrix(y_true, y_pred, class_names)               # nombres + % par ligne
    viz.classification_report(y_true, y_pred, class_names)          # DataFrame par classe
    viz.roc_curves(is_defect, scores, mark_threshold=0.5)           # binaire
    viz.roc_curves(y_true, probs, class_names)                      # un-contre-tous (probs N x C)
    fig, table = viz.threshold_sweep(is_defect, scores, chosen=0.42)
    viz.score_histogram(scores, is_defect, threshold=0.42)          # sains vs defauts (anomalies)
    viz.errors_gallery(images, y_true, y_pred, probs, class_names)  # erreurs les plus sures d'abord
    viz.metrics_table({"resnet18": m1, "simple_cnn": m2})

y_true / y_pred : indices de classe (ou noms presents dans class_names), listes, numpy ou
tenseurs. scores binaires : plus grand = plus "defaut".
"""

from __future__ import annotations  # permet les annotations de type modernes

from collections.abc import Mapping, Sequence  # Mapping : type "dictionnaire" ; Sequence : type "liste ou tuple"
from pathlib import Path  # Path : chemin de fichier independant du systeme
from typing import Any  # Any : n'importe quel type

import numpy as np  # numpy : calcul sur des tableaux de nombres
import pandas as pd  # pandas : tableaux de donnees (rapports par classe)
from matplotlib.figure import Figure  # Figure : la page qui contient les graphiques

from .images import as_numpy, grid, read_image, to_display  # conversion en numpy, grille d'images, lecture, image affichable
from .style import KO_COLOR, finish, new_axes, new_figure, to_mpl  # couleur "erreur", fin de figure, creation d'axes / figure, couleurs


# convertit des etiquettes (indices ou noms) en tableau d'indices de classe ; renvoie un tableau numpy d'entiers
def _indices(values: Any, class_names: Sequence[str] | None = None) -> np.ndarray:
    array = np.asarray(as_numpy(values)).reshape(-1)  # liste, tenseur ou tableau -> tableau numpy a une dimension
    if array.dtype.kind in "USO":  # si ce sont des textes (U : unicode, S : octets, O : objets Python)
        if class_names is None:  # il faut alors la liste des noms pour trouver les indices
            raise ValueError("etiquettes textuelles : donner class_names")  # erreur claire sinon
        lookup = {str(n): i for i, n in enumerate(class_names)}  # table nom -> indice
        return np.array([lookup[str(v)] for v in array], np.int64)  # chaque nom remplace par son indice
    return array.astype(np.int64)  # deja des nombres : simple conversion en entiers


def _binary(values: Any) -> np.ndarray:  # convertit des etiquettes 0/1 en tableau de booleens (vrai = defaut / positif)
    return np.asarray(as_numpy(values)).reshape(-1).astype(bool)  # a plat, puis 0 -> False, autre -> True


def _floats(values: Any) -> np.ndarray:  # convertit des scores en tableau numpy de nombres a virgule (double precision)
    return np.asarray(as_numpy(values), np.float64)  # garde la forme (1D ou N x C)


# ---------------------------------------------------------------------------------------------
# classification


def confusion_matrix(  # dessine la matrice de confusion (qui est predit quoi) ; renvoie l'axe dessine
    y_true: Any,  # classes vraies
    y_pred: Any,  # classes predites
    class_names: Sequence[str],  # noms des classes, dans l'ordre des indices
    *,  # les arguments suivants doivent etre nommes
    normalize: str | None = "true",  # pourcentage par ligne ("true"), par colonne ("pred"), du total ("all") ou aucun (None)
    ax: Any = None,  # axe existant ou dessiner (None = nouvelle figure)
    cmap: str = "Blues",  # palette de couleurs des cases
    save: str | Path | None = None,  # fichier ou enregistrer la figure
) -> Any:  # renvoie l'axe
    """Nombre ET pourcentage dans chaque case. normalize : "true" (par ligne = rappel), "pred", "all", None."""
    t, p = _indices(y_true, class_names), _indices(y_pred, class_names)  # etiquettes en indices
    n = len(class_names)  # nombre de classes
    counts = np.zeros((n, n), np.int64)  # matrice vide : lignes = vrai, colonnes = predit
    np.add.at(counts, (t, p), 1)  # ajoute 1 dans la case (vrai, predit) de chaque exemple (gere les repetitions)
    if normalize == "true":  # pourcentage par ligne : part de chaque classe vraie (la diagonale = rappel)
        ratio = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)  # divise par le total de la ligne (au moins 1)
    elif normalize == "pred":  # pourcentage par colonne (la diagonale = precision)
        ratio = counts / np.maximum(counts.sum(axis=0, keepdims=True), 1)  # divise par le total de la colonne
    elif normalize == "all":  # pourcentage du total
        ratio = counts / max(counts.sum(), 1)  # divise par le nombre total d'exemples
    else:  # pas de pourcentage : seulement pour la couleur
        ratio = counts / max(counts.max(), 1)  # ramene a 0-1 en divisant par la plus grande case
    side = max(3.5, 0.55 * n + 2)  # taille de la figure : grandit avec le nombre de classes
    fig, ax = new_axes(ax, figsize=(side + 0.6, side))  # figure et axe (un peu plus large pour la barre de couleur)
    shown = ax.imshow(ratio, cmap=cmap, vmin=0, vmax=1)  # dessine les cases colorees selon le pourcentage
    small = n > 12  # beaucoup de classes : texte plus petit, cases vides sans texte
    for i in range(n):  # pour chaque ligne (classe vraie)
        for j in range(n):  # pour chaque colonne (classe predite)
            if counts[i, j] == 0 and small:  # case vide avec beaucoup de classes
                continue  # on n'ecrit rien (lisibilite)
            text = f"{counts[i, j]}" if normalize is None else f"{counts[i, j]}\n{100 * ratio[i, j]:.0f}%"  # nombre, et pourcentage dessous
            # ecrit le texte au centre de la case ; blanc sur case foncee, noir sur case claire
            ax.text(j, i, text, ha="center", va="center", fontsize=7 if small else 8, color="white" if ratio[i, j] > 0.55 else "black")
    ax.set_xticks(range(n), [str(c) for c in class_names], rotation=45, ha="right", fontsize=8)  # noms en bas, inclines
    ax.set_yticks(range(n), [str(c) for c in class_names], fontsize=8)  # noms a gauche
    ax.set_xlabel("predit")  # titre de l'axe horizontal
    ax.set_ylabel("vrai")  # titre de l'axe vertical
    accuracy = np.trace(counts) / max(counts.sum(), 1)  # exactitude : somme de la diagonale (bien classes) / total
    # texte expliquant le pourcentage affiche (get : "nombres" si normalize est None)
    label = {"true": "% par ligne (rappel)", "pred": "% par colonne (precision)", "all": "% du total"}.get(normalize or "", "nombres")
    ax.set_title(f"exactitude {accuracy:.3f} - {label}", fontsize=10)  # titre
    fig.colorbar(shown, ax=ax, fraction=0.046, pad=0.04)  # barre de couleur a droite (echelle des pourcentages)
    finish(fig, save)  # enregistre si demande
    return ax  # l'axe dessine


# rapport par classe : precision, rappel, F1, effectif, puis les moyennes ; renvoie un tableau pandas
def classification_report(y_true: Any, y_pred: Any, class_names: Sequence[str]) -> pd.DataFrame:
    """precision, rappel, f1, effectif par classe, puis exactitude, moyenne macro et ponderee."""
    t, p = _indices(y_true, class_names), _indices(y_pred, class_names)  # etiquettes en indices
    rows = {}  # une ligne par classe
    for i, name in enumerate(class_names):  # pour chaque classe (i = indice, name = nom)
        tp = int(np.sum((t == i) & (p == i)))  # vrais positifs : de cette classe et predits dans cette classe
        fp = int(np.sum((t != i) & (p == i)))  # faux positifs : predits dans cette classe a tort
        fn = int(np.sum((t == i) & (p != i)))  # faux negatifs : de cette classe mais predits ailleurs
        precision = tp / (tp + fp) if tp + fp else 0.0  # precision : part des predictions de cette classe qui sont justes
        recall = tp / (tp + fn) if tp + fn else 0.0  # rappel : part des exemples de cette classe retrouves
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0  # F1 : moyenne harmonique des deux
        rows[str(name)] = {"precision": precision, "recall": recall, "f1": f1, "support": int(np.sum(t == i))}  # support = effectif
    table = pd.DataFrame(rows).T  # tableau ; .T transpose : une ligne par classe
    support = table["support"].to_numpy(float)  # effectifs en nombres a virgule (pour la moyenne ponderee)
    total = support.sum()  # nombre total d'exemples
    accuracy = float(np.mean(t == p)) if len(t) else 0.0  # exactitude globale
    macro = table[["precision", "recall", "f1"]].mean()  # moyenne macro : chaque classe compte autant
    # moyenne ponderee par l'effectif de chaque classe (0 partout s'il n'y a aucun exemple)
    weighted = (table[["precision", "recall", "f1"]].mul(support, axis=0).sum() / total) if total else macro * 0
    table.loc["accuracy"] = [np.nan, np.nan, accuracy, total]  # ligne exactitude (placee dans la colonne f1, comme scikit-learn)
    table.loc["macro avg"] = [*macro.to_numpy(), total]  # ligne moyenne macro
    table.loc["weighted avg"] = [*weighted.to_numpy(), total]  # ligne moyenne ponderee
    table["support"] = table["support"].astype(int)  # effectifs en entiers
    return table.round(4)  # arrondi a 4 decimales


def per_class_bars(  # barres horizontales, une par classe ; renvoie l'axe dessine
    values: Mapping[str, float] | pd.Series,  # valeur par classe (nom -> nombre)
    *,  # les arguments suivants doivent etre nommes
    title: str = "",  # titre
    reference: float | Mapping[str, float] | pd.Series | None = None,  # valeur de reference : une ligne, ou un marqueur par classe
    sort: bool = False,  # trier les barres par valeur
    ax: Any = None,  # axe existant (None = nouvelle figure)
    xlim: tuple[float, float] | None = None,  # bornes de l'axe horizontal (None = automatique)
    save: str | Path | None = None,  # fichier ou enregistrer la figure
) -> Any:  # renvoie l'axe
    """Barres horizontales par classe (AP, IoU, rappel...) ; reference : ligne (nombre) ou marqueurs (serie)."""
    series = pd.Series(values, dtype=float)  # valeurs en serie pandas (index = noms de classe)
    if sort:  # si on veut trier
        series = series.sort_values()  # de la plus petite a la plus grande
    fig, ax = new_axes(ax, figsize=(6, max(2.0, 0.35 * len(series) + 1)))  # hauteur qui grandit avec le nombre de classes
    positions = np.arange(len(series))  # position verticale de chaque barre : 0, 1, 2...
    ax.barh(positions, series.to_numpy(), color=to_mpl((31, 119, 180)), alpha=0.85, label="valeur")  # barres bleues
    raw = series.to_numpy()  # valeurs brutes
    # effectifs entiers (et au moins une valeur) : affiches sans decimales
    counts = bool(raw.size and np.all(np.isfinite(raw)) and np.all(raw == np.round(raw)) and np.abs(raw).max() > 1)
    for y, v in zip(positions, raw):  # pour chaque barre (position, valeur)
        ax.text(v, y, f" {v:.0f}" if counts else f" {v:.3f}", va="center", fontsize=8)  # ecrit la valeur au bout de la barre
    if isinstance(reference, int | float):  # reference = un seul nombre
        ax.axvline(reference, color="black", ls="--", lw=1, label=f"reference {reference:.3f}")  # ligne verticale en tirets
    elif reference is not None:  # reference = une valeur par classe
        ref = pd.Series(reference, dtype=float).reindex(series.index)  # remise dans l'ordre des barres
        ax.plot(ref.to_numpy(), positions, "D", color=to_mpl((214, 39, 40)), label="reference")  # losanges rouges
    ax.set_yticks(positions, [str(i) for i in series.index], fontsize=8)  # noms des classes a gauche
    if xlim is not None:  # bornes imposees
        ax.set_xlim(*xlim)  # on les applique
    else:  # bornes automatiques
        # de 0 a 15 % au-dela de la plus grande valeur (au moins 1)
        ax.set_xlim(0, max(1.0, float(np.nanmax(series.to_numpy())) * 1.15) if len(series) else 1)
    ax.grid(axis="x", alpha=0.3)  # quadrillage vertical leger
    if title:  # si un titre est donne
        ax.set_title(title, fontsize=10)  # on l'affiche
    if reference is not None:  # s'il y a une reference
        ax.legend(fontsize=8, loc="lower right")  # legende pour la distinguer des barres
    finish(fig, save)  # enregistre si demande
    return ax  # l'axe


# ---------------------------------------------------------------------------------------------
# courbes ROC et precision-rappel


# prepare les courbes a tracer : une seule en binaire, une par classe en multi-classe ; renvoie (nom, vrais positifs, scores)
def _curve_sets(y_true: Any, scores: Any, class_names: Sequence[str] | None) -> list[tuple[str, np.ndarray, np.ndarray]]:
    s = _floats(scores)  # scores en tableau
    if s.ndim == 2 and s.shape[1] > 1:  # scores N x C : multi-classe
        t = _indices(y_true, class_names)  # classes vraies en indices
        names = class_names or [str(i) for i in range(s.shape[1])]  # noms, ou numeros si absents
        return [(str(names[c]), t == c, s[:, c]) for c in range(s.shape[1])]  # un-contre-tous : classe c vs toutes les autres
    return [("defaut" if class_names is None else str(class_names[-1]), _binary(y_true), s.reshape(-1))]  # binaire : une courbe


# rappel, taux de fausses alarmes et precision a un seuil donne ; renvoie ces 3 nombres
def _point_at(y: np.ndarray, s: np.ndarray, threshold: float) -> tuple[float, float, float]:
    pred = s >= threshold  # decision : positif si le score atteint le seuil
    positives, negatives = max(int(y.sum()), 1), max(int((~y).sum()), 1)  # nombres de positifs et de negatifs (au moins 1)
    tp = int(np.sum(pred & y))  # vrais positifs
    fp = int(np.sum(pred & ~y))  # faux positifs (fausses alarmes)
    precision = tp / (tp + fp) if tp + fp else 1.0  # precision (1 par convention si rien n'est predit positif)
    return tp / positives, fp / negatives, precision  # rappel, taux de fausses alarmes, precision


def roc_curves(  # courbe ROC : rappel en fonction des fausses alarmes quand le seuil varie ; renvoie l'axe
    y_true: Any,  # verite : 0/1 en binaire, ou indices de classe
    scores: Any,  # scores 1D (binaire) ou probabilites N x C (multi-classe)
    class_names: Sequence[str] | None = None,  # noms des classes (facultatif)
    *,  # les arguments suivants doivent etre nommes
    mark_threshold: float | None = None,  # seuil a marquer d'un point sur la courbe
    ax: Any = None,  # axe existant (None = nouvelle figure)
    save: str | Path | None = None,  # fichier ou enregistrer la figure
) -> Any:  # renvoie l'axe
    """ROC binaire (scores 1D) ou un-contre-tous (scores N x C) ; AUC en legende ; seuil marque."""
    # roc_auc_score : AUC, aire sous la courbe ROC (1 = parfait, 0,5 = hasard) ; roc_curve : points de la courbe
    from sklearn.metrics import roc_auc_score, roc_curve

    fig, ax = new_axes(ax, figsize=(5, 4.6))  # figure et axe
    aucs = []  # AUC de chaque courbe
    for name, y, s in _curve_sets(y_true, scores, class_names):  # pour chaque courbe (nom, vrais positifs, scores)
        if y.all() or not y.any():  # que des positifs ou aucun : la ROC n'est pas definie
            ax.plot([], [], label=f"{name} (une seule classe presente)")  # simple ligne de legende pour le signaler
            continue  # courbe suivante
        fpr, tpr, _ = roc_curve(y, s)  # taux de faux positifs et de vrais positifs pour tous les seuils
        auc = roc_auc_score(y, s)  # aire sous la courbe
        aucs.append(auc)  # memorisee pour la moyenne
        (line,) = ax.plot(fpr, tpr, lw=1.6, label=f"{name} AUC {auc:.3f}")  # trace la courbe ; garde la ligne pour sa couleur
        if mark_threshold is not None:  # si un seuil est a marquer
            recall, false_alarm, _ = _point_at(y, s, mark_threshold)  # point de fonctionnement a ce seuil
            ax.plot(false_alarm, recall, "o", color=line.get_color(), ms=6)  # point rond de la couleur de la courbe
            if len(aucs) == 1 or _floats(scores).ndim == 1:  # etiquette seulement pour la premiere courbe (sinon illisible)
                ax.annotate(  # etiquette de texte pres du point
                    f"seuil {mark_threshold:.3g}\nTPR {recall:.2f} FPR {false_alarm:.2f}",  # seuil, rappel, fausses alarmes
                    (false_alarm, recall),  # point vise
                    xytext=(10, -28),  # decalage du texte
                    textcoords="offset points",  # decalage en points d'ecran
                    fontsize=8,  # petite police
                )
    ax.plot([0, 1], [0, 1], color="gray", ls=":", lw=1)  # diagonale = classeur au hasard
    ax.set_xlabel("taux de faux positifs (fausses alarmes)")  # titre de l'axe horizontal
    ax.set_ylabel("taux de vrais positifs (rappel)")  # titre de l'axe vertical
    title = "ROC" if len(aucs) <= 1 else f"ROC un-contre-tous, AUC macro {np.mean(aucs):.3f}"  # titre (AUC moyenne en multi-classe)
    ax.set_title(title, fontsize=10)  # affiche le titre
    ax.set_xlim(-0.01, 1.01)  # bornes horizontales (petite marge)
    ax.set_ylim(-0.01, 1.01)  # bornes verticales
    ax.grid(alpha=0.3)  # quadrillage leger
    ax.legend(fontsize=8, loc="lower right")  # legende en bas a droite
    finish(fig, save)  # enregistre si demande
    return ax  # l'axe


def pr_curves(  # courbe precision-rappel quand le seuil varie ; renvoie l'axe
    y_true: Any,  # verite : 0/1 en binaire, ou indices de classe
    scores: Any,  # scores 1D ou probabilites N x C
    class_names: Sequence[str] | None = None,  # noms des classes (facultatif)
    *,  # les arguments suivants doivent etre nommes
    mark_threshold: float | None = None,  # seuil a marquer
    ax: Any = None,  # axe existant (None = nouvelle figure)
    save: str | Path | None = None,  # fichier ou enregistrer la figure
) -> Any:  # renvoie l'axe
    """Precision-rappel (AP en legende, ligne de base = proportion de positifs) ; seuil marque."""
    # average_precision_score : AP, aire sous la courbe precision-rappel ; precision_recall_curve : points de la courbe
    from sklearn.metrics import average_precision_score, precision_recall_curve

    fig, ax = new_axes(ax, figsize=(5, 4.6))  # figure et axe
    aps = []  # AP de chaque courbe
    for name, y, s in _curve_sets(y_true, scores, class_names):  # pour chaque courbe
        if not y.any():  # aucun positif : courbe non definie
            ax.plot([], [], label=f"{name} (aucun positif)")  # ligne de legende pour le signaler
            continue  # courbe suivante
        precision, recall, _ = precision_recall_curve(y, s)  # precision et rappel pour tous les seuils
        ap = average_precision_score(y, s)  # aire sous la courbe
        aps.append(ap)  # memorisee pour la moyenne
        (line,) = ax.plot(recall, precision, lw=1.6, label=f"{name} AP {ap:.3f}")  # trace la courbe ; garde sa couleur
        if _floats(scores).ndim == 1:  # en binaire seulement
            ax.axhline(y.mean(), color="gray", ls=":", lw=1, label=f"hasard {y.mean():.2f}")  # niveau du hasard = part de positifs
        if mark_threshold is not None:  # si un seuil est a marquer
            r, _, p = _point_at(y, s, mark_threshold)  # rappel et precision a ce seuil
            ax.plot(r, p, "o", color=line.get_color(), ms=6)  # point rond
            if _floats(scores).ndim == 1:  # etiquette en binaire seulement
                ax.annotate(  # etiquette de texte pres du point
                    f"seuil {mark_threshold:.3g}\nP {p:.2f} R {r:.2f}",  # seuil, precision, rappel
                    (r, p),  # point vise
                    xytext=(-80, -30),  # decalage du texte (en bas a gauche)
                    textcoords="offset points",  # decalage en points d'ecran
                    fontsize=8,  # petite police
                )
    ax.set_xlabel("rappel")  # titre de l'axe horizontal
    ax.set_ylabel("precision")  # titre de l'axe vertical
    title = "precision-rappel" if len(aps) <= 1 else f"precision-rappel, AP moyenne {np.mean(aps):.3f}"  # titre
    ax.set_title(title, fontsize=10)  # affiche le titre
    ax.set_xlim(-0.01, 1.01)  # bornes horizontales
    ax.set_ylim(-0.01, 1.01)  # bornes verticales
    ax.grid(alpha=0.3)  # quadrillage leger
    ax.legend(fontsize=8, loc="lower left")  # legende en bas a gauche
    finish(fig, save)  # enregistre si demande
    return ax  # l'axe


def threshold_sweep(  # balaye les seuils possibles ; renvoie (figure, tableau des metriques par seuil)
    y_true: Any,  # verite 0/1
    scores: Any,  # scores (plus grand = plus "defaut")
    *,  # les arguments suivants doivent etre nommes
    chosen: float | None = None,  # seuil choisi a marquer
    points: int = 200,  # nombre maximal de seuils testes
    ax: Any = None,  # axe existant (None = nouvelle figure)
    save: str | Path | None = None,  # fichier ou enregistrer la figure
) -> tuple[Figure, pd.DataFrame]:  # renvoie la figure et le tableau
    """Precision, rappel, F1 et fausses alarmes en fonction du seuil ; renvoie aussi le tableau.

    Choisir le seuil sur la VALIDATION (rappel minimal exige, fausses alarmes tolerables), puis
    le reporter tel quel sur le test.
    """
    y, s = _binary(y_true), _floats(scores).reshape(-1)  # verite en booleens, scores a plat
    unique = np.unique(s)  # valeurs distinctes des scores (triees)
    # seuils testes : toutes les valeurs distinctes si peu nombreuses, sinon des quantiles regulierement espaces
    thresholds = unique if len(unique) <= points else np.unique(np.quantile(s, np.linspace(0, 1, points)))
    rows = []  # une ligne par seuil
    positives, negatives = int(y.sum()), int((~y).sum())  # nombres de positifs et de negatifs
    for t in thresholds:  # pour chaque seuil
        pred = s >= t  # decision a ce seuil
        tp = int(np.sum(pred & y))  # vrais positifs
        fp = int(np.sum(pred & ~y))  # faux positifs
        fn = positives - tp  # faux negatifs : positifs rates
        tn = negatives - fp  # vrais negatifs : negatifs bien laisses passer
        precision = tp / (tp + fp) if tp + fp else 1.0  # precision (1 si rien n'est predit positif)
        recall = tp / positives if positives else 0.0  # rappel
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0  # F1
        rows.append(  # ajoute la ligne de ce seuil
            {  # dictionnaire des valeurs
                "threshold": float(t),  # le seuil
                "precision": precision,  # precision
                "recall": recall,  # rappel
                "f1": f1,  # F1
                "false_alarm": fp / negatives if negatives else 0.0,  # taux de fausses alarmes
                "tp": tp,  # vrais positifs
                "fp": fp,  # faux positifs
                "fn": fn,  # faux negatifs
                "tn": tn,  # vrais negatifs
            }
        )
    table = pd.DataFrame(rows)  # tableau des resultats
    fig, ax = new_axes(ax, figsize=(7, 4))  # figure et axe
    for column, color in (  # pour chaque metrique a tracer, avec sa couleur
        ("precision", (31, 119, 180)),  # precision en bleu
        ("recall", (44, 160, 44)),  # rappel en vert
        ("f1", (148, 103, 189)),  # F1 en violet
        ("false_alarm", (214, 39, 40)),  # fausses alarmes en rouge
    ):
        ax.plot(table["threshold"], table[column], label=column, color=to_mpl(color), lw=1.6)  # courbe metrique vs seuil
    if len(table):  # si le tableau n'est pas vide
        best = table.loc[table["f1"].idxmax()]  # ligne du meilleur F1
        # ligne verticale pointillee au seuil du meilleur F1
        ax.axvline(best["threshold"], color="gray", ls=":", lw=1, label=f"F1 max {best['f1']:.3f} a {best['threshold']:.3g}")
    if chosen is not None:  # si un seuil choisi est donne
        recall, false_alarm, precision = _point_at(y, s, chosen)  # ses performances
        ax.axvline(  # ligne verticale en tirets au seuil choisi
            chosen,  # position : le seuil
            color="black",  # noire
            ls="--",  # en tirets
            lw=1.2,  # epaisseur
            label=f"choisi {chosen:.3g} : R {recall:.2f} P {precision:.2f} FA {false_alarm:.2f}",  # legende : rappel, precision, FA
        )
    ax.set_xlabel("seuil sur le score")  # titre de l'axe horizontal
    ax.set_ylim(-0.02, 1.02)  # bornes verticales (toutes ces metriques sont entre 0 et 1)
    ax.grid(alpha=0.3)  # quadrillage leger
    ax.legend(fontsize=8, loc="best")  # legende la ou elle gene le moins
    ax.set_title("balayage du seuil", fontsize=10)  # titre
    finish(fig, save)  # enregistre si demande
    return fig, table  # la figure et le tableau


def score_histogram(  # histogrammes des scores des sains et des defauts ; renvoie l'axe
    scores: Any,  # scores
    is_defect: Any,  # verite 0/1 (1 = defaut)
    *,  # les arguments suivants doivent etre nommes
    threshold: float | None = None,  # seuil a marquer
    bins: int = 50,  # nombre d'intervalles de l'histogramme
    log: bool = False,  # axe vertical en echelle logarithmique
    ax: Any = None,  # axe existant (None = nouvelle figure)
    names: tuple[str, str] = ("sain", "defaut"),  # noms des deux groupes
    save: str | Path | None = None,  # fichier ou enregistrer la figure
) -> Any:  # renvoie l'axe
    """Distribution des scores des sains et des defauts ; seuil, rappel et fausses alarmes."""
    s, y = _floats(scores).reshape(-1), _binary(is_defect)  # scores a plat, verite en booleens
    fig, ax = new_axes(ax, figsize=(7, 3.6))  # figure et axe
    edges = np.histogram_bin_edges(s, bins=bins)  # bornes des intervalles, communes aux deux histogrammes
    ax.hist(s[~y], bins=edges, alpha=0.6, color=to_mpl((31, 119, 180)), label=f"{names[0]} (n={int((~y).sum())})")  # sains en bleu
    ax.hist(s[y], bins=edges, alpha=0.6, color=to_mpl((214, 39, 40)), label=f"{names[1]} (n={int(y.sum())})")  # defauts en rouge
    if threshold is not None:  # si un seuil est donne
        recall, false_alarm, _ = _point_at(y, s, threshold)  # rappel et fausses alarmes a ce seuil
        ax.axvline(  # ligne verticale au seuil
            threshold,  # position
            color="black",  # noire
            ls="--",  # en tirets
            lw=1.2,  # epaisseur
            label=f"seuil {threshold:.4g} : rappel {recall:.2f}, fausses alarmes {false_alarm:.2f}",  # legende
        )
    if log:  # echelle logarithmique demandee
        ax.set_yscale("log")  # utile quand un groupe est bien plus nombreux que l'autre
    ax.set_xlabel("score")  # titre de l'axe horizontal
    ax.set_ylabel("images")  # titre de l'axe vertical
    ax.grid(alpha=0.3)  # quadrillage leger
    ax.legend(fontsize=8)  # legende
    finish(fig, save)  # enregistre si demande
    return ax  # l'axe


def calibration(  # diagramme de fiabilite : la confiance annoncee correspond-elle au taux de reussite ? renvoie l'axe
    y_true: Any,  # verite
    probs: Any,  # probabilites predites (1D binaire ou N x C)
    *,  # les arguments suivants doivent etre nommes
    bins: int = 10,  # nombre de tranches de confiance
    ax: Any = None,  # axe existant (None = nouvelle figure)
    save: str | Path | None = None,  # fichier ou enregistrer la figure
) -> Any:  # renvoie l'axe
    """Diagramme de fiabilite : exactitude par tranche de confiance ; ECE dans le titre.

    probs 1D : probabilite de la classe positive (y_true 0/1) ; probs N x C : confiance = max,
    juste = argmax == y_true.
    """
    p = _floats(probs)  # probabilites en tableau
    if p.ndim == 2 and p.shape[1] > 1:  # multi-classe
        confidence = p.max(axis=1)  # confiance = probabilite de la classe choisie
        correct = p.argmax(axis=1) == _indices(y_true)  # juste si la classe la plus probable est la vraie
    else:  # binaire
        prob = p.reshape(-1)  # probabilite de la classe positive
        y = _binary(y_true)  # verite en booleens
        confidence = np.where(prob >= 0.5, prob, 1 - prob)  # confiance dans la decision prise (toujours >= 0,5)
        correct = (prob >= 0.5) == y  # juste si la decision a 0,5 correspond a la verite
    edges = np.linspace(0, 1, bins + 1)  # bornes des tranches de confiance
    which = np.clip(np.digitize(confidence, edges[1:-1]), 0, bins - 1)  # numero de tranche de chaque exemple
    centers, accuracies, means, counts = [], [], [], []  # par tranche : centre, exactitude, confiance moyenne, effectif
    ece = 0.0  # ECE : ecart moyen entre confiance et exactitude (0 = parfaitement calibre)
    for b in range(bins):  # pour chaque tranche
        selected = which == b  # exemples de cette tranche
        if not selected.any():  # tranche vide
            continue  # on passe
        acc, conf = float(correct[selected].mean()), float(confidence[selected].mean())  # exactitude et confiance moyenne
        ece += selected.mean() * abs(acc - conf)  # ecart pondere par la part d'exemples de la tranche
        centers.append((edges[b] + edges[b + 1]) / 2)  # centre de la tranche
        accuracies.append(acc)  # exactitude
        means.append(conf)  # confiance moyenne
        counts.append(int(selected.sum()))  # effectif
    fig, ax = new_axes(ax, figsize=(5, 4.6))  # figure et axe
    # barres : exactitude par tranche
    ax.bar(centers, accuracies, width=1 / bins * 0.9, alpha=0.7, color=to_mpl((31, 119, 180)), label="exactitude")
    ax.plot(means, accuracies, "o", color="black", ms=4, label="confiance moyenne")  # points : confiance moyenne reelle de la tranche
    for c, a, n in zip(centers, accuracies, counts):  # pour chaque tranche
        ax.text(c, a + 0.02, str(n), ha="center", fontsize=7)  # ecrit l'effectif au-dessus de la barre
    ax.plot([0, 1], [0, 1], color="gray", ls=":", lw=1, label="calibration parfaite")  # diagonale : confiance = exactitude
    ax.set_xlabel("confiance")  # titre de l'axe horizontal
    ax.set_ylabel("exactitude")  # titre de l'axe vertical
    ax.set_xlim(0, 1)  # bornes horizontales
    ax.set_ylim(0, 1.08)  # bornes verticales (place pour les effectifs)
    ax.set_title(f"fiabilite, ECE = {ece:.3f}", fontsize=10)  # titre avec l'ECE
    ax.grid(alpha=0.3)  # quadrillage leger
    ax.legend(fontsize=8, loc="upper left")  # legende en haut a gauche
    finish(fig, save)  # enregistre si demande
    return ax  # l'axe


def errors_gallery(  # galerie des erreurs les plus sures ; renvoie la figure
    images: Sequence[Any],  # images (tableaux ou chemins), dans l'ordre de y_true
    y_true: Any,  # classes vraies
    y_pred: Any,  # classes predites
    probs: Any,  # probabilites (N x C) ou confiance (N)
    class_names: Sequence[str],  # noms des classes
    *,  # les arguments suivants doivent etre nommes
    k: int = 12,  # nombre d'erreurs a montrer
    bgr: bool = False,  # vrai si les images sont en ordre BGR (OpenCV) plutot que RGB
    ncols: int = 6,  # images par ligne
    save: str | Path | None = None,  # fichier ou enregistrer la figure
) -> Figure:  # renvoie la figure
    """Les k erreurs les plus SURES (confiance elevee dans la mauvaise classe) : etiquettes fausses,
    classes confondues, raccourcis appris. images : tableaux ou chemins, dans l'ordre de y_true."""
    t, p = _indices(y_true, class_names), _indices(y_pred, class_names)  # etiquettes en indices
    pr = _floats(probs)  # probabilites en tableau
    confidence = pr[np.arange(len(p)), p] if pr.ndim == 2 else pr.reshape(-1)  # probabilite de la classe predite pour chaque image
    wrong = np.flatnonzero(t != p)  # indices des images mal classees
    if not len(wrong):  # aucune erreur
        fig = new_figure((4, 1.2))  # petite figure
        fig.text(0.5, 0.5, "aucune erreur", ha="center", va="center", fontsize=12)  # message au centre
        finish(fig, save)  # enregistre si demande
        return fig  # la figure
    order = wrong[np.argsort(-confidence[wrong])][:k]  # erreurs triees par confiance decroissante, les k premieres
    shown, titles = [], []  # images a afficher et leurs titres
    for i in order:  # pour chaque erreur retenue
        image = images[int(i)]  # l'image (ou son chemin)
        image = read_image(image) if isinstance(image, str | Path) else image  # lit le fichier si c'est un chemin
        shown.append(to_display(image, bgr=bgr))  # image convertie pour l'affichage
        titles.append(f"vrai {class_names[t[i]]}\npredit {class_names[p[i]]} {confidence[i]:.2f}")  # titre : vrai, predit, confiance
    return grid(  # grille d'images
        shown,  # les images
        titles,  # leurs titres
        ncols=ncols,  # images par ligne
        borders=[KO_COLOR] * len(shown),  # cadre rouge autour de chaque image (ce sont des erreurs)
        suptitle=f"{len(wrong)} erreurs, les {len(order)} plus sures",  # titre general
        save=save,  # enregistre si demande
    )


# ---------------------------------------------------------------------------------------------
# tableaux et comparaisons


def _scalar(value: Any) -> bool:  # vrai si la valeur est simple (nombre, texte, booleen ou None), pas une liste ni un dictionnaire
    return isinstance(value, int | float | np.integer | np.floating | bool | str) or value is None  # types acceptes


# met des metriques en tableau lisible ; renvoie un tableau pandas
def metrics_table(metrics: Mapping[str, Any] | Sequence[Mapping[str, Any]], *, digits: int = 3) -> pd.DataFrame:
    """dict de scalaires -> une colonne "valeur" ; dict de dicts ou liste de dicts -> une ligne par essai.
    Les valeurs non scalaires (listes, sous-dicts) sont ignorees ; les flottants sont arrondis."""
    if isinstance(metrics, Mapping) and not any(isinstance(v, Mapping) for v in metrics.values()):  # un seul essai (dict de nombres)
        table = pd.DataFrame({"valeur": pd.Series({k: v for k, v in metrics.items() if _scalar(v)}, dtype=object)})  # une colonne
    else:  # plusieurs essais
        if isinstance(metrics, Mapping):  # dictionnaire nom d'essai -> metriques
            rows = {str(k): {kk: vv for kk, vv in dict(v).items() if _scalar(vv)} for k, v in metrics.items()}  # garde les scalaires
            table = pd.DataFrame(rows).T  # une ligne par essai
        else:  # liste de dictionnaires
            records = [{k: v for k, v in dict(m).items() if _scalar(v)} for m in metrics]  # garde les scalaires de chaque essai
            table = pd.DataFrame(records)  # une ligne par essai
            if "name" in table.columns:  # si chaque essai a un nom
                table = table.set_index("name")  # le nom sert d'etiquette de ligne
    return table.map(lambda v: round(float(v), digits) if isinstance(v, float | np.floating) else v)  # arrondit les nombres a virgule


def compare_runs(  # barres groupees pour comparer plusieurs essais ; renvoie la figure
    runs: pd.DataFrame | Mapping[str, Mapping[str, Any]],  # essais : tableau (une ligne par essai) ou dictionnaire
    metrics: str | Sequence[str],  # metrique(s) a comparer
    *,  # les arguments suivants doivent etre nommes
    ax: Any = None,  # axe existant (None = nouvelle figure)
    save: str | Path | None = None,  # fichier ou enregistrer la figure
) -> Figure:  # renvoie la figure
    """Barres groupees : un groupe par essai, une barre par metrique (valeur ecrite au bout)."""
    # essais en tableau (une ligne par essai)
    table = runs.copy() if isinstance(runs, pd.DataFrame) else pd.DataFrame({k: dict(v) for k, v in runs.items()}).T
    metrics = [metrics] if isinstance(metrics, str) else list(metrics)  # une seule metrique -> liste d'un element
    table = table[metrics].astype(float)  # garde les colonnes demandees, en nombres
    n_runs, n_metrics = len(table), len(metrics)  # nombre d'essais et de metriques
    height = 0.8 / n_metrics  # epaisseur d'une barre : le groupe occupe 80 % de l'espace
    fig, ax = new_axes(ax, figsize=(7, max(2.2, 0.45 * n_runs * max(1, n_metrics * 0.6) + 1)))  # hauteur selon le nombre de barres
    positions = np.arange(n_runs)  # position verticale de chaque groupe
    for j, metric in enumerate(metrics):  # pour chaque metrique
        y = positions + (j - (n_metrics - 1) / 2) * height  # decale les barres de cette metrique dans chaque groupe
        values = table[metric].to_numpy()  # valeurs de tous les essais
        ax.barh(y, values, height=height * 0.95, label=metric)  # barres horizontales
        for yy, v in zip(y, values):  # pour chaque barre
            ax.text(v, yy, f" {v:.3g}", va="center", fontsize=7)  # ecrit la valeur au bout
    ax.set_yticks(positions, [str(i) for i in table.index], fontsize=8)  # noms des essais a gauche
    ax.invert_yaxis()  # premier essai en haut
    ax.grid(axis="x", alpha=0.3)  # quadrillage vertical leger
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)  # hors du cadre : ne cache aucune barre
    # de 0 a 18 % au-dela de la plus grande valeur (1 si aucune valeur finie)
    ax.set_xlim(0, float(np.nanmax(table.to_numpy())) * 1.18 if np.isfinite(np.nanmax(table.to_numpy())) else 1)
    finish(fig, save)  # enregistre si demande
    return fig  # la figure


# barres de latence (temps de calcul par image) ; renvoie l'axe
def latency_bars(latencies: Mapping[str, float], *, ax: Any = None, save: str | Path | None = None) -> Any:
    """Latence par moteur / modele en ms (triee), avec le debit image/s correspondant."""
    series = pd.Series(dict(latencies), dtype=float).sort_values()  # latences triees de la plus rapide a la plus lente
    fig, ax = new_axes(ax, figsize=(6, max(1.8, 0.4 * len(series) + 1)))  # hauteur selon le nombre de barres
    positions = np.arange(len(series))  # position verticale de chaque barre
    ax.barh(positions, series.to_numpy(), color=to_mpl((255, 127, 14)))  # barres orange
    for y, v in zip(positions, series.to_numpy()):  # pour chaque barre
        # ecrit la latence et le debit (1000 / ms = images par seconde) ; "-" si la latence est nulle
        ax.text(v, y, f" {v:.1f} ms ({1000 / v:.0f} img/s)" if v > 0 else " -", va="center", fontsize=8)
    ax.set_yticks(positions, [str(i) for i in series.index], fontsize=8)  # noms des moteurs / modeles a gauche
    ax.set_xlabel("ms par image")  # titre de l'axe horizontal
    ax.set_xlim(0, float(series.max()) * 1.45 if len(series) else 1)  # marge a droite pour le texte
    ax.grid(axis="x", alpha=0.3)  # quadrillage vertical leger
    finish(fig, save)  # enregistre si demande
    return ax  # l'axe
