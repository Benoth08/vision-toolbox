"""Statistiques pour juger un resultat : intervalles de confiance, comparaison de deux modeles, choix du seuil.

    from vtb.toolkit import stats
    value, low, high = stats.bootstrap_ci(accuracy_score, y_true, y_pred)       # 0.95 [0.92, 0.98]
    stats.mcnemar(y_true, pred_a, pred_b)["p_value"]                            # A et B different-ils vraiment ?
    stats.threshold_for(val_scores, val_is_defect, min_recall=0.95)            # seuil choisi sur la VALIDATION

Un test de 270 images donne une exactitude a +/- 2 ou 3 points pres : deux modeles a 0,96 et 0,97
ne sont PAS departages par ce test. Toujours donner l'intervalle avec le chiffre.
"""

from __future__ import annotations  # permet d'ecrire les types "float | None" partout sans les evaluer a l'execution

import math  # fonctions mathematiques de base (racine carree...)
from collections.abc import Callable  # type "fonction" pour annoter un argument qui est une fonction
from typing import NamedTuple  # NamedTuple : tuple dont les cases ont un nom (x.value au lieu de x[0])

import numpy as np  # numpy : calcul rapide sur des tableaux de nombres
import pandas as pd  # pandas : tableaux avec des colonnes nommees (DataFrame), pratiques a afficher


class Interval(NamedTuple):  # resultat d'un intervalle de confiance : la valeur mesuree et sa fourchette plausible
    value: float  # la valeur mesuree sur le jeu de test (ex. exactitude 0.95)
    low: float  # borne basse de l'intervalle : la vraie valeur est probablement au-dessus
    high: float  # borne haute de l'intervalle : la vraie valeur est probablement en dessous

    def __str__(self) -> str:  # texte affiche par print(intervalle) ; renvoie "0.950 [0.920, 0.980]"
        return f"{self.value:.3f} [{self.low:.3f}, {self.high:.3f}]"  # 3 chiffres apres la virgule, bornes entre crochets


def bootstrap_ci(  # intervalle de confiance d'une metrique par bootstrap ; renvoie un Interval
    metric: Callable[..., float],  # la fonction qui calcule la metrique (ex. accuracy_score(y_true, y_pred))
    *arrays: np.ndarray,  # les tableaux a passer a la metrique (ex. vraies etiquettes puis predictions)
    n: int = 1000,  # nombre de retirages au hasard : plus grand = intervalle plus stable mais plus lent
    alpha: float = 0.05,  # risque accepte : 0.05 donne un intervalle a 95 %
    seed: int = 0,  # germe du hasard : meme germe = memes tirages = meme resultat
    groups: np.ndarray | None = None,  # identifiant de piece par image, pour retirer des pieces entieres
) -> Interval:  # renvoie un Interval : valeur mesuree, borne basse, borne haute
    """Intervalle de confiance par bootstrap (percentiles) d'une metrique sur un jeu de test.

    metric(*arrays) -> nombre (ex. sklearn.metrics.f1_score avec functools.partial(average="macro")).
    Les arrays sont reechantillonnes avec les MEMES indices (ils restent alignes). groups : si
    plusieurs images viennent de la meme piece, reechantillonner par piece (bootstrap par groupe),
    sinon l'intervalle est trop etroit. Les tirages ou la metrique echoue (ex. AUROC avec une seule
    classe) sont ignores ; erreur si plus de 10 % echouent. Reproductible (seed).
    """
    # Bootstrap : on refait le jeu de test en tirant ses images au hasard AVEC remise, on recalcule la
    # metrique, et on regarde de combien elle varie d'un tirage a l'autre.
    arrays = tuple(np.asarray(a) for a in arrays)  # convertit chaque entree (liste...) en tableau numpy pour l'indexer
    if not arrays:  # si aucun tableau n'a ete donne
        raise ValueError("au moins un tableau")  # on arrete avec un message clair
    size = len(arrays[0])  # nombre d'exemples du jeu de test
    if any(len(a) != size for a in arrays):  # si un tableau n'a pas autant d'exemples que le premier
        raise ValueError("les tableaux n'ont pas la meme longueur")  # erreur : ils doivent etre alignes exemple par exemple
    value = float(metric(*arrays))  # la valeur mesuree sur le vrai jeu de test (le centre de l'intervalle)
    rng = np.random.default_rng(seed)  # generateur de nombres aleatoires reproductible
    if groups is not None:  # si on a donne un groupe (piece) pour chaque image
        groups = np.asarray(groups)  # en tableau numpy pour pouvoir comparer d'un coup
        unique = np.unique(groups)  # liste des pieces differentes
        members = [np.flatnonzero(groups == g) for g in unique]  # pour chaque piece : les indices de ses images
    values, failures = [], 0  # valeurs de la metrique sur chaque tirage, et nombre de tirages rates
    for _ in range(n):  # n retirages du jeu de test
        if groups is None:  # sans groupes : on tire des images une par une
            index = rng.integers(0, size, size)  # "size" indices au hasard avec remise (certaines images 2 fois, d'autres 0)
        else:  # avec groupes : on tire des pieces entieres, avec toutes leurs images
            index = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])  # met bout a bout leurs indices
        try:  # la metrique peut echouer sur un tirage (ex. une seule classe presente)
            values.append(float(metric(*(a[index] for a in arrays))))  # recalcule la metrique sur le jeu retire (memes indices partout)
        except ValueError:  # si la metrique a refuse ce tirage
            failures += 1  # on le compte et on passe au suivant
    if failures > 0.1 * n:  # si plus de 10 % des tirages ont echoue, l'intervalle ne veut plus rien dire
        raise ValueError(f"la metrique echoue sur {failures} tirages sur {n} (jeu trop petit ou une seule classe ?)")  # on arrete
    # percentiles : valeurs sous lesquelles tombent 2.5 % et 97.5 % des tirages (pour alpha = 0.05)
    low, high = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(value, float(low), float(high))  # renvoie la valeur mesuree et sa fourchette


def wilson_ci(successes: int, total: int, alpha: float = 0.05) -> Interval:  # intervalle d'une proportion ; renvoie un Interval
    """Intervalle de Wilson d'une proportion (rappel 48/79, taux de fausses alarmes 3/190...) : exact sans bootstrap."""
    from scipy.stats import norm  # loi normale de scipy (importee ici : scipy n'est charge que si on en a besoin)

    if total <= 0:  # si le total est nul ou negatif, la proportion n'existe pas
        raise ValueError("total doit etre > 0")  # on arrete avec un message clair
    z = float(norm.ppf(1 - alpha / 2))  # nombre d'ecarts-types qui couvrent 95 % de la loi normale (environ 1.96)
    p = successes / total  # proportion observee (ex. 48/79)
    denominator = 1 + z**2 / total  # terme de correction de la formule de Wilson
    centre = (p + z**2 / (2 * total)) / denominator  # centre de l'intervalle, tire un peu vers 0.5 quand il y a peu d'exemples
    half = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator  # demi-largeur de l'intervalle
    return Interval(p, max(0.0, centre - half), min(1.0, centre + half))  # bornes gardees entre 0 et 1


# test de McNemar : compare deux modeles sur les memes images ; renvoie un dict avec la p-value et un verdict
def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, *, alpha: float = 0.05) -> dict:
    """Test de McNemar : deux modeles evalues sur les MEMES images different-ils vraiment ?

    Seules comptent les images ou un modele a raison et l'autre tort (b : A juste / B faux ;
    c : A faux / B juste). Test binomial exact si b + c < 25, sinon khi-deux avec correction de
    continuite. p_value < alpha : la difference n'est probablement pas due au hasard du jeu de test.
    """
    from scipy.stats import binomtest, chi2  # deux tests statistiques de scipy : binomial exact et loi du khi-deux

    y_true, pred_a, pred_b = (np.asarray(v) for v in (y_true, pred_a, pred_b))  # les trois entrees en tableaux numpy
    a_ok, b_ok = pred_a == y_true, pred_b == y_true  # pour chaque image : True si A (resp. B) a la bonne reponse
    b = int(np.sum(a_ok & ~b_ok))  # nombre d'images ou A a raison et B a tort (~ = "non")
    c = int(np.sum(~a_ok & b_ok))  # nombre d'images ou A a tort et B a raison
    if b + c == 0:  # si les deux modeles ne sont jamais en desaccord
        statistic, p_value, method = 0.0, 1.0, "aucun desaccord"  # rien a tester : p-value 1 (aucune preuve de difference)
    elif b + c < 25:  # peu de desaccords : on utilise le test binomial exact (pile ou face)
        # p-value : probabilite d'observer un ecart aussi grand si les deux modeles etaient en fait equivalents
        statistic, p_value, method = float(min(b, c)), float(binomtest(b, b + c, 0.5).pvalue), "binomial exact"
    else:  # beaucoup de desaccords : l'approximation du khi-deux est assez precise
        statistic = (abs(b - c) - 1) ** 2 / (b + c)  # statistique de McNemar avec correction de continuite (le "- 1")
        p_value, method = float(chi2.sf(statistic, 1)), "khi-deux corrige"  # p-value lue sur la loi du khi-deux a 1 degre de liberte
    accuracy_a, accuracy_b = float(a_ok.mean()), float(b_ok.mean())  # exactitude de chaque modele (part de bonnes reponses)
    if p_value < alpha:  # si la p-value est sous le risque accepte : difference jugee reelle
        best = "A" if b > c else "B"  # le meilleur est celui qui a raison plus souvent quand ils ne sont pas d'accord
        verdict = f"difference significative (p = {p_value:.3g} < {alpha}) : {best} est meilleur"  # phrase de conclusion
    else:  # sinon : on ne peut pas dire qu'ils different
        verdict = f"pas de difference significative (p = {p_value:.3g}) : ce jeu de test ne departage pas A et B"  # conclusion
    return {  # renvoie tout le detail dans un dictionnaire
        "n": int(len(y_true)),  # nombre d'images du jeu de test
        "accuracy_a": round(accuracy_a, 4),  # exactitude de A, arrondie a 4 chiffres
        "accuracy_b": round(accuracy_b, 4),  # exactitude de B
        "a_right_b_wrong": b,  # images ou A a raison et B tort
        "a_wrong_b_right": c,  # images ou A a tort et B raison
        "statistic": round(float(statistic), 4),  # valeur de la statistique du test
        "p_value": p_value,  # p-value : petite = difference probablement reelle
        "method": method,  # quel test a ete utilise
        "verdict": verdict,  # conclusion en francais
    }


def operating_points(scores: np.ndarray, is_defect: np.ndarray) -> pd.DataFrame:  # tableau des seuils ; renvoie un DataFrame
    """Pour chaque seuil candidat (decision : score >= seuil -> defaut) : rappel, fausses alarmes, precision."""
    # Vocabulaire : rappel = part des vrais defauts detectes ; fausses alarmes = part des pieces saines
    # declarees en defaut ; precision = part des alertes qui sont de vrais defauts.
    scores = np.asarray(scores, dtype=np.float64)  # les scores du modele (plus haut = plus suspect), en nombres decimaux
    positive = np.asarray(is_defect).astype(bool)  # True pour les images qui ont vraiment un defaut
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())  # nombre d'images avec defaut et d'images saines
    candidates = np.unique(scores)[::-1]  # seuils candidats : chaque score distinct, du plus grand au plus petit
    candidates = np.concatenate([[np.inf], candidates])  # inf : tout est declare sain
    order = np.argsort(-scores, kind="stable")  # ordre des images du score le plus haut au plus bas
    sorted_scores, sorted_pos = scores[order], positive[order]  # scores et etiquettes ranges dans cet ordre
    tp_cum = np.cumsum(sorted_pos)  # vrais positifs cumules : defauts trouves parmi les k images les plus suspectes
    fp_cum = np.cumsum(~sorted_pos)  # faux positifs cumules : images saines parmi les k plus suspectes
    # nombre d'exemples de score >= t pour chaque t candidat
    counts = np.searchsorted(-sorted_scores, -candidates, side="right")  # recherche rapide dans le tableau trie
    tp = np.where(counts > 0, tp_cum[np.maximum(counts - 1, 0)], 0)  # vrais positifs a chaque seuil (0 si rien n'est au-dessus)
    fp = np.where(counts > 0, fp_cum[np.maximum(counts - 1, 0)], 0)  # faux positifs (fausses alarmes) a chaque seuil
    with np.errstate(invalid="ignore", divide="ignore"):  # coupe les avertissements numpy de division par zero dans ce bloc
        precision = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 1.0)  # precision ; 1.0 par convention si aucune alerte
    return pd.DataFrame(  # renvoie un tableau : une ligne par seuil candidat
        {  # dictionnaire nom de colonne -> valeurs de la colonne
            "threshold": candidates,  # le seuil
            "recall": tp / n_pos if n_pos else np.zeros(len(candidates)),  # rappel (0 partout s'il n'y a aucun defaut)
            "false_alarm_rate": fp / n_neg if n_neg else np.zeros(len(candidates)),  # taux de fausses alarmes
            "precision": precision,  # precision
            "tp": tp,  # nombre de defauts detectes
            "fp": fp,  # nombre de fausses alarmes
        }
    )


def threshold_for(  # choisit le seuil de decision qui respecte des contraintes ; renvoie un dict avec le seuil et ses chiffres
    scores: np.ndarray,  # scores du modele sur la VALIDATION (plus haut = plus suspect)
    is_defect: np.ndarray,  # verite : True / 1 si l'image a vraiment un defaut
    *,  # les arguments suivants doivent etre nommes (min_recall=0.95), pour eviter les confusions
    min_recall: float | None = None,  # rappel minimum voulu (None = pas de contrainte)
    max_false_alarm: float | None = None,  # taux de fausses alarmes maximum accepte
    min_precision: float | None = None,  # precision minimum voulue
) -> dict:  # renvoie un dict : seuil, rappel, fausses alarmes, precision, feasible, message
    """Seuil de decision (score >= seuil -> defaut) qui respecte les contraintes donnees, a choisir sur la VALIDATION.

    min_recall seul      : le seuil le plus haut qui garde ce rappel (le moins de fausses alarmes) ;
    max_false_alarm seul : le seuil le plus bas sous ce taux (le meilleur rappel) ;
    min_precision seul   : le seuil le plus bas qui garde cette precision.
    Plusieurs contraintes : parmi les seuils qui les respectent toutes, le meilleur rappel puis le
    moins de fausses alarmes. Si aucun seuil ne les respecte : feasible=False, le seuil le plus
    proche (plus petit manque total) et un message qui dit ce qui est atteignable.
    """
    if min_recall is None and max_false_alarm is None and min_precision is None:  # si aucune contrainte n'est donnee
        raise ValueError("donner au moins une contrainte : min_recall, max_false_alarm ou min_precision")  # on ne sait pas choisir
    table = operating_points(scores, is_defect)  # rappel / fausses alarmes / precision pour chaque seuil candidat
    shortfall = np.zeros(len(table))  # "manque" de chaque seuil par rapport aux contraintes (0 = toutes respectees)
    if min_recall is not None:  # si un rappel minimum est demande
        shortfall += np.maximum(0.0, min_recall - table["recall"].to_numpy())  # ajoute ce qui manque de rappel (0 si atteint)
    if max_false_alarm is not None:  # si un taux de fausses alarmes maximum est demande
        shortfall += np.maximum(0.0, table["false_alarm_rate"].to_numpy() - max_false_alarm)  # ajoute le depassement
    if min_precision is not None:  # si une precision minimum est demandee
        shortfall += np.maximum(0.0, min_precision - table["precision"].to_numpy())  # ajoute ce qui manque de precision
    table["shortfall"] = shortfall  # range ce manque dans une colonne du tableau
    feasible = table[table["shortfall"] <= 1e-12]  # les seuils qui respectent tout (1e-12 : tolerance aux arrondis)
    if len(feasible):  # s'il existe au moins un seuil acceptable
        if min_recall is not None and max_false_alarm is None and min_precision is None:  # si seul le rappel est impose
            # le moins de fausses alarmes, puis le seuil le plus haut
            best = feasible.sort_values(["false_alarm_rate", "threshold"], ascending=[True, False]).iloc[0]
        else:  # plusieurs contraintes ou une autre contrainte seule
            # le meilleur rappel, puis le moins de fausses alarmes, puis le seuil le plus haut
            best = feasible.sort_values(["recall", "false_alarm_rate", "threshold"], ascending=[False, True, False]).iloc[0]
    else:  # aucun seuil ne respecte tout : on prend le plus proche
        # le plus petit manque, puis le meilleur rappel, puis le moins de fausses alarmes
        best = table.sort_values(["shortfall", "recall", "false_alarm_rate"], ascending=[True, False, True]).iloc[0]
    result = {  # dictionnaire de resultat avec les chiffres du seuil choisi
        "threshold": float(best["threshold"]),  # le seuil choisi
        "recall": round(float(best["recall"]), 4),  # son rappel
        "false_alarm_rate": round(float(best["false_alarm_rate"]), 4),  # son taux de fausses alarmes
        "precision": round(float(best["precision"]), 4),  # sa precision
        "feasible": bool(len(feasible)),  # True si les contraintes sont toutes respectees
    }
    wanted = ", ".join(  # texte des contraintes demandees, separees par des virgules
        text  # on garde le texte de la contrainte...
        for text, value in (  # ...pour chaque couple (texte, valeur) ci-dessous...
            (f"rappel >= {min_recall}", min_recall),  # contrainte de rappel
            (f"fausses alarmes <= {max_false_alarm}", max_false_alarm),  # contrainte de fausses alarmes
            (f"precision >= {min_precision}", min_precision),  # contrainte de precision
        )
        if value is not None  # ...seulement si cette contrainte a ete donnee
    )
    # texte des chiffres atteints au seuil choisi, pour le message
    reached = f"rappel {result['recall']}, fausses alarmes {result['false_alarm_rate']}, precision {result['precision']}"
    if result["feasible"]:  # si les contraintes sont respectees
        result["message"] = f"contraintes respectees ({wanted}) au seuil {result['threshold']:.4g} : {reached}"  # message de succes
    else:  # sinon : aucun seuil ne respecte tout
        # message d'echec qui dit ce qu'on peut atteindre au mieux
        result["message"] = f"IMPOSSIBLE sur ces donnees ({wanted}) ; le plus proche : seuil {result['threshold']:.4g} -> {reached}"
    return result  # renvoie le seuil, ses chiffres et le message


def _confidence_and_correct(y_true: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  # renvoie (confiance, juste ?)
    """Confiance de la classe predite et justesse ; probs (N,) = probabilite de la classe positive, ou (N, C)."""
    y_true, probs = np.asarray(y_true), np.asarray(probs, dtype=np.float64)  # entrees en tableaux numpy (probas en decimaux)
    if probs.ndim == 1:  # cas binaire : une seule probabilite par image (celle de la classe positive)
        predicted = (probs >= 0.5).astype(int)  # classe predite : 1 si la probabilite depasse 0.5, sinon 0
        confidence = np.where(predicted == 1, probs, 1 - probs)  # confiance = probabilite de la classe choisie
    else:  # cas multi-classes : un tableau (N images, C classes) de probabilites
        predicted = probs.argmax(axis=1)  # classe predite = celle de plus forte probabilite
        confidence = probs.max(axis=1)  # confiance = cette plus forte probabilite
    return confidence, predicted == y_true  # renvoie la confiance et un tableau True/False "prediction juste"


def expected_calibration_error(  # erreur de calibration (ECE) ; renvoie un nombre, ou (nombre, tableau) si return_table
    y_true: np.ndarray,  # vraies classes
    probs: np.ndarray,  # probabilites predites par le modele
    bins: int = 15,  # nombre de tranches de confiance (0-0.067, 0.067-0.133, ...)
    return_table: bool = False,  # True : renvoyer aussi le detail par tranche
) -> float | tuple[float, pd.DataFrame]:  # ECE seule, ou (ECE, tableau par tranche)
    """Erreur de calibration attendue (ECE) : ecart moyen entre la confiance annoncee et la justesse reelle.

    0 : quand le modele dit 0,9, il a raison 9 fois sur 10. Un reseau profond est souvent trop sur
    de lui (ECE de 0,05 a 0,15) ; a corriger (temperature) avant d'utiliser la probabilite comme
    un vrai niveau de confiance. return_table : le detail par tranche (pour un diagramme de fiabilite).
    """
    confidence, correct = _confidence_and_correct(y_true, probs)  # confiance annoncee et justesse de chaque image
    edges = np.linspace(0.0, 1.0, bins + 1)  # bornes des tranches, regulierement espacees de 0 a 1
    index = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, bins - 1)  # numero de tranche de chaque image
    rows, ece = [], 0.0  # lignes du tableau de detail, et somme de l'erreur
    for b in range(bins):  # pour chaque tranche de confiance
        member = index == b  # True pour les images qui tombent dans cette tranche
        if not member.any():  # si la tranche est vide
            continue  # on passe a la suivante
        accuracy, mean_conf = float(correct[member].mean()), float(confidence[member].mean())  # justesse reelle et confiance moyenne
        ece += member.mean() * abs(accuracy - mean_conf)  # ajoute l'ecart, pondere par la part d'images dans la tranche
        rows.append(  # garde le detail de la tranche pour le tableau
            # bornes de la tranche, nombre d'images, confiance moyenne annoncee, justesse reelle
            {"bin_low": edges[b], "bin_high": edges[b + 1], "count": int(member.sum()), "confidence": mean_conf, "accuracy": accuracy}
        )
    ece = float(ece)  # en nombre Python ordinaire
    return (ece, pd.DataFrame(rows)) if return_table else ece  # renvoie l'ECE, avec le tableau si demande


def worst_errors(y_true: np.ndarray, probs: np.ndarray, k: int = 12) -> np.ndarray:  # renvoie les indices des k pires erreurs
    """Indices des k erreurs les plus sures d'elles (a regarder en premier : souvent des erreurs d'etiquette)."""
    confidence, correct = _confidence_and_correct(y_true, probs)  # confiance et justesse de chaque image
    wrong = np.flatnonzero(~correct)  # indices des images mal classees
    return wrong[np.argsort(-confidence[wrong], kind="stable")][:k]  # triees de la plus confiante a la moins, on garde les k premieres
