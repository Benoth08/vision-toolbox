"""Derive de domaine : les images d'aujourd'hui ressemblent-elles encore a celles de l'entrainement ?

Changement de camera, d'eclairage, de nuance d'acier, objectif sale : le modele ne previent pas,
il se trompe en silence. Deux mesures complementaires, sans etiquette :

1. classifieur de domaine : une regression logistique essaie de distinguer "reference" et
   "courant" a partir des plongements. AUC ~ 0,5 : indiscernables (pas de derive detectable) ;
   AUC proche de 1 : les deux periodes sont faciles a separer -> derive. C'est le test le plus
   sensible, et il dit QUELLES images sont les plus "nouvelles" (probabilite d'etre courantes).
2. statistiques simples (luminosite, contraste, nettete, taille) : test de Kolmogorov-Smirnov
   colonne par colonne ; moins sensible, mais il dit QUOI a change (ex. l'image est plus sombre).
"""

from __future__ import annotations  # permet d'ecrire les types (ex. "list[str] | None") sans qu'ils soient evalues au chargement

import numpy as np  # numpy : calcul sur des tableaux de nombres (ici les plongements, un vecteur par image)
import pandas as pd  # pandas : tableaux avec colonnes nommees (DataFrame), pratiques pour les statistiques par image

from vtb.quality.checks import STAT_COLUMNS  # noms des statistiques d'image calculees par le controle qualite (luminosite...)

VERDICTS = (  # seuils d'AUC et phrase a afficher : on prend la premiere ligne dont le seuil depasse l'AUC
    (0.6, "pas de derive detectable"),  # AUC < 0.6 : le classifieur devine presque au hasard, periodes indiscernables
    (0.75, "derive moderee : surveiller, verifier quelques images"),  # 0.6 <= AUC < 0.75 : difference legere
    (1.01, "derive forte : reevaluer le modele avant de lui faire confiance"),  # AUC >= 0.75 (1.01 pour inclure AUC = 1)
)


# renvoie l'AUC (qualite de separation reference / courant) et, pour chaque image courante, sa probabilite d'etre "courante"
def domain_classifier(reference: np.ndarray, current: np.ndarray, seed: int = 0) -> tuple[float, np.ndarray]:
    """AUC en validation croisee (5 plis) et probabilite "courant" de chaque image courante (hors pli)."""
    from sklearn.linear_model import LogisticRegression  # regression logistique : classifieur lineaire simple qui donne une proba
    from sklearn.metrics import roc_auc_score  # AUC : 0.5 = hasard, 1 = separation parfaite des deux groupes

    # StratifiedKFold : decoupe en plis en gardant la meme proportion de chaque groupe ;
    # cross_val_predict : predit chaque image avec un modele qui ne l'a pas vue a l'entrainement
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline  # enchaine plusieurs etapes (mise a l'echelle puis classifieur) en un seul modele
    from sklearn.preprocessing import StandardScaler  # centre chaque colonne et la ramene a un ecart-type de 1

    x = np.concatenate([reference, current])  # empile tous les plongements : d'abord la reference, puis le courant
    y = np.r_[np.zeros(len(reference)), np.ones(len(current))]  # etiquettes : 0 = reference, 1 = courant
    # validation croisee en 5 plis (moins si un groupe a moins de 5 images), melanges de facon reproductible avec la graine
    folds = StratifiedKFold(n_splits=min(5, len(reference), len(current)), shuffle=True, random_state=seed)
    # C=0.1 : forte regularisation (evite d'apprendre par coeur) ; max_iter=2000 : assez d'iterations pour converger
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    # probabilite d'etre "courant" pour chaque image, predite par le pli qui ne l'a pas vue ([:, 1] = colonne de la classe 1)
    proba = cross_val_predict(model, x, y, cv=folds, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, proba)), proba[len(reference) :]  # renvoie l'AUC et les probas des seules images courantes


# compare chaque statistique d'image entre les deux periodes et renvoie un tableau trie du plus grand ecart au plus petit
def statistics_drift(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Test KS par statistique d'image (sorties de vtb.quality.checks.scan) : ecart et sens du changement."""
    from scipy.stats import ks_2samp  # test de Kolmogorov-Smirnov : les deux series viennent-elles de la meme distribution ?

    rows = []  # une ligne de resultat par statistique, remplie dans la boucle
    for column in columns or STAT_COLUMNS:  # pour chaque statistique demandee (toutes par defaut)
        # valeurs de la statistique cote reference (a) et cote courant (b), sans les cases vides, en nombres decimaux
        a, b = reference[column].dropna().to_numpy(float), current[column].dropna().to_numpy(float)
        if len(a) < 2 or len(b) < 2:  # si moins de 2 valeurs d'un cote, le test n'a pas de sens
            continue  # on passe a la statistique suivante
        test = ks_2samp(a, b)  # statistic = plus grand ecart entre les deux distributions (0 a 1), pvalue = risque de hasard
        rows.append(  # ajoute le resultat de cette statistique au tableau
            {  # dictionnaire = une ligne du futur tableau
                "statistic": column,  # nom de la statistique testee
                "ks": round(float(test.statistic), 3),  # ecart KS arrondi : plus il est grand, plus la statistique a change
                "p_value": float(test.pvalue),  # petite p-value (< 0.05) : la difference n'est probablement pas due au hasard
                "reference_median": round(float(np.median(a)), 4),  # valeur typique avant (mediane)
                "current_median": round(float(np.median(b)), 4),  # valeur typique maintenant : comparer donne le sens du changement
            }
        )
    # construit le tableau, met les plus forts ecarts en haut, renumerote les lignes 0, 1, 2...
    return pd.DataFrame(rows).sort_values("ks", ascending=False).reset_index(drop=True)


def verdict(auc: float) -> str:  # traduit l'AUC du classifieur de domaine en phrase lisible
    return next(text for limit, text in VERDICTS if auc < limit)  # premiere phrase dont le seuil est au-dessus de l'AUC
