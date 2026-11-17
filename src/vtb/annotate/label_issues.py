"""Etiquettes suspectes : laisser le modele designer les images a relire.

Principe (apprentissage confiant, simplifie) : on obtient pour chaque image des probabilites
HORS PLI, c'est-a-dire predites par un modele qui ne l'a pas vue a l'entrainement (validation
croisee). Une image dont la classe donnee recoit une probabilite tres faible alors qu'une autre
classe depasse le seuil de confiance de sa propre classe est probablement mal etiquetee.
En pratique 30 a 60 % des images ainsi designees sont de vraies erreurs d'etiquetage.

Sans probabilites hors pli (probabilites predites sur le train par le modele entraine dessus),
le modele a appris les erreurs et ne les signale plus : la methode ne vaut alors rien.

cleanlab (pip install cleanlab) implemente la version complete ; il est utilise si demande et
installe, sinon la version simplifiee ci-dessous suffit pour trier une relecture.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[str], X | None) partout

import logging  # messages d'information et d'avertissement, plus souples que print
from collections.abc import Sequence  # Sequence : type "liste ou tuple", pour annoter les parametres

import numpy as np  # numpy : calcul sur des tableaux de nombres
import pandas as pd  # pandas : tableaux a lignes et colonnes nommees (DataFrame)

logger = logging.getLogger(__name__)  # journal propre a ce module


# repere les images probablement mal etiquetees ; renvoie un tableau trie de la plus suspecte a la moins suspecte
def find_label_issues(
    labels: Sequence[int],  # classe donnee par l'annotateur pour chaque image (entier 0..C-1)
    probabilities: np.ndarray,  # probabilites hors pli : N images x C classes, chaque ligne somme a 1
    class_names: Sequence[str] | None = None,  # noms des classes pour l'affichage ; None = numeros
    use_cleanlab: bool = False,  # True : utilise la bibliotheque cleanlab (version complete) si elle est installee
) -> pd.DataFrame:  # renvoie un DataFrame pandas (une ligne par image)
    """Tableau des images suspectes, de la plus suspecte a la moins suspecte.

    labels : classe donnee par image (entiers 0..C-1) ; probabilities : (N, C) hors pli.
    Colonnes : index, given, predicted, self_confidence (probabilite de la classe donnee),
    predicted_confidence, suspect (bool).
    """
    labels = np.asarray(labels, dtype=int)  # convertit en tableau numpy d'entiers (permet l'indexation vectorisee)
    probs = np.asarray(probabilities, dtype=np.float64)  # tableau de flottants N x C
    n_classes = probs.shape[1]  # C = nombre de colonnes = nombre de classes
    names = list(class_names) if class_names is not None else [str(k) for k in range(n_classes)]  # noms, ou "0", "1"...
    self_conf = probs[np.arange(len(labels)), labels]  # pour chaque image i : probabilite de SA classe donnee, probs[i, labels[i]]
    predicted = probs.argmax(axis=1)  # classe la plus probable selon le modele, pour chaque image

    if use_cleanlab:  # si la version complete est demandee
        try:  # cleanlab n'est peut-etre pas installe
            from cleanlab.filter import find_label_issues as cleanlab_issues  # import a la demande, renomme pour eviter le conflit

            suspect = cleanlab_issues(labels, probs, filter_by="prune_by_noise_rate")  # tableau True/False : image suspecte ?
        except ImportError:  # si cleanlab n'est pas installe
            logger.warning("cleanlab non installe : version simplifiee utilisee")  # previent l'utilisateur
            use_cleanlab = False  # bascule sur la version simplifiee ci-dessous
    if not use_cleanlab:  # version simplifiee (apprentissage confiant)
        # Seuil par classe = confiance moyenne des images de cette classe (t_j de l'apprentissage confiant).
        # classe sans aucune image : seuil 1.0, donc jamais depasse
        thresholds = np.array([self_conf[labels == k].mean() if np.any(labels == k) else 1.0 for k in range(n_classes)])
        above = probs >= thresholds[None, :]  # tableau N x C : la proba de la classe k depasse-t-elle le seuil de k ?
        above[np.arange(len(labels)), labels] = False  # une autre classe que la classe donnee depasse son seuil
        suspect = above.any(axis=1) & (predicted != labels)  # suspecte : une autre classe est sure d'elle ET gagne la prediction

    table = pd.DataFrame(  # construit le tableau de resultats, une colonne par information
        {  # dictionnaire : nom de colonne -> valeurs de la colonne
            "index": np.arange(len(labels)),  # numero de l'image dans le jeu d'origine
            "given": [names[k] for k in labels],  # classe donnee par l'annotateur
            "predicted": [names[k] for k in predicted],  # classe predite par le modele
            "self_confidence": self_conf,  # probabilite de la classe donnee (faible = douteux)
            "predicted_confidence": probs.max(axis=1),  # probabilite de la classe predite
            "suspect": suspect,  # True : image a relire en priorite
        }
    )
    # suspectes d'abord, puis par confiance croissante dans la classe donnee ; numerotation des lignes refaite
    return table.sort_values(["suspect", "self_confidence"], ascending=[False, True]).reset_index(drop=True)
