"""Decoupage train / val / test : calcule une fois, controle contre les fuites, enregistre en CSV.

Strategies (split.strategy dans le YAML) :
- stratified : proportions de classes conservees (le cas par defaut) ;
- grouped    : stratifie ET toutes les images d'un meme groupe (piece, bobine, brame, camera)
               du meme cote ; indispensable des que plusieurs images viennent de la meme source ;
- temporal   : le passe pour entrainer, le plus recent pour tester (mesure la tenue dans le temps,
               la vraie question en production ; les groupes ne sont jamais coupes) ;
- random     : sans stratification (jeux sans etiquette) ;
- provided   : le rangement d'origine fixe deja le decoupage (dossiers train/val/test, MVTec) ;
               s'il manque val, on la preleve dans train.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (ex. pd.Series | None)

import logging  # journal : messages horodates sur la console et dans un fichier
from collections.abc import Iterable  # type "objet qu'on peut parcourir" (liste, tuple...) pour annoter
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import numpy as np  # numpy : calcul sur des tableaux de nombres
import pandas as pd  # pandas : tableaux de donnees (DataFrame) ; ici le catalogue des images

# scikit-learn : StratifiedGroupKFold coupe en plis sans couper les groupes en gardant les proportions de classes ;
# train_test_split preleve une part des lignes au hasard (stratifiee si demande)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from vtb.core.config import SplitConfig  # section "split" de la configuration (strategie, fractions)

logger = logging.getLogger(__name__)  # journal propre a ce module
SUBSETS = ("train", "val", "test")  # les 3 sous-ensembles : entrainement, validation (reglages), test (note finale)


def _can_stratify(labels: pd.Series, fraction: float) -> bool:  # dit si un tirage stratifie est possible (True / False)
    """La stratification exige au moins 2 images par classe et assez d'images pour chaque classe dans la part prelevee."""
    counts = labels.value_counts()  # nombre d'images par classe
    # au moins 2 classes, au moins 2 images dans la plus petite, et la part prelevee assez grande pour une image par classe
    return len(counts) > 1 and counts.min() >= 2 and fraction * len(labels) >= len(counts)


# preleve au hasard une part des lignes (stratifiee si possible) ; renvoie les numeros de lignes prelevees
def _holdout_random(index: pd.Index, labels: pd.Series | None, fraction: float, seed: int) -> pd.Index:
    """Preleve une fraction des lignes, stratifiee si possible."""
    if fraction <= 0:  # rien a prelever
        return index[:0]  # index vide (meme type que l'entree)
    stratify = labels if labels is not None and _can_stratify(labels, fraction) else None  # classes a respecter, ou None (pur hasard)
    _, held = train_test_split(index, test_size=fraction, stratify=stratify, random_state=seed)  # coupe en 2 ; on garde la part prelevee
    return pd.Index(held)  # renvoie les lignes prelevees sous forme d'index pandas


def _holdout_grouped(frame: pd.DataFrame, fraction: float, seed: int) -> pd.Index:  # preleve une part sans couper de groupe
    """Preleve environ une fraction des lignes sans couper aucun groupe, en equilibrant les classes.

    StratifiedGroupKFold a n plis donne des plis de taille 1/n : on prend le premier pli avec
    n = round(1 / fraction). La fraction obtenue est approximative, c'est le prix de groupes entiers.
    """
    if fraction <= 0:  # rien a prelever
        return frame.index[:0]  # index vide
    n_splits = max(2, round(1 / fraction))  # nombre de plis : 15 % -> 7 plis ; au moins 2
    if frame["group"].nunique() < n_splits:  # s'il y a moins de groupes que de plis, impossible sans couper un groupe
        raise ValueError(f"{frame['group'].nunique()} groupes : trop peu pour prelever {fraction:.0%} sans couper de groupe")  # erreur
    folds = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)  # decoupeur en plis, melange reproductible
    # classes a equilibrer ; sans colonne label, une classe unique "x" pour tout le monde
    labels = frame["label"].astype(str) if "label" in frame else pd.Series("x", index=frame.index)
    _, held = next(folds.split(frame, labels, groups=frame["group"]))  # prend seulement le premier pli : positions des lignes prelevees
    return frame.index[held]  # convertit les positions en numeros de lignes du tableau


def _holdout_temporal(frame: pd.DataFrame, fraction: float) -> pd.Index:  # preleve les groupes les plus recents ; renvoie l'index
    """Les groupes les plus recents (date de leur premiere image) jusqu'a atteindre la fraction demandee."""
    if fraction <= 0:  # rien a prelever
        return frame.index[:0]  # index vide
    group_start = frame.groupby("group")["timestamp"].min().sort_values()  # date de debut de chaque groupe, du plus ancien au plus recent
    sizes = frame.groupby("group").size().reindex(group_start.index)  # nombre d'images par groupe, dans le meme ordre
    target, taken, held_groups = fraction * len(frame), 0, []  # objectif en images, images deja prises, groupes retenus
    for group in reversed(group_start.index):  # pour chaque groupe, du plus recent au plus ancien
        if taken >= target:  # si on a atteint la quantite voulue
            break  # on arrete de prendre des groupes
        held_groups.append(group)  # retient ce groupe entier
        taken += sizes[group]  # compte ses images
    return frame.index[frame["group"].isin(held_groups)]  # lignes dont le groupe a ete retenu


def make_split(catalog: pd.DataFrame, config: SplitConfig, seed: int) -> pd.DataFrame:  # decoupe le catalogue (colonne split)
    """Ajoute (ou complete) la colonne split selon la strategie ; ajoute fold si n_folds est demande."""
    catalog = catalog.copy().reset_index(drop=True)  # copie avec des numeros de lignes 0, 1, 2... (sans trous)
    # classes des images si la colonne label existe et contient au moins une etiquette non vide ; sinon None
    labels = catalog["label"].astype(str) if "label" in catalog and catalog["label"].astype(str).str.len().gt(0).any() else None

    if config.strategy == "provided":  # strategie "provided" : le rangement d'origine donne deja le decoupage
        if "split" not in catalog:  # sans colonne split, impossible
            raise ValueError("strategy provided : le catalogue n'a pas de colonne split")  # erreur claire
        if not (catalog["split"] == "val").any() and config.val_fraction > 0:  # pas de validation fournie mais on en veut une
            train = catalog[catalog["split"] == "train"]  # lignes d'entrainement
            # preleve la validation dans l'entrainement (stratifiee si les classes sont connues)
            held = _holdout_random(train.index, labels[train.index] if labels is not None else None, config.val_fraction, seed)
            catalog.loc[held, "split"] = "val"  # ces lignes passent en validation
    else:  # toutes les autres strategies : on calcule le decoupage nous-memes
        catalog["split"] = "train"  # par defaut tout est en entrainement ; on preleve ensuite test puis val
        if config.strategy == "grouped":  # strategie par groupes (aucun groupe coupe)
            test = _holdout_grouped(catalog, config.test_fraction, seed)  # preleve le test par groupes entiers
            catalog.loc[test, "split"] = "test"  # marque ces lignes comme test
            rest = catalog[catalog["split"] == "train"]  # ce qui reste pour entrainement + validation
            # preleve la validation dans le reste ; fraction rapportee au reste (15 % du total = 15/85 du reste)
            val = _holdout_grouped(rest, config.val_fraction / (1 - config.test_fraction), seed)
        elif config.strategy == "temporal":  # strategie temporelle : les plus recents en test
            if "timestamp" not in catalog:  # sans dates, impossible
                raise ValueError("strategy temporal : il faut une colonne timestamp (data.timestamp_regex)")  # erreur claire
            test = _holdout_temporal(catalog, config.test_fraction)  # les groupes les plus recents vont en test
            catalog.loc[test, "split"] = "test"  # marque ces lignes comme test
            rest = catalog[catalog["split"] == "train"]  # ce qui reste
            val = _holdout_temporal(rest, config.val_fraction / (1 - config.test_fraction))  # les plus recents du reste vont en validation
        else:  # strategies "stratified" et "random"
            use_labels = labels if config.strategy == "stratified" else None  # classes a respecter seulement en stratified
            test = _holdout_random(catalog.index, use_labels, config.test_fraction, seed)  # preleve le test
            catalog.loc[test, "split"] = "test"  # marque ces lignes comme test
            rest = catalog[catalog["split"] == "train"]  # ce qui reste
            val = _holdout_random(  # preleve la validation dans le reste
                rest.index,  # lignes candidates
                use_labels[rest.index] if use_labels is not None else None,  # leurs classes (ou None)
                config.val_fraction / (1 - config.test_fraction),  # fraction rapportee au reste
                seed,  # graine : meme tirage a chaque execution
            )
        catalog.loc[val, "split"] = "val"  # marque les lignes de validation

    if config.n_folds:  # si une validation croisee est demandee
        catalog = assign_folds(catalog, config.n_folds, seed)  # ajoute la colonne fold
    return catalog  # renvoie le catalogue decoupe


def assign_folds(catalog: pd.DataFrame, n_folds: int, seed: int) -> pd.DataFrame:  # ajoute la colonne fold ; renvoie une copie
    """Colonne fold (0..n-1) pour la validation croisee, sur train + val seulement ; le test garde fold = -1."""
    catalog = catalog.copy()  # copie : on ne modifie pas le tableau recu
    catalog["fold"] = -1  # -1 = hors validation croisee (cas du test)
    dev = catalog[catalog["split"] != "test"]  # lignes de developpement : train + val
    labels = dev["label"].astype(str) if "label" in dev else pd.Series("x", index=dev.index)  # classes (ou une classe unique "x")
    groups = dev["group"] if "group" in dev else pd.Series(np.arange(len(dev)), index=dev.index)  # groupes (ou un groupe par image)
    folds = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)  # decoupeur en n plis, groupes respectes
    for k, (_, held) in enumerate(folds.split(dev, labels, groups=groups)):  # pour chaque pli k : positions des lignes de ce pli
        catalog.loc[dev.index[held], "fold"] = k  # ces lignes recoivent le numero de pli k
    return catalog  # renvoie le catalogue avec la colonne fold


def check_no_leakage(catalog: pd.DataFrame, normal_only_train: bool = False) -> None:  # verifie l'absence de fuite ; leve une erreur sinon
    """Refuse un decoupage dangereux : doublon de chemin, sous-ensemble inconnu ou vide, groupe coupe en deux.

    normal_only_train : pour les anomalies, refuse aussi toute image non "good" dans train / val.
    """
    if not catalog["path"].is_unique:  # si un meme chemin apparait plusieurs fois
        raise ValueError("des images apparaissent deux fois dans le catalogue")  # erreur : risque de la voir en train et en test
    unknown = set(catalog["split"]) - set(SUBSETS)  # valeurs de split autres que train / val / test
    if unknown:  # s'il y en a
        raise ValueError(f"valeurs de split inconnues : {sorted(unknown)}")  # erreur avec la liste
    if not (catalog["split"] == "test").any():  # si aucune ligne n'est en test
        raise ValueError("aucune image de test")  # impossible de mesurer la performance finale
    if "group" in catalog:  # si les groupes sont connus
        spread = catalog.groupby("group")["split"].nunique()  # pour chaque groupe : dans combien de sous-ensembles il apparait
        crossing = spread[spread > 1]  # groupes presents dans plus d'un sous-ensemble = fuite
        if len(crossing):  # s'il y a au moins un groupe coupe
            # erreur avec le nombre de groupes fautifs et 3 exemples
            raise ValueError(f"{len(crossing)} groupes repartis sur plusieurs sous-ensembles (fuite), ex. {list(crossing.index[:3])}")
    if normal_only_train and "label" in catalog:  # cas anomalies : train / val ne doivent contenir que des images normales
        dirty = catalog[catalog["split"].isin(["train", "val"]) & (catalog["label"] != "good")]  # images non normales en train / val
        if len(dirty):  # s'il y en a
            # erreur : un detecteur d'anomalies apprend seulement ce qui est normal
            raise ValueError(f"{len(dirty)} images non normales dans train / val : un detecteur d'anomalies n'apprend que le normal")


def summarize_split(catalog: pd.DataFrame) -> pd.DataFrame:  # renvoie un tableau de comptage par classe et sous-ensemble
    """Tableau classes x sous-ensembles (a mettre dans le journal)."""
    if "label" in catalog and catalog["label"].astype(str).str.len().gt(0).any():  # si des classes sont connues
        return pd.crosstab(catalog["label"], catalog["split"], margins=True)  # tableau croise classes x sous-ensembles, avec totaux
    return catalog["split"].value_counts().to_frame("images")  # sinon : simple nombre d'images par sous-ensemble


def save_split(catalog: pd.DataFrame, path: str | Path) -> None:  # enregistre le decoupage dans un CSV
    Path(path).parent.mkdir(parents=True, exist_ok=True)  # cree le dossier s'il n'existe pas
    catalog.to_csv(path, index=False)  # ecrit le CSV (sans la colonne des numeros de lignes)
    logger.info("decoupage enregistre : %s\n%s", path, summarize_split(catalog))  # journal : chemin + tableau de comptage


# relit le decoupage enregistre ; renvoie seulement les sous-ensembles demandes
def load_split(path: str | Path, subsets: Iterable[str] | str | None = None) -> pd.DataFrame:
    """Relit le decoupage fige ; subsets choisit explicitement ce qu'on lit (un script d'entrainement ne demande jamais test)."""
    catalog = pd.read_csv(path, keep_default_na=False)  # lit le CSV ; les cases vides restent "" (pas NaN)
    if subsets is None:  # aucun filtre demande
        return catalog  # renvoie tout
    wanted = [subsets] if isinstance(subsets, str) else list(subsets)  # "train" -> ["train"] ; une liste reste une liste
    return catalog[catalog["split"].isin(wanted)].reset_index(drop=True)  # garde les sous-ensembles voulus, renumerote les lignes
