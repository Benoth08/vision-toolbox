"""Base de comparaison sans apprentissage : une carte de reponse, un seuil, un filtrage des blobs.

Deux methodes, choisies dans le YAML (method) :
- response  : le pretraitement fabrique une carte ou les defauts ressortent (correction
              d'illumination puis black-hat pour des defauts sombres, par exemple) ;
- reference : comparaison a une piece "bonne" de reference (mediane d'images normales), apres
              alignement : |image - reference|. La methode reine quand les pieces sont
              identiques et bien positionnees.
Dans les deux cas la carte est un flottant dans [0, 1] et les seuils sont des fractions de la
dynamique. Le seuil et l'aire minimale des blobs sont choisis sur la validation (grille), selon
le critere industriel voulu ; le test n'est lu qu'une fois, pour le rapport. C'est le chiffre
qu'un modele appris doit battre pour justifier sa complexite.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[str], X | None) partout

import itertools  # outils d'iteration ; ici product : toutes les combinaisons (seuil, aire)
import logging  # messages d'information et d'avertissement, plus souples que print
from collections.abc import Sequence  # Sequence : type "liste ou tuple", pour annoter les parametres
from typing import Literal  # Literal : type qui n'accepte que quelques valeurs texte precises

import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)
import pandas as pd  # pandas : tableaux a lignes et colonnes nommees (DataFrame)
from pydantic import Field  # pydantic : valide la configuration lue dans le YAML ; Field fixe une valeur par defaut
from sklearn.model_selection import train_test_split  # coupe une liste en deux parties (ici val / test), stratifiee

from vtb.classical.blobs import detect_blobs  # detection et filtrage des taches d'un masque
from vtb.classical.compare import compare_to_reference  # carte de difference avec une piece de reference

# sections de configuration communes (donnees, pretraitement, execution, decoupage) et modele strict (refuse les cles inconnues)
from vtb.core.config import DataConfig, PreprocessStep, RuntimeConfig, SplitConfig, StrictModel
from vtb.io.images import read_image, read_mask  # lecture d'une image ; lecture d'un masque en 0/1
from vtb.preprocess.dtypes import as_float  # convertit une image en flottants entre 0 et 1
from vtb.preprocess.ops.geometry import resize  # redimensionnement d'image
from vtb.preprocess.pipeline import Pipeline  # suite d'operations de pretraitement decrite dans le YAML

logger = logging.getLogger(__name__)  # journal propre a ce module


class ReferenceConfig(StrictModel):  # reglages de la methode "reference" (section reference du YAML)
    """Methode reference : comment fabriquer la piece de reference et comparer."""

    n_images: int = 60  # images normales de train fusionnees en une reference
    statistic: Literal["median", "mean"] = "median"  # la mediane ignore un defaut present sur une image
    align: Literal["ecc", "orb", "none"] = "ecc"  # alignement de chaque image sur la reference avant soustraction
    blur_sigma: float = 2.0  # flou applique avant soustraction : absorbe bruit et petit decalage


class ClassicalConfig(StrictModel):  # configuration complete de la recette classical (tout le fichier YAML)
    """Configuration de la recette classical."""

    data: DataConfig  # ou sont les images (obligatoire)
    # decoupage par defaut : 20 % validation, 30 % test, enregistre dans outputs/classical/split.csv
    split: SplitConfig = Field(default_factory=lambda: SplitConfig(file="outputs/classical/split.csv", val_fraction=0.2, test_fraction=0.3))
    runtime: RuntimeConfig = Field(default_factory=lambda: RuntimeConfig(output_dir="outputs/classical"))  # dossier de sortie
    method: Literal["response", "reference"] = "response"  # methode : carte de reponse ou comparaison a une reference
    max_per_class: int = 60  # sous-echantillon par classe (la recette doit tourner en quelques minutes)
    preprocess: list[PreprocessStep] = Field(default_factory=list)  # response : la carte ; reference : mise a l'echelle commune
    reference: ReferenceConfig = Field(default_factory=ReferenceConfig)  # reglages de la methode reference
    invert: bool = False  # response : True si les defauts sont plus SOMBRES que le fond dans la carte
    thresholds: list[float] = Field(default_factory=lambda: [0.05, 0.08, 0.1, 0.12, 0.16, 0.2, 0.25, 0.3])  # fractions de [0, 1]
    min_areas: list[float] = Field(default_factory=lambda: [20, 60, 150])  # aire minimale des blobs (pixels)
    select: Literal["dice", "recall_at_max_false_alarm"] = "dice"  # critere de choix du reglage sur la validation
    max_false_alarm: float = 0.0  # critere recall_at_max_false_alarm : taux de fausses alarmes tolere sur val
    assets_dir: str | None = "docs/assets/classical"  # dossier des figures pour la documentation ; None = pas de figures


# qualite de segmentation sur tout le jeu ; renvoie {"iou": ..., "dice": ...}
def mask_scores(predicted: Sequence[np.ndarray], truth: Sequence[np.ndarray]) -> dict[str, float]:
    """IoU et Dice CUMULES sur le jeu (intersections et unions sommees), jamais moyennes image par image."""
    inter = sum(int((p.astype(bool) & t.astype(bool)).sum()) for p, t in zip(predicted, truth))  # pixels predits ET vrais
    pred = sum(int(p.astype(bool).sum()) for p in predicted)  # total de pixels predits defaut
    true = sum(int(t.astype(bool).sum()) for t in truth)  # total de pixels vraiment defaut
    # IoU = intersection / union ; Dice = 2 x intersection / (predits + vrais) ; max(..., 1) evite la division par 0
    return {"iou": inter / max(pred + true - inter, 1), "dice": 2 * inter / max(pred + true, 1)}


# qualite au niveau image ; renvoie {"recall": ..., "false_alarm_rate": ...}
def image_scores(predicted: Sequence[np.ndarray], has_defect: Sequence[bool]) -> dict[str, float]:
    """Niveau image : une image est declaree defectueuse si au moins un blob est retenu."""
    flagged = np.array([bool(p.any()) for p in predicted])  # True si le masque predit contient au moins un pixel
    truth = np.array(has_defect, dtype=bool)  # True si l'image a vraiment un defaut
    return {  # dictionnaire des deux taux
        # rappel : part des images defectueuses detectees (NaN = "pas un nombre" s'il n'y en a aucune)
        "recall": float(flagged[truth].mean()) if truth.any() else float("nan"),
        # fausses alarmes : part des images bonnes signalees a tort (~ = NON logique)
        "false_alarm_rate": float(flagged[~truth].mean()) if (~truth).any() else float("nan"),
    }


def segment(response: np.ndarray, value: float, min_area: float, invert: bool = False) -> np.ndarray:  # renvoie le masque 0/1 retenu
    """Carte flottante [0, 1] -> masque uint8 {0, 1} des blobs assez grands."""
    mask = (response < value) if invert else (response > value)  # pixels au-dela du seuil (en dessous si invert)
    return detect_blobs(mask.astype(np.uint8), response, min_area=min_area)[1]  # [1] = masque des blobs assez grands


# decoupe le catalogue en train / val / test pour la recette ; renvoie le catalogue avec la colonne split
def split_for_baseline(catalog: pd.DataFrame, config: ClassicalConfig, seed: int) -> pd.DataFrame:
    """Decoupage : si le rangement fournit deja train (normales) / test (MVTec), on coupe le test en val / test ;
    sinon sous-echantillon par classe puis decoupage stratifie classique."""
    from vtb.preprocess.benchmark import subsample_per_class  # tire au plus N images par classe
    from vtb.split import make_split  # decoupage train / val / test standard de la toolbox

    if "split" in catalog and set(catalog["split"]) == {"train", "test"}:  # rangement deja coupe (cas MVTec)
        catalog = catalog.copy()  # copie : on ne modifie pas le tableau de l'appelant
        evaluation = catalog[catalog["split"] == "test"]  # images de test fournies
        share = config.split.val_fraction / (config.split.val_fraction + config.split.test_fraction)  # part du test qui part en val
        # coupe le test en deux, en gardant la proportion de chaque classe (stratify) ; graine fixe pour rejouer
        val_idx, _ = train_test_split(evaluation.index, train_size=share, stratify=evaluation["label"], random_state=seed)
        catalog.loc[val_idx, "split"] = "val"  # ces images deviennent la validation
        return catalog.reset_index(drop=True)  # renumerote les lignes
    return make_split(subsample_per_class(catalog, config.max_per_class, seed), config.split, seed)  # cas general


def normal_rows(frame: pd.DataFrame) -> pd.DataFrame:  # renvoie seulement les lignes des images sans defaut
    """Les images sans defaut d'un tableau : has_defect faux s'il existe, sinon label good."""
    if "has_defect" in frame:  # colonne has_defect presente
        return frame[~frame["has_defect"].astype(bool)]  # lignes ou has_defect est faux
    return frame[frame["label"] == "good"] if "label" in frame else frame  # sinon classe "good" ; sans etiquette : tout


# fabrique la piece de reference ; renvoie une image du meme type que les images d'entree
def build_reference(paths: Sequence[str], pipeline: Pipeline, color: str, statistic: str = "median") -> np.ndarray:
    """Piece de reference : mediane (ou moyenne) pixel a pixel d'images normales pretraitees."""
    stack = np.stack([pipeline(read_image(p, color=color)) for p in paths])  # pile N x H x W (x C) d'images pretraitees
    merged = np.median(stack, axis=0) if statistic == "median" else stack.mean(axis=0)  # pixel a pixel sur les N images
    return merged.astype(stack.dtype)  # remet le type d'origine (median et mean donnent des flottants)


# calcule une carte de reponse et un masque de verite par image ; renvoie (cartes, masques)
def response_maps(
    frame: pd.DataFrame,  # images a traiter (colonnes path, mask_path...)
    pipeline: Pipeline,  # pretraitement a appliquer a chaque image
    config: ClassicalConfig,  # configuration de la recette
    reference: np.ndarray | None = None,  # piece de reference (methode reference seulement)
) -> tuple[list[np.ndarray], list[np.ndarray]]:  # renvoie deux listes, dans l'ordre du tableau
    """Cartes de reponse flottantes [0, 1] et masques de verite (a la taille des cartes), dans l'ordre du tableau."""
    responses, masks = [], []  # cartes de reponse et masques de verite
    for row in frame.itertuples():  # pour chaque image (itertuples : row.path, row.mask_path...)
        image = pipeline(read_image(row.path, color=config.data.color))  # lit puis pretraite l'image
        if config.method == "reference":  # methode reference
            response = compare_to_reference(  # carte |image - reference| apres alignement
                image,  # image pretraitee
                reference,  # piece de reference
                config.reference.align,  # methode d'alignement
                config.reference.blur_sigma,  # flou avant soustraction
                threshold=1.0,  # seuil inutilise ici : seule la carte "diff" nous interesse
                min_area=0,  # pas de filtrage des blobs a cette etape
            )["diff"]  # du dictionnaire renvoye, on garde seulement la carte de difference
        else:  # methode response : la carte est l'image pretraitee elle-meme
            # couleur -> moyenne des canaux (gris), puis flottants 0-1 ; [0] = l'image (sans la valeur max du type)
            response = as_float(image if image.ndim == 2 else image.mean(axis=-1).astype(image.dtype))[0]
        # masque de verite s'il existe (chemin non vide), sinon masque vide (image sans defaut)
        mask = read_mask(row.mask_path) if isinstance(row.mask_path, str) and row.mask_path else np.zeros(response.shape, np.uint8)
        if mask.shape != response.shape:  # la carte a pu etre redimensionnee par le pretraitement
            mask = resize(mask, list(response.shape), interpolation="nearest")  # "nearest" garde des valeurs 0/1 franches
        responses.append(response.astype(np.float32))  # garde la carte en flottants 32 bits
        masks.append(mask)  # garde le masque
    return responses, masks  # renvoie les deux listes


# evalue un reglage (seuil, aire minimale) ; renvoie un dict de scores (IoU, Dice, rappel, fausses alarmes)
def evaluate(responses: Sequence[np.ndarray], masks: Sequence[np.ndarray], value: float, min_area: float, invert: bool) -> dict[str, float]:
    predicted = [segment(r, value, min_area, invert) for r in responses]  # masque predit pour chaque carte
    return {  # une ligne de la grille de resultats
        "threshold": value,  # seuil teste
        "min_area": min_area,  # aire minimale testee
        **mask_scores(predicted, masks),  # ** recopie les cles iou et dice dans ce dictionnaire
        **image_scores(predicted, [bool(m.any()) for m in masks]),  # recall et false_alarm_rate
    }


# teste toutes les combinaisons de reglages sur la validation ; renvoie (meilleur reglage, tableau de la grille)
def choose_parameters(
    responses: Sequence[np.ndarray],  # cartes de reponse de la validation
    masks: Sequence[np.ndarray],  # masques de verite de la validation
    config: ClassicalConfig,  # configuration (seuils, aires, critere)
) -> tuple[dict[str, float], pd.DataFrame]:  # renvoie le reglage retenu et toute la grille
    """Grille seuils x aires minimales sur la validation ; critere dice, ou meilleur rappel a fausses alarmes bornees."""
    grid = pd.DataFrame(  # tableau : une ligne par combinaison (seuil, aire minimale)
        # itertools.product donne toutes les paires (t, a) : chaque seuil avec chaque aire
        [evaluate(responses, masks, t, a, config.invert) for t, a in itertools.product(config.thresholds, config.min_areas)]
    )
    if config.select == "dice":  # critere : meilleure qualite de segmentation
        best = grid.loc[grid["dice"].idxmax()]  # ligne ou le Dice est maximal
    else:  # critere : meilleur rappel sans depasser le taux de fausses alarmes tolere
        allowed = grid[grid["false_alarm_rate"] <= config.max_false_alarm + 1e-9]  # reglages qui respectent la limite
        if allowed.empty:  # aucun reglage ne respecte la limite
            # previent l'utilisateur
            logger.warning("aucun reglage ne tient %.2f de fausses alarmes sur val : le moins bavard est retenu", config.max_false_alarm)
            allowed = grid[grid["false_alarm_rate"] == grid["false_alarm_rate"].min()]  # garde ceux qui en font le moins
        best = allowed.sort_values(["recall", "dice"], ascending=False).iloc[0]  # meilleur rappel, puis meilleur Dice
    return {"threshold": float(best["threshold"]), "min_area": float(best["min_area"])}, grid  # reglage retenu + grille
