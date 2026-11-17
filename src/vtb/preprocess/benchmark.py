"""Juger un pretraitement a la METRIQUE : sonde lineaire sur des caracteristiques d'un ResNet18 gele.

Un pretraitement qui "rend l'image plus belle" n'aide pas forcement le modele (et l'inverse).
Le test le plus rapide : pour chaque pipeline, extraire les caracteristiques d'un reseau
pre-entraine gele, entrainer une regression logistique sur train, mesurer le F1 macro sur val.
Quelques minutes sur CPU, et le classement des pipelines est en general le meme qu'avec un vrai
entrainement. Le test n'est jamais utilise ici : on choisit, on ne rapporte pas.

Les caracteristiques sont extraites une seule fois par pipeline ; la regression logistique est
ensuite evaluee en validation croisee stratifiee (n_folds plis) sur train + val. On obtient un F1
moyen ET son ecart-type : sur quelques centaines d'images, un ecart de 1 a 2 points entre deux
pipelines est souvent du bruit, et l'ecart-type le montre.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes

import logging  # journal : messages horodates sur la console et dans un fichier
import time  # mesure du temps (duree du pretraitement)
from collections.abc import Mapping, Sequence  # types "dict en lecture" et "suite d'elements" pour les annotations
from typing import Any  # Any = n'importe quel type

import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)
import pandas as pd  # pandas : tableaux de donnees (DataFrame)
import torch  # PyTorch : tenseurs (tableaux de nombres sur CPU ou GPU) et reseaux de neurones
from pydantic import Field  # Field : valeur par defaut et contraintes d'un champ de configuration
from sklearn.linear_model import LogisticRegression  # regression logistique : classifieur lineaire simple et rapide
from sklearn.metrics import f1_score  # F1 : compromis entre precision et rappel (macro = moyenne sur les classes)
from sklearn.model_selection import StratifiedKFold  # decoupe en k plis en gardant les proportions de classes

from vtb.core.config import DataConfig, PreprocessStep, RuntimeConfig, SplitConfig, StrictModel  # sections de configuration
from vtb.io.images import read_image  # lecture d'une image en tableau numpy
from vtb.preprocess.ops.geometry import resize  # redimensionnement d'image
from vtb.preprocess.ops.illumination import apply_vignetting  # assombrit les bords (simule un eclairage inegal)
from vtb.preprocess.pipeline import Pipeline  # suite d'operations de pretraitement
from vtb.preprocess.tensor import IMAGENET_MEAN, IMAGENET_STD, normalize, to_float_chw  # passage image -> tenseur normalise

logger = logging.getLogger(__name__)  # journal propre a ce module


class GalleryConfig(StrictModel):  # reglages des planches avant / apres
    """Planches avant / apres."""

    images: list[str] = Field(default_factory=list)  # chemins des images a montrer ; vide = les premieres images du catalogue
    n_images: int = 3  # nombre d'images par planche
    families: list[str] | None = None  # familles d'operations a dessiner ; None = toutes
    skip: list[str] = Field(default_factory=list)  # operations a ne pas dessiner
    params: dict[str, dict[str, Any]] = Field(default_factory=dict)  # parametres par operation
    assets_dir: str | None = "docs/assets/preprocessing"  # copie pour les articles du site (None = pas de copie)


class CompareConfig(StrictModel):  # reglages de la comparaison des pipelines
    """Comparaison des pipelines par sonde lineaire."""

    image_size: int = 128  # taille des images donnees au reseau (petite = rapide)
    max_per_class: int = 100  # au plus 100 images par classe (test rapide et equilibre)
    batch_size: int = 32  # lot (batch) : 32 images traitees ensemble par le reseau ; plus grand = plus rapide mais plus de memoire
    logreg_c: float = 1.0  # force de la regularisation de la regression logistique (plus petit = plus regularise)
    n_folds: int = 5  # validation croisee de la sonde lineaire (moyenne et ecart-type du F1)
    simulate_vignetting: float = 0.0  # > 0 : vignettage aleatoire ajoute avant le pipeline (eclairage inegal simule)


class PreprocessingConfig(StrictModel):  # configuration complete de la recette "preprocessing"
    """Configuration de la recette preprocessing (atelier de choix du pretraitement)."""

    data: DataConfig  # ou sont les images (obligatoire)
    split: SplitConfig = Field(  # decoupage ; par defaut 20 % validation, 20 % test, enregistre dans outputs/preprocessing
        # default_factory : la valeur par defaut est creee a la demande, une nouvelle a chaque configuration
        default_factory=lambda: SplitConfig(file="outputs/preprocessing/split.csv", val_fraction=0.2, test_fraction=0.2)
    )
    runtime: RuntimeConfig = Field(default_factory=lambda: RuntimeConfig(output_dir="outputs/preprocessing"))  # sortie, graine...
    pipelines: dict[str, list[PreprocessStep]] = Field(default_factory=lambda: {"none": []})  # pipelines a comparer ; "none" = aucun
    gallery: GalleryConfig = Field(default_factory=GalleryConfig)  # reglages des planches
    compare: CompareConfig = Field(default_factory=CompareConfig)  # reglages de la comparaison

    def build_pipelines(self) -> dict[str, Pipeline]:  # construit les objets Pipeline ; renvoie nom -> Pipeline
        """Valide toutes les operations des le chargement (une faute de frappe echoue avant tout calcul)."""
        return {name: Pipeline(steps) for name, steps in self.pipelines.items()}  # un Pipeline par entree du YAML


def frozen_resnet18(device: torch.device) -> torch.nn.Module:  # renvoie un ResNet18 pre-entraine, sans sa derniere couche, fige
    """ResNet18 ImageNet sans sa tete : un vecteur de 512 valeurs par image."""
    from torchvision import models  # torchvision : modeles de vision pre-entraines

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # ResNet18 avec ses poids appris sur ImageNet (telecharges)
    model.fc = torch.nn.Identity()  # remplace la tete (classifieur final) par "ne rien faire" : sortie = 512 caracteristiques
    return model.eval().to(device)  # eval() = mode prediction (pas de dropout) ; .to(device) = sur CPU ou GPU


def _simulated(image: np.ndarray, strength: float, index: int) -> np.ndarray:  # ajoute un vignettage reproductible ; renvoie l'image
    """Vignettage reproductible (centre tire par image) : simule un eclairage inegal d'une image a l'autre."""
    if strength <= 0:  # pas de simulation demandee
        return image  # image inchangee
    rng = np.random.default_rng(index)  # generateur aleatoire dont la graine est le numero de l'image (meme tirage a chaque fois)
    return apply_vignetting(  # assombrit les bords autour d'un centre tire au hasard
        image,  # image a assombrir
        strength=float(rng.uniform(0.5, 1.0) * strength),  # force tiree entre 50 % et 100 % de la force demandee
        center=(float(rng.uniform(0.2, 0.8)), float(rng.uniform(0.2, 0.8))),  # centre (x, y) tire entre 20 % et 80 % de l'image
    )


@torch.no_grad()  # pas de calcul de gradient (derivees pour l'apprentissage) : plus rapide et moins de memoire
def extract_features(  # passe toutes les images dans le reseau ; renvoie (caracteristiques, ms de pretraitement par image)
    paths: Sequence[str],  # chemins des images
    pipeline: Pipeline,  # pretraitement a tester
    model: torch.nn.Module,  # reseau fige qui produit les caracteristiques
    config: CompareConfig,  # reglages (taille d'image, taille de lot...)
    color: str,  # lecture en "rgb" ou "gray"
    device: torch.device,  # CPU ou GPU
) -> tuple[np.ndarray, float]:  # renvoie le tableau (N, 512) et la duree moyenne de pretraitement
    """Caracteristiques (N, 512) normalisees L2 et duree moyenne du pretraitement par image (ms)."""
    features, preprocess_seconds = [], 0.0  # caracteristiques de chaque lot ; temps cumule du pretraitement
    for start in range(0, len(paths), config.batch_size):  # pour chaque lot : start = 0, 32, 64...
        batch = []  # tenseurs des images de ce lot
        for offset, path in enumerate(paths[start : start + config.batch_size]):  # pour chaque image du lot
            image = _simulated(read_image(path, color=color), config.simulate_vignetting, start + offset)  # lit + vignettage eventuel
            t0 = time.perf_counter()  # depart du chronometre
            image = resize(pipeline(image), config.image_size)  # applique le pretraitement puis redimensionne
            preprocess_seconds += time.perf_counter() - t0  # ajoute la duree du pretraitement
            x = to_float_chw(image)  # tenseur (C, H, W) entre 0 et 1
            # image grise (1 canal) recopiee sur 3 canaux, puis normalisee avec les statistiques d'ImageNet
            batch.append(normalize(x.repeat(3, 1, 1) if x.shape[0] == 1 else x, IMAGENET_MEAN, IMAGENET_STD))
        out = model(torch.stack(batch).to(device))  # empile le lot (N, 3, H, W), l'envoie sur le device, passe dans le reseau
        features.append(torch.nn.functional.normalize(out, dim=1).cpu().numpy())  # normalise chaque vecteur a longueur 1 (L2), en numpy
    return np.concatenate(features), 1000.0 * preprocess_seconds / max(len(paths), 1)  # tous les lots colles ; ms moyennes par image


def compare_pipelines(  # compare les pipelines par validation croisee ; renvoie un tableau de scores trie
    dev: pd.DataFrame,  # images de developpement (train + val), avec leurs classes
    pipelines: Mapping[str, Pipeline],  # pipelines a comparer (nom -> Pipeline)
    config: CompareConfig,  # reglages de la comparaison
    color: str,  # lecture en "rgb" ou "gray"
    device: torch.device,  # CPU ou GPU
    seed: int = 42,  # graine : memes plis a chaque execution
) -> pd.DataFrame:  # tableau : une ligne par pipeline
    """F1 macro moyen et ecart-type par pipeline, en validation croisee sur dev (= train + val, jamais le test)."""
    model = frozen_resnet18(device)  # reseau fige, charge une seule fois
    # plis de validation croisee calcules une fois : tous les pipelines sont juges sur les memes decoupages
    folds = list(StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=seed).split(dev, dev["label"]))
    labels = dev["label"].to_numpy()  # classes sous forme de tableau numpy
    rows = []  # une ligne de resultats par pipeline
    for name, pipeline in pipelines.items():  # pour chaque pipeline a tester
        t0 = time.perf_counter()  # depart du chronometre
        features, ms = extract_features(dev["path"].tolist(), pipeline, model, config, color, device)  # caracteristiques de chaque image
        scores = []  # F1 de chaque pli
        for train_idx, val_idx in folds:  # pour chaque pli : lignes d'entrainement et lignes de validation
            # entraine la regression logistique (sonde lineaire) sur la partie entrainement du pli
            classifier = LogisticRegression(C=config.logreg_c, max_iter=3000, random_state=seed).fit(features[train_idx], labels[train_idx])
            scores.append(f1_score(labels[val_idx], classifier.predict(features[val_idx]), average="macro"))  # F1 macro sur la validation
        rows.append(  # ajoute la ligne de resultats du pipeline
            {  # contenu de la ligne
                "pipeline": name,  # nom du pipeline
                "f1_macro_mean": float(np.mean(scores)),  # F1 moyen sur les plis
                "f1_macro_std": float(np.std(scores)),  # ecart-type du F1 (dispersion d'un pli a l'autre)
                "preprocess_ms_per_image": round(ms, 2),  # cout du pretraitement en millisecondes par image
                "seconds": round(time.perf_counter() - t0, 1),  # duree totale pour ce pipeline
                "steps": " -> ".join(step["name"] for step in pipeline.describe()) or "(aucun)",  # etapes du pipeline en texte
            }
        )
        logger.info(  # resume dans le journal
            "%-16s F1 macro %.3f +/- %.3f | %.1f ms de pretraitement / image",  # gabarit du message
            name,  # nom du pipeline
            rows[-1]["f1_macro_mean"],  # F1 moyen
            rows[-1]["f1_macro_std"],  # ecart-type
            ms,  # duree du pretraitement par image
        )
    return pd.DataFrame(rows).sort_values("f1_macro_mean", ascending=False).reset_index(drop=True)  # du meilleur au moins bon


# sous-echantillon equilibre : au plus max_per_class images par classe ; renvoie le tableau reduit
def subsample_per_class(catalog: pd.DataFrame, max_per_class: int, seed: int) -> pd.DataFrame:
    """Au plus max_per_class images par classe, tirees au hasard (graine fixe) : un test rapide et equilibre."""
    # pour chaque classe : tire au hasard au plus max_per_class lignes (toutes si la classe est plus petite)
    parts = [group.sample(n=min(len(group), max_per_class), random_state=seed) for _, group in catalog.groupby("label")]
    return pd.concat(parts).sort_index().reset_index(drop=True)  # recolle les classes, remet l'ordre d'origine, renumerote
