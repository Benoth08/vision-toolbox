"""Configuration d'une experience de classification : une section par brique du pipeline."""

from __future__ import annotations  # les annotations de type restent du texte : permet d'ecrire "float | None" partout

from typing import Literal  # Literal : le champ n'accepte qu'une liste fermee de valeurs (sinon erreur de validation)

from pydantic import Field  # Field : valeur par defaut calculee a la creation (une nouvelle section a chaque fois)

# sections communes a toutes les taches : donnees, augmentations, export ONNX, execution, decoupage, base stricte
from vtb.core.config import AugmentStep, DataConfig, ExportConfig, PreprocessStep, RuntimeConfig, SplitConfig, StrictModel
from vtb.models.transfer import ModelConfig  # section "model" : quel reseau, pre-entraine ou non, quelles couches geler


class TrainConfig(StrictModel):  # section "train" du YAML : tous les reglages de la boucle d'entrainement
    """Hyperparametres d'entrainement ; les valeurs par defaut sont les valeurs de depart du cours (module 17)."""

    batch_size: int = 32  # 32 images traitees ensemble a chaque pas (un lot) : plus grand = plus rapide mais plus de memoire
    epochs: int = 15  # nombre maximal d'epoques (une epoque = un passage complet sur toutes les images du train)
    lr: float = 1e-3  # taux d'apprentissage de la tete (ou de tout le modele s'il part de zero) : taille des pas de correction
    weight_decay: float = 1e-4  # penalite sur les poids trop grands : limite le sur-apprentissage
    optimizer: Literal["adamw", "sgd"] = "adamw"  # optimiseur = methode qui corrige les poids a partir du gradient
    scheduler: Literal["plateau", "cosine", "none"] = "plateau"  # comment le taux baisse au fil des epoques
    warmup_epochs: int = 0  # montee lineaire du taux au debut (utile en fine-tuning complet pour ne pas tout casser)
    patience: int = 4  # arret anticipe : on s'arrete si la perte de validation ne baisse plus pendant 4 epoques
    label_smoothing: float = 0.0  # adoucit les cibles (0.9 au lieu de 1) : rend le modele moins sur de lui
    class_weighting: bool = False  # poids inverses des effectifs du train dans la perte (aide les classes rares)
    grad_clip: float | None = None  # plafond de la norme du gradient (evite les pas geants) ; None = pas de plafond
    amp: bool = True  # precision mixte (calculs en float16, plus rapides), seulement si GPU CUDA
    # normalisation = (x - moyenne) / ecart-type par canal ; auto : stats ImageNet si pre-entraine, sinon stats du train
    normalization: Literal["auto", "imagenet", "dataset", "none"] = "auto"
    stats_max_images: int = 500  # nombre d'images du train utilisees pour calculer mean / std (suffisant et rapide)


class ClassificationConfig(StrictModel):  # le fichier YAML complet d'une experience de classification, valide par pydantic
    task: Literal["classification"] = "classification"  # nom de la tache : evite de charger un YAML d'une autre tache
    data: DataConfig  # ou sont les images et comment elles sont rangees (obligatoire : pas de valeur par defaut)
    split: SplitConfig = Field(default_factory=SplitConfig)  # decoupage train / val / test
    preprocess: list[PreprocessStep] = Field(default_factory=list)  # etapes de pretraitement OpenCV (vide = aucune)
    augment: list[AugmentStep] = Field(default_factory=list)  # augmentations aleatoires du train (vide = aucune)
    model: ModelConfig = Field(default_factory=ModelConfig)  # choix du reseau et strategie de transfert
    train: TrainConfig = Field(default_factory=TrainConfig)  # reglages d'entrainement definis juste au-dessus
    export: ExportConfig = Field(default_factory=ExportConfig)  # export ONNX (format de modele lisible hors Python, ex. C#)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)  # dossier de sortie, graine, CPU / GPU
