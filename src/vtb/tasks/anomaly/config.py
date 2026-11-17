"""Configuration d'une experience de detection d'anomalies (apprendre le normal, signaler l'ecart).

Trois familles de methodes partagent exactement le meme protocole (decoupage, calibration du seuil
sur des normales, metriques, export) ; seule la section de la methode change :

    method: autoencoder      reconstruction ; entrainement sur les normales
    method: patchcore_lite   PatchCore simplifie ecrit dans la toolbox (pas d'entrainement)
    method: anomalib         n'importe quel modele d'anomalib par son nom (Patchcore, Padim, Stfpm, Fastflow...)
"""

from __future__ import annotations  # permet les annotations de type modernes (X | None, list[str])

from typing import Any, Literal  # Any : n'importe quel type ; Literal : seules quelques valeurs exactes sont permises

from pydantic import Field  # pydantic verifie les types de la configuration ; Field donne une valeur par defaut calculee

# briques de configuration communes a toutes les taches (donnees, pretraitement, augmentation, export, execution)
from vtb.core.config import AugmentStep, DataConfig, ExportConfig, PreprocessStep, RuntimeConfig, StrictModel


class AnomalySplitConfig(StrictModel):  # reglages du decoupage train / val / test (StrictModel refuse les cles inconnues)
    """Train et val ne contiennent QUE des normales ; le test contient le reste des normales et tous les defauts."""

    file: str = "outputs/anomaly/split.csv"  # fichier CSV ou le decoupage est enregistre (une ligne par image)
    val_fraction: float = 0.15  # part des normales gardees pour calibrer le seuil
    test_fraction: float = 0.2  # part des normales mises dans le test (si le rangement ne fournit pas de test)


class AutoencoderConfig(StrictModel):  # reglages de la methode auto-encodeur
    latent_channels: int = 32  # largeur du goulot : LE reglage (trop large : recopie les defauts ; trop etroit : floute tout)
    # canaux de chaque etage de l'encodeur (une liste neuve par config)
    widths: list[int] = Field(default_factory=lambda: [32, 64, 128, 256])
    epochs: int = 30  # nombre maximal d'epoques (passages complets sur les donnees)
    batch_size: int = 16  # images traitees ensemble a chaque pas : plus grand = plus rapide mais plus de memoire
    lr: float = 1e-3  # taux d'apprentissage : taille des pas de correction des poids
    weight_decay: float = 1e-5  # penalite sur la taille des poids (limite le sur-apprentissage)
    patience: int = 8  # epoques sans amelioration avant l'arret anticipe


class PatchcoreLiteConfig(StrictModel):  # reglages de PatchCore simplifie (memoire de vecteurs normaux)
    layers: list[str] = Field(default_factory=lambda: ["layer2", "layer3"])  # couches du ResNet dont on lit les caracteristiques
    memory_fraction: float = 0.1  # part des vecteurs normaux gardes en memoire
    coreset: Literal["random", "greedy"] = "greedy"  # greedy : k-center (le coreset du papier), plus fidele, un peu plus lent
    batch_size: int = 16  # images traitees ensemble pour remplir la memoire
    pretrained: bool = True  # False seulement pour les tests (poids aleatoires, rien a telecharger)


class AnomalibConfig(StrictModel):  # reglages quand on utilise un modele de la bibliotheque anomalib
    model: str = "Patchcore"  # nom de classe dans anomalib.models
    params: dict[str, Any] = Field(default_factory=dict)  # arguments du modele (backbone, layers, coreset_sampling_ratio...)
    max_epochs: int | None = None  # None : la valeur prevue par anomalib pour ce modele
    batch_size: int = 16  # images par lot pendant l'entrainement anomalib
    use_datamodule: bool = False  # EfficientAd : exige un datamodule (validation tiree des normales du train)


class ScoringConfig(StrictModel):  # reglages communs : lissage de la carte, score image, quantiles des seuils
    gaussian_sigma: float = 4.0  # lissage de la carte (pixels) ; 0 = aucun
    image_score: Literal["max", "top_mean"] = "max"  # max : une seule zone anormale suffit a rejeter la piece
    top_fraction: float = 0.01  # top_mean : moyenne du 1 % de pixels les plus anormaux (plus robuste au bruit)
    image_quantile: float = 0.99  # seuil image = ce quantile des scores des normales de validation
    pixel_quantile: float = 0.999  # seuil pixel = ce quantile des valeurs de carte des normales de validation


class AnomalyConfig(StrictModel):  # la configuration complete d'une experience d'anomalies (lue depuis un fichier YAML)
    task: Literal["anomaly"] = "anomaly"  # nom de la tache : sert a verifier qu'on a donne le bon fichier
    data: DataConfig  # ou sont les images et comment les lire (obligatoire : pas de valeur par defaut)
    normal_labels: list[str] = Field(default_factory=lambda: ["good"])  # etiquettes considerees normales (ex. MT_Free)
    split: AnomalySplitConfig = Field(default_factory=AnomalySplitConfig)  # reglages du decoupage
    preprocess: list[PreprocessStep] = Field(default_factory=list)  # doit finir par un resize : toutes les images a la meme taille
    augment: list[AugmentStep] = Field(default_factory=list)  # auto-encodeur seulement (les normales doivent rester normales)
    method: Literal["autoencoder", "patchcore_lite", "anomalib"] = "patchcore_lite"  # la methode choisie (une des trois familles)
    autoencoder: AutoencoderConfig = Field(default_factory=AutoencoderConfig)  # reglages lus si method = autoencoder
    patchcore_lite: PatchcoreLiteConfig = Field(default_factory=PatchcoreLiteConfig)  # reglages lus si method = patchcore_lite
    anomalib: AnomalibConfig = Field(default_factory=AnomalibConfig)  # reglages lus si method = anomalib
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)  # reglages du score et des seuils (toutes methodes)
    export: ExportConfig = Field(default_factory=ExportConfig)  # reglages de l'export ONNX (format d'echange de modeles)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)  # graine aleatoire, nombre de processus de lecture, etc.
