"""Configuration d'une experience de segmentation semantique : un masque de classes par pixel.

Quand la choisir plutot que la detection : quand la FORME et la SURFACE du defaut comptent
(longueur d'une fissure, aire d'une oxydation, taux de couverture), ou quand les defauts n'ont pas
de contour net (traces, taches diffuses). Pour compter des objets separes, preferer la detection
ou la segmentation d'instances (YOLO11-seg, recette detection).
"""

from __future__ import annotations  # permet les annotations de type modernes (X | None, list[str])

from typing import Literal  # Literal : seules quelques valeurs exactes sont permises

from pydantic import Field  # pydantic verifie les types de la configuration ; Field donne une valeur par defaut calculee

# briques de configuration communes a toutes les taches (donnees, decoupage, pretraitement, augmentation, export, execution)
from vtb.core.config import AugmentStep, DataConfig, ExportConfig, PreprocessStep, RuntimeConfig, SplitConfig, StrictModel


class SegModelConfig(StrictModel):  # reglages du reseau de segmentation (StrictModel refuse les cles inconnues)
    # architecture du reseau : forme du decodeur qui reconstruit le masque (unet = la plus classique)
    architecture: Literal["unet", "unetplusplus", "fpn", "deeplabv3plus", "pspnet", "linknet", "manet", "segformer"] = "unet"
    encoder: str = "resnet18"  # tout encodeur de segmentation_models_pytorch (resnet34, efficientnet-b0, mit_b0...)
    encoder_weights: Literal["imagenet", "none"] = "imagenet"  # resnet* : poids torchvision (cache local) ; autres : smp
    in_channels: int | None = None  # None : 1 si data.color = gray, sinon 3
    decoder_dropout: float = 0.0  # dropout : met a zero des valeurs au hasard pendant l'entrainement (limite le par-coeur)


class SegTrainConfig(StrictModel):  # reglages de l'entrainement
    epochs: int = 40  # nombre maximal d'epoques (passages complets sur les donnees)
    batch_size: int = 8  # images traitees ensemble a chaque pas : plus grand = plus rapide mais plus de memoire
    lr: float = 3e-4  # taux d'apprentissage : taille des pas de correction des poids
    encoder_lr_factor: float = 0.1  # encodeur pre-entraine : pas plus petits que le decodeur neuf
    weight_decay: float = 1e-4  # penalite sur la taille des poids (limite le sur-apprentissage)
    loss: Literal["dice_ce", "ce", "dice", "focal", "lovasz"] = "dice_ce"  # dice_ce : robuste au desequilibre fond / defaut
    class_weights: list[float] | None = None  # ponderation de l'entropie croisee (fond en premier)
    patience: int = 8  # epoques sans amelioration avant l'arret anticipe
    amp: bool = False  # precision mixte (calculs en 16 bits sur GPU : plus rapide, moins de memoire)
    grad_clip: float | None = 1.0  # plafond de la norme du gradient (evite les pas de correction geants) ; None = pas de plafond


class SegEvalConfig(StrictModel):  # reglages de l'evaluation
    threshold: float = 0.5  # binaire : probabilite de defaut au-dessus de laquelle un pixel est defaut
    min_defect_area: int = 20  # pixels (a la taille du modele) : en dessous, l'image n'est pas signalee
    overlay_alpha: float = 0.45  # transparence du masque colore pose sur l'image dans les figures (0 = invisible, 1 = opaque)


class SegmentationConfig(StrictModel):  # la configuration complete d'une experience de segmentation (lue depuis un YAML)
    task: Literal["segmentation"] = "segmentation"  # nom de la tache : verifie qu'on a donne le bon fichier
    data: DataConfig  # ou sont les images et comment les lire (obligatoire)
    mask_mode: Literal["binary", "from_label", "from_values"] = "binary"  # facon de transformer un masque lu en classes de pixels
    # binary      : tout pixel non nul est "defect" (2 classes : fond, defect)
    # from_label  : masques binaires, la classe vient de l'etiquette de l'image (dossier MT_Crack...) ;
    #               les etiquettes de normal_labels n'ont pas de classe (images saines)
    # from_values : la valeur du pixel est l'indice de classe (0 = fond) ; donner class_names
    class_names: list[str] | None = None  # from_values : noms des classes 1..C (le fond est ajoute)
    normal_labels: list[str] = Field(default_factory=list)  # from_label : etiquettes d'images saines (MT_Free)
    split: SplitConfig = Field(default_factory=SplitConfig)  # reglages du decoupage train / val / test
    preprocess: list[PreprocessStep] = Field(default_factory=list)  # DOIT finir par un resize (lots de taille fixe)
    augment: list[AugmentStep] = Field(default_factory=list)  # le masque suit les transformations geometriques
    model: SegModelConfig = Field(default_factory=SegModelConfig)  # reglages du reseau
    train: SegTrainConfig = Field(default_factory=SegTrainConfig)  # reglages de l'entrainement
    evaluation: SegEvalConfig = Field(default_factory=SegEvalConfig)  # reglages de l'evaluation
    export: ExportConfig = Field(default_factory=ExportConfig)  # reglages de l'export ONNX (format d'echange de modeles)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)  # graine, processus de lecture, etc.
