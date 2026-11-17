"""Configuration d'une experience de detection : quatre moteurs derriere le meme contrat.

    backend: ultralytics   YOLO11 (detect, obb, segment) ; licence AGPL-3.0 (commerciale pour un produit ferme)
    backend: yolox         YOLOX (Megvii), Apache-2.0 ; boucle d'entrainement ecrite dans la toolbox
    backend: fasterrcnn    Faster R-CNN de torchvision, BSD ; 100 % PyTorch, le squelette du cours s'applique
    backend: rfdetr        RF-DETR (Roboflow), Apache-2.0 ; transformer de detection a dos DINOv2, sans NMS

Le decoupage, le format pivot des annotations, l'evaluation COCO (mAP) et la carte du modele
sont communs : les moteurs se comparent sur les memes images avec le meme evaluateur.
"""

from __future__ import annotations  # les annotations de type restent du texte : syntaxe moderne "X | None" partout

from typing import Any, Literal  # Any : n'importe quel type ; Literal : liste fermee de valeurs autorisees

from pydantic import Field  # Field : valeur par defaut calculee a la creation (une nouvelle section a chaque fois)

# sections communes a toutes les taches : donnees, augmentations, export ONNX, execution, decoupage, base stricte
from vtb.core.config import AugmentStep, DataConfig, ExportConfig, PreprocessStep, RuntimeConfig, SplitConfig, StrictModel


class UltralyticsConfig(StrictModel):  # section "ultralytics" du YAML : reglages du moteur YOLO11 d'Ultralytics
    weights: str = "yolo11n.pt"  # telecharge depuis GitHub par Ultralytics s'il manque ; yolo11n-obb.pt, yolo11n-seg.pt
    task: Literal["detect", "obb", "segment"] = "detect"  # boites droites, boites orientees (obb) ou contours (segment)
    epochs: int = 60  # nombre maximal d'epoques (une epoque = un passage sur toutes les images du train)
    imgsz: int = 640  # taille (en pixels) a laquelle les images sont mises avant d'entrer dans le reseau
    batch: int = 16  # images traitees ensemble a chaque pas (un lot) : plus grand = plus rapide mais plus de memoire
    patience: int = 20  # arret anticipe : stop si le score de validation ne progresse plus pendant 20 epoques
    close_mosaic: int = 10  # coupe l'augmentation mosaique (4 images collees) les 10 dernieres epoques : fin sur des images reelles
    warmup_epochs: float = 3.0  # epoques de montee progressive du taux d'apprentissage au debut
    extra: dict[str, Any] = Field(default_factory=dict)  # tout autre argument de model.train (lr0, degrees, mosaic...)


class YoloxConfig(StrictModel):  # section "yolox" du YAML : reglages du moteur YOLOX
    variant: Literal["nano", "tiny", "s", "m", "l"] = "tiny"  # taille du reseau : nano = le plus petit et rapide, l = le plus gros
    weights: str | None = "weights/yolox_tiny.pth"  # poids COCO officiels (GitHub, release 0.1.1rc0) ; None = depart de zero
    input_size: int = 320  # cote (pixels) de l'image carree en entree du reseau
    epochs: int = 40  # nombre maximal d'epoques
    batch_size: int = 16  # images par lot
    basic_lr_per_img: float = 1.5625e-4  # 0.01 / 64 : valeur officielle ; taux reel = ce nombre x taille du lot
    warmup_epochs: float = 2.0  # epoques de montee progressive du taux au debut
    min_lr_ratio: float = 0.05  # le taux finit a 5 % de sa valeur maximale (decroissance en cosinus)
    weight_decay: float = 5e-4  # penalite sur les poids trop grands : limite le sur-apprentissage
    ema: bool = True  # EMA : moyenne glissante des poids, plus stable que les poids du dernier pas
    flip_prob: float = 0.5  # probabilite de retourner l'image gauche-droite (augmentation)
    hsv_prob: float = 1.0  # probabilite de varier teinte / saturation / luminosite (augmentation de couleur)
    max_labels: int = 50  # nombre maximal de boites par image gardees pour l'entrainement
    patience: int = 8  # arret anticipe : epoques sans progres du mAP avant d'arreter


class FasterRCNNConfig(StrictModel):  # section "fasterrcnn" du YAML : reglages du Faster R-CNN de torchvision
    # backbone = partie du reseau qui extrait les caracteristiques ; FPN = pyramide qui combine plusieurs echelles
    backbone: Literal["resnet18_fpn", "resnet50_fpn", "mobilenet_v3_fpn"] = "resnet18_fpn"
    pretrained: Literal["imagenet", "coco", "none"] = "imagenet"  # coco : detecteur complet pre-entraine (resnet50 / mobilenet)
    trainable_layers: int = 3  # blocs du backbone degeles (0 a 5) : les autres gardent leurs poids pre-entraines
    min_size: int = 320  # le plus petit cote de l'image redimensionnee par le modele
    max_size: int = 512  # le plus grand cote ne depasse jamais cette valeur
    epochs: int = 20  # nombre maximal d'epoques
    batch_size: int = 4  # images par lot (petit : Faster R-CNN consomme beaucoup de memoire)
    lr: float = 0.01  # taux d'apprentissage (taille des pas de correction des poids)
    weight_decay: float = 1e-4  # penalite sur les poids trop grands
    patience: int = 5  # arret anticipe : epoques sans progres avant d'arreter


class RFDETRConfig(StrictModel):  # section "rfdetr" du YAML : reglages de RF-DETR (detecteur a base de transformer)
    size: Literal["nano", "small", "medium", "base", "large"] = "nano"  # taille du modele
    weights: str | None = "weights/rf-detr-nano.pth"  # poids COCO (telecharges par rfdetr s'ils manquent) ; null = depart de zero
    resolution: int | None = None  # None : celle de la taille (nano 384, small 512, medium 576, base 560) ; multiple de 64
    epochs: int = 50  # nombre maximal d'epoques
    batch_size: int = 4  # images par lot
    grad_accum_steps: int = 4  # lot effectif = batch_size x grad_accum_steps (16 conseille) : cumule 4 lots avant un pas
    lr: float = 1e-4  # taux d'apprentissage du decodeur (partie qui produit les boites)
    lr_encoder: float = 1.5e-4  # taux d'apprentissage de l'encodeur (partie qui lit l'image, pre-entrainee DINOv2)
    patience: int = 10  # arret anticipe sur le mAP de validation (0 = desactive)
    extra: dict[str, Any] = Field(default_factory=dict)  # tout autre argument de RFDETR.train (warmup_epochs, multi_scale...)


class EvalConfig(StrictModel):  # section "evaluation" : seuils utilises pour mesurer et pour predire
    conf_eval: float = 0.001  # evaluation : toute la courbe precision-rappel (le mAP en a besoin)
    conf_predict: float = 0.25  # production : seuil de decision (une boite moins sure est ignoree)
    # NMS = suppression des boites en double ; IoU = recouvrement de deux boites (intersection / union, de 0 a 1)
    iou_nms: float = 0.6  # deux boites de meme classe qui se recouvrent a plus de 60 % = doublon : on garde la plus sure


class TilingConfig(StrictModel):  # section "tiling" : decoupage d'une grande image en tuiles pour l'inference
    """Inference par tuiles pour les grandes images (bande d'acier en camera lineaire)."""

    enabled: bool = False  # True : prediction tuile par tuile au lieu de l'image entiere
    tile: int = 640  # cote d'une tuile en pixels
    overlap: float = 0.2  # recouvrement entre tuiles voisines (20 %) : un objet coupe au bord est vu entier ailleurs
    iou_merge: float = 0.5  # NMS entre tuiles : une boite coupee par deux tuiles n'est gardee qu'une fois


class DetectionConfig(StrictModel):  # le fichier YAML complet d'une experience de detection, valide par pydantic
    task: Literal["detection"] = "detection"  # nom de la tache : evite de charger un YAML d'une autre tache
    data: DataConfig  # ou sont les images et les annotations (obligatoire)
    split: SplitConfig = Field(default_factory=SplitConfig)  # decoupage train / val / test
    preprocess: list[PreprocessStep] = Field(default_factory=list)  # applique avant le moteur (qui fait son propre letterbox)
    augment: list[AugmentStep] = Field(default_factory=list)  # fasterrcnn seulement (Ultralytics, YOLOX et RF-DETR ont les leurs)
    backend: Literal["ultralytics", "yolox", "fasterrcnn", "rfdetr"] = "ultralytics"  # quel moteur de detection utiliser
    ultralytics: UltralyticsConfig = Field(default_factory=UltralyticsConfig)  # reglages si backend = ultralytics
    yolox: YoloxConfig = Field(default_factory=YoloxConfig)  # reglages si backend = yolox
    fasterrcnn: FasterRCNNConfig = Field(default_factory=FasterRCNNConfig)  # reglages si backend = fasterrcnn
    rfdetr: RFDETRConfig = Field(default_factory=RFDETRConfig)  # reglages si backend = rfdetr
    evaluation: EvalConfig = Field(default_factory=EvalConfig)  # seuils de confiance et de NMS
    tiling: TilingConfig = Field(default_factory=TilingConfig)  # inference par tuiles (desactivee par defaut)
    export: ExportConfig = Field(default_factory=ExportConfig)  # export ONNX (format de modele lisible hors Python, ex. C#)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)  # dossier de sortie, graine, CPU / GPU
