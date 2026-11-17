"""Augmentations : operations nommees (YAML), presets, melanges de lot, copy-paste, defauts synthetiques, outils.

Module autonome (torch, torchvision, numpy, OpenCV, pandas ; aucun autre module vtb) : le dossier
se copie tel quel dans un autre projet.

    from vtb.augment import build_augmentation, preview, check_alignment
    aug = build_augmentation([{"name": "preset", "preset": "steel_strip"}], task="detection")
    image, boxes = aug(image, boxes)
"""

# Ce fichier rassemble les fonctions utiles des sous-modules : on ecrit "from vtb.augment import X"
# sans savoir dans quel fichier X est range.
# mixing.py : banque de defauts et copy-paste, MixUp / CutMix au niveau du lot, mosaique de 4 images
from .mixing import CopyPaste, DefectBank, DefectInstance, MixCollate, batch_augmentation, copy_paste, mosaic
from .ops import SyntheticDefect, synthesize_defect  # ops.py : ajout d'un defaut synthetique dans l'image et le masque
from .presets import PRESETS, preset_steps  # presets.py : listes d'augmentations pretes et extraction de leurs etapes
from .tools import (  # tools.py : outils pour decrire, regarder, controler et chronometrer les augmentations
    AlbumentationsAdapter,  # adaptateur pour utiliser une chaine Albumentations
    alignment_pattern,  # image test avec boites et masque connus
    augmentation_catalog,  # tableau de toutes les operations disponibles
    benchmark_ops,  # mesure du temps de chaque operation
    check_alignment,  # verifie que boites et masques suivent l'image
    describe,  # tableau des etapes d'une chaine d'augmentation
    from_albumentations,  # raccourci vers AlbumentationsAdapter
    presets_table,  # tableau de tous les presets
    preview,  # affiche plusieurs tirages d'augmentation sur une image
    worker_init_fn,  # seme numpy et random dans chaque worker du DataLoader
)

# transforms.py : le registre des operations (OPS), les listes de ce qui existe, la construction de la chaine
# a partir du YAML (build_augmentation, build_step, expand_steps) et le nettoyage des boites trop petites
from .transforms import OPS, available_augmentations, available_presets, build_augmentation, build_step, expand_steps, sanitize_boxes

__all__ = [  # noms publics du module : ce que "from vtb.augment import *" importe
    "OPS",  # registre : nom d'operation -> fiche
    "PRESETS",  # registre : nom de preset -> preset
    "AlbumentationsAdapter",  # adaptateur Albumentations
    "CopyPaste",  # copy-paste applique avec une probabilite
    "DefectBank",  # banque de defauts a coller
    "DefectInstance",  # un defaut decoupe (pixels, opacite, classe)
    "MixCollate",  # MixUp / CutMix comme collate_fn du DataLoader
    "SyntheticDefect",  # operation qui ajoute un defaut synthetique (segmentation)
    "alignment_pattern",  # image test pour controler les cibles
    "augmentation_catalog",  # catalogue des operations
    "available_augmentations",  # parametres par defaut de chaque operation
    "available_presets",  # explication de chaque preset
    "batch_augmentation",  # raccourci vers MixCollate
    "benchmark_ops",  # chronometrage des operations
    "build_augmentation",  # liste du YAML -> chaine d'augmentation
    "build_step",  # une etape du YAML -> une transformation
    "check_alignment",  # controle que boites et masques suivent l'image
    "copy_paste",  # colle des defauts sur une image
    "describe",  # tableau des etapes d'une chaine
    "expand_steps",  # remplace les presets par leurs etapes
    "from_albumentations",  # raccourci vers AlbumentationsAdapter
    "mosaic",  # mosaique de 4 images (detection)
    "preset_steps",  # etapes d'un preset pour une tache
    "presets_table",  # tableau des presets
    "preview",  # affiche des tirages d'augmentation
    "sanitize_boxes",  # retire les boites devenues trop petites
    "synthesize_defect",  # ajoute un defaut synthetique a une image
    "worker_init_fn",  # semence de numpy / random par worker
]
