"""Augmentations aleatoires (entrainement seulement), construites depuis la liste `augment:` du YAML.

    augment:
      - {name: preset, preset: steel_strip}          # un preset (voir presets.py)...
      - {name: hflip}                                # ... et/ou des operations nommees
      - {name: rot90, p: 0.5}
      - {name: brightness_contrast, brightness: 0.15, contrast: 0.15}

Elles s'appliquent APRES le pretraitement et la mise en tenseur, sur un tenseur float (C, H, W)
dans [0, 1] (ordre fixe de la toolbox : lecture -> Pipeline -> to_float_chw -> augmentation ->
normalisation). Torchvision v2 transforme en meme temps les boites (tv_tensors.BoundingBoxes)
et les masques (tv_tensors.Mask) : un retournement retourne aussi les boites et le masque.

    aug = build_augmentation(config.augment, task="classification")
    x = aug(x)                      # classification
    x, boxes = aug(x, boxes)        # detection (boxes : tv_tensors.BoundingBoxes)
    x, mask = aug(x, mask)          # segmentation (mask : tv_tensors.Mask)

Une augmentation est une hypothese : "l'etiquette ne change pas si...". Un retournement est sans
risque pour une rayure, pas pour une fleche ou un texte ; une rotation de 90 degres change la
direction de laminage d'une bande. A verifier pour chaque jeu de donnees, en REGARDANT le
resultat (vtb.augment.preview) et en controlant les cibles (vtb.augment.check_alignment).

Le catalogue complet : augmentation_catalog() (nom, famille, taches, parametres, resume).
Reproductibilite : tout est tire avec le generateur torch ; torch.manual_seed(n) fixe la suite,
et DataLoader donne une graine differente a chaque worker (voir tools.worker_init_fn pour numpy).
"""

from __future__ import annotations  # permet d'ecrire les types (ex. "X | None") sans qu'ils soient evalues au chargement

import inspect  # inspect : lit la signature d'une fonction (noms et valeurs par defaut de ses parametres)
import warnings  # warnings : gere les avertissements (ici pour en faire taire un, voulu)
from collections.abc import Callable, Mapping, Sequence  # types : fonction appelable, dictionnaire, liste ou tuple
from dataclasses import dataclass  # dataclass : cree une petite classe "fiche" a partir de ses seuls champs
from typing import Any, Literal  # Any : n'importe quel type ; Literal : liste fermee de valeurs permises

import torch  # PyTorch : tenseurs (tableaux de nombres multidimensionnels)

# tv_tensors : tenseurs etiquetes de torchvision (Image, BoundingBoxes = boites, Mask = masque) que les transformations reconnaissent
from torchvision import tv_tensors
from torchvision.transforms import v2  # v2 : transformations torchvision qui deplacent aussi boites et masques avec l'image
from torchvision.transforms.v2 import functional as F  # F : versions "fonction" des transformations (appliquees une fois, sans hasard)

from . import ops  # ops.py : les operations ecrites pour la toolbox (bruit, eclairage, recadrage sur...)
from ._base import pair  # pair : transforme 3 en (3, 3) et garde (1, 5) tel quel (valeur fixe ou intervalle de tirage)
from .ops import Clamp01, RandomGamma, RandomRot90  # noqa: F401  (reexportes : anciens chemins d'import)
from .presets import PRESETS, preset_steps  # listes d'augmentations pretes et fonction qui en extrait les etapes

Task = Literal["classification", "detection", "segmentation"]  # type : les 3 taches possibles
TASKS: tuple[str, ...] = ("classification", "detection", "segmentation")  # les memes, sous forme de tuple a parcourir
ALL = TASKS  # raccourci : operation permise pour toutes les taches
CLASSIFICATION = ("classification",)  # operation reservee a la classification (la virgule fait un tuple d'un element)


# --- constructeurs : un nom du YAML -> une transformation ------------------------------------------
# geometrie (les boites et les masques suivent l'image)


def _hflip(p: float = 0.5) -> v2.Transform:  # p : probabilite d'appliquer l'operation a une image (0.5 = une fois sur deux)
    return v2.RandomHorizontalFlip(p)  # retournement gauche-droite (miroir)


def _vflip(p: float = 0.5) -> v2.Transform:  # renvoie un retournement haut-bas aleatoire
    return v2.RandomVerticalFlip(p)  # retournement haut-bas


def _rot90(p: float = 0.5) -> v2.Transform:  # renvoie une rotation aleatoire de 90, 180 ou 270 degres
    return RandomRot90(p)  # operation maison : pas d'interpolation, pas de coins vides


def _rotate(degrees: float = 10.0, p: float = 0.5) -> v2.Transform:  # renvoie une rotation d'angle quelconque
    # RandomApply : applique la liste avec la probabilite p ; RandomRotation tire un angle dans [-degrees, degrees]
    return v2.RandomApply([v2.RandomRotation(degrees)], p=p)


def _affine(  # renvoie une transformation affine aleatoire (rotation + decalage + zoom + cisaillement)
    degrees: float = 0.0,  # angle de rotation maximal, en degres
    translate: float = 0.05,  # decalage maximal, en fraction de la taille de l'image (0.05 = 5 %)
    scale: Sequence[float] = (0.9, 1.1),  # intervalle du facteur de zoom
    shear: float = 0.0,  # cisaillement maximal en degres (l'image penche comme un parallelogramme)
    p: float = 0.5,  # probabilite d'appliquer l'operation
) -> v2.Transform:  # renvoie une transformation torchvision
    # RandomAffine tire les quatre parametres a chaque appel ; RandomApply ne l'applique qu'avec la probabilite p
    return v2.RandomApply([v2.RandomAffine(degrees=degrees, translate=(translate, translate), scale=pair(scale), shear=shear)], p=p)


def _random_resized_crop(  # renvoie un recadrage aleatoire suivi d'un redimensionnement a taille fixe
    size: int | Sequence[int] = 224,  # taille de sortie en pixels
    scale: Sequence[float] = (0.8, 1.0),  # fraction de la surface de l'image gardee
    ratio: Sequence[float] = (0.9, 1.1),  # rapport largeur / hauteur de la fenetre
) -> v2.Transform:  # renvoie une transformation torchvision
    # antialias : lisse avant de reduire pour eviter le crenelage (escaliers)
    return v2.RandomResizedCrop(size=size, scale=pair(scale), ratio=pair(ratio), antialias=True)


def _perspective(distortion: float = 0.2, p: float = 0.5) -> v2.Transform:  # renvoie une deformation de perspective aleatoire
    return v2.RandomPerspective(distortion_scale=distortion, p=p)  # distortion : 0 = aucune, 1 = tres forte


def _elastic(alpha: float = 30.0, sigma: float = 5.0, p: float = 0.3) -> v2.Transform:  # renvoie une deformation elastique
    # alpha : amplitude du deplacement des pixels ; sigma : douceur du champ de deplacement (grand = ondulations larges)
    return v2.RandomApply([v2.ElasticTransform(alpha=alpha, sigma=sigma)], p=p)


def _zoom(scale: Sequence[float] = (0.85, 1.15), p: float = 0.5) -> v2.Transform:  # renvoie un zoom aleatoire
    return v2.RandomApply([v2.RandomAffine(degrees=0.0, scale=pair(scale))], p=p)  # affine sans rotation : seulement l'echelle


def _translate(fraction: float = 0.1, p: float = 0.5) -> v2.Transform:  # renvoie un decalage aleatoire
    # affine sans rotation : decalage jusqu'a fraction x largeur (horizontal) et fraction x hauteur (vertical)
    return v2.RandomApply([v2.RandomAffine(degrees=0.0, translate=(fraction, fraction))], p=p)


def _shear(degrees: float = 8.0, p: float = 0.5) -> v2.Transform:  # renvoie un cisaillement aleatoire
    # cisaillement tire dans [-degrees, degrees] en x puis en y (les 4 nombres : min x, max x, min y, max y)
    return v2.RandomApply([v2.RandomAffine(degrees=0.0, shear=(-degrees, degrees, -degrees, degrees))], p=p)


def _safe_crop(  # renvoie un recadrage aleatoire qui ne coupe aucun objet
    area: Sequence[float] = (0.5, 1.0),  # fraction de la surface gardee
    aspect: Sequence[float] = (0.8, 1.25),  # variation du rapport largeur / hauteur
    keep_objects: bool = True,  # True : la fenetre contient toujours toutes les boites / le masque
    p: float = 0.5,  # probabilite d'appliquer l'operation
) -> v2.Transform:  # renvoie une transformation torchvision
    return ops.SafeCrop(area, aspect, keep_objects, p)  # operation maison (voir ops.py)


# renvoie un changement aleatoire de la taille de l'image (detection seulement)
def _scale_jitter(target_size: Sequence[int] = (640, 640), scale_range: Sequence[float] = (0.5, 1.5)) -> v2.Transform:
    # taille de reference (hauteur, largeur) multipliee par un facteur tire dans scale_range ; la taille de sortie varie
    return v2.ScaleJitter(target_size=tuple(int(s) for s in target_size), scale_range=pair(scale_range), antialias=True)


# photometrie (l'image seule change)


# renvoie une variation aleatoire de luminosite et de contraste
def _brightness_contrast(brightness: float = 0.2, contrast: float = 0.2, p: float = 0.8) -> v2.Transform:
    # ColorJitter : facteurs tires dans [1 - 0.2, 1 + 0.2] pour la luminosite et pour le contraste
    return v2.RandomApply([v2.ColorJitter(brightness=brightness, contrast=contrast)], p=p)


def _gamma(gamma: Sequence[float] = (0.8, 1.25), p: float = 0.5) -> v2.Transform:  # renvoie une correction gamma aleatoire
    return RandomGamma(gamma, p)  # gamma < 1 eclaircit les tons sombres, > 1 les assombrit


# renvoie une variation aleatoire de teinte et de saturation (images couleur)
def _hue_saturation(hue: float = 0.05, saturation: float = 0.3, p: float = 0.5) -> v2.Transform:
    return ops.RandomHueSaturation(hue, saturation, p)  # operation maison : une image grise passe telle quelle


def _sharpness(factor: Sequence[float] = (0.5, 2.0), p: float = 0.3) -> v2.Transform:  # renvoie un changement de nettete
    return ops.RandomSharpness(factor, p)  # facteur < 1 adoucit, > 1 accentue les contours


def _autocontrast(p: float = 0.2) -> v2.Transform:  # renvoie un auto-contraste aleatoire
    return v2.RandomAutocontrast(p=p)  # etire les valeurs pour que le plus sombre devienne noir et le plus clair blanc


def _equalize(p: float = 0.1) -> v2.Transform:  # renvoie une egalisation d'histogramme aleatoire
    # l'histogramme compte les pixels de chaque niveau de gris ; l'egaliser repartit les niveaux uniformement
    return v2.RandomEqualize(p=p)


# renvoie une egalisation d'histogramme locale (CLAHE) aleatoire
def _clahe(clip_limit: Sequence[float] = (1.0, 4.0), tile_grid: int = 8, p: float = 0.3) -> v2.Transform:
    # CLAHE : egalisation par tuiles (grille 8 x 8) ; clip_limit borne le renforcement du contraste
    return ops.RandomCLAHE(clip_limit, tile_grid, p)


def _posterize(bits: int = 4, p: float = 0.2) -> v2.Transform:  # renvoie une reduction du nombre de niveaux
    return v2.RandomPosterize(bits=bits, p=p)  # garde 4 bits par canal : 16 niveaux au lieu de 256


def _invert(p: float = 0.1) -> v2.Transform:  # renvoie une inversion aleatoire (negatif photo)
    return v2.RandomInvert(p=p)  # chaque pixel x devient 1 - x


# capteur et eclairage (maison, voir ops.py)


def _illumination_gradient(  # renvoie un eclairage non uniforme aleatoire
    strength: Sequence[float] = (0.1, 0.4),  # amplitude du gain (0.3 : le gain varie de 0.7 a 1.3)
    mode: str = "both",  # "linear" (rampe), "radial" (point chaud) ou "both" (tire au hasard)
    angle: Sequence[float] = (0.0, 360.0),  # direction de la rampe en degres
    p: float = 0.5,  # probabilite d'appliquer l'operation
) -> v2.Transform:  # renvoie une transformation torchvision
    return ops.IlluminationGradient(strength, mode, angle, p)  # operation maison (voir ops.py)


# renvoie un vignettage aleatoire (bords plus sombres)
def _vignetting(strength: Sequence[float] = (0.2, 0.5), center_jitter: float = 0.1, p: float = 0.3) -> v2.Transform:
    return ops.Vignetting(strength, center_jitter, p)  # center_jitter : le centre peut bouger de 10 % de l'image


# renvoie une variation aleatoire d'exposition : pixel x gain + decalage
def _exposure(gain: Sequence[float] = (0.75, 1.3), offset: Sequence[float] = (-0.05, 0.05), p: float = 0.5) -> v2.Transform:
    return ops.Exposure(gain, offset, p)  # simule temps de pose, gain camera ou puissance d'eclairage differents


# renvoie des stries de camera lineaire aleatoires
def _line_stripes(strength: Sequence[float] = (0.01, 0.05), axis: str = "columns", p: float = 0.3) -> v2.Transform:
    return ops.LineStripes(strength, axis, p)  # chaque colonne (ou ligne) recoit son propre gain


def _gaussian_noise(sigma: float | Sequence[float] = 0.02, p: float = 0.5) -> v2.Transform:  # renvoie un bruit gaussien
    return ops.RandomGaussianNoise(sigma, p)  # sigma : ecart-type du bruit ajoute (0.02 = 2 % de la plage 0-1)


def _poisson_noise(peak: Sequence[float] = (30.0, 300.0), p: float = 0.3) -> v2.Transform:  # renvoie un bruit de photons
    return ops.PoissonNoise(peak, p)  # peak : photons recus pour un pixel blanc (petit = image tres bruitee)


def _speckle_noise(sigma: Sequence[float] = (0.02, 0.08), p: float = 0.3) -> v2.Transform:  # renvoie un bruit multiplicatif
    return ops.SpeckleNoise(sigma, p)  # bruit proportionnel a la valeur du pixel (granularite)


def _salt_pepper(amount: Sequence[float] = (0.0005, 0.005), p: float = 0.2) -> v2.Transform:  # renvoie des pixels morts / chauds
    return ops.SaltPepper(amount, p)  # amount : fraction des pixels forces a 0 (noir) ou 1 (blanc)


def _blur(kernel_size: int = 5, sigma: Sequence[float] = (0.1, 1.5), p: float = 0.3) -> v2.Transform:  # renvoie un flou gaussien
    # noyau = petite fenetre de 5 x 5 pixels moyennee ; sigma tire dans l'intervalle : plus grand = plus flou
    return v2.RandomApply([v2.GaussianBlur(kernel_size=kernel_size, sigma=pair(sigma))], p=p)


# renvoie un flou de mouvement aleatoire
def _motion_blur(length: Sequence[int] = (3, 9), angle: Sequence[float] = (0.0, 180.0), p: float = 0.3) -> v2.Transform:
    return ops.MotionBlur(length, angle, p)  # length : longueur de la trainee en pixels ; angle : 0 horizontal, 90 vertical


def _defocus_blur(radius: Sequence[float] = (1.0, 3.0), p: float = 0.2) -> v2.Transform:  # renvoie un flou de mise au point
    return ops.DefocusBlur(radius, p)  # radius : rayon du disque de flou en pixels


def _jpeg(quality: Sequence[int] = (30, 90), p: float = 0.3) -> v2.Transform:  # renvoie une compression JPEG aleatoire
    return ops.JpegCompression(quality, p)  # quality : 100 = presque sans perte, 30 = artefacts visibles (blocs)


def _low_resolution(scale: Sequence[float] = (0.3, 0.7), p: float = 0.2) -> v2.Transform:  # renvoie une perte de resolution
    return ops.LowResolution(scale, p)  # reduit a 30-70 % puis remet a la taille d'origine : les details fins disparaissent


def _specular(  # renvoie un reflet speculaire aleatoire (tache claire sur metal poli)
    intensity: Sequence[float] = (0.2, 0.6),  # luminosite ajoutee au centre du reflet
    size: Sequence[float] = (0.05, 0.25),  # taille du reflet en fraction du petit cote
    elongation: Sequence[float] = (1.0, 4.0),  # 1 = reflet rond, 4 = longue trainee
    p: float = 0.2,  # probabilite d'appliquer l'operation
) -> v2.Transform:  # renvoie une transformation torchvision
    return ops.SpecularHighlight(intensity, size, elongation, p)  # operation maison (voir ops.py)


def _dust(  # renvoie des poussieres / taches aleatoires
    count: Sequence[int] = (1, 8),  # nombre de taches
    radius: Sequence[float] = (1.0, 4.0),  # rayon de chaque tache en pixels
    contrast: Sequence[float] = (0.1, 0.4),  # assombrissement (ou eclaircissement) au centre de la tache
    p: float = 0.2,  # probabilite d'appliquer l'operation
) -> v2.Transform:  # renvoie une transformation torchvision
    return ops.DustSpots(count, radius, contrast, p)  # operation maison (voir ops.py)


# occlusion


# renvoie l'effacement aleatoire d'un rectangle (Random Erasing)
def _random_erasing(p: float = 0.25, scale: Sequence[float] = (0.02, 0.1), value: float = 0.0) -> v2.Transform:
    # scale : surface du rectangle en fraction de l'image ; value : valeur de remplissage (0 = noir)
    return v2.RandomErasing(p=p, scale=pair(scale), value=value)


def _coarse_dropout(  # renvoie plusieurs rectangles masques au hasard
    holes: Sequence[int] = (1, 6),  # nombre de trous
    size: Sequence[float] = (0.05, 0.15),  # cote d'un trou en fraction de l'image
    fill: Any = "mean",  # remplissage : un nombre, "mean" (moyenne de l'image) ou "noise" (bruit)
    p: float = 0.3,  # probabilite d'appliquer l'operation
) -> v2.Transform:  # renvoie une transformation torchvision
    return ops.CoarseDropout(holes, size, fill, p)  # operation maison (voir ops.py)


# renvoie une grille reguliere de carres masques
def _grid_mask(spacing: Sequence[float] = (0.15, 0.35), ratio: float = 0.5, fill: Any = 0.0, p: float = 0.2) -> v2.Transform:
    return ops.GridMask(spacing, ratio, fill, p)  # spacing : pas de la grille ; ratio : part de la surface masquee


# politiques automatiques (classification : elles melangent des operations sans regarder les cibles)


def _randaugment(num_ops: int = 2, magnitude: int = 9) -> v2.Transform:  # renvoie la politique RandAugment
    return v2.RandAugment(num_ops=num_ops, magnitude=magnitude)  # tire 2 operations parmi une liste, de force 9 sur 30


def _trivialaugment() -> v2.Transform:  # renvoie la politique TrivialAugment (aucun reglage)
    return v2.TrivialAugmentWide()  # une seule operation tiree au hasard, avec une force tiree au hasard


def _augmix(severity: int = 3, mixture_width: int = 3) -> v2.Transform:  # renvoie la politique AugMix
    # melange 3 chaines d'augmentations (mixture_width) de force severity : rend le modele plus robuste
    return v2.AugMix(severity=severity, mixture_width=mixture_width)


def _autoaugment(policy: str = "imagenet") -> v2.Transform:  # renvoie la politique AutoAugment
    return v2.AutoAugment(policy=v2.AutoAugmentPolicy(policy))  # politique apprise sur ImageNet, CIFAR10 ou SVHN


# defauts synthetiques (le masque change avec l'image)


# renvoie l'operation qui ajoute un defaut synthetique dans l'image et dans le masque
def _synthetic_defect(kinds: Sequence[str] = ("perlin", "cutpaste", "scratch"), class_index: int = 1, p: float = 0.5) -> v2.Transform:
    return ops.SyntheticDefect(kinds, class_index, p)  # kinds : types de defauts ; class_index : classe ecrite dans le masque


@dataclass(frozen=True)  # frozen : les champs ne peuvent plus etre modifies apres creation
class OpInfo:  # fiche d'une operation du registre
    """Une entree du registre : constructeur, famille, taches permises, resume d'une ligne."""

    builder: Callable[..., v2.Transform]  # fonction qui fabrique la transformation a partir des parametres du YAML
    family: str  # famille : geometrie, photometrie, capteur, occlusion, politique, synthese
    tasks: tuple[str, ...]  # taches pour lesquelles l'operation est permise
    summary: str  # resume d'une ligne, affiche dans le catalogue et les messages d'erreur


# registre : nom utilise dans le YAML -> fiche de l'operation (constructeur, famille, taches, resume)
OPS: dict[str, OpInfo] = {
    "hflip": OpInfo(_hflip, "geometrie", ALL, "retournement gauche-droite"),  # sans risque si aucun sens gauche / droite
    "vflip": OpInfo(_vflip, "geometrie", ALL, "retournement haut-bas"),  # sans risque pour une vue de dessus
    "rot90": OpInfo(_rot90, "geometrie", ALL, "rotation de 90, 180 ou 270 degres (pas pour une bande orientee)"),  # rotation exacte
    "rotate": OpInfo(_rotate, "geometrie", ALL, "rotation quelconque dans [-degrees, degrees] (coins noirs)"),  # angle libre
    "affine": OpInfo(_affine, "geometrie", ALL, "rotation + translation + echelle + cisaillement"),  # tout en un
    # recadrage puis taille fixe : attention, un defaut peut sortir du cadre
    "random_resized_crop": OpInfo(_random_resized_crop, "geometrie", ALL, "recadrage aleatoire puis taille fixe (peut couper un defaut)"),
    "perspective": OpInfo(_perspective, "geometrie", ALL, "deformation de perspective (camera inclinee)"),  # vue de biais
    "elastic": OpInfo(_elastic, "geometrie", ALL, "deformation elastique locale (matiere souple, lent)"),  # ondulations locales
    "zoom": OpInfo(_zoom, "geometrie", ALL, "zoom avant / arriere, taille de sortie inchangee"),  # change la taille apparente
    "translate": OpInfo(_translate, "geometrie", ALL, "decalage horizontal et vertical"),  # deplace le contenu
    "shear": OpInfo(_shear, "geometrie", ALL, "cisaillement"),  # l'image penche comme un parallelogramme
    # recadrage "sur" : la fenetre contient toujours tous les objets, puis retour a la taille d'origine
    "safe_crop": OpInfo(_safe_crop, "geometrie", ALL, "recadrage qui garde tous les objets, retour a la taille d'origine"),
    # detection seulement : les images du lot n'ont plus la meme taille, ce que Faster R-CNN accepte
    "scale_jitter": OpInfo(_scale_jitter, "geometrie", ("detection",), "change la taille de l'image (Faster R-CNN : tailles variables)"),
    "brightness_contrast": OpInfo(_brightness_contrast, "photometrie", ALL, "luminosite et contraste"),  # variation de lumiere
    "gamma": OpInfo(_gamma, "photometrie", ALL, "gamma (eclaircit ou assombrit les tons moyens)"),  # courbe des tons
    "hue_saturation": OpInfo(_hue_saturation, "photometrie", ALL, "teinte et saturation (couleur seulement)"),  # couleurs
    "sharpness": OpInfo(_sharpness, "photometrie", ALL, "nettete adoucie ou accentuee"),  # contours plus ou moins nets
    "autocontrast": OpInfo(_autocontrast, "photometrie", ALL, "etirement min-max de l'histogramme"),  # contraste maximal
    "equalize": OpInfo(_equalize, "photometrie", ALL, "egalisation globale de l'histogramme"),  # niveaux repartis
    "clahe": OpInfo(_clahe, "photometrie", ALL, "egalisation locale (CLAHE)"),  # contraste renforce par zones
    "posterize": OpInfo(_posterize, "photometrie", ALL, "reduction du nombre de niveaux"),  # moins de nuances
    "invert": OpInfo(_invert, "photometrie", ALL, "negatif (rarement utile : seulement si la polarite varie)"),  # clair <-> sombre
    # eclairage plus fort d'un cote (rampe) ou au centre (radial) : defaut d'acquisition tres courant
    "illumination_gradient": OpInfo(_illumination_gradient, "capteur", ALL, "eclairage non uniforme lineaire ou radial"),
    "vignetting": OpInfo(_vignetting, "capteur", ALL, "bords assombris"),  # effet d'objectif ou d'eclairage annulaire
    "exposure": OpInfo(_exposure, "capteur", ALL, "gain et decalage d'exposition"),  # image plus claire ou plus sombre
    "line_stripes": OpInfo(_line_stripes, "capteur", ALL, "stries de camera lineaire (gain par colonne)"),  # stries regulieres
    "gaussian_noise": OpInfo(_gaussian_noise, "capteur", ALL, "bruit gaussien additif"),  # bruit electronique du capteur
    "poisson_noise": OpInfo(_poisson_noise, "capteur", ALL, "bruit de photons (faible lumiere)"),  # plus fort dans les zones claires
    "speckle_noise": OpInfo(_speckle_noise, "capteur", ALL, "bruit multiplicatif"),  # granularite
    "salt_pepper": OpInfo(_salt_pepper, "capteur", ALL, "pixels morts et chauds"),  # points noirs et blancs isoles
    "blur": OpInfo(_blur, "capteur", ALL, "flou gaussien"),  # flou doux et uniforme
    "motion_blur": OpInfo(_motion_blur, "capteur", ALL, "flou de mouvement oriente (defilement)"),  # trainee dans une direction
    "defocus_blur": OpInfo(_defocus_blur, "capteur", ALL, "flou de mise au point (disque)"),  # optique mal reglee
    "jpeg": OpInfo(_jpeg, "capteur", ALL, "artefacts de compression JPEG"),  # blocs 8 x 8 visibles
    "low_resolution": OpInfo(_low_resolution, "capteur", ALL, "reduction puis agrandissement"),  # perte de details fins
    "specular": OpInfo(_specular, "capteur", ALL, "reflet speculaire sur metal poli"),  # tache claire allongee
    "dust": OpInfo(_dust, "capteur", ALL, "poussieres et taches (pas si elles sont une classe de defaut)"),  # petites taches floues
    "random_erasing": OpInfo(_random_erasing, "occlusion", ALL, "un rectangle efface (Random Erasing)"),  # cache une zone
    "coarse_dropout": OpInfo(_coarse_dropout, "occlusion", ALL, "plusieurs rectangles masques"),  # cache plusieurs zones
    "grid_mask": OpInfo(_grid_mask, "occlusion", ALL, "grille de carres masques"),  # cache une grille reguliere
    # politiques : classification seulement, car elles deplacent l'image sans mettre a jour boites ou masques
    "randaugment": OpInfo(_randaugment, "politique", CLASSIFICATION, "RandAugment : num_ops operations tirees, force magnitude"),
    # politique sans aucun reglage
    "trivialaugment": OpInfo(_trivialaugment, "politique", CLASSIFICATION, "TrivialAugment : une operation, force aleatoire, sans reglage"),
    # politique qui vise la robustesse aux images abimees
    "augmix": OpInfo(_augmix, "politique", CLASSIFICATION, "AugMix : melange de chaines d'augmentations (robustesse)"),
    # politique trouvee par recherche automatique sur des jeux publics
    "autoaugment": OpInfo(_autoaugment, "politique", CLASSIFICATION, "AutoAugment : politique apprise sur ImageNet / CIFAR / SVHN"),
    # segmentation seulement : le defaut ajoute doit aussi etre dessine dans le masque
    "synthetic_defect": OpInfo(_synthetic_defect, "synthese", ("segmentation",), "ajoute un defaut synthetique ET l'ecrit dans le masque"),
}

# raccourci : nom -> seulement le constructeur (sans famille ni resume)
BUILDERS: dict[str, Callable[..., v2.Transform]] = {name: info.builder for name, info in OPS.items()}


def available_augmentations() -> dict[str, dict[str, Any]]:  # renvoie, pour chaque operation, ses parametres et leurs defauts
    """Nom -> parametres par defaut (lus dans la signature des constructeurs)."""
    # inspect.signature lit "def _blur(kernel_size=5, ...)" : on en tire {"kernel_size": 5, ...} pour chaque constructeur
    return {name: {p.name: p.default for p in inspect.signature(fn).parameters.values()} for name, fn in BUILDERS.items()}


def available_presets() -> dict[str, str]:  # renvoie, pour chaque preset, son explication
    """Nom du preset -> son raisonnement."""
    return {name: preset.summary for name, preset in PRESETS.items()}  # dictionnaire nom -> resume


def _as_dict(step: Any) -> dict[str, Any]:  # convertit une etape (objet de configuration ou dictionnaire) en dictionnaire simple
    if hasattr(step, "model_dump"):  # si c'est un objet pydantic (lu depuis le YAML valide)
        return dict(step.model_dump())  # model_dump le convertit en dictionnaire
    if isinstance(step, Mapping):  # si c'est deja un dictionnaire
        return dict(step)  # on en fait une copie (pour ne pas modifier l'original)
    raise TypeError(f"etape d'augmentation inattendue : {step!r}")  # autre chose : erreur claire


# renvoie la liste des etapes ou chaque preset est remplace par ses propres etapes
def expand_steps(steps: Sequence[Any] | None, task: Task = "classification") -> list[dict[str, Any]]:
    """Liste du YAML -> liste d'etapes simples (les presets sont remplaces par leurs etapes pour la tache)."""
    out: list[dict[str, Any]] = []  # etapes simples, dans l'ordre
    for step in steps or []:  # pour chaque etape de la liste (aucune si la liste vaut None)
        params = _as_dict(step)  # l'etape sous forme de dictionnaire
        if params.get("name") == "preset":  # si l'etape est un preset
            unknown = set(params) - {"name", "preset", "exclude"}  # cles non prevues pour un preset
            if unknown or "preset" not in params:  # si une cle est en trop ou si le nom du preset manque
                raise ValueError(f"etape preset : cles attendues preset (et exclude), recu {sorted(params)}")  # erreur claire
            out.extend(preset_steps(params["preset"], task, params.get("exclude") or ()))  # ajoute les etapes du preset
        else:  # sinon : etape ordinaire
            out.append(params)  # ajoutee telle quelle
    return out  # renvoie la liste a plat


# renvoie la transformation d'une etape, apres avoir verifie son nom, sa tache et ses parametres
def build_step(params: Mapping[str, Any], task: Task = "classification") -> v2.Transform:
    """Une etape {name: ..., parametres} -> transformation ; verifie le nom, la tache et les parametres."""
    params = dict(params)  # copie : on va retirer "name" sans toucher a l'original
    name = params.pop("name")  # nom de l'operation ; il reste seulement ses parametres
    if name not in OPS:  # si le nom n'est pas dans le registre (faute de frappe ?)
        raise ValueError(f"augmentation inconnue : {name} ; disponibles : {sorted(OPS)} (et preset : {sorted(PRESETS)})")  # erreur claire
    info = OPS[name]  # fiche de l'operation
    if task not in info.tasks:  # si l'operation n'est pas permise pour cette tache
        # erreur claire avec les taches permises et le resume
        raise ValueError(f"augmentation {name} non prevue pour la tache {task} (taches : {list(info.tasks)}) : {info.summary}")
    try:  # on verifie les parametres AVANT de construire, pour un message d'erreur lisible
        inspect.signature(info.builder).bind(**params)  # bind echoue si un parametre n'existe pas dans le constructeur
    except TypeError as error:  # si un parametre est inconnu ou en trop
        # erreur claire qui liste les parametres acceptes ; "from error" garde l'erreur d'origine
        raise ValueError(f"parametres invalides pour {name} : {error} ; acceptes : {available_augmentations()[name]}") from error
    return info.builder(**params)  # construit et renvoie la transformation


# renvoie la chaine complete des augmentations (ou None s'il n'y en a pas)
def build_augmentation(steps: Sequence[Any] | None, task: Task = "classification") -> v2.Compose | None:
    """Compose torchvision v2 a partir de la liste du YAML ; None si la liste est vide (pas d'augmentation).

    Les noms, parametres et taches sont verifies a la construction : une faute de frappe echoue au
    demarrage. En detection, les boites sont ramenees dans l'image a la fin (ClampBoundingBoxes) ;
    les boites devenues trop petites se retirent ensuite avec sanitize_boxes.
    """
    if not steps:  # si la liste est vide ou absente
        return None  # pas d'augmentation
    if task not in TASKS:  # si la tache est inconnue
        raise ValueError(f"tache inconnue : {task}")  # erreur claire
    transforms = [build_step(params, task) for params in expand_steps(steps, task)]  # une transformation par etape
    if not transforms:  # si toutes les etapes ont ete retirees (ex. preset sans etape pour cette tache)
        return None  # pas d'augmentation
    transforms.append(Clamp01())  # a la fin : ramene les pixels dans [0, 1] (le bruit ou le gain peuvent en sortir)
    if task == "detection":  # en detection
        transforms.append(v2.ClampBoundingBoxes())  # coupe les boites qui depassent du bord de l'image
    return v2.Compose(transforms)  # Compose : applique les transformations les unes apres les autres


def sanitize_boxes(  # retire les boites trop petites et renvoie (boites gardees, etiquettes gardees)
    boxes: tv_tensors.BoundingBoxes,  # boites apres augmentation
    labels: torch.Tensor | None = None,  # classe de chaque boite (facultatif)
    min_size: float = 1.0,  # largeur et hauteur minimales en pixels
) -> tuple[tv_tensors.BoundingBoxes, torch.Tensor | None]:  # renvoie les boites et les etiquettes filtrees
    """Retire les boites devenues degenerees (largeur ou hauteur < min_size apres recadrage) et leurs etiquettes."""
    # format XYXY = (x gauche, y haut, x droite, y bas) : pratique pour calculer largeur et hauteur
    xyxy = F.convert_bounding_box_format(boxes, new_format=tv_tensors.BoundingBoxFormat.XYXY)
    keep = ((xyxy[:, 2] - xyxy[:, 0]) >= min_size) & ((xyxy[:, 3] - xyxy[:, 1]) >= min_size)  # vrai si largeur ET hauteur suffisantes
    kept = tv_tensors.wrap(boxes[keep], like=boxes)  # garde ces boites en conservant leur format et la taille de l'image
    return kept, (labels[keep] if labels is not None else None)  # renvoie les boites et les etiquettes correspondantes


# RandomErasing previent qu'il laisse passer boites et masques : c'est le comportement voulu ici
# (on cache une partie de l'image, la verite terrain ne change pas).
warnings.filterwarnings("ignore", message=".*RandomErasing.*passing through.*")
