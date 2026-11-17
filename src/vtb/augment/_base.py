"""Aides internes des augmentations : reconnaitre l'image parmi les entrees, tirer au hasard avec torch.

Toutes les operations du module tirent leurs nombres avec le generateur de TORCH (jamais numpy ni
`random` directement) : DataLoader re-seme torch dans chaque worker, donc deux workers ne
produisent jamais les memes augmentations et `torch.manual_seed(n)` rend tout reproductible.
Quand une operation a besoin de numpy (OpenCV, defauts synthetiques), elle cree un generateur
numpy a partir d'un entier tire par torch (`numpy_rng`).
"""

from __future__ import annotations  # permet d'ecrire les types (ex. "X | None") sans qu'ils soient evalues au chargement

import math  # fonctions mathematiques de base (exp, log) sur des nombres simples
from collections.abc import Sequence  # Sequence : type "liste ou tuple" (on peut indexer et compter les elements)
from typing import Any  # Any : "n'importe quel type", quand les entrees peuvent etre de nature variee

import numpy as np  # numpy : calcul sur des tableaux de nombres (format attendu par OpenCV)
import torch  # PyTorch : tenseurs (tableaux de nombres multidimensionnels) et generateur aleatoire

# tv_tensors : tenseurs "etiquetes" de torchvision (Image, BoundingBoxes = boites, Mask = masque) ;
# les transformations savent ainsi quoi deplacer avec l'image (boites, masques) et quoi laisser tel quel
from torchvision import tv_tensors


def is_image(inpt: Any) -> bool:  # renvoie vrai si l'entree est l'image elle-meme
    """Vrai pour l'image (tenseur flottant pur ou tv_tensors.Image), faux pour les boites, masques et etiquettes."""
    if isinstance(inpt, (tv_tensors.BoundingBoxes, tv_tensors.Mask)):  # si c'est des boites ou un masque
        return False  # ce n'est pas l'image
    return isinstance(inpt, torch.Tensor) and inpt.is_floating_point()  # l'image est le seul tenseur a valeurs decimales


def find_image(flat_inputs: Sequence[Any]) -> torch.Tensor:  # renvoie l'image trouvee parmi toutes les entrees
    for inpt in flat_inputs:  # pour chaque entree (image, boites, masque, etiquette...)
        if is_image(inpt):  # si c'est l'image
            return inpt  # on la renvoie tout de suite
    raise TypeError("aucune image flottante (C, H, W) parmi les entrees de l'augmentation")  # aucune image : erreur claire


def find_boxes(flat_inputs: Sequence[Any]) -> tv_tensors.BoundingBoxes | None:  # renvoie les boites, ou None s'il n'y en a pas
    for inpt in flat_inputs:  # pour chaque entree
        if isinstance(inpt, tv_tensors.BoundingBoxes):  # si ce sont des boites englobantes (detection d'objets)
            return inpt  # on les renvoie
    return None  # pas de boites (classification, segmentation)


def image_size(flat_inputs: Sequence[Any]) -> tuple[int, int]:  # renvoie (hauteur, largeur) de l'image
    """(H, W) de l'image."""
    image = find_image(flat_inputs)  # retrouve l'image parmi les entrees
    return int(image.shape[-2]), int(image.shape[-1])  # les deux dernieres dimensions d'un tenseur (C, H, W) : H et W


def pair(value: float | Sequence[float]) -> tuple[float, float]:  # renvoie toujours un couple (minimum, maximum)
    """3 -> (3, 3) ; (1, 5) -> (1, 5) : un parametre est soit fixe soit un intervalle de tirage."""
    if isinstance(value, (int, float)):  # si on a donne un seul nombre
        return float(value), float(value)  # intervalle reduit a ce nombre : valeur fixe
    return float(value[0]), float(value[1])  # sinon : les deux bornes donnees


def coin(p: float) -> bool:  # pile ou face truque : renvoie vrai avec la probabilite p
    """Vrai avec la probabilite p (generateur torch)."""
    # p >= 1 : toujours vrai ; p <= 0 : toujours faux ; sinon tirage d'un nombre entre 0 et 1 compare a p
    return p >= 1.0 or (p > 0.0 and torch.rand(()).item() < p)


def uniform(low: float, high: float) -> float:  # renvoie un nombre au hasard entre low et high (toutes valeurs aussi probables)
    return low + (high - low) * torch.rand(()).item()  # torch.rand(()) : un nombre entre 0 et 1 ; .item() le convertit en float


def log_uniform(low: float, high: float) -> float:  # renvoie un facteur au hasard, tire en echelle logarithmique
    """Tirage uniforme en echelle logarithmique (facteurs multiplicatifs : 0,5 et 2 aussi probables)."""
    return math.exp(uniform(math.log(low), math.log(high)))  # tire uniformement le logarithme, puis revient a l'echelle normale


def randint(low: int, high: int) -> int:  # renvoie un entier au hasard entre low et high
    """Entier dans [low, high] (bornes comprises)."""
    return int(torch.randint(int(low), int(high) + 1, ()).item())  # torch.randint exclut la borne haute, d'ou le + 1


def choice(options: Sequence[Any]) -> Any:  # renvoie un element de la liste choisi au hasard
    return options[randint(0, len(options) - 1)]  # tire un indice valide puis renvoie l'element


def numpy_rng() -> np.random.Generator:  # renvoie un generateur aleatoire numpy derive de torch
    """Generateur numpy seme par torch : suit torch.manual_seed et la graine de chaque worker."""
    return np.random.default_rng(int(torch.randint(0, 2**31 - 1, ()).item()))  # graine = un entier tire par torch


def to_hwc(image: torch.Tensor) -> np.ndarray:  # convertit un tenseur torch en tableau numpy pour OpenCV
    """Tenseur (C, H, W) float -> tableau float32 (H, W) ou (H, W, C) pour OpenCV."""
    # detach : sort du suivi des gradients ; cpu : ramene en memoire centrale ; numpy : tableau numpy ; float32 sans copie inutile
    array = image.detach().cpu().numpy().astype(np.float32, copy=False)
    # 1 canal : tableau 2D (H, W) ; sinon on remet les canaux a la fin (H, W, C), en memoire contigue comme l'exige OpenCV
    return array[0] if array.shape[0] == 1 else np.ascontiguousarray(array.transpose(1, 2, 0))


def from_hwc(array: np.ndarray, like: torch.Tensor) -> torch.Tensor:  # reconvertit un tableau numpy en tenseur torch
    """Inverse de to_hwc : meme forme, type et appareil que `like`."""
    chw = array[None] if array.ndim == 2 else array.transpose(2, 0, 1)  # remet la dimension des canaux en premier (C, H, W)
    # tenseur torch sur le meme appareil (cpu/gpu) et avec le meme type de nombres que le tenseur de depart
    out = torch.from_numpy(np.ascontiguousarray(chw)).to(device=like.device, dtype=like.dtype)
    # si le tenseur de depart etait un tv_tensor (ex. Image), on rhabille le resultat du meme type ; sinon tenseur simple
    return tv_tensors.wrap(out, like=like) if isinstance(like, tv_tensors.TVTensor) else out
