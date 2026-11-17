"""Passage de l'image pretraitee (numpy) au tenseur du reseau : la seule etape apres le Pipeline.

Ordre fixe dans toute la toolbox :
    lecture -> Pipeline (OpenCV, deterministe) -> to_float_chw ([0, 1]) -> augmentation (train)
    -> normalize (mean / std)
C'est aussi l'ordre que la carte du modele decrit au programme de production.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes

from collections.abc import Sequence  # type "suite d'elements" (liste, tuple) pour les annotations

import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)
import torch  # PyTorch : tenseurs (tableaux de nombres utilisables sur CPU ou GPU) et reseaux de neurones

from vtb.io.images import max_value  # valeur maximale d'un pixel selon le type (255, 65535 ou 1)

IMAGENET_MEAN = [0.485, 0.456, 0.406]  # moyenne R, G, B des images ImageNet (base d'apprentissage des reseaux pre-entraines)
IMAGENET_STD = [0.229, 0.224, 0.225]  # ecart-type R, G, B d'ImageNet (sert a la normalisation)


def to_float_chw(image: np.ndarray) -> torch.Tensor:  # convertit une image numpy en tenseur (canaux, hauteur, largeur) entre 0 et 1
    """(H, W) ou (H, W, C) uint8 / uint16 -> tenseur float32 (C, H, W) dans [0, 1]."""
    peak = max_value(image)  # 255 en 8 bits, 65535 en 16 bits
    array = image.astype(np.float32) / peak  # valeurs ramenees entre 0 et 1
    if array.ndim == 2:  # image grise (H, W)
        array = array[None, :, :]  # ajoute un axe canal en tete : (1, H, W)
    else:  # image couleur (H, W, C)
        array = array.transpose(2, 0, 1)  # PyTorch veut les canaux en premier : (C, H, W)
    return torch.from_numpy(np.ascontiguousarray(array))  # tableau range de facon continue en memoire, puis tenseur


def normalize(tensor: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:  # normalise canal par canal
    """(x - mean) / std par canal ; accepte (C, H, W) ou (N, C, H, W)."""
    # normalisation = centrer (moins la moyenne) et reduire (divise par l'ecart-type) pour avoir des valeurs autour de 0
    shape = (-1, 1, 1) if tensor.ndim == 3 else (1, -1, 1, 1)  # forme qui aligne une valeur par canal (avec ou sans axe du lot N)
    m = torch.tensor(mean, dtype=tensor.dtype).view(shape)  # moyennes par canal, mises en forme pour la soustraction
    s = torch.tensor(std, dtype=tensor.dtype).view(shape)  # ecarts-types par canal, mis en forme pour la division
    return (tensor - m) / s  # applique (x - moyenne) / ecart-type a chaque pixel de chaque canal


def image_to_input(image: np.ndarray, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:  # image -> entree du modele exporte
    """Image pretraitee -> tableau (1, C, H, W) float32 pret pour onnxruntime (ce que fait le C#)."""
    # onnxruntime = moteur qui execute un modele ONNX ; unsqueeze(0) ajoute l'axe du lot (1 image), numpy() le reconvertit en tableau
    return normalize(to_float_chw(image), mean, std).unsqueeze(0).numpy()


def channel_stats(images: Sequence[np.ndarray]) -> tuple[list[float], list[float]]:  # renvoie (moyennes, ecarts-types) par canal
    """Moyenne et ecart-type par canal d'images pretraitees (a calculer sur le train seulement)."""
    total, total_sq, n = None, None, 0  # somme des valeurs, somme des carres, nombre de pixels par canal
    for image in images:  # pour chaque image
        x = to_float_chw(image).double()  # tenseur (C, H, W) en double precision (evite les erreurs d'arrondi sur de grosses sommes)
        s, sq = x.sum(dim=(1, 2)), (x**2).sum(dim=(1, 2))  # somme et somme des carres de chaque canal
        total = s if total is None else total + s  # cumule les sommes
        total_sq = sq if total_sq is None else total_sq + sq  # cumule les sommes de carres
        n += x.shape[1] * x.shape[2]  # ajoute le nombre de pixels de l'image (hauteur x largeur)
    mean = total / n  # moyenne par canal
    std = (total_sq / n - mean**2).clamp_min(1e-12).sqrt()  # ecart-type = racine(moyenne des carres - carre de la moyenne), jamais 0
    return [float(v) for v in mean], [float(v) for v in std]  # renvoie deux listes de nombres Python
