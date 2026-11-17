"""Aides communes aux operations : passage en flottant et retour, application par canal.

Beaucoup d'operations calculent en flottant (division, filtrage) mais doivent rendre le type
d'entree (uint8 ou uint16). Ces deux fonctions font l'aller et le retour sans perte de dynamique.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes

from collections.abc import Callable  # type "fonction" pour annoter un parametre qui recoit une fonction

import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)

from vtb.io.images import max_value  # valeur maximale d'un pixel selon le type (255, 65535 ou 1)


def as_float(image: np.ndarray) -> tuple[np.ndarray, float]:  # convertit en flottant [0, 1] ; renvoie (image, valeur max d'origine)
    """Image -> (float32 dans [0, 1], valeur maximale du type d'origine)."""
    peak = max_value(image)  # 255 en 8 bits, 65535 en 16 bits
    return image.astype(np.float32) / peak, peak  # image divisee par le max (donc entre 0 et 1) et ce max


def from_float(values: np.ndarray, like: np.ndarray) -> np.ndarray:  # reconvertit [0, 1] vers le type d'une image modele
    """float32 dans [0, 1] -> meme type que `like` (ecrete : une operation ne doit jamais faire deborder)."""
    if like.dtype in (np.uint8, np.uint16):  # si le type d'arrivee est un entier 8 ou 16 bits
        peak = max_value(like)  # sa valeur maximale
        return (np.clip(values, 0.0, 1.0) * peak + 0.5).astype(like.dtype)  # coupe hors [0, 1], remet a l'echelle, arrondit
    return values.astype(like.dtype)  # type flottant : simple conversion


def per_channel(func: Callable[[np.ndarray], np.ndarray], image: np.ndarray) -> np.ndarray:  # applique func a chaque canal
    """Applique une operation definie pour une image grise a chaque canal d'une image couleur."""
    if image.ndim == 2:  # image grise : un seul canal
        return func(image)  # on applique directement
    return np.stack([func(image[:, :, c]) for c in range(image.shape[2])], axis=-1)  # applique a chaque canal puis les recolle
