"""Pretraitement deterministe : catalogue d'operations OpenCV, pipeline configurable, tuiles.

Importer ce paquet inscrit toutes les operations dans le registre (chaque module de `ops`
utilise le decorateur `register`).
"""

from vtb.preprocess.ops import (  # noqa: F401  (l'import inscrit les operations dans le registre)
    alignment,  # recalage sur une image de reference (ECC, ORB) et difference avec elle
    color,  # conversions et choix de canaux de couleur
    contrast,  # reglages de contraste (egalisation d'histogramme, CLAHE...)
    denoise,  # reduction du bruit (flous, filtres)
    edges,  # detection de contours
    frequency,  # filtres dans le domaine des frequences (transformee de Fourier)
    geometry,  # recadrage, rotation, redimensionnement
    illumination,  # correction d'eclairage non uniforme
    intensity,  # echelle des niveaux de gris (etirement, gamma, fenetre / niveau...)
    morphology,  # morphologie mathematique (erosion, dilatation...)
    scale,  # echelle et orientation : structures d'une taille ou d'une direction donnees (DoG, Gabor)
    sharpen,  # renforcement de la nettete
)
from vtb.preprocess.pipeline import Pipeline  # Pipeline : enchaine plusieurs operations dans l'ordre du YAML

# REGISTRY : dict nom -> operation ; get_op : retrouve une operation ; list_ops : catalogue ; register : inscrit une operation
from vtb.preprocess.registry import REGISTRY, get_op, list_ops, register

__all__ = ["Pipeline", "REGISTRY", "get_op", "list_ops", "register"]  # noms exportes par "from vtb.preprocess import *"
