"""Configurations de l'exploration : controle du jeu, plongements, derive."""

from __future__ import annotations  # permet d'ecrire les types (ex. "str | None") sans qu'ils soient evalues au chargement

from typing import Literal  # Literal : le champ n'accepte qu'une liste fermee de valeurs exactes

from pydantic import Field  # Field : precise la valeur par defaut d'un champ (ici une fonction qui la fabrique)

# sections de configuration communes a toute la boite a outils, deja validees par pydantic
from vtb.core.config import DataConfig, PreprocessStep, RuntimeConfig, StrictModel


def _resize_224() -> list[PreprocessStep]:  # fabrique le pretraitement par defaut : une liste avec un seul redimensionnement
    return [PreprocessStep(name="resize", size=224)]  # 224 x 224 pixels : la taille attendue par les reseaux du zoo


class ExploreConfig(StrictModel):  # reglages de l'exploration d'un jeu ; StrictModel refuse toute cle inconnue du YAML
    task: Literal["explore"] = "explore"  # nom de la tache : sert a reconnaitre ce type de configuration
    data: DataConfig  # ou sont les images et comment elles sont rangees (obligatoire, pas de valeur par defaut)
    split_file: str | None = None  # split.csv existant : verifie aussi les doublons a cheval sur deux sous-ensembles
    # pour les plongements (vecteurs de nombres qui resument chaque image) il faut une taille fixe
    preprocess: list[PreprocessStep] = Field(default_factory=_resize_224)  # default_factory : une liste neuve par config
    model: str = "resnet18"  # reseau du zoo utilise sans sa tete (la derniere couche qui classe)
    n_clusters: int = 6  # nombre de groupes (clusters) d'images visuellement proches a former
    projection: Literal["pca", "tsne"] = "tsne"  # methode pour dessiner les plongements en 2D : pca (lineaire) ou tsne
    max_hamming: int = 6  # quasi-doublons candidats : empreintes a <= 6 bits differents sur 64
    min_correlation: float = 0.9  # confirmation : correlation des details >= 0.9
    gallery_per_cluster: int = 8  # nombre d'images montrees par groupe dans la galerie
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)  # dossier de sortie, graine aleatoire, cpu/gpu


class DriftConfig(StrictModel):  # reglages de la mesure de derive (images d'aujourd'hui vs images d'entrainement)
    task: Literal["drift"] = "drift"  # nom de la tache : sert a reconnaitre ce type de configuration
    reference: DataConfig  # images de l'entrainement (ou d'une periode connue saine)
    current: DataConfig  # images recentes de la ligne
    simulate: list[PreprocessStep] = Field(default_factory=list)  # DEMONSTRATION : alteration appliquee aux images courantes
    preprocess: list[PreprocessStep] = Field(default_factory=_resize_224)  # taille fixe pour calculer les plongements
    model: str = "resnet18"  # reseau du zoo qui transforme chaque image en vecteur (plongement)
    max_images: int = 500  # par cote : la derive se mesure tres bien sur quelques centaines d'images
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)  # dossier de sortie, graine aleatoire, cpu/gpu
