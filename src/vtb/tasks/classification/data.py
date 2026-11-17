"""Donnees de classification : Dataset, statistiques de normalisation, DataLoader.

Ordre des operations pour une image (le meme partout, carte du modele comprise) :
    read_image(color) -> Pipeline (OpenCV) -> to_float_chw -> augmentation (train) -> normalize
"""

from __future__ import annotations  # les annotations de type restent du texte : syntaxe moderne "X | None" partout

import logging  # journal des messages (info, avertissements) au lieu de print
from collections.abc import Callable, Sequence  # types pour les annotations : une fonction appelable, une suite de valeurs

import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)
import pandas as pd  # pandas : tableaux de donnees ; ici le catalogue des images (chemin, etiquette)
import torch  # PyTorch : tenseurs (tableaux de nombres, sur CPU ou GPU) et reseaux de neurones
from torch.utils.data import DataLoader, Dataset  # Dataset : donne une image par indice ; DataLoader : les regroupe en lots

from vtb.core.runtime import make_generator, seed_worker  # outils de reproductibilite : melange et graines identiques a chaque run
from vtb.io.images import read_image  # lit une image sur disque en tableau numpy (RGB ou niveaux de gris)
from vtb.preprocess import Pipeline  # suite d'operations OpenCV (redimension, recadrage...) appliquee a chaque image
from vtb.preprocess.tensor import channel_stats, normalize, to_float_chw  # stats par canal, normalisation, image -> tenseur

logger = logging.getLogger(__name__)  # journal propre a ce module (son nom apparait dans les messages)


class ClassificationDataset(Dataset):  # jeu de donnees PyTorch : sait donner l'image numero i et sa classe
    """Lit une image du catalogue, la pretraite et la transforme en tenseur ; renvoie (tenseur, indice de classe)."""

    def __init__(  # constructeur : garde tout ce qu'il faut pour lire une image plus tard, sans rien lire maintenant
        self,  # l'objet en cours de creation
        frame: pd.DataFrame,  # catalogue : une ligne par image avec les colonnes "path" et "label"
        class_to_idx: dict[str, int],  # nom de classe -> numero (le reseau travaille avec des numeros)
        pipeline: Pipeline,  # pretraitement OpenCV a appliquer a chaque image
        color: str,  # "rgb" ou "gray" : comment lire les images
        mean: Sequence[float] | None,  # moyenne par canal pour la normalisation (None = pas de normalisation)
        std: Sequence[float] | None,  # ecart-type par canal pour la normalisation
        augment: Callable[[torch.Tensor], torch.Tensor] | None = None,  # augmentation aleatoire (train seulement), None sinon
    ) -> None:  # un constructeur ne renvoie rien
        self.paths = frame["path"].tolist()  # listes legeres des chemins : rien n'est lu avant __getitem__
        self.targets = [class_to_idx[label] for label in frame["label"]]  # numero de classe de chaque image, dans le meme ordre
        self.pipeline, self.color, self.augment = pipeline, color, augment  # garde les reglages pour la lecture
        self.mean, self.std = mean, std  # garde les statistiques de normalisation

    def __len__(self) -> int:  # nombre d'images du jeu (le DataLoader en a besoin pour savoir quand une epoque finit)
        return len(self.paths)  # une image par chemin

    def load(self, index: int) -> np.ndarray:  # lit et pretraite l'image numero index ; renvoie un tableau numpy
        """L'image pretraitee, avant mise en tenseur (pour l'affichage et la carte du modele)."""
        return self.pipeline(read_image(self.paths[index], color=self.color))  # lecture disque puis pretraitement OpenCV

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:  # appele par le DataLoader : renvoie (tenseur, classe)
        tensor = to_float_chw(self.load(index))  # image -> tenseur float (canaux, hauteur, largeur) avec valeurs dans [0, 1]
        if self.augment is not None:  # si une augmentation est prevue (train seulement)
            tensor = self.augment(tensor)  # transformation aleatoire (flip, rotation...) : nouvelle variante a chaque epoque
        if self.mean is not None and self.std is not None:  # si des statistiques de normalisation sont fournies
            tensor = normalize(tensor, self.mean, self.std)  # (x - moyenne) / ecart-type : valeurs centrees, comme a l'entrainement
        return tensor, self.targets[index]  # le tenseur d'entree et le numero de classe attendu


# calcule moyenne et ecart-type par canal sur le train ; renvoie deux listes (une valeur par canal)
def compute_stats(frame: pd.DataFrame, pipeline: Pipeline, color: str, max_images: int, seed: int) -> tuple[list[float], list[float]]:
    """mean / std par canal sur (un echantillon du) train, apres pretraitement : jamais sur val ou test."""
    sample = frame.sample(n=min(max_images, len(frame)), random_state=seed)  # tirage reproductible d'au plus max_images images
    return channel_stats(pipeline(read_image(p, color=color)) for p in sample["path"])  # lit, pretraite et cumule les stats


def class_weights(frame: pd.DataFrame, class_names: list[str]) -> torch.Tensor:  # renvoie un poids par classe pour la perte
    """Poids inverses des effectifs du train : une erreur sur une classe rare coute plus cher."""
    # nombre d'images par classe, dans l'ordre de class_names (0 si une classe est absente), en tableau numpy
    counts = frame["label"].value_counts().reindex(class_names).fillna(0).to_numpy(dtype=np.float64)
    weights = len(frame) / (len(class_names) * np.maximum(counts, 1))  # total / (nb classes x effectif) ; max(.,1) evite /0
    return torch.tensor(weights, dtype=torch.float32)  # en tenseur float32, le format attendu par la fonction de perte


# fabrique le DataLoader (distributeur de lots) ; renvoie un objet a parcourir avec "for images, labels in loader"
def build_loader(dataset: Dataset, batch_size: int, train: bool, num_workers: int, seed: int) -> DataLoader:
    """shuffle et drop_last seulement pour le train ; generateur et graines de workers fixes : ordre reproductible."""
    return DataLoader(  # cree et renvoie le distributeur de lots
        dataset,  # ou prendre les images
        batch_size=batch_size,  # combien d'images par lot
        shuffle=train,  # melange a chaque epoque pour le train seulement (val / test : ordre fixe)
        drop_last=train and len(dataset) > batch_size,  # train : jette le dernier lot incomplet (lot trop petit = stats instables)
        num_workers=num_workers,  # processus paralleles de lecture (0 = lecture dans le processus principal)
        worker_init_fn=seed_worker,  # donne une graine propre a chaque worker (augmentations differentes et reproductibles)
        generator=make_generator(seed),  # generateur aleatoire fixe : meme ordre de melange d'un run a l'autre
        persistent_workers=num_workers > 0,  # garde les workers vivants entre epoques (evite de les relancer a chaque fois)
    )
