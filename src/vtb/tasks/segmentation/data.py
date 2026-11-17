"""Donnees de segmentation : image + masque de classes (0 = fond), meme pretraitement, meme augmentation.

Le masque est ramene a la taille de l'image pretraitee au PLUS PROCHE VOISIN : une interpolation
lineaire creerait des classes intermediaires (1,5 entre fond et defaut) qui n'existent pas.
Seuls les pretraitements photometriques et resize sont compatibles avec les masques (un crop ou
un letterbox deplacerait l'image sans deplacer le masque).
"""

from __future__ import annotations  # permet les annotations de type modernes

import cv2  # OpenCV : traitement d'images (ici redimensionner les masques)
import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)
import pandas as pd  # pandas : tableaux de donnees (le catalogue : une ligne par image)
import torch  # PyTorch : tenseurs (tableaux de nombres utilises par les reseaux)
from torch.utils.data import DataLoader, Dataset  # Dataset : acces a un exemple par son numero ; DataLoader : regroupe en lots
from torchvision import tv_tensors  # tv_tensors.Mask : marque un tenseur comme masque (l'augmentation le traite en consequence)

from vtb.io.images import read_image  # lecture d'une image depuis un fichier
from vtb.preprocess import Pipeline  # enchainement des etapes de pretraitement
from vtb.preprocess.tensor import to_float_chw  # image (H, W, C) en 0-255 -> tenseur (C, H, W) en 0-1


# liste des noms de classes de pixels, "background" (fond) toujours en premier (indice 0) ; renvoie cette liste
def class_names_for(catalog: pd.DataFrame, mode: str, class_names: list[str] | None, normal_labels: list[str]) -> list[str]:
    """Noms des classes de pixels, fond compris en premier."""
    if mode == "binary":  # mode binaire : fond ou defaut
        return ["background", "defect"]  # deux classes
    if mode == "from_values":  # mode ou la valeur du pixel donne la classe
        if not class_names:  # les noms doivent alors etre fournis dans la config
            raise ValueError("mask_mode from_values : donner class_names (classes 1..C)")  # erreur claire sinon
        return ["background", *class_names]  # fond puis les classes donnees
    labels = sorted(set(catalog["label"].astype(str)) - set(normal_labels))  # mode from_label : etiquettes d'images, sauf les saines
    return ["background", *labels]  # fond puis une classe par etiquette, en ordre alphabetique


# convertit un masque lu (niveaux de gris) en indices de classe par pixel, selon mask_mode ; renvoie un tableau uint8
def encode_mask(raw: np.ndarray, mode: str, label: str, class_names: list[str]) -> np.ndarray:
    """Masque lu (uint8) -> indices de classe uint8 selon mask_mode."""
    if mode == "from_values":  # la valeur du pixel est deja l'indice de classe
        return raw.astype(np.uint8)  # on la garde telle quelle
    defect = raw > 127 if raw.max() > 1 else raw > 0  # pixel defaut : blanc (> 127) si masque 0-255, ou non nul si masque 0/1
    # indice de la classe : 1 en binaire ; sinon l'indice de l'etiquette de l'image (0 si elle n'est pas une classe)
    index = 1 if mode == "binary" else (class_names.index(label) if label in class_names else 0)
    return (defect * index).astype(np.uint8)  # pixels defaut -> index, autres -> 0 (fond)


class SegmentationDataset(Dataset):  # donne, pour un numero, l'image en tenseur et son masque de classes
    """(image float (C, H, W) dans [0, 1], masque long (H, W) d'indices de classe)."""

    # frame : lignes du catalogue ; pipeline : pretraitement ; color : rgb / gray ; mode : mask_mode ; augment : transformations
    def __init__(self, frame: pd.DataFrame, pipeline: Pipeline, color: str, mode: str, class_names: list[str], augment=None) -> None:
        self.paths = frame["path"].tolist()  # chemins des images
        self.masks = frame["mask_path"].fillna("").astype(str).tolist()  # chemins des masques ("" = pas de masque)
        self.labels = frame["label"].astype(str).tolist()  # etiquette de chaque image (utile en mode from_label)
        # garde les reglages pour la lecture
        self.pipeline, self.color, self.mode, self.class_names, self.augment = pipeline, color, mode, class_names, augment

    def __len__(self) -> int:  # nombre d'images (demande par le DataLoader)
        return len(self.paths)  # une image par chemin

    def load(self, i: int) -> tuple[np.ndarray, np.ndarray]:  # renvoie (image pretraitee, masque d'indices) en numpy
        """Image pretraitee et masque d'indices a sa taille (sans augmentation)."""
        image = self.pipeline(read_image(self.paths[i], color=self.color))  # lecture puis pretraitement (resize...)
        if self.masks[i]:  # si l'image a un masque
            raw = read_image(self.masks[i], color="gray")  # lit le masque en niveaux de gris
            mask = encode_mask(raw, self.mode, self.labels[i], self.class_names)  # niveaux de gris -> indices de classe
            # met le masque a la taille de l'image, au plus proche voisin (pas de classe intermediaire inventee)
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:  # sinon : image sans masque
            mask = np.zeros(image.shape[:2], np.uint8)  # image saine sans masque
        return image, mask  # le couple image / masque

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:  # renvoie (image tenseur, masque tenseur) pour le DataLoader
        image, mask = self.load(i)  # image et masque en numpy
        x, y = to_float_chw(image), torch.from_numpy(mask.astype(np.int64))  # image -> tenseur 0-1 ; masque -> tenseur d'entiers
        if self.augment is not None:  # si une augmentation est prevue (entrainement)
            x, wrapped = self.augment(x, tv_tensors.Mask(y))  # meme transformation geometrique sur l'image ET le masque
            y = wrapped.as_subclass(torch.Tensor)  # redevient un tenseur ordinaire
        return x, y.long()  # masque en entiers longs : le type attendu par la perte d'entropie croisee


# cree le DataLoader qui fournit les lots ; renvoie le DataLoader
def build_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int, seed: int) -> DataLoader:
    return DataLoader(  # le chargeur de lots
        dataset,  # le jeu de donnees
        batch_size=batch_size,  # images par lot
        shuffle=shuffle,  # melange a chaque epoque (en entrainement)
        num_workers=num_workers,  # processus de lecture en parallele
        # en entrainement, jette le dernier lot incomplet (BatchNorm n'aime pas les lots de 1)
        drop_last=shuffle and len(dataset) > batch_size,
        generator=torch.Generator().manual_seed(seed),  # melange reproductible
    )
