"""Donnees d'anomalies : etiquettes normal / defaut, decoupage qui n'expose jamais un defaut, Dataset avec masques.

Trois regles du module 19 : train et val ne contiennent que des normales ; le seuil se fixe sur
les normales de validation ; il faut des defauts pour EVALUER, et ils sont tous dans le test.
"""

from __future__ import annotations  # permet les annotations de type modernes

import logging  # journal des messages

import cv2  # OpenCV : traitement d'images (ici redimensionner les masques)
import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)
import pandas as pd  # pandas : tableaux de donnees (le catalogue : une ligne par image)
import torch  # PyTorch : tenseurs (tableaux de nombres utilises par les reseaux)
from sklearn.model_selection import train_test_split  # coupe un tableau en deux parties au hasard (avec une graine)
from torch.utils.data import DataLoader, Dataset  # Dataset : acces a une image par son numero ; DataLoader : regroupe en lots

from vtb.io.images import read_image, read_mask  # lecture d'une image et d'un masque (image noir / blanc du defaut)
from vtb.preprocess import Pipeline  # enchainement des etapes de pretraitement (resize...) decrit dans la config
from vtb.preprocess.tensor import to_float_chw  # image (H, W, C) en 0-255 -> tenseur (C, H, W) en 0-1
from vtb.split import check_no_leakage  # verifie qu'aucune image n'est a la fois dans deux parties (fuite)

logger = logging.getLogger(__name__)  # journal propre a ce module


def to_good_defect(catalog: pd.DataFrame, normal_labels: list[str]) -> pd.DataFrame:  # ramene les etiquettes a good / defect
    """label devient good / defect ; l'etiquette d'origine est gardee dans defect_type (pour le rappel par type)."""
    catalog = catalog.copy()  # copie : on ne modifie pas le tableau de l'appelant
    # type de defaut d'origine : colonne defect_type si elle existe, sinon l'etiquette elle-meme
    original = catalog["defect_type"] if "defect_type" in catalog else catalog["label"].astype(str)
    is_normal = catalog["label"].astype(str).isin(normal_labels) | (catalog["label"] == "good")  # vrai pour chaque image normale
    catalog["defect_type"] = np.where(is_normal, "good", original)  # normale -> "good", sinon type d'origine (rayure, trou...)
    catalog["label"] = np.where(is_normal, "good", "defect")  # etiquette binaire : good ou defect
    if "mask_path" not in catalog:  # si le catalogue n'a pas de colonne de masques
        catalog["mask_path"] = ""  # on la cree vide (pas de masque)
    catalog["mask_path"] = catalog["mask_path"].fillna("").astype(str)  # cases vides -> chaine vide, tout en texte
    catalog.loc[catalog["label"] == "good", "mask_path"] = ""  # une normale n'a pas de defaut a localiser
    return catalog  # le catalogue transforme


# decoupe le catalogue en train / val / test sans jamais mettre un defaut hors du test ; renvoie le catalogue avec une colonne split
def make_anomaly_split(catalog: pd.DataFrame, val_fraction: float, test_fraction: float, seed: int) -> pd.DataFrame:
    """Normales -> train / val / test ; defauts -> test. Si le rangement fournit deja train / test (MVTec), il est respecte."""
    catalog = catalog.copy().reset_index(drop=True)  # copie avec des numeros de ligne 0, 1, 2... (utilises comme reperes)
    normals = catalog[catalog["label"] == "good"]  # seulement les images normales
    if "split" in catalog and set(catalog["split"]) >= {"train", "test"}:  # si le jeu fournit deja train et test (ex. MVTec)
        pool = normals[normals["split"] == "train"]  # les normales du train d'origine seront partagees en train / val
    else:  # sinon on fait le decoupage nous-memes
        catalog["split"] = "test"  # par defaut tout va en test
        pool = normals  # toutes les normales sont candidates au train / val
        if test_fraction > 0:  # si on veut aussi des normales dans le test
            pool, held = train_test_split(normals, test_size=test_fraction, random_state=seed)  # held = normales mises de cote
            catalog.loc[held.index, "split"] = "test"  # les normales mises de cote vont dans le test
    catalog.loc[pool.index, "split"] = "train"  # le reste des normales va en train
    _, val = train_test_split(pool, test_size=val_fraction, random_state=seed)  # tire une part du train pour la validation
    catalog.loc[val.index, "split"] = "val"  # ces normales passent en validation (servent a calibrer le seuil)
    catalog.loc[catalog["label"] == "defect", "split"] = "test"  # tous les defauts vont en test, sans exception
    # controle : une meme image (groupe = chemin) jamais dans deux parties, et train sans aucun defaut
    check_no_leakage(catalog.assign(group=catalog["path"]), normal_only_train=True)
    return catalog  # catalogue avec la colonne split remplie


class AnomalyDataset(Dataset):  # donne, pour un numero, l'image en tenseur, son masque de defaut et son etiquette
    """Renvoie (image (C, S, S) dans [0, 1], masque (1, S, S) en 0/1, etiquette 0 normal / 1 defaut).

    Pas de normalisation ici : l'auto-encodeur reproduit une image dans [0, 1] et les extracteurs
    pre-entraines (PatchCore, anomalib) normalisent eux-memes, a l'interieur du modele exporte.
    Le masque est ramene a la taille de l'image pretraitee au plus proche voisin (pas de valeurs
    intermediaires) ; un pretraitement qui deplace les pixels (crop, letterbox) n'est donc pas
    applique au masque : seul resize est compatible avec les masques.
    """

    # frame : lignes du catalogue ; pipeline : pretraitement ; color : "rgb" ou "gray" ; augment : transformations aleatoires
    def __init__(self, frame: pd.DataFrame, pipeline: Pipeline, color: str = "rgb", augment=None) -> None:
        self.paths = frame["path"].tolist()  # chemins des images
        # chemins des masques ("" = pas de masque) ; si la colonne manque, aucune image n'a de masque
        self.masks = frame["mask_path"].fillna("").astype(str).tolist() if "mask_path" in frame else [""] * len(frame)
        self.labels = (frame["label"] == "defect").astype(int).tolist()  # etiquettes en nombres : 1 defaut, 0 normal
        self.pipeline, self.color, self.augment = pipeline, color, augment  # garde les reglages pour la lecture

    def __len__(self) -> int:  # nombre d'images (demande par le DataLoader)
        return len(self.paths)  # une image par chemin

    def load(self, index: int) -> np.ndarray:  # lit et pretraite l'image numero index ; renvoie un tableau numpy
        return self.pipeline(read_image(self.paths[index], color=self.color))  # lecture puis pretraitement (resize...)

    def load_mask(self, index: int, height: int, width: int) -> np.ndarray:  # masque de l'image index, a la taille demandee
        if not self.masks[index]:  # si l'image n'a pas de masque (normale)
            return np.zeros((height, width), np.uint8)  # masque tout noir : aucun pixel defectueux
        # lit le masque et le redimensionne au plus proche voisin (garde des valeurs 0/1 nettes)
        return cv2.resize(read_mask(self.masks[index]), (width, height), interpolation=cv2.INTER_NEAREST)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:  # renvoie (image, masque, etiquette)
        image = self.load(index)  # image pretraitee (H, W, C)
        tensor = to_float_chw(image)  # tenseur (C, H, W) en 0-1
        if self.augment is not None:  # si une augmentation est prevue (entrainement de l'auto-encodeur)
            tensor = self.augment(tensor)  # applique les transformations aleatoires
        # masque a la taille de l'image, en nombres a virgule ; [None] ajoute la dimension canal : (1, H, W)
        mask = torch.from_numpy(self.load_mask(index, image.shape[0], image.shape[1])).float()[None]
        return tensor, mask, self.labels[index]  # le trio attendu par les boucles d'entrainement et de score


# cree le DataLoader : lots de batch_size images, melanges seulement en entrainement ; renvoie le DataLoader
def anomaly_loader(dataset: Dataset, batch_size: int, train: bool, num_workers: int, seed: int) -> DataLoader:
    # shuffle : ordre aleatoire a chaque epoque ; num_workers : processus de lecture en parallele ; generator : hasard reproductible
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers, generator=torch.Generator().manual_seed(seed))
