"""anomalib derriere le meme contrat que les autres methodes : n'importe quel modele par son nom.

    method: anomalib
    anomalib:
      model: Padim                       # Patchcore, Padim, Stfpm, Fastflow, ReverseDistillation, EfficientAd, Draem, Cflow, Dfm...
      params: {backbone: resnet18, layers: [layer1, layer2, layer3]}
      max_epochs: 1

Ce que la toolbox garde pour elle, pour que toutes les methodes restent comparables :
- le decoupage (anomalib ne voit QUE les normales d'entrainement : ni validation, ni test) ;
- les scores bruts (on appelle le reseau torch d'anomalib nous-memes) puis la calibration du
  seuil sur nos normales de validation, les metriques et l'export (avec notre carte du modele).
anomalib fournit le modele, son pre-traitement interne (redimensionnement + normalisation) et
sa boucle d'entrainement quand il en a une.
"""

from __future__ import annotations  # permet les annotations de type modernes

import logging  # journal des messages
from pathlib import Path  # Path : chemin de fichier independant du systeme

import pandas as pd  # pandas : tableaux de donnees (le decoupage)
import torch  # PyTorch : tenseurs et reseaux de neurones
import torch.nn as nn  # nn : modules de reseau

from vtb.core.runtime import quiet_libraries  # fait taire les messages trop bavards des bibliotheques (Lightning, anomalib)
from vtb.io.images import read_image, write_image  # lecture et ecriture d'images
from vtb.preprocess import Pipeline  # enchainement des etapes de pretraitement

logger = logging.getLogger(__name__)  # journal propre a ce module


# ecrit les normales d'entrainement pretraitees sur disque (anomalib lit un dossier) ; renvoie le chemin du dossier
def materialize_normals(frame: pd.DataFrame, pipeline: Pipeline, color: str, out_dir: Path) -> Path:
    """Ecrit les normales d'entrainement pretraitees dans un dossier : c'est ce que le datamodule Folder lit."""
    folder = out_dir / "train" / "good"  # rangement attendu par anomalib : train/good
    folder.mkdir(parents=True, exist_ok=True)  # cree le dossier (et ses parents) s'il n'existe pas
    for i, path in enumerate(frame["path"]):  # pour chaque image normale (i = numero d'ordre)
        # nom unique (numero sur 5 chiffres + nom d'origine), image lue puis pretraitee, ecrite en PNG : sans perte
        write_image(folder / f"{i:05d}_{Path(path).stem}.png", pipeline(read_image(path, color=color)))
    return folder  # le dossier rempli


def build_model(name: str, params: dict, image_size: tuple[int, int]):  # cree un modele anomalib a partir de son nom ; le renvoie
    """Instancie un modele anomalib par son nom de classe, avec un pre-traitement a la taille de nos images."""
    import anomalib.models as models  # import ici : anomalib est une dependance facultative, chargee seulement si utilisee

    quiet_libraries()  # coupe les messages verbeux declenches par l'import
    if not hasattr(models, name):  # si anomalib n'a pas de classe de ce nom
        raise ValueError(f"modele anomalib inconnu : {name}")  # erreur claire plutot qu'un plantage obscur
    cls = getattr(models, name)  # recupere la classe (ex. models.Padim)
    pre_processor = cls.configure_pre_processor(image_size=image_size)  # pretraitement d'anomalib regle a la taille de nos images
    return cls(pre_processor=pre_processor, **params)  # cree le modele avec ses parametres de la config (backbone, layers...)


def fit_anomalib(  # entraine le modele anomalib sur les seules normales ; ne renvoie rien (le modele est modifie sur place)
    model,  # le modele anomalib cree par build_model
    train_dir: Path,  # dossier des normales d'entrainement ecrites par materialize_normals
    batch_size: int,  # images par lot
    max_epochs: int | None,  # nombre d'epoques (None : valeur par defaut d'anomalib)
    output: Path,  # dossier de sortie de l'experience
    seed: int,  # graine aleatoire pour des resultats reproductibles
    num_workers: int = 0,  # processus de lecture en parallele (0 = lecture dans le processus principal)
    use_datamodule: bool = False,  # True pour les modeles qui exigent un datamodule (EfficientAd)
) -> None:  # ne renvoie rien
    """Entraine (ou construit la memoire) sur les normales seules : aucune validation, aucun test vus par anomalib.

    On donne a l'Engine un DataLoader d'entrainement et rien d'autre : sans donnees de validation,
    Lightning saute la boucle de validation, et anomalib ne fixe aucun seuil (on le fait ensuite).
    """
    from anomalib.data.datasets import FolderDataset  # jeu de donnees anomalib qui lit un dossier d'images
    from anomalib.engine import Engine  # Engine : la boucle d'entrainement d'anomalib (basee sur Lightning)
    from lightning.pytorch import seed_everything  # fixe toutes les graines aleatoires d'un coup
    from torch.utils.data import DataLoader  # regroupe les images en lots

    dataset = FolderDataset(name="vtb", normal_dir=train_dir, split="train")  # lit les normales du dossier
    loader = DataLoader(  # lots d'entrainement
        dataset,  # le jeu de normales
        batch_size=batch_size,  # images par lot
        shuffle=True,  # ordre melange a chaque epoque
        num_workers=num_workers,  # processus de lecture
        collate_fn=dataset.collate_fn,  # fonction d'anomalib qui assemble les images d'un lot dans son format
        generator=torch.Generator().manual_seed(seed),  # melange reproductible
    )
    kwargs = {"max_epochs": max_epochs} if max_epochs is not None else {}  # ne passe max_epochs que s'il est fixe
    engine = Engine(  # cree la boucle d'entrainement
        default_root_dir=output / "anomalib",  # dossier ou anomalib ecrit ses propres fichiers
        accelerator="gpu" if torch.cuda.is_available() else "cpu",  # GPU s'il y en a un, sinon CPU
        devices=1,  # un seul processeur de calcul
        logger=False,  # pas de journal Lightning (on a le notre)
        enable_progress_bar=False,  # pas de barre de progression (sorties plus propres)
        enable_model_summary=False,  # pas de tableau resume du modele
        **kwargs,  # max_epochs eventuel
    )
    quiet_libraries()  # recoupe les messages verbeux (Engine en rajoute)
    seed_everything(seed, verbose=False)  # PatchCore tire le premier vecteur du coreset au hasard
    if use_datamodule:  # si le modele exige un datamodule
        # Quelques modeles (EfficientAd) lisent trainer.datamodule pendant l'entrainement : on leur en donne
        # un, dont la validation est tiree des NORMALES du train (toujours aucun defaut vu). Non execute ici.
        from anomalib.data import Folder  # datamodule anomalib : gere lui-meme train / val a partir d'un dossier

        datamodule = Folder(  # datamodule construit sur le meme dossier de normales
            name="vtb",  # nom du jeu (libre)
            root=train_dir.parent.parent,  # racine : le dossier qui contient train/good
            normal_dir=train_dir.relative_to(train_dir.parent.parent),  # chemin des normales relatif a la racine (train/good)
            train_batch_size=batch_size,  # images par lot en entrainement
            eval_batch_size=batch_size,  # images par lot en evaluation
            num_workers=num_workers,  # processus de lecture
            test_split_mode="none",  # aucun test : anomalib ne voit jamais de defaut
            val_split_mode="from_train",  # la validation est prise dans les normales du train
            val_split_ratio=0.1,  # 10 % des normales pour cette validation interne
            seed=seed,  # tirage reproductible
        )
        engine.fit(model, datamodule=datamodule)  # entrainement avec le datamodule
        return  # fini pour ce cas
    engine.fit(model, train_dataloaders=loader)  # cas general : entrainement (ou memoire) avec le seul DataLoader des normales


class AnomalibRawMap(nn.Module):  # enveloppe : modele anomalib -> carte brute, comme les autres methodes de la toolbox
    """Le reseau torch d'anomalib + son pre-traitement : image [0, 1] -> carte brute (N, 1, h, w), sans post-traitement.

    Le post-traitement d'anomalib normalise les scores avec SES seuils ; on le court-circuite pour
    calibrer le seuil nous-memes, sur nos normales de validation, comme pour les autres methodes.
    """

    def __init__(self, anomalib_model) -> None:  # recoit le modele anomalib entraine
        super().__init__()  # initialise nn.Module
        self.pre = anomalib_model.pre_processor  # pretraitement d'anomalib (redimensionnement + normalisation)
        self.net = anomalib_model.model  # le reseau torch nu, sans le post-traitement d'anomalib

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # image [0, 1] -> carte brute (N, 1, h, w)
        if x.shape[1] == 1:  # si l'image est grise (1 canal)
            x = x.repeat(1, 3, 1, 1)  # recopie sur 3 canaux : les reseaux pre-entraines attendent du RGB
        out = self.net(self.pre(x))  # pretraitement puis reseau
        anomaly_map = out.anomaly_map if hasattr(out, "anomaly_map") else out[1]  # carte d'anomalie (champ nomme ou 2e element)
        return anomaly_map if anomaly_map.ndim == 4 else anomaly_map[:, None]  # garantit la forme (N, 1, h, w)
