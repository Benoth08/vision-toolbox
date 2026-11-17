"""Ce que tous les scripts font en premier : graines, device, journal, chronometre.

Ce module n'importe rien du projet : il est tout en bas de la pile des dependances.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (ex. Path | None)

import logging  # journal : messages horodates sur la console et dans un fichier
import os  # acces au systeme (ici : variables d'environnement)
import random  # generateur aleatoire standard de Python
import time  # mesure du temps (chronometre)
from collections.abc import Iterator  # type "objet qu'on parcourt element par element" (pour annoter le chronometre)
from contextlib import contextmanager  # transforme une fonction en bloc utilisable avec "with"
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)
import torch  # PyTorch : tenseurs (tableaux de nombres sur CPU ou GPU) et reseaux de neurones

logger = logging.getLogger(__name__)  # journal propre a ce module (son nom apparait dans les messages)

# bibliotheques qui ecrivent beaucoup de messages : on les fera taire (niveau WARNING)
NOISY_LIBRARIES = ("onnxscript", "onnx_ir", "torch.onnx", "lightning", "lightning.pytorch", "anomalib", "ultralytics", "PIL", "matplotlib")


def seed_everything(seed: int, deterministic: bool = False) -> None:  # fixe toutes les graines aleatoires pour des resultats reproductibles
    """Fixe les quatre generateurs aleatoires ; deterministic force aussi les algorithmes GPU reproductibles."""
    random.seed(seed)  # graine du hasard de Python (melanges, tirages)
    np.random.seed(seed)  # graine du hasard de numpy
    torch.manual_seed(seed)  # graine du hasard de PyTorch sur CPU (poids initiaux, dropout...)
    torch.cuda.manual_seed_all(seed)  # graine du hasard de PyTorch sur toutes les cartes graphiques
    if deterministic:  # si on exige des resultats strictement identiques d'une execution a l'autre
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # reglage demande par CUDA pour des calculs matriciels reproductibles
        torch.use_deterministic_algorithms(True, warn_only=True)  # impose les algorithmes reproductibles (simple avertissement sinon)
        torch.backends.cudnn.benchmark = False  # desactive le choix automatique de l'algorithme le plus rapide (il varie)


def seed_worker(worker_id: int) -> None:  # donne une graine differente a chaque processus de chargement des images
    """Graine propre a chaque worker de DataLoader (sinon les augmentations se repetent entre workers)."""
    worker_seed = torch.initial_seed() % 2**32  # graine deja propre au worker, ramenee a la plage acceptee par numpy
    np.random.seed(worker_seed)  # applique cette graine a numpy dans ce worker
    random.seed(worker_seed)  # et au hasard standard de Python


def make_generator(seed: int) -> torch.Generator:  # cree un generateur aleatoire PyTorch avec une graine donnee
    """Generateur dedie au melange du DataLoader : meme ordre d'une execution a l'autre."""
    return torch.Generator().manual_seed(seed)  # renvoie le generateur deja initialise avec la graine


def get_device(requested: str = "auto") -> torch.device:  # choisit ou faire les calculs (CPU ou carte graphique) et le renvoie
    """auto : GPU si disponible, sinon CPU. Le reste du code ne fait que .to(device)."""
    if requested == "auto":  # si l'utilisateur laisse le choix
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU NVIDIA si present, sinon processeur
    device = torch.device(requested)  # sinon on prend le materiel demande (ex. "cpu", "cuda:0")
    if device.type == "cuda" and not torch.cuda.is_available():  # si on demande un GPU alors qu'il n'y en a pas
        raise RuntimeError(f"device {requested} demande mais aucun GPU CUDA n'est disponible")  # on arrete avec un message clair
    return device  # renvoie le materiel choisi


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> None:  # configure le journal (console + fichier optionnel)
    """Journal sur la console et, si demande, dans un fichier a cote des resultats."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]  # destinations des messages : d'abord la console
    if log_file is not None:  # si un fichier de journal est demande
        log_file.parent.mkdir(parents=True, exist_ok=True)  # cree son dossier s'il n'existe pas
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))  # ajoute le fichier comme seconde destination
    # format : date | niveau | message ; force=True remplace une configuration deja faite (utile dans un notebook)
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s", handlers=handlers, force=True)
    quiet_libraries()  # fait taire les bibliotheques trop bavardes


def quiet_libraries() -> None:  # baisse le volume des bibliotheques bavardes
    """Ramene au niveau WARNING les bibliotheques bavardes (a rappeler apres leur import : certaines le remettent a INFO)."""
    for name in NOISY_LIBRARIES:  # pour chaque bibliotheque de la liste
        logging.getLogger(name).setLevel(logging.WARNING)  # n'affiche plus que ses avertissements et erreurs


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:  # compte les poids du modele et renvoie ce nombre
    """Nombre de parametres (entrainables par defaut) : la preuve chiffree qu'un gel a pris effet."""
    # numel = nombre de valeurs d'un tenseur ; requires_grad = le poids sera modifie a l'entrainement (non gele)
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


@contextmanager  # rend la fonction utilisable dans un bloc "with"
def timer(label: str, log: bool = True) -> Iterator[dict[str, float]]:  # chronometre un bloc de code ; fournit un dict avec la duree
    """Chronometre : `with timer("train") as t: ...` puis t["seconds"]."""
    result: dict[str, float] = {}  # dict rempli a la fin avec la duree en secondes
    start = time.perf_counter()  # heure de depart (horloge precise)
    try:  # on execute le bloc "with" ; le "finally" s'execute meme en cas d'erreur
        yield result  # rend la main au bloc "with" (qui recoit result) ; reprend ici a sa sortie
    finally:  # a la sortie du bloc, qu'il ait reussi ou non
        result["seconds"] = time.perf_counter() - start  # duree ecoulee = maintenant - depart
        if log:  # si on veut afficher la duree dans le journal
            logger.info("%s : %.1f s", label, result["seconds"])  # ex. "train : 12.3 s"
