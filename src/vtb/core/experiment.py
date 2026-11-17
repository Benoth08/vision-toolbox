"""Demarrage commun de tous les scripts : dossier de sortie, journal, copie de la config, graines, device."""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (ex. list[...] | None)

import argparse  # lit les arguments donnes en ligne de commande (ex. --config fichier.yaml)
import logging  # journal : messages horodates sur la console et dans un fichier
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import torch  # PyTorch : ici pour le type torch.device (CPU ou carte graphique)

from vtb.core.config import StrictModel, save_config  # classe de base des configs et fonction qui les enregistre
from vtb.core.runtime import get_device, seed_everything, setup_logging  # choix du materiel, graines, journal

logger = logging.getLogger(__name__)  # journal propre a ce module


# lit --config (obligatoire) et d'eventuels autres arguments ; renvoie les valeurs lues
def parse_config_path(description: str, extra_args: list[tuple[str, dict]] | None = None) -> argparse.Namespace:
    """`--config chemin.yaml` (obligatoire) et d'eventuels arguments en plus (ex. --source pour predict)."""
    parser = argparse.ArgumentParser(description=description)  # cree le lecteur d'arguments (description affichee par --help)
    parser.add_argument("--config", required=True, type=Path, help="fichier YAML de l'experience")  # argument obligatoire, converti en Path
    for name, kwargs in extra_args or []:  # pour chaque argument supplementaire demande (liste vide si aucun)
        parser.add_argument(name, **kwargs)  # l'ajoute avec ses options (type, aide, valeur par defaut...)
    return parser.parse_args()  # lit la ligne de commande et renvoie un objet avec un attribut par argument


# prepare une experience : dossier de sortie, journal, copie de la config, graines ; renvoie (dossier, device)
def start(config: StrictModel, log_name: str, save: bool = True) -> tuple[Path, torch.device]:
    """Cree output_dir, ouvre le journal <log_name>.log, copie config_used.yaml, fixe les graines, choisit le device."""
    # section runtime de la config (le commentaire "type: ignore" fait taire le verificateur de types)
    runtime = config.runtime  # type: ignore[attr-defined]
    output = Path(runtime.output_dir)  # dossier de sortie sous forme de chemin
    output.mkdir(parents=True, exist_ok=True)  # le cree s'il n'existe pas (avec ses dossiers parents)
    setup_logging(output / f"{log_name}.log")  # journal sur la console et dans <dossier>/<log_name>.log
    if save:  # si on veut garder une trace des reglages utilises
        save_config(config, output / "config_used.yaml")  # copie la config complete a cote des resultats
    seed_everything(runtime.seed, runtime.deterministic)  # fixe les graines aleatoires (resultats reproductibles)
    device = get_device(runtime.device)  # choisit CPU ou GPU selon la config
    logger.info("%s | sortie %s | device %s", log_name, output, device)  # note dans le journal le debut de l'experience
    return output, device  # renvoie le dossier de sortie et le materiel de calcul
