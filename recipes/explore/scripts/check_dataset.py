"""Controle d'un jeu d'images : illisibles, tailles, doublons, fuites entre sous-ensembles, raccourcis, images atypiques.

    python recipes/explore/scripts/check_dataset.py --config recipes/explore/configs/neu_cls.yaml

Produit quality_report.md (alertes en tete), quality_report.json, image_stats.csv, duplicates.csv,
outliers.csv, shortcuts.csv, stats_by_class.png, duplicates.png. A lancer avant tout entrainement.
"""

from vtb.core.config import load_config  # lit le fichier YAML de l'experience et verifie chaque reglage
from vtb.core.experiment import parse_config_path, start  # lecture de --config, puis preparation (dossier, journal, graines)
from vtb.explore import workflow  # les traitements d'exploration (controle du jeu, plongements, derive)
from vtb.explore.config import ExploreConfig  # le modele de configuration : la liste des cles autorisees dans le YAML


def main() -> None:  # point d'entree : controle la qualite du jeu d'images ; ne renvoie rien
    args = parse_config_path(__doc__.splitlines()[0])  # lit --config ; l'aide reprend la 1re ligne de la docstring
    config = load_config(args.config, ExploreConfig)  # charge le YAML ; une cle inconnue ou mal typee arrete tout ici
    # dossier de sortie, journal check_dataset.log, copie de la config, graines ; le device est ignore (_) : pas de reseau ici
    output, _ = start(config, "check_dataset")
    # cherche les fichiers illisibles, les doublons, les fuites (meme image dans train et test), les raccourcis
    # (statistique simple qui devine la classe) et les images atypiques ; ecrit le rapport et les figures
    workflow.check_dataset(config, output)


if __name__ == "__main__":  # vrai seulement quand on lance le fichier directement (pas quand on l'importe)
    main()  # lance le script
