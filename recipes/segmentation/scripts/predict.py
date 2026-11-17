"""Script 4 sur 5 : masque de defaut d'une image ou d'un dossier ; surfaces dans le CSV, masques et superpositions en option.

    python recipes/segmentation/scripts/predict.py --config recipes/segmentation/configs/unet_resnet18_tile.yaml \
        --source data/raw/magnetic_tile/MT_Crack/Imgs --csv surfaces.csv --masks outputs/segmentation/masks
"""

import logging  # logging : affiche des messages de suivi (mieux que print dans un script)
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

from vtb.core.config import load_config  # lit le fichier YAML de l'experience et verifie chaque reglage
from vtb.core.experiment import parse_config_path, start  # lecture de --config, puis preparation (dossier, journal, graines)
from vtb.tasks.segmentation import workflow  # les etapes de la recette segmentation (prepare, train, evaluate, predict, export)
from vtb.tasks.segmentation.config import SegmentationConfig  # modele de configuration : les cles autorisees dans le YAML

logger = logging.getLogger(__name__)  # "journal" propre a ce fichier : logger.info(...) ecrit un message de suivi


def main() -> None:  # point d'entree : predit le masque de defaut de nouvelles images ; ne renvoie rien
    args = parse_config_path(  # lit --config et les arguments en plus listes ci-dessous
        __doc__.splitlines()[0],  # texte d'aide : 1re ligne de la docstring
        [  # arguments supplementaires : (nom, options d'argparse)
            ("--source", {"required": True, "type": Path, "help": "image ou dossier"}),  # image(s) a analyser (obligatoire)
            ("--csv", {"type": Path, "default": None}),  # optionnel : fichier CSV des surfaces de defaut par image
            # optionnel : dossier ou ecrire les masques (un numero de classe par pixel) et les images superposees
            ("--masks", {"type": Path, "default": None, "help": "dossier ou ecrire masques (indices) et superpositions"}),
        ],
    )
    config = load_config(args.config, SegmentationConfig)  # charge le YAML ; une cle inconnue ou mal typee arrete tout ici
    output, device = start(config, "predict", save=False)  # dossier de sortie, journal predict.log, CPU ou GPU (device)
    # recharge le modele entraine et predit un masque par image ; renvoie un tableau (image, surface de defaut par classe...)
    table = workflow.predict(config, output, device, args.source, args.masks)
    logger.info("\n%s", table.head(12).to_string(index=False))  # affiche les 12 premieres lignes du tableau
    if args.csv:  # si un fichier CSV a ete demande
        table.to_csv(args.csv, index=False)  # ecrit tout le tableau (sans la colonne de numeros de ligne)


if __name__ == "__main__":  # vrai seulement quand on lance le fichier directement (pas quand on l'importe)
    main()  # lance le script
