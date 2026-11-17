"""Script 4 sur 5 : detecte les objets d'une image ou d'un dossier ; une ligne par objet dans le CSV, images annotees en option.

    python recipes/detection/scripts/predict.py --config recipes/detection/configs/yolo11n_neudet.yaml \
        --source data/raw/neu_det/Validation_Images --csv detections.csv --images outputs/detection/predictions
"""

import logging  # logging : affiche des messages de suivi (mieux que print dans un script)
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

from vtb.core.config import load_config  # lit le fichier YAML de l'experience et verifie chaque reglage
from vtb.core.experiment import parse_config_path, start  # lecture de --config, puis preparation (dossier, journal, graines)
from vtb.tasks.detection import workflow  # les etapes de la recette detection (prepare, train, evaluate, predict, export)
from vtb.tasks.detection.config import DetectionConfig  # le modele de configuration : la liste des cles autorisees dans le YAML

logger = logging.getLogger(__name__)  # "journal" propre a ce fichier : logger.info(...) ecrit un message de suivi


def main() -> None:  # point d'entree : detecte les objets de nouvelles images ; ne renvoie rien
    args = parse_config_path(  # lit --config et les arguments en plus listes ci-dessous
        __doc__.splitlines()[0],  # texte d'aide : 1re ligne de la docstring
        [  # arguments supplementaires : (nom, options d'argparse)
            ("--source", {"required": True, "type": Path, "help": "image ou dossier"}),  # image(s) a analyser (obligatoire)
            ("--csv", {"type": Path, "default": None}),  # optionnel : fichier CSV des detections (une ligne par objet)
            # optionnel : dossier ou ecrire chaque image avec ses boites dessinees
            ("--images", {"type": Path, "default": None, "help": "dossier ou ecrire les images annotees"}),
        ],
    )
    config = load_config(args.config, DetectionConfig)  # charge le YAML ; une cle inconnue ou mal typee arrete tout ici
    output, device = start(config, "predict", save=False)  # dossier de sortie, journal predict.log, CPU ou GPU (device)
    # recharge le detecteur entraine et detecte les objets ; renvoie un tableau (image, classe, score, coordonnees de la boite)
    table = workflow.predict(config, output, device, args.source, args.images)
    logger.info("\n%s", table.head(15).to_string(index=False))  # affiche les 15 premieres detections
    if args.csv:  # si un fichier CSV a ete demande
        table.to_csv(args.csv, index=False)  # ecrit toutes les detections (sans la colonne de numeros de ligne)


if __name__ == "__main__":  # vrai seulement quand on lance le fichier directement (pas quand on l'importe)
    main()  # lance le script
