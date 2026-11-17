"""Script 4 sur 5 : score et decision pour une image ou un dossier ; cartes superposees en option.

    python recipes/anomaly/scripts/predict.py --config recipes/anomaly/configs/patchcore_lite_bottle.yaml \
        --source data/raw/mvtec_bottle/test/broken_large --csv scores.csv --maps outputs/anomaly/maps
"""

import logging  # logging : affiche des messages de suivi (mieux que print dans un script)
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

from vtb.core.config import load_config  # lit le fichier YAML de l'experience et verifie chaque reglage
from vtb.core.experiment import parse_config_path, start  # lecture de --config, puis preparation (dossier, journal, graines)
from vtb.tasks.anomaly import workflow  # les etapes de la recette anomalies (prepare, train, evaluate, predict, export)
from vtb.tasks.anomaly.config import AnomalyConfig  # le modele de configuration : la liste des cles autorisees dans le YAML

logger = logging.getLogger(__name__)  # "journal" propre a ce fichier : logger.info(...) ecrit un message de suivi


def main() -> None:  # point d'entree : score d'anomalie et decision pour de nouvelles images ; ne renvoie rien
    args = parse_config_path(  # lit --config et les arguments en plus listes ci-dessous
        __doc__.splitlines()[0],  # texte d'aide : 1re ligne de la docstring
        [  # arguments supplementaires : (nom, options d'argparse)
            ("--source", {"required": True, "type": Path, "help": "image ou dossier"}),  # image(s) a analyser (obligatoire)
            ("--csv", {"type": Path, "default": None}),  # optionnel : fichier CSV ou enregistrer les scores
            # optionnel : dossier ou ecrire chaque image avec sa carte d'anomalie posee dessus
            ("--maps", {"type": Path, "default": None, "help": "dossier ou ecrire les cartes superposees"}),
        ],
    )
    config = load_config(args.config, AnomalyConfig)  # charge le YAML ; une cle inconnue ou mal typee arrete tout ici
    output, device = start(config, "predict", save=False)  # dossier de sortie, journal predict.log, CPU ou GPU (device)
    # recharge le modele entraine, calcule score et decision (normal / defaut) pour chaque image ; renvoie un tableau
    table = workflow.predict(config, output, device, args.source, args.maps)
    logger.info("\n%s", table.head(12).to_string(index=False))  # affiche les 12 premieres lignes du tableau
    if args.csv:  # si un fichier CSV a ete demande
        table.to_csv(args.csv, index=False)  # ecrit tout le tableau (index=False : sans la colonne de numeros de ligne)


if __name__ == "__main__":  # vrai seulement quand on lance le fichier directement (pas quand on l'importe)
    main()  # lance le script
