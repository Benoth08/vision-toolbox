"""Script 4 sur 5 : prediction sur une image ou un dossier ; le pretraitement vient de model.pt.

    python recipes/classification/scripts/predict.py --config recipes/classification/configs/resnet18_partial.yaml \
        --source chemin/dossier --csv predictions.csv
"""

import logging  # logging : affiche des messages de suivi (mieux que print dans un script)
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

from vtb.core.config import load_config  # lit le fichier YAML de l'experience et verifie chaque reglage
from vtb.core.experiment import parse_config_path, start  # lecture de --config, puis preparation (dossier, journal, graines)
from vtb.tasks.classification import workflow  # les etapes de la recette classification (prepare, train, evaluate...)
from vtb.tasks.classification.config import ClassificationConfig  # modele de configuration : les cles autorisees dans le YAML

logger = logging.getLogger(__name__)  # "journal" propre a ce fichier : logger.info(...) ecrit un message de suivi


def main() -> None:  # point d'entree : predit la classe de nouvelles images ; ne renvoie rien
    args = parse_config_path(  # lit --config et les arguments en plus listes ci-dessous
        __doc__.splitlines()[0],  # texte d'aide : 1re ligne de la docstring
        [  # arguments supplementaires : (nom, options d'argparse)
            ("--source", {"required": True, "type": Path, "help": "image ou dossier"}),  # image(s) a classer (obligatoire)
            ("--csv", {"type": Path, "default": None}),  # optionnel : fichier CSV ou enregistrer les predictions
        ],
    )
    config = load_config(args.config, ClassificationConfig)  # charge le YAML ; une cle inconnue ou mal typee arrete tout ici
    output, device = start(config, "predict", save=False)  # dossier de sortie, journal predict.log, CPU ou GPU (device)
    # recharge model.pt (poids + pretraitement + classes) et predit chaque image ; renvoie un tableau (image, classe, probabilite)
    predictions = workflow.predict(config, output, device, args.source)
    logger.info("\n%s", predictions.head(12).to_string(index=False))  # affiche les 12 premieres predictions
    if args.csv:  # si un fichier CSV a ete demande
        predictions.to_csv(args.csv, index=False)  # ecrit toutes les predictions (sans la colonne de numeros de ligne)
        logger.info("%d predictions -> %s", len(predictions), args.csv)  # indique combien et ou


if __name__ == "__main__":  # vrai seulement quand on lance le fichier directement (pas quand on l'importe)
    main()  # lance le script
