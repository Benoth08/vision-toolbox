"""Script 3 sur 5 : test, une seule fois : mAP COCO commun aux moteurs, precision / rappel au seuil de production, galerie.

    python recipes/detection/scripts/evaluate.py --config recipes/detection/configs/yolo11n_neudet.yaml

Produit test_metrics.json, per_class.csv et test_predictions.png.
"""

from vtb.core.config import load_config  # lit le fichier YAML de l'experience et verifie chaque reglage
from vtb.core.experiment import parse_config_path, start  # lecture de --config, puis preparation (dossier, journal, graines)
from vtb.tasks.detection import workflow  # les etapes de la recette detection (prepare, train, evaluate, predict, export)
from vtb.tasks.detection.config import DetectionConfig  # le modele de configuration : la liste des cles autorisees dans le YAML


def main() -> None:  # point d'entree : evalue le detecteur entraine sur le jeu de test ; ne renvoie rien
    args = parse_config_path(__doc__.splitlines()[0])  # lit --config ; l'aide reprend la 1re ligne de la docstring
    config = load_config(args.config, DetectionConfig)  # charge le YAML ; une cle inconnue ou mal typee arrete tout ici
    # cree le dossier de sortie et le journal evaluate.log, choisit CPU ou GPU (device) ; save=False : pas de copie de la config
    output, device = start(config, "evaluate", save=False)
    # mAP (precision moyenne sur toute la courbe precision-rappel) + precision / rappel au seuil de production
    workflow.evaluate(config, output, device)


if __name__ == "__main__":  # vrai seulement quand on lance le fichier directement (pas quand on l'importe)
    main()  # lance le script
