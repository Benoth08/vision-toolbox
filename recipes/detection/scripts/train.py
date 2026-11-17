"""Script 2 sur 5 : entraine le moteur du YAML (ultralytics, yolox, fasterrcnn ou rfdetr) sur train, choisit la meilleure epoque sur val.

    python recipes/detection/scripts/train.py --config recipes/detection/configs/yolo11n_neudet.yaml

Produit best.pt, history.csv / history.png et detector.json (classes, couleur, pretraitement).
"""

from vtb.core.config import load_config  # lit le fichier YAML de l'experience et verifie chaque reglage
from vtb.core.experiment import parse_config_path, start  # lecture de --config, puis preparation (dossier, journal, graines)
from vtb.tasks.detection import workflow  # les etapes de la recette detection (prepare, train, evaluate, predict, export)
from vtb.tasks.detection.config import DetectionConfig  # le modele de configuration : la liste des cles autorisees dans le YAML


def main() -> None:  # point d'entree : entraine le detecteur ; ne renvoie rien
    args = parse_config_path(__doc__.splitlines()[0])  # lit --config ; l'aide reprend la 1re ligne de la docstring
    config = load_config(args.config, DetectionConfig)  # charge le YAML ; une cle inconnue ou mal typee arrete tout ici
    # dossier de sortie, journal train.log, copie de la config, graines du hasard fixees ; device = CPU ou GPU
    output, device = start(config, "train")
    # entraine le moteur choisi dans le YAML sur plusieurs epoques (passages complets sur train), garde la meilleure sur val ;
    # ecrit best.pt, history.csv / history.png et detector.json
    workflow.train(config, output, device)


if __name__ == "__main__":  # vrai seulement quand on lance le fichier directement (pas quand on l'importe)
    main()  # lance le script
