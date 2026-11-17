"""Script 1 sur 5 : lit les annotations (VOC, YOLO, COCO, LabelMe, Label Studio, CVAT, masques), decoupe par image -> split.csv.

    python recipes/detection/scripts/prepare.py --config recipes/detection/configs/yolo11n_neudet.yaml

Le pivot split.vtb.json garde toutes les annotations dans un seul format : les quatre moteurs le lisent.
"""

from vtb.core.config import load_config  # lit le fichier YAML de l'experience et verifie chaque reglage
from vtb.core.experiment import parse_config_path, start  # lecture de --config, puis preparation (dossier, journal, graines)
from vtb.tasks.detection import workflow  # les etapes de la recette detection (prepare, train, evaluate, predict, export)
from vtb.tasks.detection.config import DetectionConfig  # le modele de configuration : la liste des cles autorisees dans le YAML


def main() -> None:  # point d'entree : lit les annotations et decoupe le jeu ; ne renvoie rien
    args = parse_config_path(__doc__.splitlines()[0])  # lit --config ; l'aide reprend la 1re ligne de la docstring
    config = load_config(args.config, DetectionConfig)  # charge le YAML ; une cle inconnue ou mal typee arrete tout ici
    start(config, "prepare", save=False)  # cree le dossier de sortie et le journal prepare.log, fixe les graines du hasard
    # convertit les annotations au format pivot (format interne commun), decoupe train / val / test par image,
    # ecrit split.csv et split.vtb.json
    workflow.prepare(config)


if __name__ == "__main__":  # vrai seulement quand on lance le fichier directement (pas quand on l'importe)
    main()  # lance le script
