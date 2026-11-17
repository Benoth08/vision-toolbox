"""Script 1 sur 5 : catalogue, etiquettes good / defect, decoupage (aucun defaut dans train ni val).

python recipes/anomaly/scripts/prepare.py --config recipes/anomaly/configs/patchcore_lite_bottle.yaml
"""

from vtb.core.config import load_config  # lit le fichier YAML de l'experience et verifie chaque reglage
from vtb.core.experiment import parse_config_path, start  # lecture de --config, puis preparation (dossier, journal, graines)
from vtb.tasks.anomaly import workflow  # les etapes de la recette anomalies (prepare, train, evaluate, predict, export)
from vtb.tasks.anomaly.config import AnomalyConfig  # le modele de configuration : la liste des cles autorisees dans le YAML


def main() -> None:  # point d'entree : construit le catalogue et le decoupage train / val / test ; ne renvoie rien
    args = parse_config_path(__doc__.splitlines()[0])  # lit --config ; l'aide reprend la 1re ligne de la docstring
    config = load_config(args.config, AnomalyConfig)  # charge le YAML ; une cle inconnue ou mal typee arrete tout ici
    start(config, "prepare")  # cree le dossier de sortie, le journal prepare.log, copie la config, fixe les graines du hasard
    # liste les images (catalogue), les etiquette normale / defaut, puis les repartit en train / val / test (split.csv) ;
    # les defauts vont tous en test : la methode n'apprend que sur des images normales
    workflow.prepare(config)


if __name__ == "__main__":  # vrai seulement quand on lance le fichier directement (pas quand on l'importe)
    main()  # lance le script
