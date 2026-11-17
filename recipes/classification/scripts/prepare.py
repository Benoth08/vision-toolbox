"""Script 1 sur 5 : catalogue des images et decoupage train / val / test, enregistre (une fois par jeu de donnees).

python recipes/classification/scripts/prepare.py --config recipes/classification/configs/resnet18_partial.yaml
"""

from vtb.core.config import load_config  # lit le fichier YAML de l'experience et verifie chaque reglage
from vtb.core.experiment import parse_config_path, start  # lecture de --config, puis preparation (dossier, journal, graines)
from vtb.tasks.classification import workflow  # les etapes de la recette classification (prepare, train, evaluate...)
from vtb.tasks.classification.config import ClassificationConfig  # modele de configuration : les cles autorisees dans le YAML


def main() -> None:  # point d'entree : construit le catalogue et le decoupage train / val / test ; ne renvoie rien
    args = parse_config_path(__doc__.splitlines()[0])  # lit --config ; l'aide reprend la 1re ligne de la docstring
    config = load_config(args.config, ClassificationConfig)  # charge le YAML ; une cle inconnue ou mal typee arrete tout ici
    start(config, "prepare")  # cree le dossier de sortie, le journal prepare.log, copie la config, fixe les graines du hasard
    # liste les images et leur classe (catalogue), puis les repartit en train / val / test et enregistre split.csv
    workflow.prepare(config)


if __name__ == "__main__":  # vrai seulement quand on lance le fichier directement (pas quand on l'importe)
    main()  # lance le script
