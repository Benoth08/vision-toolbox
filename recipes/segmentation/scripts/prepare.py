"""Script 1 sur 5 : catalogue image + masque et decoupage stratifie -> split.csv.

python recipes/segmentation/scripts/prepare.py --config recipes/segmentation/configs/unet_resnet18_tile.yaml
"""

from vtb.core.config import load_config  # lit le fichier YAML de l'experience et verifie chaque reglage
from vtb.core.experiment import parse_config_path, start  # lecture de --config, puis preparation (dossier, journal, graines)
from vtb.tasks.segmentation import workflow  # les etapes de la recette segmentation (prepare, train, evaluate, predict, export)
from vtb.tasks.segmentation.config import SegmentationConfig  # modele de configuration : les cles autorisees dans le YAML


def main() -> None:  # point d'entree : construit le catalogue et le decoupage train / val / test ; ne renvoie rien
    args = parse_config_path(__doc__.splitlines()[0])  # lit --config ; l'aide reprend la 1re ligne de la docstring
    config = load_config(args.config, SegmentationConfig)  # charge le YAML ; une cle inconnue ou mal typee arrete tout ici
    start(config, "prepare", save=False)  # cree le dossier de sortie et le journal prepare.log, fixe les graines du hasard
    # associe chaque image a son masque (catalogue), puis decoupe en train / val / test en gardant les memes
    # proportions de classes dans chaque partie (stratifie) ; enregistre split.csv
    workflow.prepare(config)


if __name__ == "__main__":  # vrai seulement quand on lance le fichier directement (pas quand on l'importe)
    main()  # lance le script
