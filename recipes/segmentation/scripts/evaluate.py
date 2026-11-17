"""Script 3 sur 5 : test, une seule fois : IoU et Dice par classe, decision par image, galerie.

    python recipes/segmentation/scripts/evaluate.py --config recipes/segmentation/configs/unet_resnet18_tile.yaml

Produit test_metrics.json et test_predictions.png.
"""

from vtb.core.config import load_config  # lit le fichier YAML de l'experience et verifie chaque reglage
from vtb.core.experiment import parse_config_path, start  # lecture de --config, puis preparation (dossier, journal, graines)
from vtb.tasks.segmentation import workflow  # les etapes de la recette segmentation (prepare, train, evaluate, predict, export)
from vtb.tasks.segmentation.config import SegmentationConfig  # modele de configuration : les cles autorisees dans le YAML


def main() -> None:  # point d'entree : evalue le modele de segmentation sur le jeu de test ; ne renvoie rien
    args = parse_config_path(__doc__.splitlines()[0])  # lit --config ; l'aide reprend la 1re ligne de la docstring
    config = load_config(args.config, SegmentationConfig)  # charge le YAML ; une cle inconnue ou mal typee arrete tout ici
    # cree le dossier de sortie et le journal evaluate.log, choisit CPU ou GPU (device) ; save=False : pas de copie de la config
    output, device = start(config, "evaluate", save=False)
    # IoU (surface commune / surface totale) et Dice (recouvrement) par classe de pixels, decision defaut / normal par image
    workflow.evaluate(config, output, device)


if __name__ == "__main__":  # vrai seulement quand on lance le fichier directement (pas quand on l'importe)
    main()  # lance le script
