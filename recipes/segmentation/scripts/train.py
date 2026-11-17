"""Script 2 sur 5 : entraine le reseau de segmentation sur train, garde l'epoque au meilleur IoU de validation.

    python recipes/segmentation/scripts/train.py --config recipes/segmentation/configs/unet_resnet18_tile.yaml

Produit model.pt (normalisation et softmax inclus), history.csv / history.png, train_summary.json.
"""

from vtb.core.config import load_config  # lit le fichier YAML de l'experience et verifie chaque reglage
from vtb.core.experiment import parse_config_path, start  # lecture de --config, puis preparation (dossier, journal, graines)
from vtb.tasks.segmentation import workflow  # les etapes de la recette segmentation (prepare, train, evaluate, predict, export)
from vtb.tasks.segmentation.config import SegmentationConfig  # modele de configuration : les cles autorisees dans le YAML


def main() -> None:  # point d'entree : entraine le reseau de segmentation ; ne renvoie rien
    args = parse_config_path(__doc__.splitlines()[0])  # lit --config ; l'aide reprend la 1re ligne de la docstring
    config = load_config(args.config, SegmentationConfig)  # charge le YAML ; une cle inconnue ou mal typee arrete tout ici
    # dossier de sortie, journal train.log, copie de la config, graines du hasard fixees ; device = CPU ou GPU
    output, device = start(config, "train")
    # segmentation = une classe par pixel ; entraine sur plusieurs epoques (passages complets sur train) et garde celle
    # au meilleur IoU de validation ; ecrit model.pt, history.csv / history.png et train_summary.json
    workflow.train(config, output, device)


if __name__ == "__main__":  # vrai seulement quand on lance le fichier directement (pas quand on l'importe)
    main()  # lance le script
