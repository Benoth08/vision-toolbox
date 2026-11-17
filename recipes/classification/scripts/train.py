"""Script 2 sur 5 : entrainement selon la strategie du YAML (features, head, partial, full, progressive).

    python recipes/classification/scripts/train.py --config recipes/classification/configs/resnet18_partial.yaml

Ne lit que train et val. Produit dans runtime.output_dir : config_used.yaml, train.log,
history.csv, curves.png, best.pt, model.pt (poids + pretraitement + classes), val_metrics.json.
"""

from vtb.core.config import load_config  # lit le fichier YAML de l'experience et verifie chaque reglage
from vtb.core.experiment import parse_config_path, start  # lecture de --config, puis preparation (dossier, journal, graines)
from vtb.tasks.classification import workflow  # les etapes de la recette classification (prepare, train, evaluate...)
from vtb.tasks.classification.config import ClassificationConfig  # modele de configuration : les cles autorisees dans le YAML


def main() -> None:  # point d'entree : entraine le classifieur ; ne renvoie rien
    args = parse_config_path(__doc__.splitlines()[0])  # lit --config ; l'aide reprend la 1re ligne de la docstring
    config = load_config(args.config, ClassificationConfig)  # charge le YAML ; une cle inconnue ou mal typee arrete tout ici
    # dossier de sortie, journal train.log, copie de la config, graines du hasard fixees ; device = CPU ou GPU
    output, device = start(config, "train")
    # part d'un reseau pre-entraine et l'adapte aux classes du jeu sur plusieurs epoques (passages complets sur train) ;
    # garde le modele de la meilleure epoque sur val ; ecrit history.csv, curves.png, best.pt, model.pt, val_metrics.json
    workflow.train(config, output, device)


if __name__ == "__main__":  # vrai seulement quand on lance le fichier directement (pas quand on l'importe)
    main()  # lance le script
