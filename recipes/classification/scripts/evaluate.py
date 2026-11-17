"""Script 3 sur 5 : evaluation sur le test, une seule fois, avec le modele sauvegarde.

    python recipes/classification/scripts/evaluate.py --config recipes/classification/configs/resnet18_partial.yaml

Produit test_metrics.json (accuracy, balanced accuracy, F1 macro, detail par classe),
classification_report.txt, confusion_matrix.png et test_errors.png (les images mal classees).
"""

from vtb.core.config import load_config  # lit le fichier YAML de l'experience et verifie chaque reglage
from vtb.core.experiment import parse_config_path, start  # lecture de --config, puis preparation (dossier, journal, graines)
from vtb.tasks.classification import workflow  # les etapes de la recette classification (prepare, train, evaluate...)
from vtb.tasks.classification.config import ClassificationConfig  # modele de configuration : les cles autorisees dans le YAML


def main() -> None:  # point d'entree : evalue le modele entraine sur le jeu de test ; ne renvoie rien
    args = parse_config_path(__doc__.splitlines()[0])  # lit --config ; l'aide reprend la 1re ligne de la docstring
    config = load_config(args.config, ClassificationConfig)  # charge le YAML ; une cle inconnue ou mal typee arrete tout ici
    # cree le dossier de sortie et le journal evaluate.log, choisit CPU ou GPU (device) ; save=False : pas de copie de la config
    output, device = start(config, "evaluate", save=False)
    workflow.evaluate(config, output, device)  # recharge model.pt, predit le test, ecrit les mesures et les figures


if __name__ == "__main__":  # vrai seulement quand on lance le fichier directement (pas quand on l'importe)
    main()  # lance le script
