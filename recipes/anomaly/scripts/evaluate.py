"""Script 3 sur 5 : test, une seule fois : AUROC image et pixel, rappel et fausses alarmes au seuil, rappel par type.

    python recipes/anomaly/scripts/evaluate.py --config recipes/anomaly/configs/patchcore_lite_bottle.yaml

Produit test_metrics.json, scores.png (distribution normales / defauts et seuil) et maps.png.
"""

from vtb.core.config import load_config  # lit le fichier YAML de l'experience et verifie chaque reglage
from vtb.core.experiment import parse_config_path, start  # lecture de --config, puis preparation (dossier, journal, graines)
from vtb.tasks.anomaly import workflow  # les etapes de la recette anomalies (prepare, train, evaluate, predict, export)
from vtb.tasks.anomaly.config import AnomalyConfig  # le modele de configuration : la liste des cles autorisees dans le YAML


def main() -> None:  # point d'entree : evalue le modele entraine sur le jeu de test ; ne renvoie rien
    args = parse_config_path(__doc__.splitlines()[0])  # lit --config ; l'aide reprend la 1re ligne de la docstring
    config = load_config(args.config, AnomalyConfig)  # charge le YAML ; une cle inconnue ou mal typee arrete tout ici
    # cree le dossier de sortie et le journal evaluate.log, choisit CPU ou GPU (device) ; save=False : pas de copie de la config
    output, device = start(config, "evaluate", save=False)
    workflow.evaluate(config, output, device)  # calcule les mesures sur le jeu de test et ecrit test_metrics.json et les figures


if __name__ == "__main__":  # vrai seulement quand on lance le fichier directement (pas quand on l'importe)
    main()  # lance le script
