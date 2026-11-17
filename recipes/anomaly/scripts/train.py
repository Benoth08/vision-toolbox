"""Script 2 sur 5 : construit la methode du YAML (entrainement ou memoire) puis calibre le seuil sur les normales de validation.

    python recipes/anomaly/scripts/train.py --config recipes/anomaly/configs/patchcore_lite_bottle.yaml

Produit model.pt (le modele de production complet), calibration.json (seuils) et train_summary.json.
"""

from vtb.core.config import load_config  # lit le fichier YAML de l'experience et verifie chaque reglage
from vtb.core.experiment import parse_config_path, start  # lecture de --config, puis preparation (dossier, journal, graines)
from vtb.tasks.anomaly import workflow  # les etapes de la recette anomalies (prepare, train, evaluate, predict, export)
from vtb.tasks.anomaly.config import AnomalyConfig  # le modele de configuration : la liste des cles autorisees dans le YAML


def main() -> None:  # point d'entree : construit le detecteur d'anomalies et fixe son seuil ; ne renvoie rien
    args = parse_config_path(__doc__.splitlines()[0])  # lit --config ; l'aide reprend la 1re ligne de la docstring
    config = load_config(args.config, AnomalyConfig)  # charge le YAML ; une cle inconnue ou mal typee arrete tout ici
    # dossier de sortie, journal train.log, copie de la config, graines du hasard fixees ; device = CPU ou GPU
    output, device = start(config, "train")
    # entraine la methode (auto-encodeur) ou remplit sa memoire d'images normales (PatchCore), puis choisit le seuil
    # de decision sur les images normales de validation ; ecrit model.pt, calibration.json et train_summary.json
    workflow.train(config, output, device)


if __name__ == "__main__":  # vrai seulement quand on lance le fichier directement (pas quand on l'importe)
    main()  # lance le script
