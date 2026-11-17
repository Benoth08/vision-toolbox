"""Derive de domaine entre deux dossiers (entrainement / ligne) : AUC du classifieur de domaine et statistiques qui ont bouge.

    python recipes/explore/scripts/drift.py --config recipes/explore/configs/drift_simulated.yaml

Produit drift_report.json (AUC, verdict), statistics_drift.csv, current_novelty.csv (images les plus
"nouvelles"), drift_projection.png et most_novel.png.
"""

from vtb.core.config import load_config  # lit le fichier YAML de l'experience et verifie chaque reglage
from vtb.core.experiment import parse_config_path, start  # lecture de --config, puis preparation (dossier, journal, graines)
from vtb.explore import workflow  # les traitements d'exploration (controle du jeu, plongements, derive)
from vtb.explore.config import DriftConfig  # le modele de configuration de la derive : les cles autorisees dans le YAML


def main() -> None:  # point d'entree : mesure si les images recentes ont change par rapport a la reference ; ne renvoie rien
    args = parse_config_path(__doc__.splitlines()[0])  # lit --config ; l'aide reprend la 1re ligne de la docstring
    config = load_config(args.config, DriftConfig)  # charge le YAML ; une cle inconnue ou mal typee arrete tout ici
    output, device = start(config, "drift")  # dossier de sortie, journal drift.log, copie de la config, graines, CPU ou GPU
    # derive = les images de production ne ressemblent plus a celles de l'entrainement ; un classifieur essaie de
    # distinguer les deux dossiers : AUC ~ 0,5 = indiscernables (pas de derive), proche de 1 = derive nette
    workflow.drift(config, output, device)


if __name__ == "__main__":  # vrai seulement quand on lance le fichier directement (pas quand on l'importe)
    main()  # lance le script
