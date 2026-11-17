"""Plongements d'un jeu d'images : projection 2D, groupes k-means, galerie par groupe, exemple de voisins.

    python recipes/explore/scripts/embeddings.py --config recipes/explore/configs/neu_cls.yaml

Usage typique : comprendre la structure d'un jeu, pre-annoter par groupe, reperer deux acquisitions melangees.
"""

from vtb.core.config import load_config  # lit le fichier YAML de l'experience et verifie chaque reglage
from vtb.core.experiment import parse_config_path, start  # lecture de --config, puis preparation (dossier, journal, graines)
from vtb.explore import workflow  # les traitements d'exploration (controle du jeu, plongements, derive)
from vtb.explore.config import ExploreConfig  # le modele de configuration : la liste des cles autorisees dans le YAML


def main() -> None:  # point d'entree : calcule et visualise les plongements du jeu ; ne renvoie rien
    args = parse_config_path(__doc__.splitlines()[0])  # lit --config ; l'aide reprend la 1re ligne de la docstring
    config = load_config(args.config, ExploreConfig)  # charge le YAML ; une cle inconnue ou mal typee arrete tout ici
    output, device = start(config, "embeddings")  # dossier de sortie, journal embeddings.log, copie de la config, CPU ou GPU
    # plongement = vecteur de nombres qui resume une image (sortie d'un reseau pre-entraine sans sa tete) ;
    # projette les vecteurs en 2D, les regroupe (k-means) et dessine une galerie par groupe
    workflow.embeddings(config, output, device)


if __name__ == "__main__":  # vrai seulement quand on lance le fichier directement (pas quand on l'importe)
    main()  # lance le script
