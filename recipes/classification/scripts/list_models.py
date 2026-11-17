"""Le catalogue des modeles pre-entraines : parametres, precision ImageNet, latence CPU mesuree ici.

python recipes/classification/scripts/list_models.py --latency --size 224
"""

import argparse  # argparse : lit les options passees en ligne de commande (--latency, --size, --markdown)
import logging  # logging : affiche des messages de suivi (mieux que print dans un script)

from vtb.core.runtime import setup_logging  # configure l'affichage des messages de suivi sur la console
from vtb.models.zoo import list_models  # le "zoo" : la liste des reseaux pre-entraines disponibles dans la toolbox

logger = logging.getLogger(__name__)  # "journal" propre a ce fichier : logger.info(...) ecrit un message de suivi


def main() -> None:  # point d'entree : affiche le catalogue des modeles (et l'ecrit en Markdown si demande)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # l'aide reprend la 1re ligne de la docstring
    # store_true : l'option ne prend pas de valeur ; sa presence suffit a la mettre a True
    parser.add_argument("--latency", action="store_true", help="mesure la latence CPU (quelques secondes par modele)")
    parser.add_argument("--size", type=int, default=224)  # taille des images de test (pixels) pour mesurer la latence
    parser.add_argument("--markdown", default=None)  # optionnel : fichier .md ou ecrire le tableau
    args = parser.parse_args()  # lit la ligne de commande ; args.latency, args.size... contiennent les valeurs
    setup_logging()  # active l'affichage des messages de suivi
    # tableau des modeles : nombre de parametres (poids), precision sur ImageNet (grand jeu public), latence si demandee
    table = list_models(with_latency=args.latency, size=args.size)
    logger.info("\n%s", table.to_string(index=False))  # affiche le tableau en texte (sans numeros de ligne)
    if args.markdown:  # si un fichier Markdown a ete demande
        with open(args.markdown, "w", encoding="utf-8") as handle:  # ouvre le fichier en ecriture ; with le referme a la fin
            handle.write(table.to_markdown(index=False))  # ecrit le tableau au format Markdown


if __name__ == "__main__":  # vrai seulement quand on lance le fichier directement (pas quand on l'importe)
    main()  # lance le script
