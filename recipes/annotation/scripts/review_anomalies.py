"""Revue des images signalees par un detecteur d'anomalies (fenetre OpenCV) : g = normal, d = defaut, s = passer, q = quitter.

    python recipes/annotation/scripts/review_anomalies.py --csv outputs/anomaly_anomalib/predictions.csv \
        --out outputs/annotation/revue.csv

Le CSV d'entree contient path et score (et map_path pour superposer la carte) ; les images
sont presentees du score le plus eleve au plus faible. Reprise automatique depuis --out.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[str], X | None) partout

import argparse  # argparse : lit les options passees en ligne de commande (--csv, --out, ...)
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

from vtb.annotate.tools.review import run_review  # la fenetre OpenCV de revue des images signalees
from vtb.core.runtime import setup_logging  # configure l'affichage des messages de suivi sur la console


def main() -> None:  # point d'entree du script : lit les options puis ouvre la fenetre de revue ; ne renvoie rien
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # l'aide reprend la 1re ligne de la docstring
    parser.add_argument("--csv", required=True, type=Path)  # CSV des predictions du detecteur (path, score, map_path)
    parser.add_argument("--out", required=True, type=Path)  # CSV ou sont enregistrees les decisions (normal / defaut)
    # transparence de la carte d'anomalie posee sur l'image : 0 = invisible, 1 = cache l'image
    parser.add_argument("--alpha", type=float, default=0.45, help="opacite de la carte superposee")
    args = parser.parse_args()  # lit la ligne de commande ; args.csv, args.out, args.alpha contiennent les valeurs
    setup_logging()  # active l'affichage des messages de suivi
    run_review(args.csv, args.out, args.alpha)  # ouvre la fenetre et attend les touches jusqu'a q (quitter)


if __name__ == "__main__":  # vrai seulement quand on lance le fichier directement (pas quand on l'importe)
    main()  # lance le script
