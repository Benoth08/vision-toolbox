"""Banc de comparaison des methodes deja lancees : memes donnees, meme decoupage, meme protocole de seuil.

python recipes/anomaly/scripts/benchmark.py --outputs outputs/anomaly --markdown docs/assets/anomaly/benchmark.md
"""

import argparse  # argparse : lit les options passees en ligne de commande (--outputs, --markdown)
import json  # json : relit les fichiers de resultats .json ecrits par les autres scripts
import logging  # logging : affiche des messages de suivi (mieux que print dans un script)
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import pandas as pd  # pandas : tableaux de donnees (le tableau comparatif des methodes)

from vtb.core.runtime import setup_logging  # configure l'affichage des messages de suivi sur la console

logger = logging.getLogger(__name__)  # "journal" propre a ce fichier : logger.info(...) ecrit un message de suivi


def collect(root: Path) -> pd.DataFrame:  # rassemble les resultats de chaque experience ; renvoie un tableau, une ligne par experience
    rows = []  # une ligne (dict) par experience trouvee
    for folder in sorted(  # pour chaque sous-dossier d'experience, par ordre alphabetique
        p  # l'element garde : le chemin du sous-dossier
        for p in root.iterdir()  # parcourt tout ce que contient le dossier racine
        if (p / "test_metrics.json").exists() and p.name != "notebook"  # seulement les experiences evaluees, sauf "notebook"
    ):  # le notebook refait une experience : on ne le compte pas dans le banc
        test = json.loads((folder / "test_metrics.json").read_text())  # mesures sur le jeu de test (ecrites par evaluate.py)
        # resume de l'entrainement (ecrit par train.py), ou dict vide si le fichier manque
        summary = json.loads((folder / "train_summary.json").read_text()) if (folder / "train_summary.json").exists() else {}
        # carte du modele (ecrite par export.py : latences, seuils...), ou dict vide si le modele n'a pas ete exporte
        card = json.loads((folder / "model_card.json").read_text()) if (folder / "model_card.json").exists() else {}
        rows.append(  # ajoute la ligne de cette experience
            {  # une ligne = un dict {nom de colonne: valeur}
                "experience": folder.name,  # nom du dossier de l'experience
                "methode": summary.get("name", ""),  # methode utilisee ; .get(cle, defaut) evite une erreur si la cle manque
                # AUROC image : probabilite qu'une image defectueuse ait un score plus haut qu'une normale (1 = parfait, 0,5 = hasard)
                "auroc_image": round(test.get("auroc_image", float("nan")), 3),
                "auroc_pixel": round(test.get("auroc_pixel", float("nan")), 3),  # meme mesure pixel par pixel (nan = non mesuree)
                "rappel_au_seuil": round(test["recall"], 3),  # rappel : part des images defectueuses detectees au seuil choisi
                "fausses_alarmes": round(test["false_alarm_rate"], 3),  # part des images normales signalees a tort
                "construction_s": summary.get("fit_seconds", ""),  # duree de construction / d'entrainement en secondes
                "ms_par_image_torch": test.get("ms_per_image", ""),  # temps de calcul par image avec PyTorch (millisecondes)
                # temps par image du modele exporte en ONNX (format standard de modele) avec onnxruntime sur CPU
                "onnx_ms": card.get("latency_ms", {}).get("onnxruntime_cpu_ms", ""),
            }
        )
    return pd.DataFrame(rows)  # transforme la liste de lignes en tableau pandas


def main() -> None:  # point d'entree du script : construit le tableau, l'affiche et l'ecrit en Markdown si demande
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # l'aide reprend la 1re ligne de la docstring
    parser.add_argument("--outputs", type=Path, default=Path("outputs/anomaly"))  # dossier qui contient un sous-dossier par experience
    parser.add_argument("--markdown", type=Path, default=None)  # optionnel : fichier .md ou ecrire le tableau
    args = parser.parse_args()  # lit la ligne de commande ; args.outputs et args.markdown contiennent les valeurs
    setup_logging()  # active l'affichage des messages de suivi
    table = collect(args.outputs)  # tableau comparatif de toutes les experiences
    logger.info("\n%s", table.to_string(index=False))  # l'affiche en texte (sans numeros de ligne)
    if args.markdown:  # si un fichier Markdown a ete demande
        args.markdown.parent.mkdir(parents=True, exist_ok=True)  # cree son dossier s'il n'existe pas
        args.markdown.write_text(table.to_markdown(index=False), encoding="utf-8")  # ecrit le tableau au format Markdown


if __name__ == "__main__":  # vrai seulement quand on lance le fichier directement (pas quand on l'importe)
    main()  # lance le script
