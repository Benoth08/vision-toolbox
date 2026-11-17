"""Tableau comparatif des experiences deja lancees : strategie, epoques, temps, metriques de test.

python recipes/classification/scripts/compare.py --outputs outputs/classification --markdown docs/assets/classification/comparison.md
"""

import argparse  # argparse : lit les options passees en ligne de commande (--outputs, --markdown)
import json  # json : relit les fichiers de resultats .json ecrits par les autres scripts
import logging  # logging : affiche des messages de suivi (mieux que print dans un script)
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import pandas as pd  # pandas : tableaux de donnees (le tableau comparatif des experiences)
import yaml  # yaml : relit la configuration utilisee (config_used.yaml) pour connaitre le modele

from vtb.core.runtime import setup_logging  # configure l'affichage des messages de suivi sur la console

logger = logging.getLogger(__name__)  # "journal" propre a ce fichier : logger.info(...) ecrit un message de suivi


def collect(root: Path) -> pd.DataFrame:  # rassemble les resultats de chaque experience ; renvoie un tableau, une ligne par experience
    rows = []  # une ligne (dict) par experience trouvee
    for folder in sorted(  # pour chaque sous-dossier d'experience, par ordre alphabetique
        p  # l'element garde : le chemin du sous-dossier
        for p in root.iterdir()  # parcourt tout ce que contient le dossier racine
        if (p / "test_metrics.json").exists() and p.name != "notebook"  # seulement les experiences evaluees, sauf "notebook"
    ):  # le notebook refait une experience : on ne le compte pas dans le tableau
        # mesures de validation et infos d'entrainement (ecrites par train.py), ou dict vide si le fichier manque
        val = json.loads((folder / "val_metrics.json").read_text()) if (folder / "val_metrics.json").exists() else {}
        test = json.loads((folder / "test_metrics.json").read_text())  # mesures sur le jeu de test (ecrites par evaluate.py)
        # carte du modele (ecrite par export.py : latences...), ou dict vide si le modele n'a pas ete exporte
        card = json.loads((folder / "model_card.json").read_text()) if (folder / "model_card.json").exists() else {}
        # configuration utilisee (copiee au lancement), ou dict vide ; safe_load lit le YAML sans executer de code
        used = yaml.safe_load((folder / "config_used.yaml").read_text()) if (folder / "config_used.yaml").exists() else {}
        rows.append(  # ajoute la ligne de cette experience
            {  # une ligne = un dict {nom de colonne: valeur}
                "experience": folder.name,  # nom du dossier de l'experience
                "modele": used.get("model", {}).get("name", ""),  # nom du reseau ; .get(cle, defaut) evite une erreur si absent
                # strategie de transfert : quelles couches du modele pre-entraine sont reentrainees
                "strategie": val.get("strategy", ""),
                "params_entraines": val.get("trainable_params", ""),  # nombre de poids reellement modifies par l'entrainement
                "meilleure_epoque": val.get("best_epoch", ""),  # epoque (= un passage complet sur train) du meilleur score val
                "temps_s": val.get("train_seconds", ""),  # duree d'entrainement en secondes
                "test_accuracy": round(test["accuracy"], 3),  # accuracy : part des images de test bien classees
                # F1 macro : moyenne sur les classes du F1 (compromis precision / rappel) ; chaque classe compte autant
                "test_f1_macro": round(test["f1_macro"], 3),
                # temps par image du modele exporte en ONNX (format standard de modele) avec onnxruntime sur CPU
                "onnx_ms": card.get("latency_ms", {}).get("onnxruntime_cpu_ms", ""),
            }
        )
    return pd.DataFrame(rows)  # transforme la liste de lignes en tableau pandas


def main() -> None:  # point d'entree du script : construit le tableau, l'affiche et l'ecrit en Markdown si demande
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # l'aide reprend la 1re ligne de la docstring
    parser.add_argument("--outputs", type=Path, default=Path("outputs/classification"))  # dossier des experiences
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
