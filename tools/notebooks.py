"""Notebooks des recettes : source en mode cellules (# %%) -> .ipynb, execute ou non.

    python tools/notebooks.py recipes/classification/notebooks/classification.py            # convertit
    python tools/notebooks.py recipes/classification/notebooks/classification.py --execute  # convertit et execute

La source de verite est le .py (lisible dans un diff, executable cellule par cellule dans VS
Code avec Shift+Entree). Le .ipynb est genere ; execute, il garde les sorties pour la lecture.
Le notebook s'execute depuis la racine du depot, comme les scripts : les chemins des YAML
sont donc les memes partout.
"""

from __future__ import annotations  # permet d'ecrire les types modernes sans les evaluer a l'execution

import argparse  # lecture des arguments de la ligne de commande
import re  # expressions regulieres : reperer les marqueurs de cellule dans le texte
import sys  # sys.stderr : sortie des messages d'etat
import time  # mesure de la duree d'execution
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import nbformat  # creation et ecriture des fichiers notebook (.ipynb)

ROOT = Path(__file__).resolve().parents[1]  # racine du depot (le dossier au-dessus de tools/)


def py_to_notebook(source: Path) -> nbformat.NotebookNode:  # convertit un .py a cellules en notebook ; renvoie le notebook
    """Decoupe le fichier sur les marqueurs `# %%` ; `# %% [markdown]` donne une cellule de texte."""
    cells, current, kind = [], [], "code"  # cellules finies, lignes de la cellule en cours, type de la cellule en cours
    for line in source.read_text(encoding="utf-8").splitlines():  # pour chaque ligne du fichier .py
        marker = re.match(r"^# %%( \[markdown\])?(.*)$", line)  # la ligne est-elle un marqueur "# %%" (avec ou sans [markdown]) ?
        if marker:  # si oui : une nouvelle cellule commence
            if current:  # si la cellule precedente a du contenu
                cells.append((kind, "\n".join(current).strip("\n")))  # on la range (sans lignes vides au debut et a la fin)
            kind, current = ("markdown" if marker.group(1) else "code"), []  # type de la nouvelle cellule, contenu vide
            continue  # le marqueur lui-meme n'entre pas dans la cellule
        if kind == "markdown":  # dans une cellule de texte
            current.append(line[2:] if line.startswith("# ") else ("" if line == "#" else line))  # retire le "# " du debut de ligne
        else:  # dans une cellule de code
            current.append(line)  # la ligne est gardee telle quelle
    if current:  # a la fin du fichier, s'il reste une cellule en cours
        cells.append((kind, "\n".join(current).strip("\n")))  # on la range aussi
    notebook = nbformat.v4.new_notebook()  # notebook vide (format version 4)
    notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}  # noyau Python 3
    # cree les cellules (texte ou code) en sautant les cellules vides
    notebook.cells = [nbformat.v4.new_markdown_cell(t) if k == "markdown" else nbformat.v4.new_code_cell(t) for k, t in cells if t.strip()]
    return notebook  # renvoie le notebook


def execute(notebook: nbformat.NotebookNode, timeout: int) -> None:  # execute toutes les cellules du notebook ; ne renvoie rien
    from nbclient import NotebookClient  # nbclient : execute un notebook sans ouvrir Jupyter

    # execute depuis la racine du depot ; les sorties (textes, figures) sont ajoutees dans le notebook
    NotebookClient(notebook, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()


def main() -> None:  # convertit (et execute si demande) chaque fichier donne ; ne renvoie rien
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # aide = premiere ligne de la docstring
    parser.add_argument("sources", nargs="+", type=Path)  # un ou plusieurs fichiers .py a convertir
    parser.add_argument("--execute", action="store_true")  # interrupteur : executer apres conversion
    parser.add_argument("--timeout", type=int, default=3600, help="secondes par cellule")  # temps maximum par cellule
    args = parser.parse_args()  # lit les arguments tapes
    for source in args.sources:  # pour chaque fichier .py
        notebook = py_to_notebook(source)  # le convertit en notebook
        start = time.time()  # instant de depart, pour mesurer la duree
        if args.execute:  # si l'execution est demandee
            execute(notebook, args.timeout)  # execute toutes les cellules
        target = source.with_suffix(".ipynb")  # meme nom, extension .ipynb
        nbformat.write(notebook, target)  # ecrit le notebook
        status = f"execute en {time.time() - start:.0f} s" if args.execute else "non execute"  # texte d'etat
        print(f"{target} : {len(notebook.cells)} cellules, {status}", file=sys.stderr)  # bilan (sur la sortie d'erreur)


if __name__ == "__main__":  # vrai seulement quand le script est lance directement
    main()  # lance la conversion
