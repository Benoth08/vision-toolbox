# Tache : installer la toolbox sur ce poste et verifier tout ce qui tourne sur CPU

Contexte : depot `vision-toolbox` (bibliotheque `src/vtb/`, recettes `recipes/` = mode
production, notebooks autonomes `notebooks/` = mode notebook). Il a ete developpe et teste sous
Linux ; cette tache verifie qu'il fonctionne sur ce poste Windows (PowerShell, CPU seulement).
Lis d'abord `CLAUDE.md`, `README.md` et `notebooks/README.md`.

Regles : avant chaque commande longue (installation, telechargement, entrainement), dis ce que
tu lances et combien de temps cela devrait prendre. Une commande a la fois pour les calculs
lourds (pas d'entrainements en parallele : la memoire du poste est limitee).

## Etapes

1. Environnement : Python 3.11 avec `uv` (sinon `python -m venv .venv` puis
   `.venv\Scripts\Activate.ps1` et pip). Installer `uv sync --extra all` (ou
   `pip install -e .[all]`), puis YOLOX : `pip install --no-build-isolation --no-deps yolox==0.3.0`.
   Si le proxy bloque un index, le dire et proposer l'alternative (index interne, roues locales)
   sans contourner le proxy. Verifier `vtb-kit --help`.
2. Donnees publiques : lancer les quatre scripts de `tools/get_data/` (NEU-CLS, NEU-DET,
   MVTec bottle, Magnetic Tile). Si un telechargement est bloque, le noter et continuer.
3. Poids : `python tools/fetch_weights.py` (par defaut : resnet18 torchvision, YOLO11n
   boites / OBB / seg, YOLOX-tiny, RF-DETR nano). Noter ce qui est bloque.
4. Tests : `uv run pytest` (575 tests attendus, quelques minutes), puis `uv run ruff check .`
   et `uv run ruff format --check .`. Tout echec specifique a Windows (chemins, encodage,
   separateurs, multiprocessing des DataLoader, fenetres OpenCV) se corrige DANS LE CODE,
   jamais en affaiblissant un test.
5. Mode production, essais rapides : pour chaque recette,
   `python recipes/<recette>/scripts/run.py --config recipes/<recette>/configs/quick*.yaml`
   (classification, detection : les six quick, segmentation, anomaly), puis
   `check_dataset.py`, `embeddings.py` et `drift.py` de `recipes/explore`.
6. Notebooks pilotes des recettes : `python tools/notebooks.py recipes/<recette>/notebooks/<nom>.py --execute`
   pour chaque recette ; ils doivent s'executer de bout en bout.
7. Mode notebook : executer les 20 notebooks dans l'ordre, un par un, en mode QUICK (valeur par
   defaut) : `python tools/notebooks.py notebooks/00_vue_ensemble.py --execute`, puis 01 ... 19
   (18 apres 17 et apres les recettes : il relit leurs modeles exportes). Pour chacun : pas
   d'erreur, la cellule de controle "Passer en production" est OK, et les chiffres sont du meme
   ordre que la colonne "resultat de l'execution livree" de `notebooks/README.md`. Une
   correction se fait dans le `.py` (jamais dans le `.ipynb`), puis on reexecute.
8. `num_workers` : verifier que `num_workers: 2` fonctionne sous Windows (DataLoader en spawn),
   dans les recettes et dans les notebooks ; sinon corriger la cause (fonction non picklable,
   code hors `if __name__ ==`).

## Livrable

Un compte rendu : versions installees, ce qui a ete bloque par le reseau, tests (nombre,
duree), resultats des essais rapides et des 20 notebooks compares a ceux des README (memes
ordres de grandeur ?), temps d'execution sur ce poste, corrections faites (fichier, raison).
Mettre a jour les README avec les temps mesures sur ce poste s'ils different nettement.
Aucun secret, aucune donnee interne dans le depot. `pytest` et `ruff` au vert a la fin.
