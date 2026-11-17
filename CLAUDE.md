# Conventions du depot vision-toolbox

Ce fichier est relu au debut de chaque session de travail assistee. Il fixe le contrat que
toute modification doit respecter. En cas de doute, imiter la recette de reference de la meme
famille (voir plus bas) plutot qu'inventer une nouvelle structure.

## But du depot

Une boite a outils de vision industrielle : une bibliotheque `src/vtb/` (briques reutilisables,
chaque fichier copiable seul) et des recettes `recipes/<nom>/` (un cas d'usage complet, en mode
production, avec son notebook). Le code doit etre simple a lire, generique (on change de donnees
en changeant le YAML) et verifie par des tests.

## Style

- Python 3.11, PEP 8, ruff (`ruff check .` et `ruff format .`, longueur de ligne 140).
- Identifiants en anglais ; commentaires et docstrings en francais, en prose naturelle.
- Tout le code est commente LIGNE PAR LIGNE pour un lecteur debutant : chaque ligne de code a
  un commentaire (en fin de ligne, ou juste au-dessus si la ligne depasserait 140 caracteres)
  qui dit ce qu'elle fait ET pourquoi ; chaque terme technique est explique a sa premiere
  apparition dans le fichier. Tout code ajoute ou modifie suit la meme regle.
- ASCII uniquement dans le code, les commentaires, les YAML et les README (pas d'accents, pas
  d'emoji).
- Annotations de type sur toutes les fonctions publiques ; `pathlib.Path` ; `logging`, jamais
  `print` dans `src/` et `recipes/*/scripts/` (le `print` est permis dans les notebooks).
- Aucune mention d'auteur IA dans le code, les commits ou la documentation.

## Le contrat commun a toutes les recettes

1. **Cinq verbes**, un script chacun dans `recipes/<nom>/scripts/` : `prepare.py` (catalogue +
   decoupage + donnees derivees), `train.py`, `evaluate.py` (le test, une seule fois),
   `predict.py`, `export.py`. Chacun prend `--config <yaml>` et ecrit dans
   `runtime.output_dir`. Un script fait au plus 80 lignes de code (commentaires non
   comptes) : la logique est dans `src/vtb/`.
2. **Configuration** : un YAML par experience, valide par un modele pydantic (`extra="forbid"` :
   une cle inconnue est une erreur au demarrage). Aucune constante en dur dans le code.
   `config_used.yaml` est copie dans `output_dir` a chaque lancement.
3. **Catalogue standard** : un `pandas.DataFrame` avec au minimum `path`, et selon la tache
   `label`, `split`, `group`, `mask_path`, `annotation_path`, `timestamp`. Tout format d'origine
   est converti vers lui (`vtb.io.catalogs`, `vtb.annotate`).
4. **Decoupage enregistre** (`split.csv`) et relu ; train / val / test ; groupe par piece, bobine
   ou brame des que plusieurs images viennent de la meme source ; aucune image de test dans
   l'entrainement ; les statistiques (mean / std, poids de classes, seuils) sont calculees sur
   train (ou val pour les seuils), jamais sur test.
5. **Pretraitement** : liste d'operations nommees dans le YAML (`preprocess:`), appliquee par
   `vtb.preprocess.Pipeline` a l'identique a l'entrainement, en prediction et a l'export, et
   recopiee dans la carte du modele. Le pretraitement est deterministe ; l'augmentation est
   aleatoire et seulement a l'entrainement.
6. **Carte du modele** (`model_card.json`, schema `vtb.export.card.ModelCard`) a cote de chaque
   modele exporte : tache, entree, pretraitement, sorties, classes, seuils, metriques, latence.
7. **Export ONNX verifie** contre PyTorch sur de vraies images, avec un lot de taille
   differente de celle de l'export ; le script echoue au-dela de la tolerance.
8. **Deux configurations au minimum par recette** : une nominale et une `quick` (quelques
   minutes sur CPU) qui sert de verification de bout en bout.

## Recettes de reference (a imiter)

- classification : `recipes/classification/` (zoo de modeles, cinq strategies de transfert)
- detection : `recipes/detection/` (quatre moteurs : ultralytics, yolox, fasterrcnn, rfdetr ; OBB et
  segmentation d'instances par Ultralytics)
- segmentation semantique : `recipes/segmentation/`
- anomalies : `recipes/anomaly/` (auto-encodeur, PatchCore de la toolbox, modeles anomalib)
- sans entrainement : `recipes/explore/` (controle du jeu, plongements, derive),
  `recipes/preprocessing/`, `recipes/classical/`, `recipes/annotation/`

Toute nouvelle tache suit le meme decoupage : `src/vtb/tasks/<tache>/{config,workflow}.py`, la
recette `recipes/<tache>/{configs,scripts,notebooks,README.md}`, un test de bout en bout sur
donnees synthetiques dans `tests/`, et un export qui passe par `vtb.export` (carte du modele).
Le predicteur sans PyTorch `vtb.serve.predictor` doit savoir relire le modele exporte : c'est
la preuve que la carte suffit au programme de production (`csharp/` en est le portage C#).

## Le mode notebook (`notebooks/`)

Chaque recette a un notebook autonome dans `notebooks/` (00 a 19, liste dans `notebooks/README.md`).

- Source = le `.py` (cellules `# %%`) ; le `.ipynb` est genere et execute par
  `tools/notebooks.py ... --execute` depuis la racine. Ne jamais editer le `.ipynb` a la main.
- Le code metier est deroule dans les cellules ; imports permis : bibliotheques tierces et
  `vtb.viz` ; `vtb.augment` / `vtb.toolkit` seulement dans 03 et 19 ; le reste de `vtb` seulement
  dans la derniere section "Passer en production".
- Premiere cellule de code "REGLAGES" (donnees, `QUICK`, `SEED`, `DEVICE`, `OUT`) ; `QUICK=True`
  tourne en quelques minutes sur CPU, `QUICK=False` reprend le YAML nominal de la recette.
- Meme decoupage que la recette, export `model.onnx` + `model_card.json` au format `vtb.export.card`,
  et une cellule finale qui verifie que `vtb.serve.predictor.OnnxPredictor` rend les memes sorties.
- Toute modification d'un workflow de production qui change ses resultats impose de reexecuter
  le notebook correspondant (et inversement).

## Outils transverses

- `vtb.viz` : tout affichage (images, metriques, courbes) ; dossier autonome, n'importe rien de vtb.
- `vtb.toolkit` / `vtb-kit` : outils de travail (donnees, lr_find, overfit_one_batch, statistiques).
- `vtb.augment` : toute operation d'augmentation ecrite a la main passe
  `vtb.augment.check_alignment` (boites et masques suivent l'image) ; un preset se regarde
  (`preview`) avant d'entrainer. Albumentations n'est pas une dependance (AlbumentationsX : AGPL).

## Definition de fin d'un lot de travail

Un lot n'est termine que si TOUT ce qui suit passe, commandes lancees depuis la racine :

1. `uv run pytest` passe (nouveaux tests compris) ;
2. `uv run ruff check .` et `uv run ruff format --check .` passent ;
3. la configuration `quick` de chaque recette touchee tourne de `prepare` a `export` ;
4. le notebook de la recette et le notebook autonome correspondant (`notebooks/`) s'executent
   de bout en bout ;
5. le README de la recette donne les commandes et les resultats obtenus (chiffres reels).

## Ce qu'il ne faut jamais faire

- Modifier ou affaiblir un test existant pour le faire passer, sauf si la tache le demande
  explicitement. Un test qui echoue signale un probleme a corriger dans le code.
- Calculer une statistique, un seuil ou choisir un hyperparametre sur le test.
- Ecrire un chemin absolu ou une constante dans le code.
- Exporter un modele sans verification, ou reecrire le pretraitement a la main ailleurs que
  dans `vtb.preprocess`.
- Ajouter une dependance lourde dans `dependencies` : elle va dans un extra.
- Mettre une cle API, un mot de passe ou un jeton dans un fichier du depot. Les secrets sont
  lus dans les variables d'environnement (`RUNPOD_API_KEY`) ou un fichier `.env` ignore par git.
- Copier dans ce depot des donnees, du code ou des noms provenant d'un client.

## GPU distant (RunPod)

- Toujours demander confirmation avant de creer un pod, en affichant : type de carte, prix
  horaire, duree maximale prevue.
- Toute execution distante a une duree maximale et supprime le pod a la fin, y compris en cas
  d'erreur (`tools/runpod/run_remote.py` le fait dans un bloc `finally`).
- Apres chaque session : `python tools/runpod/run_remote.py --list` pour verifier qu'aucun pod
  ne tourne encore.

## Commandes utiles

```
uv sync --extra dev                          # socle
uv sync --extra all                          # tout (yolox : pip install --no-build-isolation --no-deps yolox==0.3.0)
uv run pytest                                # toute la suite (quelques minutes sur CPU)
uv run ruff check . && uv run ruff format --check .
uv run python recipes/<tache>/scripts/run.py --config recipes/<tache>/configs/quick.yaml
uv run python tools/notebooks.py recipes/<tache>/notebooks/<tache>.py --execute
uv run python tools/notebooks.py notebooks/07_classification.py --execute   # mode notebook
uv run vtb-kit --help                        # outils de travail
uv run python tools/make_golden.py --model outputs/<tache>/<experience> --images <dossier> --n 5
uv run python tools/runpod/run_remote.py --list   # apres toute session GPU : aucun pod ne doit rester
```

