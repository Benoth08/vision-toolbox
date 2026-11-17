# vision-toolbox : boite a outils de vision industrielle, mode production et mode notebook

Des briques reutilisables (`src/vtb/`, chaque fichier lisible et copiable seul) et des recettes
completes (`recipes/<tache>/` : configurations YAML, cinq scripts, un notebook, un README avec
les resultats reels). On change de donnees, de modele ou de pretraitement en changeant le YAML,
jamais le code. Tout est teste sur des donnees synthetiques, et chaque modele exporte est
verifie contre PyTorch.

Deux facons de travailler, sur les memes donnees, le meme decoupage et les memes reglages :

- **mode production** : `recipes/<tache>/` (un YAML, cinq scripts) pour livrer et refaire a l'identique ;
- **mode notebook** : `notebooks/00_...` a `19_...`, le code deroule cellule par cellule pour
  comprendre, explorer et prototyper ; chaque notebook exporte un modele que la production sert
  tel quel (voir `notebooks/README.md`).

## Le fil en 13 etapes

Un projet de vision suit toujours les memes 13 etapes, dans cet ordre. Le cours, les notebooks de
tache (00, 07 a 15 : leurs sections s'appellent "Etape N") et la bibliotheque suivent ce fil : pour
retrouver un element, partir du numero de l'etape.

| # | etape | dossier de `src/vtb/` | notebook qui l'approfondit | cle du YAML |
|---|---|---|---|---|
| 1 | Imports et reglages | `core/` | `00_vue_ensemble` | `runtime` |
| 2 | Importer et explorer les donnees | `io/`, `quality/`, `explore/`, `annotate/` | `01_donnees_decoupage`, `04_annotation`, `05_exploration_qualite` | `data.root`, `data.color` |
| 3 | Cataloguer | `io/` | `01_donnees_decoupage` | `data.layout` |
| 4 | Decouper | `split/` | `01_donnees_decoupage` | `split` |
| 5 | Pretraiter | `preprocess/` | `02_opencv_pretraitements` | `preprocess` |
| 6 | Augmenter | `augment/` | `03_augmentation` | `augment` |
| 7 | Charger par lots : Dataset et DataLoader | `tasks/` | `07_classification` | `train.batch_size`, `runtime.num_workers` |
| 8 | Construire le modele | `models/` | `06_modeles_transfert` | `model` |
| 9 | Perte, optimiseur, hyperparametres | `tasks/`, `toolkit/` | `06_modeles_transfert`, `19_boite_a_outils` | `train` |
| 10 | Entrainer | `tasks/` | `07_classification` | `train.epochs`, `train.patience` |
| 11 | Evaluer | `evaluate/`, `explain/`, `toolkit/`, `viz/` | `19_boite_a_outils` | `evaluation`, `scoring` |
| 12 | Sauvegarder et exporter | `export/` | `17_export_onnx` | `export` |
| 13 | Predire et mettre en production | `serve/` | `18_prediction_production` | `runtime.output_dir` |

Le detail de chaque etape (sections des notebooks, fonctions, cles du YAML, scripts, articles) est
dans `ETAPES.md`.

## Ce qu'il y a dedans

| tache | recette | methodes |
|---|---|---|
| classification | `recipes/classification` | zoo torchvision et timm, reseaux de zero ; strategies features, head, partial, full, progressive ; Grad-CAM |
| detection | `recipes/detection` | YOLO11 (boites, boites orientees, segmentation d'instances), YOLOX, Faster R-CNN, RF-DETR ; inference par tuiles |
| segmentation semantique | `recipes/segmentation` | U-Net, U-Net++, FPN, DeepLabV3+, SegFormer... (segmentation_models_pytorch), binaire ou multi-classes |
| anomalies (sans defaut a l'entrainement) | `recipes/anomaly` | auto-encodeur convolutif, PatchCore de la toolbox, anomalib : PatchCore, PaDiM, STFPM, FastFlow, Reverse Distillation, DFM, SuperSimpleNet, EfficientAd, DRAEM |
| pretraitement | `recipes/preprocessing` | 49 operations OpenCV en 12 familles, dont la correction d'illumination (division par le fond flou, flat field, profil de colonnes) ; comparaison a la metrique |
| vision classique | `recipes/classical` | seuillage, blobs, mesure calibree en mm, comparaison a une reference |
| annotation | `recipes/annotation` | formats VOC, YOLO, COCO, LabelMe, Label Studio, CVAT, masques ; outils clavier, revue, pre-annotation, accord entre annotateurs, detection d'erreurs d'etiquette |
| exploration | `recipes/explore` | controle du jeu (illisibles, doublons, fuites, raccourcis), plongements et groupes, derive de domaine |
| augmentation | `vtb.augment` (notebook 03) | 45 operations dont capteur et eclairage industriels, 7 presets, MixUp / CutMix, copy-paste de defauts, mosaique, controle d'alignement boites / masques |
| affichage | `vtb.viz` | images (grilles rapides, masques, cartes, boites), metriques (confusion, ROC / PR, seuil, calibration, erreurs), courbes et suivi en direct ; dossier autonome |
| outils de travail | `vtb.toolkit`, `vtb-kit` | inventaire, moyenne / ecart-type, poids de classes, redimensionnement, 16 -> 8 bits, video -> images, patchs ; LR finder, sur-apprendre un lot, resume du modele ; bootstrap, McNemar, choix du seuil |
| mode notebook | `notebooks/` | 20 notebooks autonomes et executes, de la vue d'ensemble a la prediction sans PyTorch, dont un notebook 100 % OpenCV |

Et pour la production : export ONNX (+ OpenVINO) verifie, carte du modele (`model_card.json`),
predicteur sans PyTorch (`vtb.serve`), service HTTP FastAPI + Dockerfile, portage C#
(`csharp/`, ONNX Runtime + OpenCvSharp) avec verification par fichiers de reference.

## Organisation

```
src/vtb/
  core/        configuration validee (pydantic), graines, journal, demarrage d'une experience [etape 1]
  io/          lecture d'images (8 / 16 bits, chemins Windows), catalogues (dossiers, CSV, MVTec, image + masque, annotations) [etapes 2 et 3]
  quality/     controle d'un jeu d'images [etape 2]
  explore/     plongements, groupes, voisins, derive [etapes 2 et 13]
  annotate/    format pivot, lecteurs / ecrivains, outils, controle, pre-annotation [etapes 2 et 3]
  split/       decoupage stratifie, par groupe (piece, bobine), temporel ; controle des fuites [etape 4]
  preprocess/  registre d'operations OpenCV, Pipeline, tuiles, passage au tenseur [etape 5]
  augment/     augmentations : presets, operations industrielles, melanges, copy-paste, controles (autonome) [etape 6]
  tasks/       classification, detection (4 moteurs), segmentation, anomaly : config + workflow [etapes 7, 9 et 10]
  models/      zoo, transfer learning, reseaux de zero [etape 8]
  evaluate/    metriques (classification, COCO, point de fonctionnement), figures [etape 11]
  explain/     Grad-CAM [etape 11]
  export/      ONNX verifie, OpenVINO, latence, carte du modele [etape 12]
  serve/       predicteur ONNX sans PyTorch, service FastAPI [etape 13]
  viz/         affichage des images et des metriques (autonome, copiable seul) [toutes les etapes]
  toolkit/     outils de travail : donnees, entrainement, statistiques, ligne de commande (vtb-kit) [toutes les etapes]
  classical/   vision classique [hors du fil : sans apprentissage]
recipes/<tache>/{configs, scripts, notebooks, README.md}
notebooks/     le mode notebook : 00 a 19, source .py (cellules # %%) + .ipynb execute
tools/         donnees publiques, poids, notebooks, fichiers de reference C#, RunPod
csharp/        inference C# des modeles exportes
docker/        image du service de prediction
prompts/       taches pretes a confier a Claude Code (verifications, GPU, C#, extensions)
tests/         la suite pytest (donnees synthetiques, rien a telecharger)
```

## Installation

```
uv sync --extra all                     # ou : pip install -e .[all]
pip install --no-build-isolation --no-deps yolox==0.3.0     # moteur YOLOX (facultatif)
python tools/get_data/get_neu_cls.py    # et get_neu_det, get_mvtec_bottle, get_magnetic_tile
python tools/fetch_weights.py           # poids pre-entraines dans weights/ (poste sans Internet ensuite)
uv run pytest
vtb-kit --help                          # outils de travail (ou : python -m vtb.toolkit --help)
python tools/notebooks.py notebooks/00_vue_ensemble.py --execute   # un notebook, execute
```

`uv sync` ecrit `uv.lock` (les versions exactes installees) au premier lancement : le fichier n'est pas fourni, il se calcule
sur le poste. Le mettre dans git avec le code ; `uv sync --frozen` reinstalle ensuite exactement ces versions.

Les extras : `detection`, `yolox`, `rfdetr`, `anomaly`, `segmentation`, `export` (OpenVINO), `serve`,
`runpod`, `dev` ; `all` les reunit. Le socle (classification, pretraitement, export) n'a besoin
que de PyTorch, OpenCV et scikit-learn.

## Le contrat de chaque recette

1. Cinq verbes : `prepare` (catalogue + decoupage enregistre), `train`, `evaluate` (le test,
   une seule fois), `predict`, `export` ; `run.py` les enchaine. Chacun prend `--config`.
2. Un YAML valide par pydantic : une cle inconnue est une erreur au demarrage.
3. Le pretraitement est une liste d'operations nommees, appliquee a l'identique a
   l'entrainement, en prediction et a l'export, et recopiee dans la carte du modele.
4. Les statistiques et les seuils viennent de train / val, jamais du test.
5. L'export echoue si ONNX ne reproduit pas PyTorch (et, en detection, si la chaine ONNX + post-
   traitement ne reproduit pas les boites du moteur).
6. Une configuration `quick` par recette tourne en quelques minutes sur CPU.

Detail et regles dans `CLAUDE.md` (relu par Claude Code a chaque session).

## Du modele entraine au programme de production

```
python recipes/<tache>/scripts/export.py --config <yaml>         # model.onnx + model_card.json (+ model.xml OpenVINO)
python -m vtb.serve --model outputs/<tache>/<experience>         # service HTTP : POST /predict
python tools/make_golden.py --model outputs/<tache>/<experience> --images <dossier>
dotnet run --project csharp/VtbInference -- --model <dossier> --golden <dossier>/golden.json
```

La carte du modele dit tout ce qu'il faut : couleur de lecture, pretraitement, taille et mise a
l'echelle de l'entree, normalisation, sorties, classes, seuils, post-traitement, metriques et
latence. `vtb.serve.predictor` la relit sans PyTorch : s'il rend les memes reponses que le
workflow (c'est teste), le programme C# peut s'ecrire en la lisant.

## GPU a la demande

`tools/runpod/run_remote.py` loue un GPU RunPod le temps d'une experience : confirmation avec le
prix avant creation, duree maximale, suppression du pod garantie, liste des pods restants. La cle
est lue dans `RUNPOD_API_KEY` ou `.env` (jamais dans un fichier versionne).

## Donnees et licences

Les jeux publics utilises (NEU-CLS, NEU-DET, MVTec AD, Magnetic Tile) sont telecharges par
`tools/get_data/` et ne sont pas redistribues ; MVTec AD est sous licence CC BY-NC-SA 4.0
(usage non commercial). Ultralytics est sous AGPL-3.0 (licence commerciale pour un produit
ferme) ; YOLOX et RF-DETR nano a large (Apache-2.0), torchvision (BSD), anomalib (Apache-2.0) et
segmentation_models_pytorch (MIT) n'ont pas cette contrainte. Les poids pre-entraines suivent
la licence de leur jeu d'origine.
