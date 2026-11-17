# Les 13 etapes : ou trouver chaque etape dans la toolbox

Le cours, les notebooks et les recettes suivent le meme fil de 13 etapes ; ce fichier dit, pour chaque etape, ou la
trouver ici : le notebook, le module de `src/vtb/`, la cle du YAML, le script, le projet de depart du cours et l'article.
Il est genere par `tools/make_etapes_index.py` du cours a partir de `content_etapes/toolbox_map.yaml` (la carte des
tableaux "Ou c'est dans la toolbox" du cours) : ne pas l'editer a la main. La carte est controlee par script : chaque
notebook, module, fonction, cle de YAML, script et fichier cite ici existe.

| Phase | # | Etape |
|---|---|---|
| Donnees | 1 | [Imports et reglages](#etape-1--imports-et-reglages) |
| Donnees | 2 | [Importer et explorer les donnees](#etape-2--importer-et-explorer-les-donnees) |
| Donnees | 3 | [Cataloguer](#etape-3--cataloguer) |
| Donnees | 4 | [Decouper](#etape-4--decouper) |
| Donnees | 5 | [Pretraiter](#etape-5--pretraiter) |
| Donnees | 6 | [Augmenter](#etape-6--augmenter) |
| Donnees | 7 | [Charger par lots : Dataset et DataLoader](#etape-7--charger-par-lots--dataset-et-dataloader) |
| Modele | 8 | [Construire le modele](#etape-8--construire-le-modele) |
| Modele | 9 | [Perte, optimiseur, hyperparametres](#etape-9--perte-optimiseur-hyperparametres) |
| Apprentissage | 10 | [Entrainer](#etape-10--entrainer) |
| Apprentissage | 11 | [Evaluer](#etape-11--evaluer) |
| Livraison | 12 | [Sauvegarder et exporter](#etape-12--sauvegarder-et-exporter) |
| Livraison | 13 | [Predire et mettre en production](#etape-13--predire-et-mettre-en-production) |

Le decoupage (4) vient avant le pretraitement (5) et l'augmentation (6) : tout ce qui se calcule sur les donnees
(statistiques de normalisation, seuils) ne doit voir que le train.
Dans les notebooks de tache (00, 07 a 15), les titres de section commencent par "Etape N" ; quand un notebook
regroupe des etapes, le titre est "Etapes A et B" ou "Etapes A a B".

## Etape 1 : Imports et reglages

Phase : Donnees.

**Mode notebook (`notebooks/`)**

- `07_classification`, section "Etape 1" : les imports et la cellule REGLAGES : `DATA_DIR`, `QUICK`, `SEED`, `DEVICE`, `OUT`
- `00_vue_ensemble` : les 13 etapes en un seul notebook court (moins d'une minute en `QUICK`) : a lancer en premier pour verifier l'installation

**Bibliotheque (`src/vtb/`)**

- `vtb.core.runtime` (`seed_everything`, `get_device`, `setup_logging`) : fixer les graines, choisir CPU ou GPU, ouvrir le journal
- `vtb.core.config` (`load_config`, `save_config`, `RuntimeConfig`) : lire le YAML et le valider (une cle inconnue arrete tout), le recopier a cote des resultats
- `vtb.core.experiment.start` : ce que fait chaque script en premier : dossier de sortie, journal, `config_used.yaml`, graines, device
- `vtb.viz.setup_notebook` : regler la taille et le format des figures d'un notebook

**Fichier de configuration (YAML de `recipes/<tache>/configs/`)**

- `runtime.seed`, `runtime.output_dir`, `runtime.num_workers` : graine, dossier des resultats, processus de chargement (`runtime.device` vaut `auto` par defaut)
- `recipes/classification/configs/quick.yaml` : un YAML complet et court : une section par etape (`data`, `split`, `preprocess`, `augment`, `model`, `train`, `runtime`)

**Scripts et ligne de commande**

- `uv sync --extra all` puis `uv run pytest` : installer la toolbox et verifier qu'elle tourne (donnees synthetiques, rien a telecharger)
- `python tools/fetch_weights.py` : copier les poids pre-entraines dans `weights/` (poste sans Internet ensuite)
- `python recipes/classification/scripts/run.py --config recipes/classification/configs/quick.yaml` : toute la chaine en moins de 2 minutes sur CPU : le test que l'installation est bonne
- `vtb-kit --help` : la liste des outils en ligne de commande

**Projet de depart du cours (`pytorch_starter/`, livre avec le cours, hors de ce depot)**

- `notebooks/01_projet_complet.py`, cellules `imports`, `config`, `seed_device` : les imports, le dictionnaire de reglages, la graine et le device
- `src/vision_starter/utils.py` (`seed_everything`, `get_device`, `setup_logging`) : les memes fonctions, rangees dans le projet
- `src/vision_starter/config.py` (`Config`, `load_config`) : les reglages sortis du code, dans un YAML valide
- `configs/default.yaml` : le YAML de reference du projet de depart

**Articles et pages a lire**

- **Entrainer un modele de A a Z, en autonomie** (`docs/guide_autonomie.md`) : le projet entier dans l'ordre, avec le depannage
- **Conventions du depot** (`CLAUDE.md`) : les regles communes a toutes les recettes (cinq scripts, un YAML)

## Etape 2 : Importer et explorer les donnees

Phase : Donnees.

**Mode notebook (`notebooks/`)**

- `00_vue_ensemble`, section "Etape 2" : regarder les images avant tout calcul (le notebook `07_classification` ne refait pas cette etape)
- `01_donnees_decoupage`, section "Lire une image : une fonction, toutes les situations" : 8 et 16 bits, gris ou couleur, chemins avec accents
- `05_exploration_qualite`, sections "Une ligne de statistiques par fichier", "Doublons et quasi-doublons", "Raccourcis", "Plongements" : controler le jeu avant d'entrainer : illisibles, doublons, raccourcis, groupes
- `04_annotation`, sections "Le format pivot" et "Controler un jeu annote" : quand les images ont des boites ou des masques : lire, convertir, controler les annotations
- `08_detection_yolo`, section "Etape 2" : detection : regarder les images avec leurs boites avant d'entrainer
- `19_boite_a_outils`, sections "Partie A" et "Partie B" : afficher des images (`vtb.viz`) et inventorier un dossier (`vtb.toolkit.dataset`)

**Bibliotheque (`src/vtb/`)**

- `vtb.io.images` (`read_image`, `list_images`, `image_size`, `read_mask`) : lire une image (8 / 16 bits, chemins Windows), lister un dossier dans un ordre stable
- `vtb.toolkit.dataset` (`folder_summary`, `summarize`) : l'inventaire d'un dossier : effectifs, tailles, types, fichiers illisibles
- `vtb.viz` (`class_montage`, `grid`, `histogram`, `show`) : regarder : une ligne d'exemples par classe, une planche, l'histogramme des niveaux
- `vtb.quality.checks` (`scan`, `find_duplicates`, `shortcut_scores`, `flag_outliers`) : statistiques par image, doublons, raccourcis (une statistique simple devine la classe), images atypiques
- `vtb.explore.embeddings` (`embed_images`, `project_2d`, `cluster`) : un vecteur par image sans etiquette, projection 2D, groupes
- `vtb.annotate.formats` (`read_annotations`, `convert`) : lire VOC, YOLO, COCO, LabelMe, Label Studio, CVAT ou des masques

**Fichier de configuration (YAML de `recipes/<tache>/configs/`)**

- `data.root`, `data.color`, `data.extensions` : le dossier des images, lecture `rgb` ou `gray`, extensions acceptees
- `data.annotations_dir`, `data.annotation_format` : detection : ou sont les annotations et dans quel format
- `recipes/explore/configs/neu_cls.yaml` : le YAML du controle d'un jeu (`max_hamming`, `n_clusters`, `projection`)

**Scripts et ligne de commande**

- `python tools/get_data/get_neu_cls.py` : telecharger NEU-CLS ; dans le meme dossier : `get_neu_det.py`, `get_mvtec_bottle.py`, `get_magnetic_tile.py`
- `vtb-kit summary <dossier>` : l'inventaire d'un dossier en une commande
- `python recipes/explore/scripts/check_dataset.py --config recipes/explore/configs/neu_cls.yaml` : le rapport de qualite (`quality_report.md`) : a lancer avant tout entrainement
- `python recipes/explore/scripts/embeddings.py --config recipes/explore/configs/neu_cls.yaml` : projection 2D, groupes, galerie par groupe
- `python recipes/annotation/scripts/convert.py` : convertir un jeu annote d'un format a l'autre (`--check` relit et compare)
- `vtb-kit to-8bit <src> <dst>`, `vtb-kit frames <video> <dst>` : preparer des images recues en 16 bits ou sous forme de video

**Projet de depart du cours (`pytorch_starter/`, livre avec le cours, hors de ce depot)**

- `notebooks/01_projet_complet.py`, cellules `explore_count`, `explore_show`, `explore_sizes` : compter par classe, afficher des exemples, verifier les tailles
- `scripts/get_neu_cls.py` (`layout_a`, `layout_b`, `layout_c`) : telecharger NEU-CLS et le ranger selon les trois organisations du cours
- `notebooks/05_anomalie.py`, cellule `embed_explore` : explorer un jeu sans etiquettes avec des plongements

**Articles et pages a lire**

- **Explorer un jeu d'images : controle, plongements, derive** (`docs/exploration.md`) : le controle du jeu, les doublons, les plongements
- **Annotation : outils, formats, pre-annotation, qualite** (`docs/annotation.md`) : les formats d'annotation et le controle d'un jeu annote
- **Recette exploration** (`recipes/explore/README.md`) : les commandes et les rapports obtenus sur NEU-CLS et Magnetic Tile

## Etape 3 : Cataloguer

Phase : Donnees.

**Mode notebook (`notebooks/`)**

- `07_classification`, section "Etapes 3 et 4" : le catalogue : un tableau `path`, `label`, une ligne par image
- `01_donnees_decoupage`, sections "Le catalogue : un tableau, quel que soit le rangement" et "Controler le catalogue avant de decouper" : tous les rangements (dossiers de classes, CSV, MVTec, image + masque) vers le meme tableau
- `08_detection_yolo`, section "Etapes 2 et 3" : detection : le catalogue porte les boites lues dans les annotations
- `12_segmentation`, section "Etape 3" : segmentation : une image, un masque

**Bibliotheque (`src/vtb/`)**

- `vtb.io.catalogs.build_catalog` : choisit le lecteur d'apres `data.layout`, ajoute `group` et `timestamp` si demande
- `vtb.io.catalogs` (`from_class_folders`, `from_split_folders`, `from_csv`, `from_flat_folder`) : classification : dossiers de classes, decoupage fourni, CSV, images sans etiquette
- `vtb.io.catalogs` (`from_mvtec`, `from_image_mask`, `from_annotations`) : anomalies (MVTec), segmentation (image + masque), detection (annotations)
- `vtb.io.catalogs` (`add_groups`, `add_timestamps`, `limit_catalog`) : colonnes `group` (piece, bobine) et `timestamp` lues dans le nom de fichier ; sous-echantillon d'essai
- `vtb.annotate.schema` (`ImageAnnotation`, `Box`, `Polygon`) : le format pivot des annotations : tout format d'origine est converti vers lui
- `vtb.core.config.DataConfig` : toutes les cles de la section `data` et leurs valeurs par defaut

**Fichier de configuration (YAML de `recipes/<tache>/configs/`)**

- `data.layout` : le rangement : `class_folders`, `split_folders`, `csv`, `flat`, `mvtec`, `image_mask`, `annotations`
- `data.root`, `data.extensions` : le dossier racine et les extensions lues (utile quand les masques `.png` sont a cote des images)
- `data.annotations_dir`, `data.annotation_format`, `data.object_label_from` : detection : dossier, format et origine de la classe des objets
- `data.groups_csv`, `data.max_images` : groupes fournis par un CSV ; sous-echantillon pour un essai rapide (`data.group_regex` lit le groupe dans le nom de fichier)
- `normal_labels`, `mask_mode` : anomalies et segmentation : quelles etiquettes sont saines, masque binaire ou multi-classes

**Scripts et ligne de commande**

- `python recipes/<tache>/scripts/prepare.py --config <yaml>` : construit le catalogue et l'enregistre avec le decoupage dans `split.csv` (etapes 3 et 4 ensemble)
- `python recipes/annotation/scripts/convert.py` : ramener des annotations recues dans un format que le catalogue sait lire

**Projet de depart du cours (`pytorch_starter/`, livre avec le cours, hors de ce depot)**

- `notebooks/01_projet_complet.py`, cellules `catalog`, `catalog_variants` : le catalogue en pandas, et les variantes de rangement
- `src/vision_starter/data.py` (`catalog_from_class_folders`, `catalog_from_split_folders`, `catalog_from_csv`, `build_catalog`) : un lecteur par rangement, choisi par la configuration
- `configs/layout_a_split_folders.yaml`, `configs/layout_c_csv.yaml` : les YAML des deux autres rangements
- `src/vision_starter/detection.py` (`catalog_voc`) : le catalogue d'un jeu de detection (boites VOC)
- `src/vision_starter/segmentation.py` (`catalog_magnetic_tile`) : le catalogue image + masque
- `src/vision_starter/anomaly.py` (`catalog_mvtec`) : le catalogue d'un jeu d'anomalies (arborescence MVTec)

**Articles et pages a lire**

- **Conventions du depot** (`CLAUDE.md`) : le catalogue standard : colonnes `path`, `label`, `split`, `group`, `mask_path`, `annotation_path`
- **Annotation : outils, formats, pre-annotation, qualite** (`docs/annotation.md`) : les formats d'annotation que le catalogue sait relire

## Etape 4 : Decouper

Phase : Donnees.

**Mode notebook (`notebooks/`)**

- `07_classification`, section "Etapes 3 et 4" : le decoupage stratifie train / val / test, enregistre dans `split.csv`
- `01_donnees_decoupage`, sections "Decoupage stratifie", "Decoupage groupe", "Decoupage temporel", "Validation croisee", "Controle des fuites" : les trois strategies ecrites a la main et le controle qui refuse un decoupage dangereux
- `05_exploration_qualite`, section "La fuite, mesuree" : ce que coute un quasi-doublon de part et d'autre du decoupage
- `12_segmentation`, section "Etape 4" : decoupage groupe : les quasi-doublons d'une meme tuile restent ensemble
- `14_anomalies_patchcore`, section "Etapes 3 et 4" : anomalies : aucun defaut dans train ni dans val

**Bibliotheque (`src/vtb/`)**

- `vtb.split.splits.make_split` : ajoute la colonne `split` : stratifie, groupe ou temporel
- `vtb.split.splits` (`check_no_leakage`, `summarize_split`) : refuser un doublon de chemin, un sous-ensemble vide ou un groupe coupe en deux ; tableau classes x sous-ensembles
- `vtb.split.splits` (`save_split`, `load_split`) : ecrire `split.csv` une fois, puis le relire partout (on choisit explicitement le sous-ensemble lu)
- `vtb.split.splits.assign_folds` : colonne `fold` pour la validation croisee, sur train + val seulement
- `vtb.quality.checks` (`find_duplicates`, `duplicate_groups`) : trouver les quasi-doublons et en faire des groupes a ne pas separer
- `vtb.tasks.anomaly.data.make_anomaly_split` : anomalies : normales vers train / val / test, defauts vers le test

**Fichier de configuration (YAML de `recipes/<tache>/configs/`)**

- `split.strategy` : `stratified` (defaut), `grouped`, `temporal`, `random`, `provided` (decoupage deja fourni)
- `split.val_fraction`, `split.test_fraction` : les parts de validation et de test (0.15 et 0.15 par defaut)
- `split.file` : le CSV du decoupage, partage par toutes les experiences d'une recette : elles restent comparables
- `data.groups_csv` : les groupes a garder entiers (`data.group_regex` et `data.timestamp_regex` les lisent dans le nom de fichier ; `split.n_folds` ajoute les plis)

**Scripts et ligne de commande**

- `python recipes/<tache>/scripts/prepare.py --config <yaml>` : calcule le decoupage une fois et ecrit `split.csv`
- `python recipes/explore/scripts/check_dataset.py --config recipes/explore/configs/magnetic_tile.yaml` : ecrit `groups.csv` (les quasi-doublons regroupes), a donner a `data.groups_csv` avec un decoupage `grouped` ; avec la cle `split_file`, mesure aussi les doublons a cheval sur train et test
- `vtb-kit imagefolder <split.csv> <dossier>` : ranger un decoupage en dossiers `<split>/<classe>/` pour un outil qui l'exige

**Projet de depart du cours (`pytorch_starter/`, livre avec le cours, hors de ce depot)**

- `notebooks/01_projet_complet.py`, cellules `split`, `split_check` : le decoupage stratifie, ses deux controles, la relecture
- `src/vision_starter/data.py` (`make_split`, `make_grouped_split`, `check_split`, `load_split`) : les memes fonctions dans le projet, avec la variante groupee
- `src/vision_starter/data.py` (`near_duplicate_groups`, `make_stratified_grouped_split`, `check_group_leak`) : reperer les quasi-doublons, decouper sans les separer, refuser un groupe coupe
- `scripts/make_split.py` : le script qui ecrit `split.csv`
- `configs/default.yaml` (`val_fraction`, `test_fraction`, `split_file`, `seed`) : les reglages du decoupage

**Articles et pages a lire**

- **Explorer un jeu d'images : controle, plongements, derive** (`docs/exploration.md`) : doublons et fuites entre sous-ensembles
- **Entrainer un modele de A a Z, en autonomie** (`docs/guide_autonomie.md`) : le point 3 de la checklist : decouper train / validation / test

## Etape 5 : Pretraiter

Phase : Donnees.

**Mode notebook (`notebooks/`)**

- `07_classification`, section "Etape 5" : des chaines de pretraitement candidates, choisies a la metrique et non a l'oeil
- `02_opencv_pretraitements`, sections "Geometrie", "Histogramme et contraste", "Illumination : corriger le shading", "Debruitage", "Aide-memoire" : la boite a recettes OpenCV, 100 % OpenCV, copiable partout
- `01_donnees_decoupage`, section "Moyenne et ecart-type du jeu" : les statistiques de normalisation, calculees sur le train seulement
- `00_vue_ensemble`, section "Etapes 5 a 7" : pretraitement puis passage au tenseur, en quelques lignes

**Bibliotheque (`src/vtb/`)**

- `vtb.preprocess.pipeline.Pipeline` : applique la liste `preprocess:` du YAML, a l'identique a l'entrainement, en prediction et a l'export
- `vtb.preprocess.registry` (`list_ops`, `get_op`, `register`) : le catalogue des 49 operations en 12 familles ; inscrire la sienne
- `vtb.preprocess.ops.geometry` (`resize`, `crop`, `letterbox`) : mettre a la taille du reseau, avec ou sans deformation
- `vtb.preprocess.ops.illumination` (`shading_division`, `flat_field`, `column_profile`) : corriger un eclairage non uniforme
- `vtb.preprocess.ops.contrast.clahe` : contraste local, sans faire exploser le bruit
- `vtb.preprocess.tensor` (`to_float_chw`, `normalize`, `channel_stats`, `image_to_input`) : image vers tenseur `(C, H, W)` dans [0, 1], normalisation, moyenne et ecart-type par canal
- `vtb.tasks.classification.data.compute_stats` : moyenne et ecart-type sur le train, apres pretraitement
- `vtb.viz.pipeline_steps` : voir l'image apres chaque operation de la chaine

**Fichier de configuration (YAML de `recipes/<tache>/configs/`)**

- `preprocess` : la liste d'operations nommees, dans l'ordre : `{name: resize, size: 128}`, `{name: clahe}`...
- `data.color` : lecture en `rgb` ou en `gray` (1 canal)
- `pipelines`, `compare.image_size`, `compare.max_per_class` : recette preprocessing : les chaines candidates a comparer
- `recipes/preprocessing/configs/steel_gray.yaml` : un exemple complet (acier en niveaux de gris) ; la normalisation se regle par `train.normalization` (`auto` par defaut)

**Scripts et ligne de commande**

- `python recipes/preprocessing/scripts/gallery.py --config recipes/preprocessing/configs/steel_gray.yaml` : une planche avant / apres par famille d'operations
- `python recipes/preprocessing/scripts/compare.py --config recipes/preprocessing/configs/steel_gray.yaml` : compare les chaines a la metrique (F1 macro en validation croisee)
- `vtb-kit mean-std <dossier du train>` : moyenne et ecart-type par canal
- `vtb-kit resize <src> <dst>`, `vtb-kit to-8bit <src> <dst>`, `vtb-kit patches <src> <dst>` : redimensionner, convertir ou decouper un dossier entier une fois pour toutes

**Projet de depart du cours (`pytorch_starter/`, livre avec le cours, hors de ce depot)**

- `notebooks/01_projet_complet.py`, cellules `mean_std`, `transforms` : statistiques du train, redimensionnement, passage au tenseur, normalisation
- `src/vision_starter/data.py` (`compute_mean_std`, `build_transforms`) : les memes fonctions dans le projet
- `configs/default.yaml` (`image_size`, `normalization`) : taille d'entree et choix des statistiques

**Articles et pages a lire**

- **Le pretraitement dans la toolbox** (`docs/preprocessing.md`) : le catalogue famille par famille, la correction d'illumination, juger un pretraitement a la metrique
- **Recette pretraitement** (`recipes/preprocessing/README.md`) : les planches et les comparaisons obtenues sur cinq cas (acier, microscope, thermique 16 bits...)

## Etape 6 : Augmenter

Phase : Donnees.

**Mode notebook (`notebooks/`)**

- `07_classification`, section "Etapes 6 et 7" : les augmentations du train, ecrites en clair
- `03_augmentation`, sections "Une augmentation est une hypothese", "Les presets compares", "Les boites et les masques suivent l'image", "Controler l'alignement, automatiquement", "L'effet chiffre" : tout le module `vtb.augment`, avec l'effet mesure de l'augmentation
- `12_segmentation`, section "Etapes 5 et 6" : segmentation : le masque subit la meme transformation que l'image

**Bibliotheque (`src/vtb/`)**

- `vtb.augment.transforms.build_augmentation` : la liste `augment:` du YAML devient un `Compose` torchvision v2 (image, boites et masques ensemble)
- `vtb.augment.transforms` (`available_augmentations`, `available_presets`) : les 45 operations et les 7 presets disponibles, avec leurs parametres
- `vtb.augment.presets` (`PRESETS`, `preset_steps`) : des listes pretes : `light`, `medium`, `strong`, `steel_strip`, `steel_texture`, `metal_part`, `anomaly_normals`
- `vtb.augment.tools` (`preview`, `describe`, `check_alignment`, `benchmark_ops`) : regarder des tirages, lister les etapes, verifier que boites et masques suivent, chronometrer
- `vtb.augment.ops` (`IlluminationGradient`, `Vignetting`, `MotionBlur`, `LineStripes`, `SpecularHighlight`) : les operations industrielles : eclairage, capteur, flou de defilement, reflets
- `vtb.augment.mixing` (`MixCollate`, `CopyPaste`, `mosaic`) : MixUp / CutMix au niveau du lot, copier-coller de defauts, mosaique
- `vtb.tasks.classification.workflow.augmentation_for` : ou la recette construit l'augmentation : pour le train seulement
- `vtb.viz.show_augmentations` : l'original puis n tirages de la transformation

**Fichier de configuration (YAML de `recipes/<tache>/configs/`)**

- `augment` : la liste des augmentations du train : `{name: hflip}`, `{name: vflip}`, ou un preset `{name: preset, preset: steel_strip}`
- `ultralytics.extra`, `ultralytics.close_mosaic` : detection YOLO11 : les augmentations sont celles du moteur (mosaique, retournements)
- `yolox.flip_prob`, `yolox.hsv_prob` : detection YOLOX : probabilite de retournement et de variation de couleur

**Scripts et ligne de commande**

- `python tools/notebooks.py notebooks/03_augmentation.py --execute` : pas de script dedie : l'augmentation se regle dans le YAML et se regarde dans le notebook 03

**Projet de depart du cours (`pytorch_starter/`, livre avec le cours, hors de ce depot)**

- `notebooks/01_projet_complet.py`, cellule `transforms` : transformation d'entrainement (avec augmentations) et d'evaluation (sans)
- `src/vision_starter/data.py` (`AUGMENTATIONS`, `build_transforms`) : les deux transformations, construites au meme endroit ; les augmentations du train viennent de la liste `augment`
- `configs/default.yaml` (`augment`) : la liste des augmentations : `hflip`, `vflip`, `rotate`, `brightness`
- `src/vision_starter/segmentation.py` (`build_segmentation_transforms`) : segmentation : image et masque transformes ensemble
- `src/vision_starter/anomaly.py` (`build_anomaly_transforms`) : anomalies : retournements seulement, une image saine doit rester saine

**Articles et pages a lire**

- **Augmentation de donnees : multiplier les images sans mentir sur l'etiquette** (`docs/augmentation.md`) : les operations, les presets, les boites et les masques qui suivent, l'effet chiffre

## Etape 7 : Charger par lots : Dataset et DataLoader

Phase : Donnees.

**Mode notebook (`notebooks/`)**

- `07_classification`, section "Etapes 6 et 7" : le `Dataset`, le `DataLoader`, un lot regarde avant d'entrainer
- `19_boite_a_outils`, section "Partie C" : regarder ce que le reseau recoit avec `show_batch`
- `03_augmentation`, section "Vitesse : l'augmentation ne doit pas affamer le GPU" : le cout du chargement et le nombre de workers
- `10_detection_faster_rcnn`, section "Etapes 5 a 7" : detection : un `Dataset` qui rend (image, boites) et sa `collate_fn`
- `12_segmentation`, section "Etapes 7 et 10" : segmentation : des lots (image, masque)

**Bibliotheque (`src/vtb/`)**

- `vtb.tasks.classification.data.ClassificationDataset` : lit une image du catalogue, la pretraite, l'augmente, rend (tenseur, indice de classe)
- `vtb.tasks.classification.data.build_loader` : le `DataLoader` : `shuffle` et `drop_last` pour le train seulement, ordre reproductible
- `vtb.core.runtime` (`seed_worker`, `make_generator`) : une graine par worker et un generateur pour le melange
- `vtb.tasks.classification.data.class_weights` : desequilibre : poids inverses des effectifs du train, pour la perte
- `vtb.toolkit.dataset` (`class_weights`, `sample_weights`, `sample_per_class`) : desequilibre : poids par classe, poids par exemple (`WeightedRandomSampler`), sous-ensemble equilibre
- `vtb.viz.show_batch` : afficher un lot tel que le reseau le voit, normalisation annulee
- `vtb.tasks.segmentation.data.SegmentationDataset` : segmentation : (image, masque d'indices de classe)
- `vtb.tasks.detection.fasterrcnn_backend` (`DetectionDataset`, `collate`) : detection : (image, boites et classes), lots assembles en tuples

**Fichier de configuration (YAML de `recipes/<tache>/configs/`)**

- `train.batch_size` : la taille du lot (32 par defaut)
- `runtime.num_workers` : processus de chargement : 0 dans un notebook sous Windows, 2 a 4 dans un script
- `train.class_weighting` : `true` : poids de classes dans la perte
- `ultralytics.batch`, `yolox.batch_size`, `fasterrcnn.batch_size`, `rfdetr.batch_size` : detection : la taille du lot se regle dans la section du moteur

**Scripts et ligne de commande**

- `vtb-kit class-weights --csv <split.csv>` : les poids de classes d'un jeu, en une commande
- `python recipes/<tache>/scripts/train.py --config <yaml>` : pas de script dedie : `Dataset` et `DataLoader` sont construits au debut de l'entrainement

**Projet de depart du cours (`pytorch_starter/`, livre avec le cours, hors de ce depot)**

- `notebooks/01_projet_complet.py`, cellules `dataset`, `dataloader`, `show_batch`, `class_weights` : le `Dataset` ecrit a la main, les trois `DataLoader`, un lot affiche, les poids de classes
- `src/vision_starter/data.py` (`ImageDataset`, `build_loader`, `class_weights`) : les memes briques dans le projet
- `src/vision_starter/utils.py` (`seed_worker`) : la graine de chaque processus de chargement
- `configs/default.yaml` (`batch_size`, `num_workers`, `class_weighting`) : les reglages du chargement

**Articles et pages a lire**

- **Outils de travail : afficher, diagnostiquer, juger un resultat** (`docs/outils.md`) : afficher un lot (`show_batch`), poids de classes et de tirage

## Etape 8 : Construire le modele

Phase : Modele.

**Mode notebook (`notebooks/`)**

- `07_classification`, section "Etapes 8 a 10" : un resnet18 pre-entraine (strategie partial) et, en comparaison, un reseau ecrit de zero
- `06_modeles_transfert`, sections "Le catalogue : taille, vitesse, precision ImageNet", "Anatomie d'un reseau pre-entraine", "Geler, degeler, et le piege des BatchNorm", "Images grises", "Les cinq strategies comparees sur NEU-CLS" : choisir un modele pre-entraine et la strategie de transfert
- `08_detection_yolo`, section "Etapes 6 a 10" : detection : le modele vient de la librairie, avec ses poids COCO
- `12_segmentation`, section "Etapes 8 et 9" : segmentation : un U-Net a encodeur pre-entraine
- `14_anomalies_patchcore`, section "Etape 8" : anomalies : un resnet18 gele et une banque memoire, sans entrainement

**Bibliotheque (`src/vtb/`)**

- `vtb.models.zoo` (`list_models`, `get_spec`, `instantiate`) : le catalogue des architectures (torchvision, `timm:<nom>`) : parametres, precision ImageNet, latence
- `vtb.models.transfer.build_classifier` : le modele pret a entrainer : tete remplacee, entree adaptee
- `vtb.models.transfer` (`apply_strategy`, `freeze_frozen_batchnorm`, `ProgressiveUnfreezer`) : geler et degeler selon la strategie ; BatchNorm gelees gardees en mode eval
- `vtb.models.transfer` (`replace_head`, `adapt_input_channels`, `load_backbone_weights`) : nouvelle tete, entree a 1 canal sans perdre les poids ImageNet, poids locaux
- `vtb.models.custom` (`SimpleCNN`, `SmallResNet`) : deux reseaux ecrits de zero, a lire et a copier
- `vtb.toolkit.training.model_summary` : parametres (total, entrainables, geles), taille, forme de sortie
- `vtb.tasks.segmentation.models.build_segmenter` : segmentation : U-Net, FPN, DeepLabV3+... avec la normalisation dans le graphe
- `vtb.tasks.anomaly.patchcore_lite.PatchCoreLite` : anomalies : PatchCore de la toolbox (voir aussi `vtb.tasks.anomaly.autoencoder.ConvAutoencoder`)

**Fichier de configuration (YAML de `recipes/<tache>/configs/`)**

- `model.name`, `model.pretrained`, `model.in_channels` : l'architecture, poids ImageNet ou non, nombre de canaux d'entree
- `model.transfer.strategy`, `model.transfer.unfreeze_last` : `features`, `head`, `partial`, `full` ou `progressive` ; nombre de blocs degeles
- `backend`, `ultralytics.weights`, `yolox.variant`, `fasterrcnn.backbone`, `rfdetr.size` : detection : le moteur (`ultralytics`, `yolox`, `fasterrcnn`, `rfdetr`) et sa taille de modele
- `model.architecture`, `model.encoder`, `model.encoder_weights` : segmentation : `unet`, `unetplusplus`, `deeplabv3plus`... et l'encodeur
- `method`, `autoencoder.widths`, `patchcore_lite.layers`, `anomalib.model` : anomalies : la methode et ses reglages propres

**Scripts et ligne de commande**

- `python recipes/classification/scripts/list_models.py` : le catalogue des modeles pre-entraines (`--latency` mesure la vitesse sur ce poste)
- `python tools/fetch_weights.py` : telecharger les poids une fois dans `weights/`
- `recipes/classification/configs/resnet18_head.yaml`, `recipes/classification/configs/resnet18_partial.yaml`, `recipes/classification/configs/simple_cnn.yaml` : un YAML par choix de modele : on change de modele en changeant de fichier

**Projet de depart du cours (`pytorch_starter/`, livre avec le cours, hors de ce depot)**

- `notebooks/01_projet_complet.py`, cellules `model_mlp`, `model_cnn`, `model_check`, `model_summary`, `transfer_build` : un MLP, un CNN ecrit de zero, le controle des formes, puis un resnet18 pre-entraine
- `src/vision_starter/model.py` (`SimpleCNN`, `SmallResNet`, `build_resnet18`, `build_model`, `unfreeze_layers`) : les modeles du projet et la fabrique qui choisit d'apres le YAML
- `notebooks/04_references.py`, cellules `freeze_basics`, `freeze_verify`, `freeze_batchnorm` : geler des couches et verifier que le gel a pris
- `configs/resnet18.yaml`, `configs/resnet18_finetune.yaml`, `configs/small_resnet.yaml` : les YAML des autres modeles

**Articles et pages a lire**

- **Choisir un modele pre-entraine et une strategie de transfert** (`docs/modeles_transfert.md`) : le zoo de modeles, les cinq strategies, ce qu'elles donnent sur NEU-CLS
- **Detection : quatre moteurs, un seul contrat** (`docs/detection.md`) : choisir un detecteur
- **Segmentation semantique : la forme et la surface de chaque defaut** (`docs/segmentation.md`) : choisir une architecture de segmentation
- **Detection d'anomalies : apprendre le normal, signaler l'ecart** (`docs/anomalies.md`) : choisir une methode quand on n'a pas de defauts a l'entrainement

## Etape 9 : Perte, optimiseur, hyperparametres

Phase : Modele.

**Mode notebook (`notebooks/`)**

- `07_classification`, section "Etapes 8 a 10" : la perte, l'optimiseur, le planificateur et leurs valeurs de depart
- `06_modeles_transfert`, section "Un taux d'apprentissage par groupe de couches" : la tete apprend vite, le backbone doucement
- `19_boite_a_outils`, section "Partie C" : chercher le taux d'apprentissage avec `lr_find`
- `12_segmentation`, section "Etapes 9 et 11" : segmentation : la perte CE + Dice

**Bibliotheque (`src/vtb/`)**

- `vtb.tasks.classification.config.TrainConfig` : tous les hyperparametres d'entrainement, leur valeur de depart et leur explication
- `vtb.tasks.classification.engine` (`build_optimizer`, `build_scheduler`) : AdamW ou SGD ; planificateur `plateau`, `cosine` ou aucun, avec echauffement
- `vtb.tasks.classification.workflow.train` : ou la perte est creee : entropie croisee, poids de classes et lissage des etiquettes en option
- `vtb.models.transfer.param_groups` : un taux d'apprentissage par groupe de couches
- `vtb.toolkit.training` (`lr_find`, `suggest_lr`, `max_batch_size`) : test du taux d'apprentissage, taux propose, plus grand lot qui tient en memoire
- `vtb.tasks.segmentation.config.SegTrainConfig` : segmentation : pertes `dice_ce`, `ce`, `dice`, `focal`, `lovasz`

**Fichier de configuration (YAML de `recipes/<tache>/configs/`)**

- `train.lr`, `train.weight_decay` : taux d'apprentissage et penalite sur les poids (`train.optimizer` : `adamw` par defaut, ou `sgd`)
- `train.scheduler`, `train.warmup_epochs` : comment le taux evolue : `plateau`, `cosine`, `none` ; montee progressive au debut
- `train.epochs`, `train.batch_size`, `train.patience` : duree maximale, taille du lot, patience de l'arret anticipe
- `train.class_weighting` : poids de classes dans la perte (`train.label_smoothing` et `train.grad_clip` existent aussi, a 0 et vide par defaut)
- `model.transfer.backbone_lr_factor`, `model.transfer.layer_decay` : taux du backbone = taux de la tete x facteur
- `train.loss`, `train.encoder_lr_factor` : segmentation : la perte et le taux de l'encodeur
- `ultralytics.epochs`, `yolox.basic_lr_per_img`, `fasterrcnn.lr`, `rfdetr.lr` : detection : chaque moteur a ses propres hyperparametres

**Scripts et ligne de commande**

- `python recipes/classification/scripts/compare.py` : pas de script dedie : un essai = un YAML copie avec une valeur changee ; ce script met les essais en tableau

**Projet de depart du cours (`pytorch_starter/`, livre avec le cours, hors de ce depot)**

- `notebooks/01_projet_complet.py`, cellule `train_step` : la perte, l'optimiseur et un pas d'apprentissage decompose
- `notebooks/04_references.py`, cellules `losses_classification`, `losses_other`, `lr_range_test`, `random_search` : les pertes comparees, le test du taux d'apprentissage, la recherche aleatoire d'hyperparametres
- `src/vision_starter/model.py` (`param_groups`, `build_optimizer`) : deux taux d'apprentissage (tete et backbone), et l'optimiseur choisi dans le YAML
- `configs/default.yaml` (`optimizer`, `learning_rate`, `weight_decay`, `scheduler_factor`, `scheduler_patience`, `epochs`, `patience`, `class_weighting`) : les hyperparametres du projet

**Articles et pages a lire**

- **Outils de travail : afficher, diagnostiquer, juger un resultat** (`docs/outils.md`) : `lr_find` et les diagnostics d'entrainement
- **Choisir un modele pre-entraine et une strategie de transfert** (`docs/modeles_transfert.md`) : les cinq strategies et leurs taux d'apprentissage

## Etape 10 : Entrainer

Phase : Apprentissage.

**Mode notebook (`notebooks/`)**

- `07_classification`, section "Etapes 8 a 10" : la boucle d'entrainement ecrite en clair : meilleure epoque gardee, arret anticipe, courbes
- `00_vue_ensemble`, section "Etape 10" : la meme boucle dans sa version la plus courte
- `19_boite_a_outils`, section "Partie C" : quand rien n'apprend : sur-apprendre un lot avec `overfit_one_batch`
- `10_detection_faster_rcnn`, section "Etapes 9 et 10" : detection : une boucle d'entrainement ecrite a la main
- `08_detection_yolo`, section "Etapes 6 a 10" : detection : la librairie fait la boucle
- `13_anomalies_autoencodeur`, section "Etapes 6 a 10" : anomalies : entrainer sur les images saines seulement

**Bibliotheque (`src/vtb/`)**

- `vtb.tasks.classification.engine` (`train_one_epoch`, `evaluate`, `fit`) : une epoque, une evaluation, la boucle complete avec arret anticipe et historique
- `vtb.tasks.classification.workflow.train` : l'entrainement de la recette : ecrit `model.pt`, `history.csv`, `curves.png`
- `vtb.toolkit.training` (`overfit_one_batch`, `EarlyStopping`, `AverageMeter`) : le premier diagnostic, l'arret anticipe, la moyenne d'une perte sur une epoque
- `vtb.viz` (`LivePlot`, `plot_history`) : les courbes en direct pendant l'entrainement, ou apres coup
- `vtb.tasks.classification.features.fit_logistic_head` : strategie `features` : une regression logistique sur les vecteurs, sans retropropagation
- `vtb.tasks.detection.workflow.train` : detection : lance le moteur du YAML
- `vtb.tasks.segmentation.workflow.train` : segmentation : garde l'epoque au meilleur IoU de validation
- `vtb.tasks.anomaly.workflow.train` : anomalies : entraine ou construit la memoire, puis calibre le seuil

**Fichier de configuration (YAML de `recipes/<tache>/configs/`)**

- `train.epochs`, `train.patience` : nombre maximal d'epoques et patience de l'arret anticipe
- `runtime.output_dir` : ou sont ecrits `model.pt`, `history.csv`, `curves.png`, `train.log` (`runtime.device` et `train.amp` reglent le GPU et la precision mixte)
- `model.transfer.progressive_every` : strategie `progressive` : un bloc degele de plus toutes les n epoques
- `recipes/classification/configs/quick.yaml` : 2 epoques a 64 px : pour verifier la chaine, pas pour la precision

**Scripts et ligne de commande**

- `python recipes/<tache>/scripts/train.py --config <yaml>` : entraine et sauvegarde le meilleur modele
- `python recipes/<tache>/scripts/run.py --config <yaml>` : enchaine prepare, train, evaluate, export
- `python tools/runpod/run_remote.py` : louer un GPU le temps d'une experience : prix confirme avant, pod supprime a la fin (`--list` verifie qu'aucun pod ne reste)

**Projet de depart du cours (`pytorch_starter/`, livre avec le cours, hors de ce depot)**

- `notebooks/01_projet_complet.py`, cellules `sanity_initial_loss`, `sanity_overfit`, `engine`, `fit`, `curves` : les deux controles avant d'entrainer, la boucle, l'entrainement, les courbes
- `src/vision_starter/engine.py` (`train_one_epoch`, `evaluate`, `fit`) : la boucle rangee dans le projet
- `scripts/train.py` : le script d'entrainement : donnees, modele, boucle, sauvegardes

**Articles et pages a lire**

- **Entrainer un modele de A a Z, en autonomie** (`docs/guide_autonomie.md`) : le depannage : la perte ne descend pas, sur-apprentissage, validation instable
- **Recette classification** (`recipes/classification/README.md`) : les commandes et les resultats reels de chaque strategie

## Etape 11 : Evaluer

Phase : Apprentissage.

**Mode notebook (`notebooks/`)**

- `07_classification`, section "Etape 11" : le test lu une seule fois, la matrice de confusion, Grad-CAM, le seuil de rejet choisi sur la validation
- `19_boite_a_outils`, sections "Partie D" et "Partie E" : afficher des metriques ; intervalle de confiance, test de McNemar, choix d'un seuil
- `08_detection_yolo`, section "Etape 11" : detection : mAP COCO et point de fonctionnement a un seuil
- `12_segmentation`, section "Etape 11" : segmentation : IoU et Dice par classe, decision par image
- `14_anomalies_patchcore`, section "Etape 11" : anomalies : AUROC image et pixel, seuil = quantile des images saines de validation

**Bibliotheque (`src/vtb/`)**

- `vtb.evaluate.classification` (`classification_metrics`, `threshold_for_precision`) : accuracy, balanced accuracy, F1 macro, detail par classe ; seuil qui garantit une precision
- `vtb.viz` (`confusion_matrix`, `classification_report`, `errors_gallery`, `roc_curves`, `pr_curves`, `threshold_sweep`) : les figures d'evaluation, dont la galerie des erreurs les plus sures
- `vtb.toolkit.stats` (`bootstrap_ci`, `wilson_ci`, `mcnemar`, `threshold_for`, `expected_calibration_error`) : juger un resultat : intervalle de confiance, deux modeles vraiment differents ?, seuil sous contrainte, calibration
- `vtb.explain.gradcam.GradCAM` : ou le modele a regarde
- `vtb.classical.blobs` (`detect_blobs`, `blob_table`), `vtb.classical.measure` (`strip_width`, `calibrate_from_distance`), `vtb.classical.compare` (`compare_to_reference`) : sans apprentissage : compter et mesurer des taches, mesurer en millimetres, comparer a une image de reference
- `vtb.evaluate.detection` (`coco_metrics`, `operating_point`) : detection : mAP@0.5, mAP@0.5:0.95, puis precision et rappel au seuil de production
- `vtb.tasks.segmentation.metrics` (`confusion`, `scores_from_confusion`, `image_decisions`) : segmentation : IoU et Dice depuis la matrice de confusion des pixels
- `vtb.tasks.anomaly.scoring` (`calibrate_thresholds`, `anomaly_metrics`) : anomalies : seuil calibre sur les saines, AUROC, rappel et fausses alarmes

**Fichier de configuration (YAML de `recipes/<tache>/configs/`)**

- `evaluation.conf_eval`, `evaluation.conf_predict`, `evaluation.iou_nms` : detection : seuil bas pour le mAP, seuil de production, seuil de la NMS
- `evaluation.threshold`, `evaluation.min_defect_area` : segmentation : seuil sur la probabilite de defaut, surface minimale pour signaler une image
- `scoring.image_quantile`, `scoring.image_score`, `scoring.gaussian_sigma` : anomalies : taux de fausses alarmes vise, score d'image, lissage de la carte

**Scripts et ligne de commande**

- `python recipes/<tache>/scripts/evaluate.py --config <yaml>` : le test, une seule fois : metriques, rapport, figures
- `python recipes/classification/scripts/compare.py` : le tableau comparatif des experiences deja lancees (detection : `recipes/detection/scripts/compare.py` ; anomalies : `recipes/anomaly/scripts/benchmark.py`)
- `vtb-kit ci <predictions.csv>` : intervalle de confiance d'une metrique par bootstrap
- `vtb-kit mcnemar <predictions.csv>` : deux modeles sur les memes images : la difference est-elle reelle ?
- `vtb-kit threshold <validation.csv>` : le seuil de decision qui respecte une contrainte (rappel, fausses alarmes, precision)

**Projet de depart du cours (`pytorch_starter/`, livre avec le cours, hors de ce depot)**

- `notebooks/01_projet_complet.py`, cellules `test_metrics`, `confusion`, `worst_images` : les metriques du test, la matrice de confusion, les pires erreurs
- `scripts/evaluate.py` : le script d'evaluation : `test_metrics.json` (avec intervalle de confiance et nombre d'erreurs), rapport, matrice de confusion
- `notebooks/04_references.py`, cellules `metrics_classification`, `metrics_imbalance`, `metrics_threshold` : quelle metrique choisir, l'effet du desequilibre, le choix d'un seuil
- `notebooks/05_anomalie.py`, cellule `gradcam` : Grad-CAM ecrit a la main

**Articles et pages a lire**

- **Outils de travail : afficher, diagnostiquer, juger un resultat** (`docs/outils.md`) : intervalles de confiance, McNemar, choix du seuil
- **Entrainer un modele de A a Z, en autonomie** (`docs/guide_autonomie.md`) : evaluer une seule fois, avec un intervalle ; choisir le seuil de decision

## Etape 12 : Sauvegarder et exporter

Phase : Livraison.

**Mode notebook (`notebooks/`)**

- `07_classification`, section "Etape 12" : sauvegarde du meilleur modele, export ONNX verifie, carte du modele
- `17_export_onnx`, sections "Exporter avec un lot dynamique", "Un lot fige, detecte et contourne", "Verifier ONNX contre PyTorch", "OpenVINO et latence", "La carte du modele, ecrite a la main" : l'export pas a pas, avec ses pieges
- `08_detection_yolo`, section "Etapes 12 et 13" : detection : ONNX sans NMS, post-traitement ecrit et verifie
- `14_anomalies_patchcore`, section "Etape 12" : anomalies : la banque memoire exportee d'un bloc

**Bibliotheque (`src/vtb/`)**

- `vtb.tasks.classification.workflow` (`train`, `load_trained`) : `model.pt` : les poids de la meilleure epoque et tout ce qu'il faut pour predire (classes, pretraitement, moyenne et ecart-type, taille d'entree)
- `vtb.tasks.classification.workflow.export` : l'export de la recette : ONNX a lot dynamique, verifie sur de vraies images, latences, carte
- `vtb.export.onnx` (`export_onnx`, `verify_onnx`, `OnnxVerificationError`) : exporter, puis comparer chaque sortie a PyTorch : l'export echoue si l'ecart depasse la tolerance
- `vtb.export.onnx` (`to_openvino`, `measure_latency`, `batch_size_of`) : conversion OpenVINO, millisecondes par lot, detection d'un lot fige
- `vtb.export.card` (`ModelCard`, `write_card`, `read_card`) : `model_card.json` : entree, pretraitement, sorties, classes, seuils, metriques, latence
- `vtb.core.config.ExportConfig` : toutes les cles de la section `export` et leurs valeurs par defaut

**Fichier de configuration (YAML de `recipes/<tache>/configs/`)**

- `export.openvino`, `export.latency_runs` : conversion OpenVINO en plus ; nombre de passages pour la latence (`export.opset`, `export.atol`, `export.rtol` : version ONNX et tolerances)
- `runtime.output_dir` : le dossier du modele : `model.pt`, `model.onnx`, `model_card.json`

**Scripts et ligne de commande**

- `python recipes/<tache>/scripts/export.py --config <yaml>` : `model.onnx` + `model_card.json` (+ `model.xml` OpenVINO), verifies
- `python recipes/<tache>/scripts/train.py --config <yaml>` : `model.pt` est ecrit par l'entrainement, avec les poids de la meilleure epoque
- `python tools/notebooks.py notebooks/17_export_onnx.py --execute` : rejouer l'export pas a pas

**Projet de depart du cours (`pytorch_starter/`, livre avec le cours, hors de ce depot)**

- `notebooks/01_projet_complet.py`, cellules `save_model`, `load_predict`, `resume`, `load_foreign_weights`, `onnx_export` : sauvegarder, recharger, reprendre un entrainement, charger des poids venus d'ailleurs, exporter en ONNX
- `scripts/export_onnx.py` : export ONNX (opset fixe par `onnx_opset`), verification, carte du modele avec chiffres du test et versions
- `src/vision_starter/model.py` (`rebuild_model`) : recreer le modele enregistre, sans retelecharger de poids
- `scripts/train.py` : ou `model.pt` est ecrit

**Articles et pages a lire**

- **Du modele entraine au programme de production** (`docs/production.md`) : les trois fichiers d'un modele livre et la carte du modele

## Etape 13 : Predire et mettre en production

Phase : Livraison.

**Mode notebook (`notebooks/`)**

- `07_classification`, section "Etape 13" : predire sans PyTorch (numpy + OpenCV + onnxruntime), puis passer en production
- `18_prediction_production`, sections "La carte et l'image", "Le post-traitement, tache par tache", "Le predicteur complet", "Le service HTTP", "Fichiers de reference pour le programme C#" : un predicteur ONNX ecrit a la main, le service, la verification du portage C#
- `08_detection_yolo`, section "Etape 13" : detection : predire par le chemin de production, par tuiles sur une grande image
- `05_exploration_qualite`, section "Derive de domaine" : surveiller la ligne apres la mise en production

**Bibliotheque (`src/vtb/`)**

- `vtb.tasks.classification.workflow.predict` : predire avec PyTorch depuis `model.pt` : une image ou un dossier
- `vtb.serve.predictor.OnnxPredictor` : predire sans PyTorch, pour les quatre taches, a partir de `model.onnx` et `model_card.json`
- `vtb.serve.app.create_app` : le service HTTP FastAPI : `POST /predict`
- `vtb.preprocess.tensor.image_to_input` : image pretraitee vers le tableau `(1, C, H, W)` attendu par onnxruntime
- `vtb.serve.predictor` (`nms`, `nms_per_class`, `yolox_decode`) : detection : le post-traitement en numpy, a porter tel quel
- `vtb.tasks.detection.common.predict_tiled` : grande image : inference par tuiles avec recouvrement (voir aussi `vtb.preprocess.tiles.tile_grid`)
- `vtb.explore.drift` (`domain_classifier`, `statistics_drift`) : les images de la ligne ressemblent-elles encore a celles de l'entrainement ?
- `vtb.toolkit.training.benchmark` : latence d'une inference : moyenne, mediane, p95, debit

**Fichier de configuration (YAML de `recipes/<tache>/configs/`)**

- `evaluation.conf_predict`, `evaluation.iou_nms` : detection : le seuil de production et la NMS (`tiling` active l'inference par tuiles)
- `evaluation.threshold`, `evaluation.min_defect_area` : segmentation : la decision par image
- `scoring.image_quantile` : anomalies : le seuil ecrit dans la carte du modele
- `reference.root`, `current.root`, `simulate` : derive : le dossier de reference, le dossier courant, une derive simulee pour tester

**Scripts et ligne de commande**

- `python recipes/<tache>/scripts/predict.py --config <yaml> --source <image ou dossier>` : predire avec le modele sauvegarde
- `python -m vtb.serve --model outputs/<tache>/<experience>` : le service HTTP de prediction, sans PyTorch
- `python tools/make_golden.py --model <dossier> --images <dossier>` : fichiers de reference : les reponses attendues, pour verifier un autre programme
- `csharp/VtbInference/Program.cs` : le portage C# (ONNX Runtime + OpenCvSharp) ; voir `csharp/README.md`
- `docker/Dockerfile.serve` : l'image Docker du service
- `python recipes/explore/scripts/drift.py --config recipes/explore/configs/drift_simulated.yaml` : mesurer la derive entre deux dossiers d'images

**Projet de depart du cours (`pytorch_starter/`, livre avec le cours, hors de ce depot)**

- `notebooks/01_projet_complet.py`, cellule `load_predict` : recharger le modele et predire une image
- `scripts/predict.py` (`load_saved_model`, `predict_image`) : predire une image ou un dossier depuis `model.pt`
- `scripts/export_onnx.py` : la carte du modele (`model_card.json`) que lira le programme de production

**Articles et pages a lire**

- **Du modele entraine au programme de production** (`docs/production.md`) : la reference Python sans PyTorch, le service HTTP, le programme C#
- **Explorer un jeu d'images : controle, plongements, derive** (`docs/exploration.md`) : la derive de domaine
- **Entrainer un modele de A a Z, en autonomie** (`docs/guide_autonomie.md`) : verifier la chaine de production, servir et surveiller
