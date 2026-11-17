# Entraîner un modèle de A à Z, en autonomie

Ce guide suit un projet de vision dans l'ordre : de la réception des images au modèle servi en production. Il reprend le fil en 13 étapes du cours, avec les mêmes numéros et les mêmes noms. À chaque étape : ce qu'il faut faire, où le trouver (notebook, recette, outil) et le critère pour passer à la suite. La seconde moitié est un guide de dépannage : symptôme, étape où chercher, causes probables, comment vérifier, comment corriger.

{{etapes:0}}

## Deux modes, les mêmes étapes

| | Mode notebook (`notebooks/`) | Mode production (`recipes/<tâche>/`) |
|---|---|---|
| pour | comprendre, explorer, prototyper sur de nouvelles données | livrer, refaire à l'identique, comparer des expériences |
| comment | on lit et on exécute de haut en bas ; la cellule `REGLAGES` dit quoi changer | un YAML et cinq scripts : `prepare`, `train`, `evaluate`, `predict`, `export` (`run.py` les enchaîne) |
| sortie | `model.onnx` + `model_card.json` | les mêmes, plus `split.csv`, `history.csv`, métriques, figures |

Les deux produisent la même carte de modèle : un modèle mis au point dans un notebook est servi tel quel par `vtb.serve` et par le programme C#. Démarche conseillée : prototyper dans le notebook de la tâche sur ses données, puis reporter les réglages retenus dans le YAML de la recette pour les entraînements suivis.

Les 13 étapes se retrouvent dans les deux modes : dans un notebook de tâche, chaque étape a son titre « Étape N » ; dans une recette, `prepare` fait les étapes 3 et 4, `train` les étapes 5 à 10, `evaluate` l'étape 11, `export` l'étape 12 et `predict` l'étape 13.

## Choisir la tâche

| La question de l'atelier | Tâche | Notebook | Recette |
|---|---|---|---|
| « cette image est-elle bonne, et sinon quel défaut ? » | classification | `07_classification` | `classification` |
| « où sont les défauts, combien, de quel type ? » | détection | `08_detection_yolo` à `11_detection_rfdetr` | `detection` |
| « quelle forme, quelle surface ? » | segmentation | `12_segmentation` | `segmentation` |
| « est-ce différent du normal ? » (peu ou pas d'exemples de défauts) | anomalies | `13_anomalies_autoencodeur` à `15_anomalies_anomalib` | `anomaly` |
| « la pièce est-elle dans les cotes ? » (contraste net, géométrie connue) | vision classique | `16_vision_classique` | `classical` |

Le notebook `00_vue_ensemble` parcourt toute la chaîne une fois, en court, sur NEU-CLS : c'est la carte de tout le reste.

## La checklist, dans l'ordre

Les 13 étapes sont celles du cours. Les contrôles qui n'ont pas d'étape à eux (qualité du jeu, recherche du taux d'apprentissage, sur-apprentissage d'un lot, intervalle de confiance, seuil de décision, vérification de l'export et de la chaîne de production) sont rangés dans l'étape où on les fait.

### Étape 1 : imports et réglages

Tout ce qu'on change d'un projet à l'autre tient à un seul endroit : la cellule `REGLAGES` d'un notebook, le YAML d'une recette. La graine est fixée dès le départ, pour que deux exécutions donnent le même découpage et les mêmes chiffres.

- Notebook : `00_vue_ensemble` pour voir toute la chaîne une fois ; cellule `REGLAGES` du notebook de la tâche (`DATA_DIR`, `QUICK`, `SEED`, `DEVICE`, `OUT`).
- Recette : un YAML par expérience ; `runtime.seed` et `runtime.output_dir`.
- Critère : l'essai rapide tourne de bout en bout sur les données de démonstration (`QUICK = True` dans le notebook, un YAML `quick` dans la recette) avant de brancher ses propres images.

### Étape 2 : importer et explorer les données

Combien d'images par classe, quelles tailles, quels types (8 ou 16 bits), des fichiers illisibles ?

- Outils : `python -m vtb.toolkit summary <dossier>`, `viz.class_montage`.
- Notebook : `01_donnees_decoupage` (lecture).
- Images pas encore étiquetées : article `annotation` et notebook `04_annotation` (outils, formats, contrôle).
- Critère : au moins une centaine d'images par classe pour un classifieur par transfert (moins avec des caractéristiques gelées), une centaine d'images saines pour les anomalies, quelques centaines d'objets par classe en détection. En dessous, prévoir de l'augmentation et des intervalles de confiance larges.

#### Contrôler la qualité du jeu

Fichiers illisibles, doublons et quasi-doublons (la même pièce photographiée deux fois), images aberrantes, raccourcis (une classe plus sombre que les autres parce qu'elle a été photographiée un autre jour).

- Recette : `python recipes/explore/scripts/check_dataset.py --config ...` (écrit `groups.csv`).
- Notebook : `05_exploration_qualite`.
- Critère : aucun raccourci signalé, ou un raccourci compris et neutralisé (prétraitement, nouvelles images).

### Étape 3 : cataloguer

Comment les images sont-elles rangées (dossiers de classes, CSV, image + masque, annotations) ? Quel que soit le rangement, on le ramène à un seul tableau, le catalogue : une ligne par image, avec son chemin et son étiquette. Toute la suite ne lit que ce tableau.

- Notebook : `01_donnees_decoupage` (catalogues).
- Recette : la clé `data.layout` du YAML dit le rangement (`class_folders`, `split_folders`, `csv`, `flat`, `mvtec`, `image_mask`, `annotations`) ; `prepare.py` construit le catalogue.
- Critère : le catalogue compte autant d'images que l'inventaire de l'étape 2, et la liste des classes est celle qu'on attend.

### Étape 4 : découper

Le découpage décide de la valeur de tous les chiffres qui suivent. Stratifié par classe par défaut ; **par groupe** dès que plusieurs images viennent de la même pièce, de la même bobine, ou sont des quasi-doublons (`split.strategy: grouped` avec `group_regex` ou `data.groups_csv`) ; **temporel** si le modèle doit marcher sur la production de demain.

- Notebook : `01_donnees_decoupage` ; la recette écrit `split.csv`, à conserver avec le modèle.
- Critère : aucune fuite (même pièce dans train et test), chaque classe présente dans chaque sous-ensemble.

Le découpage vient avant le prétraitement et l'augmentation : tout ce qui se calcule sur les données (moyenne et écart-type, seuils) ne doit voir que le train.

### Étape 5 : prétraiter

Le prétraitement est appliqué à l'identique à l'entraînement, en prédiction et à l'export. En sidérurgie, la correction d'illumination (division par le fond flou) et le CLAHE sont les deux premiers à essayer.

- Notebook : `02_opencv_pretraitements` (la boîte à recettes OpenCV) ; choix à la métrique dans `07_classification`.
- Recette : `preprocessing` (galerie des 49 opérations, comparaison à la métrique).
- Critère : le prétraitement améliore la métrique de validation, pas seulement l'aspect des images.

### Étape 6 : augmenter

Une augmentation est une hypothèse : « l'étiquette ne change pas si... ». Un retournement est sans risque pour une rayure ; une rotation de 90° change la direction de laminage d'une bande.

- Notebook : `03_augmentation` ; outil `viz.show_augmentations` pour regarder avant d'entraîner.
- Critère : on a regardé huit images augmentées et elles sont toutes encore de la bonne classe.

### Étape 7 : charger par lots (Dataset et DataLoader)

Le `Dataset` rend une image prête et sa cible ; le `DataLoader` les regroupe en lots, mélangés pour l'entraînement seulement. C'est le dernier endroit où regarder les données avant d'entraîner.

- Notebook : notebook de la tâche, section « Étape 7 » ; outil `viz.show_batch` pour afficher un lot avec ses étiquettes.
- Réglages : `train.batch_size`, `runtime.num_workers` (0 dans un notebook sous Windows) ; classes déséquilibrées : `class_weighting: true` (recette) ou `class_weights` / `sample_weights` (notebook).
- Outil : `max_batch_size(model, (3, H, W), device="cuda")` quand la mémoire GPU limite la taille du lot.
- Critère : le lot affiché montre des images lisibles, à la bonne taille, chacune avec la bonne étiquette.

### Étape 8 : construire le modèle

Un réseau pré-entraîné, presque toujours. Commencer petit (resnet18, YOLO11n, U-Net resnet18) ; la stratégie (`features`, `head`, `partial`, `full`, `progressive`) dépend du nombre d'images et de la distance à ImageNet.

- Notebook : `06_modeles_transfert`.
- Outils : `model_summary` (paramètres, GFLOPs), `benchmark` sur la machine cible si la latence compte.

### Étape 9 : perte, optimiseur, hyperparamètres

La perte dépend de la tâche (entropie croisée en classification, Dice + entropie croisée en segmentation) ; l'optimiseur de départ est AdamW. Le réglage qui compte le plus est le taux d'apprentissage. Les valeurs de départ sont dans « Repères chiffrés », en fin de page.

#### Trouver le taux d'apprentissage

- Outil : `lr_find(model, train_loader, loss_fn)` ; prendre la suggestion, ou 3 à 10 fois moins en fine-tuning complet.
- Repères : AdamW à 1e-3 pour une tête seule, 1e-4 pour un fine-tuning complet, l'encodeur dix fois moins vite que la tête.

### Étape 10 : entraîner

- Notebook de la tâche, ou `python recipes/<tâche>/scripts/train.py --config <yaml>`.
- Outils : `viz.LivePlot`, `EarlyStopping`, `viz.plot_history`.
- Critère : la validation s'améliore puis plafonne ; on garde la meilleure époque de validation, jamais la dernière par défaut.

#### Avant un long entraînement : vérifier que le modèle peut apprendre

- Outil : `overfit_one_batch(model, batch, loss_fn)`.
- Critère : la perte tombe sous 10 % de sa valeur initiale en une centaine de pas. Sinon, voir « La perte ne descend pas » plus bas, **avant** de lancer un long entraînement.

### Étape 11 : évaluer

Le test se lit une seule fois, à la fin. Tout réglage fait en regardant le test le transforme en validation et rend le chiffre optimiste.

- Recette : `evaluate.py` ; outils `viz.confusion_matrix`, `viz.classification_report`, `viz.errors_gallery`.

#### Donner un intervalle de confiance

- Outils : `bootstrap_ci` (par pièce si besoin), `wilson_ci` pour un rappel, `mcnemar` pour comparer deux modèles.
- Critère : on présente « 0,95 [0,92 ; 0,97] », pas « 0,95 ».

#### Choisir le seuil de décision

Le seuil est une décision métier : rappel minimal sur les défauts graves, taux de fausses alarmes tolérable par la ligne. Il se choisit sur la **validation**.

- Outils : `viz.threshold_sweep`, `viz.score_histogram`, `threshold_for(scores, is_defect, min_recall=0.95)`, ou `python -m vtb.toolkit threshold val_scores.csv --min-recall 0.95`.
- Recettes : `conf_predict` (détection), `scoring.image_quantile` (anomalies), `defect_probability` et `min_defect_area` (segmentation).

### Étape 12 : sauvegarder et exporter

On garde le modèle de la meilleure époque de validation, puis on l'exporte dans un format que la production sait lire (ONNX), avec sa carte de modèle.

- Notebook : `17_export_onnx` ; recette : `export.py` (ONNX vérifié contre PyTorch, OpenVINO, latence, carte du modèle).

#### Vérifier l'export

L'export compare les sorties ONNX à celles de PyTorch (`verify_onnx`) : un export qui réussit est un export vérifié.

- Critère : l'export réussit (il échoue tout seul si ONNX ne reproduit pas PyTorch).

### Étape 13 : prédire et mettre en production

Le modèle exporté est servi sans PyTorch, avec le prétraitement et les seuils écrits dans sa carte.

#### Vérifier la chaîne de production

- Notebook : `18_prediction_production` ; prédicteur sans PyTorch `vtb.serve.predictor.OnnxPredictor`.
- Fichiers de référence pour le C# : `python tools/make_golden.py --model <dossier> --images <dossier>`.
- Critère : le prédicteur ONNX rend les mêmes réponses que le notebook ou le workflow sur quelques images ; le programme C# rend les mêmes que les fichiers de référence.

#### Servir et surveiller

- Service : `python -m vtb.serve --model outputs/<tâche>/<expérience>` (POST `/predict`), image Docker dans `docker/`.
- Surveillance : dérive des images de production contre celles de l'entraînement (`recipes/explore/scripts/drift.py`, notebook `05_exploration_qualite`) ; taux de défauts déclarés par jour ; une revue régulière des images signalées.

## Dépannage

Chaque cas : les causes, de la plus fréquente à la plus rare, comment vérifier, comment corriger. La ligne « Où chercher » et la colonne « Étape » donnent le numéro de l'étape concernée : le même dans le cours, dans la checklist ci-dessus et dans les notebooks.

### La perte ne descend pas

**Où chercher** : étapes 7 (les cibles, le lot) et 9 (la perte, le taux d'apprentissage) d'abord.

| Étape | Cause probable | Vérifier | Corriger |
|---|---|---|---|
| 3, 7 | bug de cibles : indices décalés, classes mélangées entre dossiers | `overfit_one_batch` échoue ; afficher un lot avec ses étiquettes (`viz.show_batch`) | indices 0..C-1 dans l'ordre de `class_names`, même ordre partout |
| 9 | perte inadaptée aux sorties | softmax avant `CrossEntropyLoss`, sigmoïde avant `BCEWithLogitsLoss` | donner les **logits** à ces pertes |
| 8, 9 | tout est gelé, ou l'optimiseur ne reçoit pas les bons paramètres | `model_summary` : paramètres entraînables | dégeler ; construire l'optimiseur **après** le gel |
| 9 | taux d'apprentissage trop faible ou trop fort | `lr_find` | prendre la suggestion |
| 5 | entrées non normalisées, ou normalisées deux fois | `viz.show(batch[0])` : titre avec la plage de valeurs | une seule normalisation, celle du modèle pré-entraîné |
| 6 | augmentation trop forte | `viz.show_augmentations` | réduire, retirer l'augmentation suspecte |

### La perte vaut NaN ou inf

**Où chercher** : étape 9 d'abord (le taux d'apprentissage, la perte).

| Étape | Cause probable | Vérifier | Corriger |
|---|---|---|---|
| 9 | taux trop fort | la perte explose avant de devenir NaN | diviser le taux par 10, warmup |
| 9 | log(0), division par zéro dans une perte écrite à la main | perte personnalisée | `clamp(min=1e-7)`, pertes de torch avec logits |
| 2 | image corrompue, valeurs infinies | `summary` (illisibles), `torch.isfinite(x).all()` sur le lot fautif | retirer l'image |
| 10 | précision mixte (AMP) instable | NaN seulement avec `amp: true` | `GradScaler` (déjà dans les recettes classification et segmentation), ou `amp: false` |
| 2 | boîtes dégénérées en détection (largeur nulle) | annotations | `sanitize_boxes`, contrôle des annotations (notebook `04_annotation`) |

### Sur-apprentissage : l'entraînement monte, la validation plafonne ou descend

**Où chercher** : étapes 8 (ce qui est dégelé) et 6 (l'augmentation) d'abord.

| Étape | Cause probable | Vérifier | Corriger |
|---|---|---|---|
| 8 | trop peu d'images pour le nombre de paramètres dégelés | écart train / val qui se creuse dès les premières époques | stratégie `features` ou `head`, modèle plus petit |
| 6 | pas assez d'augmentation | aucune augmentation dans le YAML | flips, petites rotations, luminosité / contraste |
| 10 | trop d'époques | `plot_history` | `EarlyStopping`, meilleure époque de validation |
| 9 | régularisation absente | `weight_decay` à 0 | AdamW avec `weight_decay` 1e-2 à 5e-2, `label_smoothing` 0,1 |

### Sous-apprentissage : l'entraînement lui-même reste mauvais

**Où chercher** : étapes 8 (le gel, la taille du modèle) et 5 (la résolution, le prétraitement) d'abord.

| Étape | Cause probable | Vérifier | Corriger |
|---|---|---|---|
| 8 | modèle trop gelé ou trop petit | perte d'entraînement haute et plate | dégeler `layer4` (`partial`), puis tout (`full`) |
| 5 | résolution trop basse : les petits défauts disparaissent | `viz.compare` image native / image à la taille d'entrée | augmenter la taille d'entrée, ou travailler par tuiles |
| 5 | prétraitement qui efface l'information | `viz.pipeline_steps` | retirer l'opération |
| 9, 10 | taux trop faible, trop peu d'époques | `lr_find`, courbes encore descendantes | augmenter |

### Résultat trop beau pour être vrai

Un 100 % sur un vrai problème industriel est une alerte, pas une victoire.

**Où chercher** : étape 4 d'abord (une fuite dans le découpage), puis étape 2 (un raccourci dans les images).

| Étape | Cause probable | Vérifier | Corriger |
|---|---|---|---|
| 4 | fuite : la même pièce dans train et test | groupes dans `split.csv` ; `check_dataset` (quasi-doublons) | découpage par groupe (`groups_csv`, `group_regex`) |
| 2 | raccourci : la classe se lit dans la luminosité, le bord, un tampon | `check_dataset` (scores de raccourci), Grad-CAM (notebook `07_classification`) | neutraliser (recadrage, prétraitement), refaire des images |
| 4, 11 | test trop facile ou trop petit | `bootstrap_ci` : intervalle large | plus d'images de test, prises comme en production |
| 11 | seuil choisi sur le test | relire la démarche | choisir sur la validation, refaire un test propre |

### Validation instable d'une époque à l'autre

**Où chercher** : étape 4 d'abord (la taille de la validation).

| Étape | Cause probable | Vérifier | Corriger |
|---|---|---|---|
| 4 | validation trop petite | `wilson_ci` sur la métrique de validation | plus d'images de validation, validation croisée (`assign_folds`) |
| 9 | taux trop fort en fin d'entraînement | courbes | décroissance cosinus (`scheduler: cosine` en classification, par défaut en segmentation) |
| 7 | BatchNorm avec de très petits lots | `batch_size` ≤ 4 | lots plus grands, accumulation de gradient, ou geler les BatchNorm |

### Classes déséquilibrées

**Où chercher** : étape 7 (les poids de classes), puis étape 11 (le seuil).

| Étape | Symptôme | Vérifier | Corriger |
|---|---|---|---|
| 7 | bonne exactitude, la classe rare jamais trouvée | rappel par classe (`classification_report`), matrice de confusion | `class_weighting: true` (recette) ou `class_weights` / `sample_weights` (notebook) |
| 11 | le seuil par défaut ne convient pas | `threshold_sweep` sur la validation | seuil choisi pour un rappel minimal sur la classe rare |
| 6 | trop peu d'exemples rares | effectifs | augmentation ciblée, copy-paste de défauts (notebook `03_augmentation`), ou passer aux anomalies |

### Mémoire GPU insuffisante (CUDA out of memory)

**Où chercher** : étape 7 d'abord (la taille du lot).

| Étape | Cause probable | Vérifier | Corriger |
|---|---|---|---|
| 7 | lot trop grand pour la résolution | `max_batch_size(model, (3, H, W), device="cuda")` | lot plus petit et accumulation de gradient, précision mixte |
| 5 | images trop grandes | taille d'entrée | redimensionner (`resize`), tuiles (`patches`) |
| 10 | tenseurs gardés entre les itérations | perte accumulée sans `.item()`, évaluation sans `torch.no_grad()` | `loss.item()`, `torch.inference_mode()` en évaluation |

### Le GPU n'est pas utilisé, ou l'entraînement est lent

**Où chercher** : étape 1 (l'installation, le device), puis étape 7 (le chargement des données).

| Étape | Cause probable | Vérifier | Corriger |
|---|---|---|---|
| 1 | PyTorch installé sans CUDA | `torch.cuda.is_available()`, `torch.version.cuda` | réinstaller la roue CUDA (prompt `01`) |
| 1, 10 | modèle ou lot restés sur CPU | `next(model.parameters()).device` | `.to(device)` sur le modèle **et** sur chaque lot |
| 7 | chargement des données goulot : GPU à 30 % | `nvidia-smi`, `Timer` autour du DataLoader seul | `num_workers` 4 à 8, `pin_memory=True`, images pré-redimensionnées (`resize`) |
| 5 | décodage d'images énormes à chaque époque | taille des fichiers (`summary`) | convertir une fois en PNG à la bonne taille |

### Bon en validation, mauvais en production

**Où chercher** : étape 13 d'abord (la chaîne de production), puis étape 5 (le prétraitement).

| Étape | Cause probable | Vérifier | Corriger |
|---|---|---|---|
| 5, 13 | prétraitement différent entre entraînement et production | comparer la carte du modèle et le code de production ; `vtb.serve` | lire la carte du modèle ; une seule implémentation du prétraitement |
| 2, 13 | BGR / RGB inversés | une image vue par les deux chaînes (`viz.show` avec `bgr=True`) | lire en RGB partout (`vtb.io.images.read_image`), ou le dire dans la carte (`read_color`) |
| 12, 13 | normalisation oubliée ou en double | plage de valeurs à l'entrée du modèle | la normalisation est dans le graphe ONNX : la production envoie [0, 1] |
| 13 | dérive : caméra, éclairage, nouvelle nuance d'acier | `drift.py` (AUC du classifieur de domaine), histogrammes | ré-annoter un lot récent, ré-entraîner, correction d'illumination |
| 11 | seuil choisi sur des proportions qui ne sont pas celles de la ligne | taux de défauts réel | re-choisir le seuil sur des données représentatives |

### ONNX ne donne pas la même chose que PyTorch

**Où chercher** : étape 12 (l'export et sa vérification).

| Étape | Cause probable | Vérifier | Corriger |
|---|---|---|---|
| 12 | modèle exporté en mode entraînement (dropout, BatchNorm) | `model.eval()` avant l'export | l'export de la toolbox le fait |
| 12 | opérateur mal supporté, taille figée dans le graphe | message de `verify_onnx`, lot fixe dans la carte | opset plus récent ; lot fixe de 1 (`dynamic_batch: false`) |
| 12, 13 | post-traitement réécrit différemment (NMS, seuils, letterbox) | vérification des boîtes de l'export de détection | reprendre le post-traitement de `vtb.serve.predictor` |
| 12 | tolérance trop stricte en float16 / OpenVINO | écart maximal | comparer à la tolérance de la carte |

### Annotations incohérentes

**Où chercher** : étape 2 (les annotations se contrôlent avant de cataloguer).

| Étape | Symptôme | Vérifier | Corriger |
|---|---|---|---|
| 2 | deux annotateurs ne sont pas d'accord | accord entre annotateurs (recette `annotation`, notebook `04_annotation`) | guide d'annotation avec exemples limites, revue croisée |
| 11 | erreurs les plus sûres d'elles = vraies erreurs d'étiquette | `errors_gallery`, `worst_errors` | corriger les étiquettes, ré-entraîner |
| 2 | boîtes trop larges ou trop serrées selon les images | galerie de boîtes (`viz.draw_detections`) | une règle unique, contrôle automatique des annotations |

## Repères chiffrés

| Étape | Réglage | Point de départ |
|---|---|---|
| 5 | taille d'entrée | 224 (classification), 640 (YOLO), 256 à 512 (segmentation), 256 (anomalies) |
| 7 | lot | 32 en classification sur GPU, 16 en détection, 8 en segmentation |
| 9 | optimiseur | AdamW, `weight_decay` 1e-2 ; cosinus avec warmup |
| 9 | taux | 1e-3 tête, 1e-4 fine-tuning complet, encodeur × 0,1 |
| 10 | époques | 10 à 30 (classification par transfert), 60 à 100 (détection), 40 (segmentation) |
| 10 | patience | 8 à 10 époques |
| 11 | seuil d'anomalie | quantile 0,99 des scores des images saines de validation |

Les outils cités ici sont décrits dans la page `outils` et montrés dans le notebook `19_boite_a_outils`. L'index `ETAPES.md`, à la racine de la toolbox, donne pour chacune des 13 étapes les notebooks, les fonctions, les clés du YAML et les scripts.
