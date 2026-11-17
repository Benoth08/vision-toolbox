# Recette classification : une image -> une classe (type de defaut, conforme / non conforme)

Le squelette du cours en mode production, avec le choix du modele (zoo torchvision, timm, ou
reseau ecrit a la main) et cinq strategies de transfer learning. On change de donnees, de modele
ou de strategie en changeant le YAML, jamais le code.

## Commandes (depuis la racine du depot)

```
python tools/get_data/get_neu_cls.py                                   # NEU-CLS, 1800 images, 6 classes
python recipes/classification/scripts/run.py --config recipes/classification/configs/quick.yaml
python recipes/classification/scripts/prepare.py  --config recipes/classification/configs/resnet18_partial.yaml
python recipes/classification/scripts/train.py    --config recipes/classification/configs/resnet18_partial.yaml
python recipes/classification/scripts/evaluate.py --config recipes/classification/configs/resnet18_partial.yaml
python recipes/classification/scripts/predict.py  --config recipes/classification/configs/resnet18_partial.yaml --source data/raw/neu_cls_flat/crazing --csv predictions.csv
python recipes/classification/scripts/export.py   --config recipes/classification/configs/resnet18_partial.yaml
python recipes/classification/scripts/compare.py  --outputs outputs/classification --markdown comparison.md
python recipes/classification/scripts/list_models.py                    # le zoo : taille, precision ImageNet, notes
python tools/notebooks.py recipes/classification/notebooks/classification.py --execute
```

`run.py --steps prepare,train,evaluate,export` enchaine les etapes. Chaque etape ecrit dans
`runtime.output_dir` : `config_used.yaml`, le journal, puis `model.pt`, `history.csv`,
`test_metrics.json`, `classification_report.txt`, `confusion_matrix.png`, `test_errors.png`,
`model.onnx` (+ `model.xml` OpenVINO) et `model_card.json`.

## Les etapes du fil dans cette recette

Le fil en 13 etapes (le meme que dans le cours et les notebooks : `ETAPES.md`, a la racine) se
retrouve dans les scripts et dans le YAML, ou chaque cle de premier niveau est precedee d'une ligne
`# --- Etape N : ... ---`.

| script | etapes du fil couvertes | cles du YAML |
|---|---|---|
| tous | 1 Imports et reglages | `runtime` (dossier de sortie, graine, materiel) |
| `prepare.py` | 2 Importer, 3 Cataloguer, 4 Decouper (`split.csv` ecrit une fois, relu ensuite) | `data`, `split` |
| `train.py` | 5 Pretraiter, 6 Augmenter, 7 Charger par lots, 8 Modele, 9 Perte et optimiseur, 10 Entrainer ; sauvegarde `model.pt` (debut de l'etape 12) | `preprocess`, `augment`, `model`, `train` |
| `evaluate.py` | 11 Evaluer : le test, une seule fois | aucune cle propre (`split.file`, `train.batch_size`) |
| `predict.py` | 13 Predire : le pretraitement et les classes viennent de `model.pt` | `data.extensions` |
| `export.py` | 12 Sauvegarder et exporter : ONNX verifie, latence, carte du modele | `export` |

Reperes : la taille des lots (etape 7) est `train.batch_size` ; la normalisation (etape 5) est
`train.normalization` ; le taux du backbone (etape 9) est `model.transfer.backbone_lr_factor`.
Regarder les donnees (etape 2) se fait avant, avec la recette `explore`. La meme chaine, cellule
par cellule : `notebooks/07_classification.py`.

Les docstrings des scripts numerotent les cinq verbes ("Etape 1" a "Etape 5") : ce sont les etapes
de la recette, pas celles du fil. `run.py` enchaine `prepare`, `train`, `evaluate`, `export`.

## Choisir le modele et la strategie

| strategie | ce qui apprend | quand | cle YAML |
|---|---|---|---|
| `features` | rien dans le reseau : regression logistique sur les vecteurs du backbone gele, repliee en couche lineaire | tres peu d'images (dizaines par classe), premier essai en 20 s | `transfer.strategy: features` |
| `head` | la tete seule | peu d'images, domaine proche d'ImageNet | `head` |
| `partial` | la tete + les N derniers blocs, avec un taux plus faible pour le backbone | **le cas courant** | `partial`, `unfreeze_last: 1`, `backbone_lr_factor: 0.1` |
| `full` | tout, avec taux discriminatifs | beaucoup d'images ou domaine tres different (rayons X, thermique) | `full`, `layer_decay: 0.8` |
| `progressive` | un bloc de plus toutes les k epoques | jeux moyens, entrainement instable | `progressive`, `progressive_every: 2` |

Le modele : `model.name` parmi le zoo (`list_models.py`), `timm:<nom>` pour n'importe quel
modele timm, ou `simple_cnn` / `small_resnet` (reseaux de zero, ecrits dans
`src/vtb/models/custom.py`, a copier pour un reseau maison). `in_channels: 1` adapte la
premiere couche a une image grise (somme des noyaux RGB) ; `weights_path` charge des poids
locaux (poste sans Internet).

## Configurations

| fichier | modele | strategie | role |
|---|---|---|---|
| `quick.yaml` | resnet18, 64 px, 2 epoques | head | verification de bout en bout (moins d'une minute) |
| `resnet18_features.yaml` | resnet18 | features | reference "zero entrainement" |
| `resnet18_head.yaml` | resnet18 | head | |
| `resnet18_partial.yaml` | resnet18 | partial | **la configuration nominale** |
| `resnet18_full.yaml` | resnet18 | full | |
| `resnet18_progressive.yaml` | resnet18 | progressive | |
| `resnet18_gray.yaml` | resnet18 a 1 canal | partial | image grise native (pas de copie sur 3 canaux) |
| `simple_cnn.yaml`, `small_resnet.yaml` | reseaux de zero | (tout entraine) | le modele custom du cours |
| `efficientnet_b0_partial.yaml`, `mobilenet_v3_large_partial.yaml`, `convnext_tiny_partial.yaml` | zoo torchvision | partial | autres architectures (poids a telecharger) |
| `timm_example.yaml` | timm convnext_nano | partial | n'importe quel modele timm |

Toutes partagent `outputs/classification/split.csv` : les resultats sont comparables entre eux.

## Resultats obtenus (NEU-CLS, test de 270 images, CPU 2 coeurs partage)

| experience | strategie | parametres entraines | meilleure epoque | temps (s) | exactitude test | F1 macro test | ONNX (ms / image) |
|---|---|---|---|---|---|---|---|
| quick (64 px, 2 epoques) | head | 3 078 | 2 | 40 | 0.941 | 0.941 | 2.8 |
| resnet18_features | features | 3 078 | - | 10 | 0.978 | 0.978 | 10.6 |
| resnet18_head | head | 3 078 | 8 | 131 | 0.959 | 0.959 | 4.1 |
| resnet18_partial | partial | 8 396 806 | 7 | 478 | 1.000 | 1.000 | 20.6 |
| resnet18_full | full | 11 179 590 | 7 | 191 | 1.000 | 1.000 | 4.0 |
| resnet18_progressive | progressive | 11 170 054 | 7 | 599 | 1.000 | 1.000 | 4.4 |
| resnet18_gray (1 canal) | partial | 8 396 806 | 9 | 122 | 1.000 | 1.000 | 4.0 |
| simple_cnn (de zero) | tout | 390 438 | 10 | 240 | 0.970 | 0.971 | 1.6 |
| small_resnet (de zero) | tout | 1 229 030 | 15 | 1 421 | 0.952 | 0.951 | 6.4 |

Lecture : NEU-CLS est un jeu facile pour un reseau pre-entraine des qu'on degele au moins un
bloc (100 % au test). La strategie `features` donne 97,8 % en 10 secondes sans aucune
retropropagation : c'est le premier essai a faire sur un nouveau jeu. Les reseaux de zero
restent a 95-97 % avec bien plus de temps : le pre-entrainement vaut de l'or quand les donnees
sont rares. Les temps sont indicatifs (machine partagee avec d'autres calculs pendant les
essais) ; les latences ONNX aussi.

## Adapter a ses donnees

1. `data.root` et `data.layout` (`class_folders`, `split_folders`, `csv`) ; `color: gray` si
   les images sont en niveaux de gris et qu'on veut un modele a 1 canal.
2. `split.strategy: grouped` avec `data.group_regex` des que plusieurs images viennent de la
   meme piece / bobine / brame (sinon le test est optimiste) ; `temporal` pour tester sur la
   periode la plus recente.
3. `preprocess:` : choisir avec la recette `preprocessing` (a la metrique, pas a l'oeil).
4. Lancer `quick` puis la configuration nominale ; lire `test_errors.png` avant les chiffres.
