# Recette segmentation : une classe par pixel (forme et surface des defauts)

U-Net, U-Net++, FPN, DeepLabV3+, PSPNet, LinkNet, MAnet ou SegFormer (segmentation_models_pytorch,
licence MIT) avec un encodeur pre-entraine, une perte CE + Dice, des metriques par classe de
defaut et une decision par image. Pour separer des objets (une instance par defaut), voir la
segmentation d'instances YOLO11-seg dans la recette `detection`.

## Commandes (depuis la racine du depot)

```
python tools/get_data/get_magnetic_tile.py                 # Magnetic Tile : 1344 images, masques au pixel
python recipes/explore/scripts/check_dataset.py --config recipes/explore/configs/magnetic_tile.yaml   # groups.csv (quasi-doublons)
python recipes/segmentation/scripts/run.py --config recipes/segmentation/configs/quick.yaml
python recipes/segmentation/scripts/prepare.py  --config recipes/segmentation/configs/unet_resnet18_tile.yaml
python recipes/segmentation/scripts/train.py    --config recipes/segmentation/configs/unet_resnet18_tile.yaml
python recipes/segmentation/scripts/evaluate.py --config recipes/segmentation/configs/unet_resnet18_tile.yaml
python recipes/segmentation/scripts/predict.py  --config recipes/segmentation/configs/unet_resnet18_tile.yaml --source data/raw/magnetic_tile/MT_Crack/Imgs --csv surfaces.csv --masks outputs/segmentation/masks
python recipes/segmentation/scripts/export.py   --config recipes/segmentation/configs/unet_resnet18_tile.yaml
python tools/notebooks.py recipes/segmentation/notebooks/segmentation.py --execute
```

Installation : extra `segmentation`. Les encodeurs resnet prennent les poids ImageNet de
torchvision (meme cache que la classification) ; les autres encodeurs les telechargent par smp.

## Ce que fait chaque script

- `prepare` : catalogue image + masque (`layout: image_mask`, masque de meme nom a cote de
  l'image ou dans `masks_dir`), decoupage stratifie par etiquette d'image, ou groupe
  (`groups_csv`) quand une meme piece apparait plusieurs fois.
- `train` : encodeur a `lr x encoder_lr_factor`, decodeur a `lr`, AdamW + cosinus, choix de
  l'epoque au meilleur IoU moyen des classes de defaut en validation ; `model.pt` contient le
  modele complet (normalisation et softmax inclus).
- `evaluate` : matrice de confusion des pixels du test -> IoU et Dice par classe, moyenne sur
  les classes de defaut, rappel image et fausses alarmes image (au moins `min_defect_area`
  pixels) ; galerie image / verite / prediction (`test_predictions.png`).
- `predict` : surface de chaque classe par image (CSV), masques d'indices et superpositions.
- `export` : ONNX verifie (sortie : probabilites par pixel), OpenVINO, latence, carte du modele.

## Les etapes du fil dans cette recette

Le fil en 13 etapes (le meme que dans le cours et les notebooks : `ETAPES.md`, a la racine) se
retrouve dans les scripts et dans le YAML, ou chaque cle de premier niveau est precedee d'une ligne
`# --- Etape N : ... ---`.

| script | etapes du fil couvertes | cles du YAML |
|---|---|---|
| tous | 1 Imports et reglages | `runtime` (dossier de sortie, graine, materiel) |
| `prepare.py` | 2 Importer, 3 Cataloguer (image + masque), 4 Decouper (stratifie ou par groupe) | `data`, `split` |
| `train.py` | 3 Cataloguer (lecture des masques en classes), 5 Pretraiter, 6 Augmenter, 7 Charger par lots, 8 Modele, 9 Perte et optimiseur, 10 Entrainer ; sauvegarde `model.pt` (debut de l'etape 12) | `mask_mode`, `normal_labels`, `class_names`, `preprocess`, `augment`, `model`, `train` |
| `evaluate.py` | 11 Evaluer : IoU et Dice par classe, decision par image, sur le test | `evaluation` (`threshold`, `min_defect_area`) |
| `predict.py` | 13 Predire : surface de chaque classe, masques et superpositions | `evaluation`, `data.extensions` |
| `export.py` | 12 Sauvegarder et exporter : ONNX verifie, carte du modele | `export`, `evaluation` (seuils recopies dans la carte) |

Reperes : la taille des lots (etape 7) est `train.batch_size`, la perte (etape 9) `train.loss`. Les
quasi-doublons (etape 2) se cherchent avant, avec `recipes/explore/scripts/check_dataset.py`, dont
le `groups.csv` sert au decoupage par groupe (`data.groups_csv`). La meme chaine, cellule par
cellule : `notebooks/12_segmentation.py`.

Les docstrings des scripts numerotent les cinq verbes ("Etape 1" a "Etape 5") : ce sont les etapes
de la recette, pas celles du fil. `run.py` enchaine `prepare`, `train`, `evaluate`, `export`.

## Masques (`mask_mode`)

| mode | masque | classes |
|---|---|---|
| `binary` | tout pixel non nul est defaut | fond, defect |
| `from_label` | masque binaire, classe = etiquette de l'image (dossier) ; `normal_labels` sans classe | fond + une classe par dossier de defaut |
| `from_values` | valeur du pixel = indice de classe (0 = fond) ; donner `class_names` | fond + `class_names` |

Le masque suit l'image au plus proche voisin (resize) et dans les augmentations (torchvision
v2). Un recadrage dans `preprocess` le desynchroniserait : seuls resize et les operations
photometriques sont compatibles.

## Configurations

| fichier | modele | donnees | role |
|---|---|---|---|
| `quick.yaml` | U-Net resnet18, 128 px, 2 epoques | 200 images | verification de bout en bout |
| `unet_resnet18_tile_cpu.yaml` | U-Net resnet18, 128 px, 15 epoques | Magnetic Tile complet, decoupage groupe | vrais chiffres sans GPU |
| `unet_resnet18_tile.yaml` | U-Net resnet18, 256 px, 40 epoques | idem | **nominale** (GPU) |
| `deeplabv3plus_resnet18_tile.yaml` | DeepLabV3+ resnet18 | idem | comparaison d'architecture |
| `unetplusplus_resnet34_tile.yaml` | U-Net++ resnet34 | idem | plus lourd |
| `unet_resnet18_tile_multiclass.yaml` | U-Net resnet18, 6 classes | idem (`from_label`) | quel defaut, ou, quelle surface |

Les configurations Magnetic Tile completes utilisent `groups_csv` (sortie de
`recipes/explore/scripts/check_dataset.py`) : le jeu contient la meme tuile sous plusieurs
expositions (418 images en 161 groupes de quasi-doublons), un decoupage aleatoire mettrait des
copies d'une tuile dans train et test.

## Resultats obtenus

| essai | donnees | entrainement | mIoU defauts | Dice defauts | rappel image | fausses alarmes image | ONNX CPU |
|---|---|---|---|---|---|---|---|
| `quick` (2 epoques, 200 images) | sous-ensemble | 3 min | 0.009 | 0.018 | 1.00 | 1.00 | verification de la chaine seulement |
| `unet_resnet18_tile_cpu` (15 epoques, 128 px) | Magnetic Tile complet, decoupage groupe (test : 79 tuiles defectueuses, 190 saines) | 640 s | **0.705** | **0.827** | 0.61 | 0.016 | 10 ms (OpenVINO 2,5 ms) |

Lecture : avec 15 epoques sur CPU, le U-Net retrouve 70 % du recouvrement des defauts (Dice
0,83), le fond est a 0,99. La decision par image (au moins 10 pixels de defaut a 128 px) ne
signale encore que 61 % des tuiles defectueuses pour 1,6 % de fausses alarmes : les petits trous
(blowholes) disparaissent presque a 128 px. Deux leviers, a regler sur la VALIDATION : la
resolution (256 px dans `unet_resnet18_tile.yaml`, GPU) et `min_defect_area` / `threshold`.
Le premier essai rapide (2 epoques) montre l'etat de depart : le modele predit du defaut partout.

Les autres architectures (DeepLabV3+, U-Net++, multi-classes) se lancent sur GPU (`prompts/03`).
