# Recette detection : ou sont les defauts, et de quel type

Quatre moteurs derriere le meme contrat. Le decoupage, le format pivot des annotations,
l'evaluateur COCO, le point de fonctionnement et la carte du modele sont communs : les moteurs se
comparent honnetement sur les memes images.

| moteur (`backend:`) | licence | forces | taches |
|---|---|---|---|
| `ultralytics` (YOLO11) | AGPL-3.0 (licence commerciale pour un produit ferme) | le plus simple, tres bon compromis vitesse / precision, augmentations fortes | boites (`detect`), boites orientees (`obb`), segmentation d'instances (`segment`) |
| `yolox` (Megvii) | Apache-2.0 | sans ancres, proche de YOLO en precision, utilisable dans un produit ferme | boites |
| `fasterrcnn` (torchvision) | BSD | 100 % PyTorch (le squelette du cours), deux etages : bon sur petits objets | boites |
| `rfdetr` (Roboflow RF-DETR) | Apache-2.0 (nano a large) | transformer de detection a dos DINOv2 : sans ancres ni NMS, transfert tres bon hors de COCO | boites |

## Commandes (depuis la racine du depot)

```
python tools/get_data/get_neu_det.py                 # NEU-DET : 1800 images 200 x 200, 6 defauts, annotations VOC
python tools/get_data/get_magnetic_tile.py           # Magnetic Tile : masques -> boites orientees / polygones
python recipes/detection/scripts/run.py --config recipes/detection/configs/quick_yolo.yaml
python recipes/detection/scripts/prepare.py  --config recipes/detection/configs/yolo11n_neudet.yaml
python recipes/detection/scripts/train.py    --config recipes/detection/configs/yolo11n_neudet.yaml
python recipes/detection/scripts/evaluate.py --config recipes/detection/configs/yolo11n_neudet.yaml
python recipes/detection/scripts/predict.py  --config recipes/detection/configs/yolo11n_neudet.yaml --source data/raw/neu_det/Validation_Images --csv detections.csv --images outputs/detection/predictions
python recipes/detection/scripts/export.py   --config recipes/detection/configs/yolo11n_neudet.yaml
python recipes/detection/scripts/compare.py  --outputs outputs/detection --markdown comparison.md
python tools/notebooks.py recipes/detection/notebooks/detection.py --execute
```

Installation : extra `detection` (Ultralytics, pycocotools) ; pour YOLOX, extra `yolox` puis
`pip install --no-build-isolation --no-deps yolox==0.3.0` ; pour RF-DETR, extra `rfdetr`
(`pip install "rfdetr[train]"`). Les poids pre-entraines viennent de GitHub (YOLO, YOLOX) et du
stockage de Roboflow (RF-DETR) ; `tools/fetch_weights.py` les prepare pour un poste sans Internet.

## Ce que fait chaque script

- `prepare` : lit les annotations dans n'importe quel format de `vtb.annotate` (VOC, YOLO,
  COCO, LabelMe, Label Studio, CVAT, masques), les convertit au pivot, decoupe PAR IMAGE
  (stratifie sur la classe principale, `group_regex` pour garder une bobine du meme cote),
  ecrit `split.csv` et `split.vtb.json` (les annotations de toutes les images du split).
- `train` : le moteur choisi ; pour Ultralytics (format YOLO) et RF-DETR (format COCO de Roboflow),
  le jeu est regenere depuis le pivot avec le meme pretraitement qu'en production
  (`data/processed/yolo/...`, `data/processed/rfdetr/...`). Produit `best.pt`,
  `history.csv`, `history.png` et `detector.json`.
- `evaluate` : prediction du test par LE MEME CHEMIN que la production (lecture, pretraitement,
  moteur, retour aux coordonnees d'origine), puis mAP@0.5 et mAP@0.5:0.95 COCO, AP par classe,
  et le point de fonctionnement au seuil `conf_predict` : precision, rappel, rappel image et
  fausses alarmes image (`test_metrics.json`, `per_class.csv`, `test_predictions.png`).
- `predict` : une ligne par objet (et une ligne vide par image sans detection, pour la
  tracabilite) ; boites orientees (cx, cy, w, h, angle) et polygones si le modele en produit.
- `export` : ONNX + carte du modele + OpenVINO, avec DEUX verifications : sorties brutes ONNX
  contre PyTorch, et boites de la chaine ONNX + post-traitement ecrit en numpy contre celles
  du moteur (IoU moyen >= 0,98, sinon echec). RF-DETR n'a pas de NMS : son post-traitement est
  un top-k sur les couples (requete, classe).

## Les etapes du fil dans cette recette

Le fil en 13 etapes (le meme que dans le cours et les notebooks : `ETAPES.md`, a la racine) se
retrouve dans les scripts et dans le YAML, ou chaque cle de premier niveau est precedee d'une ligne
`# --- Etape N : ... ---`.

| script | etapes du fil couvertes | cles du YAML |
|---|---|---|
| tous | 1 Imports et reglages | `runtime` (dossier de sortie, graine, materiel) |
| `prepare.py` | 2 Importer (annotations de tout format), 3 Cataloguer (format pivot), 4 Decouper par image | `data`, `split` |
| `train.py` | 5 Pretraiter, puis le moteur fait 6 Augmenter, 7 Charger par lots, 8 Modele, 9 Perte et optimiseur, 10 Entrainer ; sauvegarde `best.pt` (debut de l'etape 12) | `preprocess`, `backend`, le bloc du moteur (`ultralytics`, `yolox`, `fasterrcnn` ou `rfdetr`), `augment` (fasterrcnn seulement) |
| `evaluate.py` | 11 Evaluer : mAP COCO et point de fonctionnement sur le test, par le chemin de prediction de la production | `evaluation` (`conf_eval`, `conf_predict`, `iou_nms`) |
| `predict.py` | 13 Predire : une ligne par objet, retour aux coordonnees d'origine, tuiles en option | `evaluation.conf_predict`, `tiling`, `data.extensions` |
| `export.py` | 12 Sauvegarder et exporter : ONNX, post-traitement verifie, carte du modele | `export`, `evaluation` (seuils recopies dans la carte) |

Reperes : avec Ultralytics, YOLOX et RF-DETR, les augmentations et la taille des lots (etapes 6 et 7)
se reglent dans le bloc du moteur, pas dans `augment`. Regarder les donnees (etape 2) : recette
`explore` et notebook `08_detection_yolo`. La meme chaine, cellule par cellule, un notebook par
moteur : `notebooks/08_detection_yolo.py`, `09_detection_yolox.py`, `10_detection_faster_rcnn.py`,
`11_detection_rfdetr.py`.

Les docstrings des scripts numerotent les cinq verbes ("Etape 1" a "Etape 5") : ce sont les etapes
de la recette, pas celles du fil. `run.py` enchaine `prepare`, `train`, `evaluate`, `export`.

## Configurations

| fichier | moteur | donnees | role |
|---|---|---|---|
| `quick_yolo.yaml` | YOLO11n, 320 px, 2 epoques | 240 images NEU-DET | verification de bout en bout |
| `quick_yolox.yaml` | YOLOX-tiny, 2 epoques | memes 240 images | verification |
| `quick_fasterrcnn.yaml` | Faster R-CNN resnet18, 1 epoque | memes 240 images | verification |
| `quick_rfdetr.yaml` | RF-DETR nano, 256 px, 1 epoque, sans poids | memes 240 images | verification |
| `quick_obb_tile.yaml`, `quick_seg_tile.yaml` | YOLO11n-obb / -seg, 1 epoque | 200 images Magnetic Tile | verification OBB et instances |
| `yolo11n_neudet.yaml` | YOLO11n, 60 epoques | NEU-DET complet | **nominale** (GPU conseille) |
| `yolox_tiny_neudet.yaml` | YOLOX-tiny, 40 epoques | NEU-DET | alternative Apache-2.0 |
| `fasterrcnn_resnet18_neudet.yaml` | Faster R-CNN resnet18 ImageNet | NEU-DET | 100 % PyTorch |
| `fasterrcnn_resnet50_coco_neudet.yaml` | Faster R-CNN resnet50 v2 COCO | NEU-DET | le plus precis, le plus lent |
| `rfdetr_nano_neudet.yaml` | RF-DETR nano COCO, 384 px | NEU-DET | transformer, Apache-2.0 |
| `rfdetr_base_neudet.yaml` | RF-DETR base COCO, 448 px | NEU-DET | plus precis, GPU |
| `yolo11n_obb_tile.yaml` | YOLO11n-obb | Magnetic Tile (masques -> boites orientees) | defauts fins et en biais |
| `yolo11n_seg_tile.yaml` | YOLO11n-seg | Magnetic Tile (masques -> polygones) | surface et forme de chaque defaut |

Les configurations NEU-DET partagent `outputs/detection/neu_det_split.csv`.

## Resultats obtenus

Essais rapides sur CPU (2 coeurs partages, 168 images d'entrainement, 36 de test, 1 a 2
epoques) : ils verifient la chaine complete, pas la precision.

| essai | moteur | entrainement (s) | mAP50 test | mAP50-95 test | verification export (IoU moyen ONNX / moteur) | ONNX CPU (ms) |
|---|---|---|---|---|---|---|
| quick_yolo | YOLO11n | 41 | 0.130 | 0.052 | 1.000 (27 boites / 27) | 26 |
| quick_yolox | YOLOX-tiny | 470 | 0.137 | 0.076 | 1.000 (5 / 5) | 55 |
| quick_fasterrcnn | Faster R-CNN r18 | 561 | 0.068 | 0.032 | 1.000 (118 / 118) | 343 (OpenVINO 77) |
| quick_rfdetr | RF-DETR nano, poids aleatoires | 1084 | 0.002 | 0.001 | 1.000 (1200 / 1200 ; scores a 3e-8) | 160 (OpenVINO 46) |
| quick_obb_tile | YOLO11n-obb | 13 | 0.000 | 0.000 | sorties brutes OK | 18 |
| quick_seg_tile | YOLO11n-seg | 13 | 0.000 | 0.000 | sorties brutes OK | 27 |

RF-DETR part de poids aleatoires dans cet essai (les poids COCO, sur le stockage de Roboflow,
etaient inaccessibles depuis l'environnement de test) : sans pre-entrainement, un transformer
n'apprend rien en une epoque. Les configurations completes (60 epoques et plus) sont a lancer sur GPU ; la procedure et le
tableau a remplir sont dans `prompts/` (references publiees pour NEU-DET : mAP50 de l'ordre de
0,7 a 0,8 selon le modele).

## Adapter a ses donnees

1. Annoter (recette `annotation` : outils, formats, controle) ; tout format lisible par
   `vtb.annotate` convient, `annotation_format:` le dit.
2. Masques seulement ? `annotation_format: masks` avec `annotation_options.kind` = `boxes`,
   `rotated` (OBB) ou `polygons` (instances) ; `object_label_from: folder` si la classe est le
   dossier.
3. Images tres grandes (bande en camera lineaire) : `tiling.enabled: true` (tuiles avec
   recouvrement, boites fusionnees par NMS) ; entrainer alors sur des tuiles a la meme echelle
   (`vtb.preprocess.tiles`).
4. Choisir `conf_predict` sur la validation (pas sur le test) d'apres le point de
   fonctionnement voulu : rappel minimal sur les defauts graves, fausses alarmes tolerables.
