# Détection : quatre moteurs, un seul contrat

La détection répond à deux questions à la fois : **où** est chaque défaut (une boîte) et **de quel type** il est (une classe). La toolbox propose quatre moteurs, choisis par une seule clé du YAML (`backend:`), et tout le reste est commun : le format des annotations, le découpage, l'évaluateur, la carte du modèle. C'est ce qui permet de les comparer honnêtement, sur les mêmes images, avec les mêmes chiffres.

| Moteur | Famille | Licence | Points forts |
|---|---|---|---|
| `ultralytics` (YOLO11) | un étage, sans ancres | AGPL-3.0 | le plus simple ; boîtes, boîtes orientées (OBB) et segmentation d'instances |
| `yolox` (Megvii) | un étage, sans ancres | Apache-2.0 | précision proche de YOLO, utilisable dans un produit fermé |
| `fasterrcnn` (torchvision) | deux étages | BSD | 100 % PyTorch (le squelette du cours) ; bon sur les petits objets |
| `rfdetr` (Roboflow) | transformer (DETR), sans NMS | Apache-2.0 (nano à large) | dos DINOv2 : transfert très bon loin de COCO ; aucune suppression des non-maxima |

:::warning
La licence AGPL d'Ultralytics impose de publier le code d'un produit qui l'intègre, ou d'acheter une licence commerciale. Pour un logiciel livré à un client, YOLOX, Faster R-CNN et RF-DETR (tailles nano à large) évitent la question.
:::

## RF-DETR : un transformer de détection, sans suppression des non-maxima

Les YOLO et Faster R-CNN prédisent des milliers de boîtes candidates, puis éliminent les doublons par suppression des non-maxima (NMS), un réglage de plus et une étape à réécrire en production. RF-DETR suit l'approche DETR : le décodeur porte quelques centaines de **requêtes**, chacune prédit au plus un objet, et l'entraînement apparie chaque vraie boîte à une seule requête (appariement hongrois). Une requête qui « double » une autre est pénalisée : le modèle apprend lui-même à ne pas dupliquer, et la sortie se lit directement.

Son dos est DINOv2, un vision transformer appris sans étiquettes sur 142 millions d'images. Ses caractéristiques se transfèrent remarquablement bien vers des domaines éloignés de COCO, ce qui en fait un bon candidat pour des surfaces industrielles.

| Taille | Résolution par défaut | Licence |
|---|---|---|
| nano | 384 | Apache-2.0 |
| small | 512 | Apache-2.0 |
| medium | 576 | Apache-2.0 |
| base | 560 | Apache-2.0 |
| large | 704 | Apache-2.0 |

Les tailles Atto, Femto, Pico, XLarge et 2XLarge viennent d'un paquet d'extension (`rfdetr_plus`) et sont sous une autre licence, la PML 1.0 de Roboflow : à lire avant tout usage commercial. Cette répartition est celle de la documentation de RF-DETR à la mi-novembre 2026 ; elle a déjà changé d'une version à l'autre, donc vérifie celle de la version que tu installes.

Le post-traitement de production tient en quelques lignes, sans NMS : sigmoïde de chaque logit (requête × classe), les 300 meilleurs couples, puis les boîtes `cxcywh` normalisées multipliées par la taille de l'image. L'entrée est l'image redimensionnée **sans garder les proportions** (pas de letterbox), normalisée comme ImageNet.

{{file:src/vtb/tasks/detection/rfdetr_backend.py|181-208}}

## Le format pivot des annotations

Les annotations arrivent dans tous les formats : Pascal VOC (NEU-DET), YOLO, COCO, LabelMe, Label Studio, CVAT, ou simplement des masques de défauts. `prepare` les lit avec `vtb.annotate`, les convertit en un format pivot (une liste d'`ImageAnnotation` : boîtes, boîtes orientées, polygones), découpe **par image** et écrit deux fichiers : `split.csv` et `split.vtb.json`. Chaque moteur relit le pivot et le convertit dans son propre format (Ultralytics : dossiers YOLO régénérés à chaque entraînement, avec le même prétraitement qu'en production).

Des masques de défauts suffisent pour entraîner un détecteur : `annotation_format: masks` avec `kind: boxes` (boîtes droites), `kind: rotated` (boîtes orientées, par rectangle d'aire minimale de chaque composante) ou `kind: polygons` (contours, pour la segmentation d'instances).

{{img:docs/assets/annotation/magnetic_tile_masks_to_objects.png|Un masque de défaut de Magnetic Tile converti en boîte droite, boîte orientée et polygone.}}

## Mesurer : le mAP et le point de fonctionnement

L'évaluateur est celui de COCO (pycocotools), le même pour les trois moteurs. Le **mAP@0,5** compte une détection juste si son IoU avec une vraie boîte de la même classe atteint 0,5 ; le **mAP@0,5:0,95** moyenne dix seuils d'IoU et récompense les boîtes précises. Les deux résument toute la courbe précision-rappel, d'où un seuil de confiance très bas pendant l'évaluation (`conf_eval: 0.001`).

Mais l'opérateur ne voit pas une courbe : il voit les détections au-dessus du seuil de production `conf_predict`. La toolbox calcule donc aussi le **point de fonctionnement** à ce seuil : précision, rappel, et deux chiffres par image (part des images défectueuses signalées, part des images saines signalées à tort).

{{file:src/vtb/evaluate/detection.py|105-160}}

:::tip
Choisir `conf_predict` sur la **validation**, d'après ce qui coûte le plus cher : un défaut grave manqué, ou une fausse alarme qui arrête la ligne. Le test ne sert qu'à mesurer, une fois.
:::

## Les pièges de coordonnées

La source de bugs la plus fréquente en détection n'est pas le modèle : c'est le **repère** des boîtes. Une image passe par trois repères : l'image d'origine, l'image prétraitée (après un éventuel redimensionnement), puis l'entrée du réseau (carré de 320 ou 640 pixels). Chaque moteur prépare l'entrée différemment, et se tromper ne lève aucune erreur : les boîtes sont juste fausses.

| Moteur | Mise à l'échelle | Couleur | Valeurs |
|---|---|---|---|
| YOLO11 | letterbox **centré**, bordure grise 114 | RGB | ÷ 255 |
| YOLOX | letterbox **en haut à gauche**, bordure 114 | **BGR** | 0-255, sans division |
| Faster R-CNN | redimensionnement interne au modèle | RGB | ÷ 255 |
| RF-DETR | redimensionnement **sans proportions** (carré) | RGB | ÷ 255 puis moyenne / écart-type ImageNet |

Le chemin de prédiction est le même partout (`predict_paths` dans `vtb/tasks/detection/workflow.py`) : lecture, prétraitement, moteur, retour aux coordonnées d'origine. L'évaluation l'utilise aussi, ce qui garantit que les chiffres du test sont ceux de la production.

## Exporter et vérifier deux fois

L'export ONNX des YOLO sort les prédictions **brutes**, avant la suppression des non-maxima : c'est au programme de production de décoder et de filtrer. La toolbox écrit ce post-traitement en clair, en numpy, puis vérifie deux choses : les sorties brutes ONNX contre le réseau PyTorch, et les **boîtes** obtenues par la chaîne ONNX + post-traitement contre celles du moteur. L'export échoue si l'IoU moyen descend sous 0,98.

{{file:src/vtb/tasks/detection/ultralytics_backend.py|186-215}}

Sur les essais rapides, les quatre moteurs reproduisent leurs boîtes : IoU moyen 1,000 pour YOLO11 (27 boîtes sur 27), YOLOX (5 sur 5) et Faster R-CNN (118 sur 118) . Pour RF-DETR, la vérification porte sur les 300 meilleurs couples de chaque image (1 200 boîtes sur 1 200) : scores triés identiques à 3·10⁻⁸ près, et IoU de 1,000 sur les boîtes dont le score se distingue des autres (entre deux scores ex aequo, l'ordre dépend d'un arrondi et ne prouve rien).

## Résultats des essais rapides

168 images d'entraînement de NEU-DET, 36 de test, une à deux époques sur CPU : ces chiffres vérifient la chaîne, pas la précision (les configurations complètes se lancent sur GPU). RF-DETR part ici de poids aléatoires (les poids COCO n'étaient pas téléchargeables dans l'environnement de test) : un transformer sans pré-entraînement n'apprend rien en une époque, c'est attendu.

| Essai | Moteur | Entraînement (s) | mAP50 | mAP50-95 | ONNX CPU (ms) |
|---|---|---|---|---|---|
| `quick_yolo` | YOLO11n | 41 | 0,130 | 0,052 | 26 |
| `quick_yolox` | YOLOX-tiny | 470 | 0,137 | 0,076 | 55 |
| `quick_fasterrcnn` | Faster R-CNN resnet18 | 561 | 0,068 | 0,032 | 343 (OpenVINO : 77) |
| `quick_rfdetr` | RF-DETR nano, sans poids | 1 084 | 0,002 | 0,001 | 160 (OpenVINO : 46) |

{{img:docs/assets/detection/quick_yolo_predictions.jpg|Vérité terrain (à gauche) et prédictions au seuil de production (à droite) après deux époques de YOLO11n.}}

## Grandes images : l'inférence par tuiles

Une bande d'acier vue par une caméra linéaire fait des milliers de pixels de large ; la réduire à 640 pixels efface les petits défauts. `tiling.enabled: true` découpe l'image en tuiles qui se recouvrent, prédit sur chaque tuile, ramène les boîtes dans l'image et fusionne les doublons des zones de recouvrement par NMS.

{{file:src/vtb/tasks/detection/common.py|63-81}}

:::memo
Règle de production : entraîner à la même échelle que l'inférence. Si l'on prédit par tuiles de 640 pixels à pleine résolution, on entraîne sur des tuiles de 640 pixels à pleine résolution (`vtb.preprocess.tiles`).
:::
