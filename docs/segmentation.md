# Segmentation sémantique : la forme et la surface de chaque défaut

La segmentation sémantique attribue une classe à **chaque pixel** : fond, rayure, inclusion, etc. Elle répond aux questions que ni la classification ni la détection ne traitent bien : quelle est la surface d'une oxydation, la longueur d'une fissure, la part de la bande couverte de calamine ; et elle convient aux défauts sans contour net (traces, taches diffuses) qu'une boîte englobe mal.

| Besoin | Méthode |
|---|---|
| « cette image contient-elle un défaut ? » | classification ou anomalies |
| « où sont les défauts, combien ? » (objets séparés) | détection |
| « quelle forme, quelle surface ? » (classe par pixel) | **segmentation sémantique** |
| « quelle forme pour chaque objet séparé ? » | segmentation d'instances (YOLO11-seg, recette détection) |

## Le modèle : un encodeur pré-entraîné, un décodeur neuf

La recette s'appuie sur segmentation_models_pytorch (licence MIT) : une architecture (U-Net, U-Net++, FPN, DeepLabV3+, PSPNet, LinkNet, MAnet, SegFormer) se combine avec n'importe quel encodeur (resnet18, efficientnet-b0, mit_b0...). L'encodeur est pré-entraîné sur ImageNet ; le décodeur, qui remonte à la résolution de l'image, part de zéro. D'où deux taux d'apprentissage : le décodeur au taux nominal, l'encodeur dix fois moins vite (`encoder_lr_factor: 0.1`).

Pour les encodeurs resnet, les poids viennent de torchvision (le même cache que la classification, utilisable hors ligne). Une image grise est copiée sur trois canaux **dans** le modèle, puis normalisée canal par canal : la réponse est exactement celle des poids ImageNet, et l'entrée de production reste à un canal.

{{file:src/vtb/tasks/segmentation/models.py|48-71}}

## Les masques

Trois façons de lire les masques (`mask_mode`) :

- `binary` : tout pixel non nul est « défaut » (deux classes) ;
- `from_label` : masques binaires dont la classe vient du dossier de l'image (Magnetic Tile : `MT_Crack`, `MT_Blowhole`...), les images saines (`normal_labels`) n'ayant aucune classe ;
- `from_values` : la valeur du pixel est l'indice de classe (0 = fond).

Le masque est ramené à la taille de l'image prétraitée **au plus proche voisin** : une interpolation linéaire créerait des classes intermédiaires (1,5 entre fond et défaut) qui n'existent pas. Pour la même raison, seules les opérations photométriques et le redimensionnement sont compatibles avec les masques ; un recadrage déplacerait l'image sans déplacer le masque. Les augmentations, elles, passent par torchvision v2, qui transforme l'image et le masque ensemble.

## La perte et les métriques

Sur une tuile magnétique, le défaut occupe souvent moins de 1 % des pixels. Un modèle qui répond « fond » partout a donc 99 % de pixels justes et ne sert à rien. Deux parades :

- **la perte** `dice_ce` additionne l'entropie croisée (pixel par pixel) et la perte de Dice (recouvrement global de chaque classe, insensible au nombre de pixels de fond) ;
- **les métriques** : IoU et Dice par classe, moyennés sur les classes de **défaut** (`miou_defects`), jamais sur le fond seul.

La décision de production est souvent par image : « au moins `min_defect_area` pixels de défaut ». La recette rapporte donc aussi le rappel par image et le taux de fausses alarmes par image.

## Résultats

Magnetic Tile complet (1 344 images), découpage **groupé** sur les quasi-doublons (la même tuile sous plusieurs expositions reste d'un seul côté), U-Net resnet18 à 128 pixels, 15 époques sur CPU (640 secondes) :

| Mesure (test, 269 tuiles dont 79 défectueuses) | Valeur |
|---|---|
| IoU défauts | 0,705 |
| Dice défauts | 0,827 |
| IoU fond | 0,992 |
| Rappel image (au moins 10 pixels de défaut) | 0,61 |
| Fausses alarmes image | 0,016 |
| Latence ONNX CPU (onnxruntime / OpenVINO) | 10 ms / 2,5 ms |

{{img:docs/assets/segmentation/unet_cpu_predictions.jpg|Image, vérité terrain et prédiction du U-Net sur des tuiles de test.}}

{{img:docs/assets/segmentation/unet_cpu_history.jpg|Perte d'entraînement et IoU de validation au fil des époques.}}

La segmentation elle-même est bonne (Dice 0,83), mais la décision par image reste prudente : à 128 pixels, les plus petits trous ne font que quelques pixels et passent sous `min_defect_area`. Les deux leviers se règlent sur la validation, jamais sur le test : la résolution (256 pixels dans la configuration GPU) et le couple seuil de probabilité / surface minimale.

## Export

La normalisation et le softmax sont dans le graphe ONNX : le programme de production envoie une image dans [0, 1] et reçoit une probabilité par classe et par pixel. La carte du modèle donne le seuil de probabilité (`defect_probability`) et la surface minimale (`min_defect_area`).
