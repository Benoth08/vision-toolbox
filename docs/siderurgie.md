# Cas d'usage en sidérurgie : quelle recette pour quel problème

Une usine sidérurgique offre presque tous les problèmes de vision industrielle : surfaces qui défilent à plusieurs mètres par seconde, images très grandes et en niveaux de gris, défauts rares et variés, éclairages difficiles (métal réfléchissant, calamine, vapeur), caméras thermiques 16 bits, marquages à lire, mesures à faire au dixième de millimètre. Cette page associe les cas les plus fréquents aux recettes de la toolbox et aux réglages qui comptent.

## La table de décision

| Question posée | Données disponibles | Recette | Réglages clés |
|---|---|---|---|
| Cette image de bande contient-elle un défaut, et lequel ? | images étiquetées par type | `classification` | `strategy: partial`, `color: gray`, découpage par bobine (`group_regex`) |
| Où sont les défauts sur la bande, de quel type ? | boîtes ou masques | `detection` (YOLO11, YOLOX, Faster R-CNN, RF-DETR) | tuiles pour les images de caméra linéaire, `conf_predict` réglé sur la validation ; YOLOX ou RF-DETR pour un logiciel livré (Apache-2.0) |
| Quelle est la surface ou la longueur du défaut ? | masques au pixel | `segmentation` | `dice_ce`, `min_defect_area`, découpage groupé |
| Le défaut est rare, on n'en a presque pas d'exemples | images saines seulement | `anomaly` | seuil au quantile 0,99 des saines de validation, PatchCore ou PaDiM |
| Mesurer une largeur, une position de rive, un diamètre | images calibrées | `classical` | mire de calibration, bord au sous-pixel |
| Contrôle de présence, de conformité d'une pièce fixe | une image de référence | `classical` (comparaison à la référence) | alignement ECC, seuil sur la différence |
| Le modèle marche-t-il encore ce mois-ci ? | images récentes | `explore` (dérive) | AUC du classifieur de domaine chaque semaine |

## Inspection de surface de bande (laminage à chaud et à froid)

C'est le cas de NEU-CLS et NEU-DET : calamine incrustée, rayures, piqûres, inclusions, plaques, craquelures. Les images viennent de caméras linéaires qui balaient la bande : une image fait des milliers de pixels de large.

- **Prétraitement** : correction des profils de colonnes (`column_profile`, une colonne de capteur plus sensible que sa voisine laisse une bande verticale), division par le fond flou (`shading_division`) quand l'éclairage n'est pas uniforme sur la largeur, puis éventuellement CLAHE. Toujours juger à la métrique (`recipes/preprocessing/scripts/compare.py`) : sur NEU-CLS, aucun prétraitement n'apporte de gain mesurable à un réseau pré-entraîné.
- **Grandes images** : détection par tuiles avec recouvrement (`tiling.enabled: true`), entraînement sur des tuiles à la même échelle.
- **Découpage** : toutes les images d'une même bobine du même côté (`group_regex` sur le numéro de bobine dans le nom de fichier), et un test temporel (`strategy: temporal`) pour mesurer ce qui se passera sur les bobines de demain.

## Défauts rares : l'approche par anomalies

Les défauts graves sont rares par nature : quelques exemples par an, jamais deux identiques. Les méthodes d'anomalie apprennent l'aspect **normal** de la surface et signalent l'écart, sans exemple de défaut à l'entraînement. Les défauts servent seulement à évaluer.

- **PatchCore** (mémoire de vecteurs normaux, distance au plus proche) : la référence, sans entraînement, très bon sur textures ; la version de la toolbox s'exporte en ONNX d'un bloc.
- **PaDiM** (une gaussienne par position) : plus léger, demande des images alignées.
- **Auto-encodeur** : simple, explicable, plus faible sur les textures fines.

Le seuil se règle sur des images saines de validation (quantile 0,99 : une fausse alarme sur cent pièces saines) ; le rappel se mesure ensuite par type de défaut sur le test. Sur Magnetic Tile, la surface texturée du banc la plus proche de l'acier, PatchCore atteint une AUROC image de 0,965 sans entraînement, mais ne rattrape que 65 % des défauts au seuil 0,99 (fissures et zones irrégulières surtout) : sur une texture, l'anomalie sert d'abord à trier et à collecter des défauts, puis un modèle supervisé prend le relais dès qu'il y en a quelques dizaines par type (voir l'article Anomalies).

## Marquage et traçabilité

Numéros de bobine peints, frappes à froid sur brames, étiquettes : la chaîne est **détection** de la zone (recette `detection`) puis **lecture** (OCR). La lecture n'est pas encore une recette de la toolbox ; `prompts/07_extensions_ocr_tracking.md` décrit comment l'ajouter (OCR pré-entraîné comparé à un CRNN entraîné sur des marquages synthétiques).

## Microscopie métallographique

Taille de grain, inclusions, phases : images de microscope avec un vignettage marqué et une illumination qui change d'un champ à l'autre.

- **Correction d'illumination** : `flat_field` si l'on dispose d'une image de lame vide (la meilleure correction : écart centre / coins ramené de 60 à 1 niveau de gris sur un vignettage connu), sinon `shading_division` (division par le fond flou, écart ramené à 6).
- **Segmentation** des grains ou des phases (recette `segmentation`, `mask_mode: from_values` avec un masque d'indices), puis mesures par la recette `classical` (blobs : aire, élongation, orientation).

## Caméras thermiques et images 16 bits

Les caméras thermiques et certaines caméras de surface livrent des images 16 bits dont la dynamique utile n'occupe qu'une petite partie. La toolbox lit et conserve le 16 bits de bout en bout (lecture, prétraitement, tenseur divisé par 65 535) ; `window_level` ramène une fenêtre de températures en 8 bits quand c'est nécessaire (affichage, modèle pré-entraîné). Le modèle `recipes/preprocessing/configs/thermal16.yaml` est un point de départ.

## Ce qui fait échouer un projet (et comment la toolbox s'en protège)

| Piège | Symptôme | Protection |
|---|---|---|
| images d'une même bobine dans train et test | 99 % au test, 80 % en production | découpage par groupe, contrôle des fuites, `groups_csv` des quasi-doublons |
| classes acquises à des moments différents | le modèle apprend l'éclairage | contrôle des raccourcis (`check_dataset.py`) |
| prétraitement réécrit à la main en production | réponses différentes de l'évaluation | prétraitement nommé dans la carte, vérification golden C# |
| seuil choisi sur le test | performance surestimée | seuils calculés sur la validation uniquement |
| dérive de l'acquisition | taux de défauts manqués qui monte en silence | `drift.py` chaque semaine |
| export non vérifié | boîtes décalées, scores faux | export qui échoue au-delà de la tolérance |
