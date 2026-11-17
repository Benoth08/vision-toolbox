# Vision classique : seuils, blobs, pièce de référence, mesure

Avant d'entraîner un réseau, il faut savoir ce que donne la vision classique, sans apprentissage. Deux raisons. Elle résout seule une bonne part des contrôles industriels : contraste franc entre défaut et fond, pièces identiques et bien positionnées, mesures dimensionnelles. Et quand elle ne suffit pas, elle fournit le chiffre qu'un modèle appris doit battre pour justifier sa complexité (données à annoter, entraînement, GPU, maintenance).

Le module `vtb.classical` regroupe quatre outils : le seuillage, l'analyse de blobs, la comparaison à une pièce de référence et la mesure au sous-pixel avec calibration. La recette `recipes/classical/` en fait une base de comparaison complète, réglée sur la validation et évaluée une fois sur le test, avec les mêmes métriques que les recettes de segmentation et d'anomalies.

## Quand la vision classique suffit

| Situation | Méthode classique | Quand passer à l'apprentissage |
|---|---|---|
| Défaut nettement plus sombre ou plus clair qu'un fond uniforme | correction d'illumination, seuil, blobs | fond texturé qui répond autant que les défauts |
| Pièces identiques et bien positionnées (étiquette, circuit, pièce usinée) | comparaison à une référence alignée | variations naturelles de la pièce (coulée, laminage, bois, tissu) |
| Mesure d'une largeur, d'un diamètre, d'une position | bord au sous-pixel, calibration | presque jamais : la mesure reste classique |
| Comptage d'objets séparés | seuil, blobs, filtre de forme | objets qui se touchent, se chevauchent ou varient beaucoup |
| Défauts d'aspect variés sur surface naturelle | — | classification, détection, segmentation ou anomalies |

## Le seuillage

Le seuillage transforme une image, ou une carte de réponse fabriquée par le prétraitement, en masque binaire. `vtb.classical.threshold` en propose quatre :

| Méthode | Principe | Quand l'utiliser | Piège | OpenCvSharp |
|---|---|---|---|---|
| `otsu` | sépare deux populations de niveaux en maximisant la variance entre elles | objet et fond de surfaces comparables | sur une image sans défaut, il coupe quand même en deux et invente des défauts | `Cv2.Threshold(..., ThresholdTypes.Otsu)` |
| `triangle` | un pic dominant et une queue | petits défauts sur un grand fond | même piège qu'Otsu | `ThresholdTypes.Triangle` |
| `adaptive` | seuil local : moyenne de la fenêtre moins `c` | éclairage qui varie dans l'image | la fenêtre doit être plus grande que les défauts | `Cv2.AdaptiveThreshold` |
| `fixed` | seuil constant | **production**, une fois réglé sur la validation | dépend du niveau absolu : figer l'éclairage ou corriger avant | `Cv2.Threshold(..., ThresholdTypes.Binary)` |

:::warning
Otsu et triangle choisissent un seuil **image par image**. Sur une image sans défaut, ils trouvent toujours un seuil et produisent donc toujours des « défauts ». En production, on règle un seuil fixe sur la validation ; `threshold_value` donne la valeur qu'Otsu aurait choisie, pour partir d'un ordre de grandeur.
:::

Sur une tuile magnétique, la carte de réponse est fabriquée par le prétraitement : filtre médian, correction d'illumination, puis black-hat, qui ne garde que les petites structures sombres (les trous). Les quatre seuils donnent ensuite des masques très différents :

{{img:docs/assets/classical/nb_thresholds.jpg|Tuile magnétique (blowhole) : image, vérité, puis Otsu, triangle, adaptatif et seuil fixe sur la carte black-hat.}}

## L'analyse de blobs

Chaque composante connexe du masque devient une ligne d'un tableau : aire, boîte, centre, périmètre, circularité (`4π aire / périmètre²`, 1 pour un disque), élongation (rapport des axes de l'ellipse équivalente, calculée par les moments), angle, intensité moyenne et maximale. On filtre ensuite sur la taille et la forme, ce qui suffit souvent à séparer des familles de défauts :

```python
from vtb.classical import blob_table, filter_blobs, threshold

table = blob_table(threshold(image, method="fixed", value=60), image)
pits = filter_blobs(table, min_area=20, max_elongation=2.0)      # piqûres : petites et rondes
scratches = filter_blobs(table, min_elongation=5.0)              # rayures : allongées
```

{{img:docs/assets/classical/nb_blobs.jpg|Trois disques et deux traits : le filtre d'élongation sépare piqûres et rayures.}}

Côté C#, le tableau se reconstruit avec `Cv2.ConnectedComponentsWithStats` (aire, boîte, centre), `Cv2.FindContours` et `Cv2.ArcLength` (périmètre), `Cv2.Moments` (élongation et angle).

:::tip
Le filtre d'aire minimale est le réglage le plus rentable : il retire d'un coup le bruit, les poussières et les petits résidus d'alignement. La recette le règle sur la validation en même temps que le seuil.
:::

## La comparaison à une pièce de référence

Quand les pièces sont identiques, tout écart significatif à une pièce bonne est un défaut. La chaîne est courte :

1. **Construire la référence** : la médiane d'une soixantaine d'images normales d'entraînement. La médiane ignore un défaut qui serait présent sur une seule image, et lisse le bruit.
2. **Aligner** chaque image sur la référence : ECC pour de petits décalages (translation, rotation), ORB et homographie pour des déplacements plus grands.
3. **Lisser** les deux images (sigma 1 à 2) pour absorber le bruit et le résidu d'alignement au sous-pixel, puis **soustraire**.
4. **Seuiller** l'écart (en fraction de la dynamique) et **filtrer** les blobs par aire.

{{file:src/vtb/classical/compare.py|20-46}}

{{img:docs/assets/classical/nb_reference.jpg|MVTec bottle : référence médiane, bouteille cassée, écart absolu après alignement ECC, blobs retenus.}}

:::warning
Sans alignement, un décalage d'un seul pixel crée un faux défaut tout le long des contours de la pièce. L'alignement n'est pas une option. Si l'ECC ne converge pas (image trop différente de la référence, pièce absente), OpenCV lève une erreur : une image impossible à recaler se traite à part (rejet, contrôle manuel), elle ne se compare pas telle quelle.
:::

## Les résultats des deux bases de comparaison

`recipes/classical/scripts/baseline.py` règle le seuil et l'aire minimale sur une grille, en validation, puis lit le test une seule fois. Le critère de choix se déclare dans le YAML : `dice` (la meilleure localisation) ou `recall_at_max_false_alarm` (le meilleur rappel sans dépasser un taux de fausses alarmes sur les images normales, le critère qui compte sur une ligne). Les métriques sont celles des autres recettes : IoU et Dice **cumulés** sur tout le test, rappel et taux de fausses alarmes au niveau de l'image.

| Configuration | Test | Seuil, aire min. | IoU pixel | Dice pixel | Rappel image | Fausses alarmes |
|---|---|---|---|---|---|---|
| `mvtec_bottle_reference.yaml` : référence, critère « 0 fausse alarme » | 50 images, 38 défauts | 0,25 ; 60 px | 0,044 | 0,084 | 0,605 | 0,000 |
| `magnetic_tile.yaml` : réponse black-hat, critère Dice | 99 images | 0,04 ; 50 px | 0,058 | 0,109 | 1,000 | 1,000 |

{{img:docs/assets/classical/mvtec_bottle_reference_examples.jpg|MVTec bottle, test : vérité en vert, prédiction en rouge.}}

**Sur les bouteilles**, la comparaison à une référence détecte 60 % des bouteilles défectueuses sans une seule fausse alarme, sans aucun apprentissage. Pour situer l'ordre de grandeur (les jeux de test ne sont pas les mêmes) : l'auto-encodeur convolutif du cours atteint un rappel de 0,38 à zéro fausse alarme sur ces bouteilles, PatchCore 1,00. La localisation au pixel reste grossière (Dice 0,08) : l'écart à la référence marque les contours du défaut et les bords de la bouteille plus que sa surface.

**Sur les tuiles magnétiques**, le seuillage échoue : la texture du matériau et les bords sombres de la tuile répondent au black-hat autant que les trous et les fissures. Le meilleur réglage au sens du Dice signale un défaut sur toutes les images, y compris les bonnes. C'est la limite de la vision classique sur une surface naturelle, et le chiffre que bat largement le U-Net du cours sur le même jeu (IoU de 0,57 à 0,63 sur le test).

{{img:docs/assets/classical/magnetic_tile_examples.jpg|Magnetic Tile, test : vérité en vert, prédiction en rouge. La texture déclenche partout.}}

:::memo
Une base classique qui échoue n'est pas du temps perdu : elle prouve que le problème demande un modèle appris, et donne le chiffre de départ du rapport. Une base qui réussit peut éviter tout le projet d'apprentissage.
:::

## La mesure au sous-pixel

Une mesure dimensionnelle (largeur d'une bande, diamètre d'un trou, position d'un bord) reste presque toujours une méthode classique : elle est exacte, explicable et ne demande aucune donnée annotée.

**Le bord au sous-pixel.** Sur un profil d'intensité perpendiculaire au bord, lissé, le bord est au maximum de la dérivée. La parabole qui passe par ce maximum et ses deux voisins donne sa position à quelques centièmes de pixel (`strongest_edge`). Sur un bord synthétique à 33,37 px avec du bruit, la position trouvée est 33,31 px, soit 0,07 px d'écart.

**La largeur d'une bande.** `strip_width` mesure, ligne par ligne, les bords gauche et droit d'une bande claire (ou sombre) et leur écart. La moyenne sur les lignes réduit encore le bruit ; l'écart-type entre lignes dit si la bande est droite et si la mesure est stable.

{{img:docs/assets/classical/nb_measure.jpg|Profil d'une ligne d'une bande synthétique et les deux bords trouvés au sous-pixel.}}

**La calibration.** Pour passer des pixels aux millimètres :

- `calibrate_from_distance(pixels, mm)` : une cale étalon ou une mire de longueur connue dans le champ donne l'échelle. Suffisant pour une caméra perpendiculaire à un plan, avec un objectif peu déformant.
- `calibrate_chessboard(images, pattern, square_mm)` : 10 à 20 photos d'un damier sous des angles variés donnent la matrice de la caméra, la distorsion de l'objectif, l'erreur de reprojection (sous 0,5 px pour une bonne calibration, au-delà de 1 px refaire les photos) et l'échelle. `save_calibration` l'enregistre en JSON, que l'opération de prétraitement `undistort` relit.

Le script `measure_demo.py` vérifie toute la chaîne sur des images synthétiques dont on connaît la vérité :

{{text:docs/assets/classical/measure.log|0}}

Huit vues d'un damier, erreur de reprojection 0,66 px, échelle retrouvée exactement (0,25 mm par pixel) ; une bande de 60,25 mm mesurée à 60,255 mm, avec 0,03 mm d'écart-type entre lignes.

:::note
Une mesure en millimètres n'est valable que dans le plan de la calibration. Si la pièce s'approche ou s'éloigne de la caméra (tôle qui flotte, pièce posée de travers), l'échelle change : il faut un objectif télécentrique, une mesure de distance, ou une mire dans chaque image.
:::

## Les règles

- Essayer la vision classique en premier, au moins comme base de comparaison, et rapporter son chiffre à côté de celui du modèle.
- Régler seuils et aires sur la validation, jamais sur le test ; en production, un seuil fixe, pas Otsu.
- Corriger l'illumination avant tout seuil : un seuil fixe sur un éclairage inégal ne veut rien dire.
- Toujours aligner avant de comparer à une référence.
- Juger avec les métriques des autres recettes (IoU cumulé, rappel au taux de fausses alarmes accepté), pour que la comparaison soit honnête.
