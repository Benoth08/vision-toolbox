# Le prétraitement dans la toolbox

**Dans le fil : étape 5 (Prétraiter).** C'est le même numéro dans le cours, dans les notebooks et dans l'index `ETAPES.md`.

Un prétraitement est une opération **déterministe** appliquée à l'image avant le réseau : la même image d'entrée donne toujours la même image de sortie. Il se distingue de l'augmentation, qui est **aléatoire** et n'existe qu'à l'entraînement. Cette différence fixe une règle de production : le prétraitement fait partie du modèle. Il est appliqué à l'identique à l'entraînement, en prédiction, à l'export, et dans le programme de l'usine (C#, C++), dans le même ordre et avec les mêmes paramètres.

L'ordre est fixe dans toute la toolbox :

```
lecture -> Pipeline (OpenCV, déterministe) -> to_float_chw ([0, 1]) -> augmentation (train seulement) -> normalisation
```

Un pipeline se déclare dans le YAML de n'importe quelle recette, sous `preprocess:` :

```yaml
preprocess:
  - {name: shading_division, sigma: 40}
  - {name: clahe, clip_limit: 2.0}
  - {name: resize, size: [256, 256]}
```

`vtb.preprocess.Pipeline` valide chaque nom et chaque paramètre à la construction : une faute de frappe échoue au démarrage, pas au milieu d'un entraînement. `Pipeline.describe()` donne la liste exacte, valeurs par défaut comprises, qui est recopiée dans `model_card.json`.

```python
from vtb.preprocess import Pipeline, list_ops

list_ops()                     # le catalogue : famille, nom, résumé, paramètres par défaut
pipeline = Pipeline([{"name": "shading_division", "sigma": 40}, {"name": "clahe"}])
corrected = pipeline(image)    # image : ndarray (H, W) ou (H, W, 3) RGB, uint8 ou uint16
pipeline.describe()            # pour la carte du modèle
```

Toutes les opérations respectent le même contrat : entrée en niveaux de gris ou en RGB, en 8 ou 16 bits ; sortie du même type (sauf `to_uint8` et `window_level`, qui passent en 8 bits, et `to_gray` / `extract_channel`, qui rendent une image (H, W)). Le catalogue compte 49 opérations en 12 familles ; chacune est testée dans les quatre combinaisons gris / couleur × 8 / 16 bits, et refuse avec un message clair ce qu'elle ne sait pas traiter (`extract_channel` sur une image déjà grise, par exemple).

:::note
Les équivalents OpenCvSharp indiqués dans les tableaux sont ceux à appeler côté C#. Ce sont les mêmes fonctions OpenCV, à deux pièges près : OpenCvSharp lit les images en **BGR** (la toolbox travaille en RGB), et les paramètres en niveaux de gris sont exprimés sur 0-255.
:::

## La correction d'illumination (shading)

C'est souvent le prétraitement qui compte le plus en inspection : un champ de microscope plus clair au centre, un vignettage d'objectif, un éclairage rasant inégal, une caméra linéaire dont la sensibilité varie le long du capteur. Le modèle physique est simple :

```
I = R x E + B
```

L'image mesurée `I` est la réflectance de l'objet `R` (ce qui nous intéresse), multipliée par l'éclairage `E`, plus un fond additif `B` (lumière parasite, offset du capteur). Les six variantes estiment `E` ou `B` et les retirent :

| Opération | Calcul | Quand l'utiliser |
|---|---|---|
| `shading_division` | `I / flou(I) × moyenne(flou)` | éclairage multiplicatif : microscope, vignettage, éclairage inégal |
| `shading_subtraction` | `I − flou(I) + moyenne` | fond additif (lumière parasite, offset) |
| `morphological_background` | retire un fond estimé par ouverture (objets clairs) ou par fermeture (objets sombres) | petits objets sur fond variable, à la manière du « rolling ball » d'ImageJ |
| `flat_field` | `(I − noir) / (blanc − noir) × moyenne(blanc − noir)` | la méthode exacte quand on peut acquérir une image sans objet (lame vide, plaque blanche) |
| `column_profile` | chaque colonne divisée par son profil médian | caméras linéaires : l'inhomogénéité est dans la largeur de la bande |
| `polynomial_background` | `I / surface polynomiale` ajustée en deux passes | gros objets qu'un flou « verrait » comme du fond |

**Le réglage de `sigma`.** Le flou doit être plus large que les objets, sinon le « fond » estimé contient les objets et la division les efface ; il doit rester plus petit que l'échelle de variation de l'éclairage, sinon le fond estimé est plat et rien n'est corrigé. Règle de départ : 1/10 à 1/5 de la taille de l'image pour un éclairage doux ; moins (1/16) quand l'éclairage varie vite et que les objets sont fins. La variante `method: median` résiste mieux à de gros objets brillants.

**La rapidité.** Le flou large est calculé sur l'image réduite d'un facteur `downscale` puis agrandi : pour un éclairage lisse, le résultat est le même et le calcul est des dizaines de fois plus rapide. C'est ce qui rend la correction utilisable à la cadence d'une ligne.

{{file:src/vtb/preprocess/ops/illumination.py|32-81}}

Sur un vignettage connu (l'image de gauche, propre, est assombrie sur les bords), les six corrections donnent :

{{img:docs/assets/preprocessing/nb_shading_variants.jpg|Les six corrections sur un vignettage artificiel (NEU-CLS, classe rolled, sigma 12).}}

Le profil moyen des colonnes le montre mieux que l'œil : il redevient plat.

{{img:docs/assets/preprocessing/nb_shading_profile.jpg|Profil moyen des colonnes : propre, vignetté, après shading_division et flat_field.}}

Écart moyen entre le centre et les coins, en niveaux de gris : image propre 1,1 ; vignettée 60,1 ; `shading_division` 5,9 ; `shading_subtraction` 4,7 ; `polynomial_background` 5,7 ; `morphological_background` 19,9 ; `column_profile` 34,3 (normal : le vignettage est radial, pas par colonne) ; `flat_field` 1,0. Avec une image de référence, la correction est presque parfaite : c'est la méthode à privilégier dès qu'on peut photographier une lame vide ou une plaque blanche sous le même éclairage.

Sur une vraie image, la bouteille MVTec éclairée par-dessous : la division rend le fond uniforme, la soustraction garde mieux les contrastes locaux. Le choix entre les deux se fait à la métrique.

{{img:docs/assets/preprocessing/nb_shading_bottle.jpg|Bouteille MVTec : origine, shading_division, shading_subtraction (sigma 60).}}

{{img:docs/assets/preprocessing/gallery_illumination.jpg|Les corrections d'illumination avec leurs paramètres par défaut sur cinq images d'exemple.}}

## Le catalogue, famille par famille

Chaque planche montre en première ligne les images d'origine (NEU-CLS crazing, scratches, patches ; une tuile magnétique ; une bouteille MVTec), puis une ligne par opération avec ses paramètres par défaut. Les opérations qui ont besoin d'un fichier ou de coordonnées propres à une installation (`flat_field`, `perspective`, `undistort`, alignements) n'y figurent pas.

### Intensité

| Opération | À quoi ça sert | Piège | OpenCvSharp |
|---|---|---|---|
| `percentile_stretch` | étire [p1, p99] sur toute la dynamique, robuste aux pixels aberrants | calculé image par image : l'échelle change d'une image à l'autre | percentiles sur l'histogramme puis `Mat.ConvertTo(alpha, beta)` |
| `minmax_normalize` | étire [min, max] | un seul pixel chaud écrase tout le reste | `Cv2.Normalize(..., NormTypes.MinMax)` |
| `gamma` | gamma < 1 éclaircit les ombres, > 1 les assombrit | change le contraste relatif des défauts | `Cv2.LUT` |
| `log_transform` | comprime une grande dynamique (thermique) | écrase les hautes lumières | `Cv2.Log` sur une Mat flottante |
| `window_level` | fenêtre / niveau : 16 bits → 8 bits sur la plage utile | une fenêtre automatique change l'échelle des températures ; la figer | `Mat.ConvertTo(CV_8U, alpha, beta)` |
| `to_uint8` | 16 → 8 bits (division ou étirement par percentiles) | perte de dynamique si l'image n'occupe qu'une petite plage | `Mat.ConvertTo(CV_8U, 255.0 / 65535.0)` |
| `invert` | rend clairs des défauts sombres | — | `Cv2.BitwiseNot` |

{{img:docs/assets/preprocessing/gallery_intensite.jpg|Famille intensité.}}

### Contraste

| Opération | À quoi ça sert | Piège | OpenCvSharp |
|---|---|---|---|
| `equalize_hist` | répartit les niveaux uniformément | fort, souvent excessif ; amplifie le bruit dans les zones plates | `Cv2.EqualizeHist` |
| `clahe` | contraste local par tuiles, amplification bornée | `clip_limit` > 3 fait ressortir la texture du métal autant que les défauts | `Cv2.CreateCLAHE(clip, new Size(g, g)).Apply` |

En couleur, les deux travaillent sur la luminance (canal L de LAB) pour ne pas changer les couleurs.

{{img:docs/assets/preprocessing/gallery_contraste.jpg|Famille contraste.}}

{{img:docs/assets/preprocessing/nb_clahe_shading.jpg|NEU-CLS : shading_division (sigma 25) puis CLAHE (clip_limit 2, grille 4). La texture ressort autant que les défauts : un réglage se valide à la métrique.}}

### Débruitage

| Opération | À quoi ça sert | Piège | OpenCvSharp |
|---|---|---|---|
| `gaussian_blur` | bruit fin et uniforme | adoucit aussi les bords | `Cv2.GaussianBlur` |
| `median_blur` | pixels isolés (poussières, sel et poivre) en gardant les bords | au-delà de 5 px en 16 bits, OpenCV ne sait pas : la toolbox passe par scipy | `Cv2.MedianBlur` |
| `bilateral` | lisse les zones homogènes, préserve les contours | lent sur de grandes images | `Cv2.BilateralFilter` |
| `nl_means` | le débruitage le plus fort | le plus lent : hors temps réel | `Cv2.FastNlMeansDenoising(Colored)` |

Un débruitage fort efface aussi les défauts fins (piqûres, micro-fissures) : le réglage se valide sur la métrique.

{{img:docs/assets/preprocessing/gallery_debruitage.jpg|Famille débruitage.}}

### Netteté

| Opération | À quoi ça sert | Piège | OpenCvSharp |
|---|---|---|---|
| `unsharp_mask` | accentuation classique : `image + a (image − flou)` | amplifie le bruit | `Cv2.GaussianBlur` + `Cv2.AddWeighted` |
| `laplacian_sharpen` | accentuation plus agressive des détails fins | halos autour des bords | `Cv2.Laplacian` + `Cv2.AddWeighted` |

{{img:docs/assets/preprocessing/gallery_nettete.jpg|Famille netteté.}}

### Fréquentiel

| Opération | À quoi ça sert | Piège | OpenCvSharp |
|---|---|---|---|
| `fft_notch` | éteint les pics périodiques du spectre (automatique, ou fréquences données) | un coupe-bande mal placé crée des anneaux ; en production, figer les fréquences d'un motif fixe | `Cv2.Dft` / `Cv2.Idft` |
| `fft_lowpass` | garde les variations lentes | flou | idem |
| `fft_highpass` | retire éclairage et gradients lents | amplifie le bruit | idem |

Un motif périodique (traces de cylindres de laminoir, rayure fixe d'une caméra linéaire) est un pic étroit dans le spectre ; l'éteindre laisse le reste de l'image, défauts compris.

{{img:docs/assets/preprocessing/nb_fft_notch.jpg|Rayures périodiques ajoutées à une image NEU-CLS, et leur spectre, avant et après fft_notch.}}

{{img:docs/assets/preprocessing/gallery_frequentiel.jpg|Famille fréquentiel.}}

### Échelle et orientation

| Opération | À quoi ça sert | Piège | OpenCvSharp |
|---|---|---|---|
| `difference_of_gaussians` | passe-bande : garde les structures de taille entre deux sigmas | réglage par taille de défaut | deux `Cv2.GaussianBlur` + `Cv2.Subtract` |
| `gabor` | réponse maximale sur plusieurs orientations : textures, rayures d'une période donnée | une période ne convient qu'à une taille de structure | `Cv2.GetGaborKernel` + `Cv2.Filter2D` |

{{img:docs/assets/preprocessing/gallery_echelle.jpg|Famille échelle.}}

### Contours

| Opération | À quoi ça sert | Piège | OpenCvSharp |
|---|---|---|---|
| `sobel`, `scharr` | norme du gradient | sensibles au bruit : lisser avant | `Cv2.Sobel`, `Cv2.Scharr`, `Cv2.Magnitude` |
| `laplacian_edges` | points et lignes fines, toutes directions | très sensible au bruit | `Cv2.Laplacian` |
| `canny` | arêtes fines et binaires | deux seuils à régler | `Cv2.Canny` |

Les sorties sont à une échelle **fixe** (paramètre `gain`) et jamais normalisées image par image : une même arête doit donner la même valeur d'une image à l'autre, sinon un seuil ne veut plus rien dire.

{{img:docs/assets/preprocessing/gallery_contours.jpg|Famille contours.}}

### Morphologie

| Opération | À quoi ça sert | Piège | OpenCvSharp |
|---|---|---|---|
| `morph_open` | efface les petits objets clairs | la taille de l'élément fixe ce qui disparaît | `Cv2.MorphologyEx(..., MorphTypes.Open, ...)` |
| `morph_close` | bouche trous et fissures sombres fines | idem | `MorphTypes.Close` |
| `morph_gradient` | contour épais des objets | — | `MorphTypes.Gradient` |
| `tophat` | ne garde que les petits objets **clairs** | objets plus grands que `kernel` éliminés | `MorphTypes.TopHat` |
| `blackhat` | ne garde que les petits objets **sombres** (piqûres, fissures), rendus en clair | idem | `MorphTypes.BlackHat` |

{{img:docs/assets/preprocessing/gallery_morphologie.jpg|Famille morphologie.}}

### Géométrie

| Opération | À quoi ça sert | Piège | OpenCvSharp |
|---|---|---|---|
| `resize` | taille fixe ; `INTER_AREA` pour réduire (pas de crénelage sur les textures fines) | la même échelle à l'entraînement et en production | `Cv2.Resize` |
| `crop` | région d'intérêt fixe | — | `new Mat(src, rect)` |
| `auto_roi` | boîte du plus grand objet, ou bords d'une bande (`mode: columns`) | dépend du seuil de détection de l'objet | `Cv2.Threshold`, `Cv2.FindContours`, `Cv2.BoundingRect` |
| `letterbox` | carré sans déformation, bordure 114 (convention YOLO) | ramener les boîtes avec `letterbox_params` | `Cv2.Resize` + `Cv2.CopyMakeBorder` |
| `pad` | bordure (par exemple pour un multiple de 32) | — | `Cv2.CopyMakeBorder` |
| `deskew` | remet l'axe principal de l'objet à l'horizontale | ambiguïté d'un quart de tour (`max_angle`) | `Cv2.MinAreaRect`, `Cv2.WarpAffine` |
| `perspective` | redresse une vue en biais à partir de 4 coins | coins à mesurer une fois pour une caméra fixe | `Cv2.GetPerspectiveTransform`, `Cv2.WarpPerspective` |
| `undistort` | corrige la distorsion de l'objectif (calibration) | indispensable avant une mesure en millimètres | `Cv2.Undistort` |

{{img:docs/assets/preprocessing/gallery_geometrie.jpg|Famille géométrie.}}

### Couleur

| Opération | À quoi ça sert | Piège | OpenCvSharp |
|---|---|---|---|
| `to_gray` | RGB → niveaux de gris | — | `Cv2.CvtColor(RGB2GRAY)` |
| `extract_channel` | un canal de RGB, HSV ou LAB, remis sur toute la dynamique | en 8 bits, OpenCV code la teinte sur 0-179 : la toolbox calcule en flottant | `Cv2.CvtColor` (CV_32FC3) + `Cv2.ExtractChannel` |
| `gray_world` | balance des blancs : chaque canal ramené à la moyenne commune | faux pour une scène à dominante réelle (rouille, métal chaud) | `Cv2.Mean` par canal |

{{img:docs/assets/preprocessing/gallery_couleur.jpg|Famille couleur.}}

### Alignement

| Opération | À quoi ça sert | Piège | OpenCvSharp |
|---|---|---|---|
| `align_ecc` | recale sur une image de référence (petits décalages) | échoue si les images diffèrent trop | `Cv2.FindTransformECC`, `Cv2.WarpAffine` |
| `align_orb` | recale par points ORB + homographie (déplacements plus grands) | images trop uniformes : pas de points | `ORB.Create`, `Cv2.FindHomography` |
| `absdiff_reference` | différence avec une pièce de référence, après alignement | un décalage d'un pixel crée de faux défauts sur tous les bords | `Cv2.Absdiff` |

La comparaison à une référence est développée dans l'article sur la vision classique.

## Les grandes images : tuiles

Une caméra linéaire au-dessus d'une bande produit des images de plusieurs milliers de pixels de long. On les découpe en tuiles qui se recouvrent, on traite chaque tuile à la résolution native, puis on recolle les cartes (maximum dans les recouvrements) ou on ramène les boîtes dans l'image complète (`offset_boxes`, puis une NMS pour retirer les doublons créés par le recouvrement).

```python
from vtb.preprocess.tiles import stitch_maps, tile_image

tiles, pieces = tile_image(strip, tile_size=512, overlap=0.25)
maps = [model_map(piece) for piece in pieces]              # une carte par tuile, à la taille de la tuile
full = stitch_maps(maps, tiles, strip.shape[0], strip.shape[1], mode="max")
```

{{img:docs/assets/preprocessing/nb_tiles.jpg|Une bande longue (six tuiles magnétiques bout à bout) découpée en 8 tuiles de 256 px de large avec 25 % de recouvrement, et la carte black-hat recollée (maximum dans les recouvrements).}}

:::warning
Si le modèle infère sur des tuiles, il s'entraîne sur des tuiles découpées de la même façon : même taille, même échelle. Un modèle entraîné sur des images entières réduites ne reconnaît pas un défaut vu à la résolution native.
:::

## Juger un prétraitement à la métrique

Un prétraitement qui « rend l'image plus belle » n'aide pas forcément le modèle. `recipes/preprocessing/scripts/compare.py` le mesure : pour chaque pipeline, caractéristiques d'un ResNet18 ImageNet gelé, régression logistique, F1 macro en validation croisée (5 plis sur train + val ; le test n'est jamais lu). Le F1 moyen vient avec son écart-type, qui dit si un écart entre deux pipelines est réel ou du bruit.

Sur NEU-CLS (480 images, 6 classes), avec l'éclairage d'origine (`steel_gray.yaml`) puis avec un vignettage aléatoire simulé sur chaque image (`steel_vignetted.yaml`) :

| Pipeline | F1 macro, éclairage d'origine | F1 macro, vignettage simulé | Coût (ms / image 200 × 200, réduction à 128 px comprise) |
|---|---|---|---|
| aucun | 0,956 ± 0,029 | 0,960 ± 0,014 | 0,2 |
| `percentile_stretch` | 0,952 ± 0,016 | 0,948 ± 0,017 | 1,0 |
| `clahe` | 0,950 ± 0,018 | 0,946 ± 0,011 | 0,4 à 0,6 |
| `shading_division` | 0,947 ± 0,018 | 0,956 ± 0,019 | 0,7 à 0,8 |
| `shading_division` → `clahe` | 0,947 ± 0,026 | 0,950 ± 0,020 | 1,1 à 1,3 |
| `difference_of_gaussians` | 0,946 ± 0,016 | 0,939 ± 0,020 | 0,5 à 0,7 |
| `median_blur` → `clahe` | 0,945 ± 0,025 | 0,952 ± 0,015 | 0,7 |

Tous les écarts sont plus petits que l'écart-type : pour classer NEU-CLS avec un réseau pré-entraîné, aucun prétraitement n'apporte de gain mesurable, même avec un éclairage inégal. Les caractéristiques d'un ResNet pré-entraîné y sont déjà robustes. C'est la conclusion utile de l'atelier : **un prétraitement n'est pas un gain gratuit**. Il ajoute du temps de calcul et une étape à reproduire en production ; on ne le garde que si la métrique le justifie.

Il compte vraiment dans quatre situations :

1. les méthodes sensibles au niveau absolu des pixels : seuils classiques, cartes d'anomalies, segmentation fine ;
2. les images 16 bits (thermique, radiographie) qu'un réseau pré-entraîné ne sait pas lire telles quelles ;
3. les artefacts forts et systématiques : motif périodique, bandes d'une caméra linéaire ;
4. les changements entre l'entraînement et la production (autre caméra, autre éclairage, usure d'une lampe) : une correction d'illumination ou une normalisation rend le modèle moins dépendant de ces dérives.

## Les règles

- Le prétraitement est déterministe et identique partout ; il est décrit dans la carte du modèle et reproduit tel quel en C#.
- Il se juge à la métrique de validation, jamais sur le test, et jamais seulement à l'œil.
- Les paramètres calculés image par image (percentiles, Otsu, fenêtre automatique) changent l'échelle d'une image à l'autre : les figer quand c'est possible.
- Les augmentations (retournements, rotations, luminosité aléatoire) ne sont pas des prétraitements : elles vont dans `augment:` et ne servent qu'à l'entraînement.
