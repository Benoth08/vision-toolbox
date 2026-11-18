# Augmentation de données : multiplier les images sans mentir sur l'étiquette

**Dans le fil : étape 6 (Augmenter).** C'est le même numéro dans le cours, dans les notebooks et dans l'index `ETAPES.md`.

En inspection industrielle, les jeux d'images sont petits : quelques centaines d'images saines, quelques dizaines de défauts par classe. Un réseau apprend alors les images par cœur au lieu d'apprendre les défauts. L'augmentation de données fabrique, à chaque époque, des variantes des images d'entraînement (retournées, plus sombres, plus floues…) pour que le modèle apprenne ce qui ne change pas : le défaut.

Le module `vtb.augment` réunit 45 opérations, 7 presets et les outils pour les contrôler. Il repose sur torchvision v2 (licence BSD), qui transforme l'image, les boîtes et les masques dans le même appel, et ajoute des opérations écrites pour la toolbox : éclairage non uniforme, stries de caméra linéaire, flou de défilement, reflets, copy-paste de défauts. Le dossier est autonome : il se copie tel quel dans un autre projet.

## Une augmentation est une hypothèse

Augmenter, c'est affirmer : « l'étiquette ne change pas si l'on fait ceci à l'image ». Le modèle apprend à ignorer ce changement. Si l'hypothèse est fausse, on lui apprend une erreur.

| Opération | Hypothèse | Fausse quand |
|---|---|---|
| retournement | le défaut n'a pas de sens gauche / droite | texte, flèche, pièce asymétrique en pose fixe |
| rotation de 90° | pas de direction privilégiée | bande laminée (la direction de laminage compte) |
| zoom | la taille ne décide pas de la classe | « trou de plus de 2 mm = rebut » |
| éclairage, exposition | le défaut ne se définit pas par sa luminosité | « tache claire » et « tache sombre » sont deux classes |
| flou, basse résolution | le défaut reste visible | piqûres de quelques pixels : elles disparaissent |
| trous (dropout) | une partie de l'image suffit | petit défaut caché : l'étiquette ment |
| poussières, taches | les taches ne sont pas des défauts | elles en sont une classe, ou l'on apprend le « normal » |

La règle pratique : n'augmenter qu'avec des variations qu'on verra en production (autre éclairage, autre caméra, autre position de la pièce), et toujours **regarder** le résultat avant d'entraîner.

## Les opérations

| Famille | Opérations | Ce qui change |
|---|---|---|
| géométrie (13) | hflip, vflip, rot90, rotate, affine, zoom, translate, shear, perspective, elastic, safe_crop, random_resized_crop, scale_jitter | l'image **et** les boîtes et masques |
| photométrie (9) | brightness_contrast, gamma, hue_saturation, sharpness, autocontrast, equalize, clahe, posterize, invert | l'image seule |
| capteur et éclairage (15) | illumination_gradient, vignetting, exposure, line_stripes, gaussian_noise, poisson_noise, speckle_noise, salt_pepper, blur, motion_blur, defocus_blur, jpeg, low_resolution, specular, dust | l'image seule |
| occlusion (3) | random_erasing, coarse_dropout, grid_mask | l'image seule (la cible ne bouge pas) |
| politiques (4) | randaugment, trivialaugment, augmix, autoaugment | classification seulement |
| synthèse (1) | synthetic_defect | l'image **et** le masque (un défaut est ajouté) |

La famille « capteur et éclairage » est celle qu'on rencontre réellement en ligne : un éclairage qui vieillit, une caméra changée, une bande qui défile plus vite. Chaque opération tire ses paramètres dans un intervalle et ne s'applique qu'avec une probabilité `p`.

{{img:docs/assets/augmentation/capteur.jpg|Les quinze opérations « capteur et éclairage », forcées sur la même vignette NEU-CLS (rayures).}}

Une opération maison est une classe torchvision v2 de quelques lignes. Le gradient d'éclairage, par exemple, multiplie l'image par une rampe linéaire ou par un champ radial :

{{file:src/vtb/augment/ops.py|290-333}}

## Les presets

Un preset est une liste d'opérations accompagnée de son raisonnement. Dans le YAML d'une recette :

```yaml
augment:
  - {name: preset, preset: steel_strip}                 # le preset...
  - {name: preset, preset: medium, exclude: [rot90]}    # ... ou un autre, sans une opération
  - {name: jpeg, quality: [60, 95], p: 0.2}             # et des opérations ajoutées après
```

| Preset | Pour |
|---|---|
| `light` | point de départ prudent, ou grand jeu : retournements, un peu de lumière |
| `medium` | le réglage par défaut d'un petit jeu industriel |
| `strong` | très petit jeu, modèle qui sur-apprend vite (à vérifier visuellement) |
| `steel_strip` | bande qui défile verticalement sous une caméra linéaire : pas de rot90, gradient en travers de la bande, stries de colonnes, flou de défilement |
| `steel_texture` | vignettes de texture sans direction (NEU-CLS, tuiles) : les huit symétries du carré |
| `metal_part` | pièce usinée en position fixe : pas de retournement, reflets, vignettage, défocalisation |
| `anomaly_normals` | apprendre le normal : seulement l'exposition, le bruit et 2 % de décalage |

Le preset `steel_strip` montre comment les hypothèses se traduisent en réglages :

{{file:src/vtb/augment/presets.py|77-95}}

{{img:docs/assets/augmentation/presets.jpg|Quatre tirages de chaque preset sur la même vignette : du plus prudent au plus fort.}}

Un preset est un point de départ, pas une vérité. On le regarde sur une vingtaine de ses propres images (`viz.show_augmentations`), on retire ce qui crée des images impossibles, puis on compare avec et sans augmentation sur la **validation**.

## Les boîtes et les masques suivent

En détection et en segmentation, une augmentation géométrique doit déplacer la vérité terrain avec l'image. Torchvision v2 le fait si l'on passe les cibles dans le même appel : `aug(image, boxes)` ou `aug(image, mask)`. Une boîte sortie de l'image devient vide, puis `sanitize_boxes` la retire **avec** son étiquette. Le masque est toujours rééchantillonné au plus proche voisin : jamais de classe intermédiaire.

{{img:docs/assets/augmentation/boxes.jpg|Preset strong en détection (NEU-DET, inclusions) : les boîtes suivent rotations, zoom et perspective.}}

{{img:docs/assets/augmentation/masks.jpg|Preset medium + safe_crop en segmentation (Magnetic Tile, fissure) : le masque suit, et safe_crop ne coupe jamais le défaut.}}

`safe_crop` recadre en gardant **tous** les objets (l'union des boîtes ou du masque), puis revient à la taille d'origine. `random_resized_crop`, lui, peut couper un défaut, ce qui le rend dangereux en classification de défauts.

Une opération écrite à la main peut oublier les cibles : c'est l'erreur la plus fréquente, et elle est silencieuse. `check_alignment` la détecte. Il fait passer une image test asymétrique (rectangle, L, disque) avec ses boîtes et son masque dans l'augmentation, vingt fois, et vérifie que la zone claire, le masque et les boîtes restent superposés. Les 13 opérations géométriques et les 7 presets passent ce contrôle. Un retournement écrit à la main sur l'image seule échoue (IoU médian du masque 0,28).

## Mélanger plusieurs images

- **MixUp et CutMix** (classification) mélangent deux images d'un lot et leurs étiquettes : 70 % de A et 30 % de B, ou un rectangle de B collé dans A. `batch_augmentation(nombre_de_classes)` s'utilise comme `collate_fn` du DataLoader d'entraînement. Les étiquettes deviennent des probabilités, que `CrossEntropyLoss` accepte telles quelles.
- **Copy-paste de défauts** (détection, segmentation) : c'est le remède direct au manque de défauts. `DefectBank.from_masks` (ou `from_boxes`) découpe les défauts annotés, et `CopyPaste` les colle sur d'autres images en mettant à jour masque, boîtes et étiquettes. Le bord est adouci et la luminosité accordée au fond d'accueil ; sans cela, le modèle apprend à reconnaître le collage plutôt que le défaut.
- **Mosaïque** (détection) : quatre images autour d'un centre aléatoire. Une boîte dont il reste moins d'un quart est retirée.
- **Défauts synthétiques** : tache de Perlin (façon DRAEM), morceau déplacé (CutPaste) ou rayure, écrits dans le masque. Ils servent à vérifier une chaîne avant d'avoir des défauts réels ou à l'auto-supervision, jamais à mesurer une performance finale.

{{img:docs/assets/augmentation/copy_paste.jpg|Copy-paste : des défauts de Magnetic Tile (soufflures, cassures, fissures) collés sur des tuiles saines, masque mis à jour.}}

## L'effet chiffré

Le notebook `03_augmentation` mesure l'effet dans un cas difficile : 25 images d'entraînement par classe (NEU-CLS, 6 classes, images réduites à 96 pixels), un petit réseau convolutif appris de zéro, 40 époques et trois graines. Le test porte sur 300 images jamais vues, présentées de deux façons :

- **même acquisition** : les images telles quelles ;
- **acquisition changée** : les mêmes images vues par une ligne dont l'éclairage a changé (gain 0,8, rampe de ±30 %, gamma 1,3, léger flou et bruit).

| Entraînement | Même acquisition | Acquisition changée |
|---|---|---|
| sans augmentation | 0,932 ± 0,010 | 0,431 ± 0,039 |
| symétries seules (retournements, rot90) | 0,929 ± 0,013 | 0,379 ± 0,033 |
| preset `steel_texture` | 0,888 ± 0,010 | **0,567 ± 0,050** |

Le résultat se lit en trois points.

1. **Sur la même acquisition, l'augmentation ne gagne rien ici.** Le preset complet coûte même quatre points : la tâche d'entraînement est plus dure pour le même nombre d'époques, et le test ne contient aucune des variations simulées.
2. **Quand l'éclairage change, le modèle sans augmentation s'effondre** (0,43) : il a appris l'éclairage de ses 150 images en même temps que les défauts. Le preset garde 14 points de plus.
3. **Les symétries seules ne protègent pas d'un changement d'éclairage.** Chaque opération achète une invariance précise, pas une robustesse générale.

Cette expérience a une limite : les changements testés font partie de ce que le preset simule. Elle montre le mécanisme (on devient robuste à ce qu'on a prévu), pas une garantie face à l'imprévu.

Deux règles en découlent :

- choisir les opérations d'après les variations attendues en production ;
- évaluer aussi sur des images d'autres conditions (autre jour, autre réglage) quand on en a, et pas seulement sur un test tiré du même lot que l'entraînement.

## Reproductibilité, vitesse, Windows

- Toutes les opérations tirent leurs paramètres avec le générateur de **torch**. `torch.manual_seed(n)` rend donc les tirages reproductibles, et DataLoader donne une graine différente à chaque worker. Les versions actuelles de PyTorch sèment aussi numpy et le module `random` dans chaque worker (vérifié avec la version installée : tirages distincts d'un worker à l'autre, identiques d'une exécution à l'autre) ; avec une vieille version (avant 1.9), tous les workers produisaient les mêmes tirages numpy, et `worker_init_fn` reste fourni pour ce cas.
- `benchmark_ops` mesure le temps de chaque opération. Sur CPU, le chargement est souvent le goulot de l'entraînement ; l'opération `elastic` (environ 20 ms par image en 200 × 200) est la seule vraiment lente.
- Les classes de lot et de copy-paste sont picklables. Elles fonctionnent avec `num_workers > 0` sous Windows, où les workers démarrent par « spawn ».

## Et Kornia ?

Kornia est une autre bibliothèque d'augmentation (licence Apache-2.0). Elle ne travaille pas image par image dans le `Dataset`, mais sur un lot entier de tenseurs `(N, C, H, W)`, dans la boucle d'entraînement, sur la carte graphique quand il y en a une ; chaque image du lot reçoit son propre tirage. Elle devient utile quand le processeur n'arrive plus à préparer les lots assez vite. La toolbox augmente avec `torchvision.transforms.v2`, dans le `Dataset` ; le moteur RF-DETR sait utiliser Kornia sur carte graphique (clé `augmentation_backend` de la librairie). Le cours compare les deux à l'étape 6, et le projet type « Réseau custom, version complète » propose les deux au choix.

## En production

L'augmentation se règle dans la clé `augment:` du YAML. C'est le cas pour les recettes classification, segmentation, Faster R-CNN et auto-encodeur d'anomalies. YOLO, YOLOX et RF-DETR ont leurs propres augmentations internes (mosaïque, HSV…), réglées dans leur section. L'augmentation ne s'applique qu'à l'entraînement ; la validation et le test voient les images telles que la production les verra.
