# Explorer un jeu d'images : contrôle, plongements, dérive

**Dans le fil : étape 2 (Importer et explorer les données)** pour le contrôle du jeu et les plongements ; les quasi-doublons servent à l'étape 4 (Découper), la dérive se surveille à l'étape 13 (Prédire et mettre en production). Ce sont les mêmes numéros dans le cours, dans les notebooks et dans l'index `ETAPES.md`.

Une bonne part des « excellents résultats » qui s'effondrent en production vient des données, pas du modèle : un doublon de part et d'autre du découpage qui truque le test, des classes prises à des heures différentes que le réseau distingue à leur luminosité, une caméra changée en cours d'année. La recette `recipes/explore` regroupe trois analyses qui ne demandent ni étiquette ni entraînement, et qu'il faut lancer **avant** d'entraîner (puis régulièrement en production).

## Le contrôle du jeu

`check_dataset.py` lit chaque image une fois et en tire une ligne de statistiques : lisibilité, taille, type (8 ou 16 bits), canaux, luminosité, contraste, part de pixels saturés ou noirs, netteté (variance du laplacien), empreinte MD5 et empreinte perceptive. Le rapport commence par les alertes, en clair :

{{text:docs/assets/explore/magnetic_tile_quality_report.md|0}}

| Alerte | Pourquoi c'est grave | Que faire |
|---|---|---|
| fichiers illisibles | l'entraînement plante au milieu de la nuit | retirer ou réexporter |
| types ou canaux mélangés | un uint16 parmi des uint8, une image RGB parmi des grises | normaliser à la lecture |
| doublon à cheval sur deux sous-ensembles | le modèle a déjà vu l'image du test : le score est faux | découper par groupe |
| doublon à deux étiquettes | erreur d'annotation | revoir l'annotation |
| raccourci | une statistique simple prédit la classe : le modèle apprendra l'acquisition | vérifier l'acquisition, équilibrer |
| images atypiques | caméra mal réglée, capot fermé, mauvaise pièce | regarder `outliers.csv` |

## Des doublons qui n'en sont pas, et l'inverse

L'empreinte perceptive dHash compare le sens des gradients d'une vignette 9 × 8 : deux copies d'une image (réenregistrement JPEG, luminosité un peu changée) ont la même empreinte à quelques bits près. Toutes les paires se comparent en une seconde pour dix mille images. Mais sur des surfaces texturées, l'empreinte ne voit que l'éclairage : sur NEU-CLS, 9 724 paires d'images différentes ont des empreintes proches. La toolbox confirme donc chaque candidat par la corrélation des **détails** de deux vignettes 64 × 64, basses fréquences retirées : deux textures différentes ont des détails indépendants (corrélation proche de 0), deux copies gardent une corrélation proche de 1. Il reste deux vrais doublons sur NEU-CLS, dont un à cheval sur l'entraînement et le test.

Sur Magnetic Tile, le contrôle révèle un problème de fond : la même tuile y est photographiée sous plusieurs expositions. 418 images forment 161 groupes de quasi-doublons.

{{img:docs/assets/explore/magnetic_tile_duplicates.jpg|Paires de quasi-doublons confirmées dans Magnetic Tile : la même tuile, deux expositions.}}

:::warning
Avec un découpage aléatoire, une même tuile se retrouve dans l'entraînement et dans le test : le score du test mesure alors la mémoire du modèle, pas sa capacité à généraliser. `check_dataset.py` écrit `groups.csv` (les quasi-doublons regroupés) ; on le donne à `data.groups_csv` avec `split.strategy: grouped`, et toutes les copies d'une tuile tombent du même côté.
:::

## Les plongements

Un réseau pré-entraîné sans sa tête transforme chaque image en un vecteur de 512 nombres (resnet18), comparable entre images : deux images proches pour le réseau ont des vecteurs proches. Sans aucune étiquette, on peut alors projeter le jeu en deux dimensions, le regrouper en k groupes et chercher les voisins d'une image.

{{img:docs/assets/explore/neu_cls_projection.jpg|NEU-CLS projeté en 2D (t-SNE des plongements resnet18), coloré par classe.}}

Sur NEU-CLS, k-means à six groupes retrouve les six classes avec une pureté de 0,913, sans avoir vu une étiquette. Deux usages pratiques : **pré-annoter par groupe** (on étiquette un groupe entier d'un coup, puis on corrige), et **repérer deux acquisitions mélangées** (un groupe inattendu qui correspond à une caméra ou un éclairage).

{{img:docs/assets/explore/neu_cls_clusters.jpg|Quelques images de chaque groupe k-means : chaque ligne est un groupe.}}

## La dérive

En production, le modèle ne prévient pas quand les images changent : il se trompe en silence. `drift.py` compare un dossier de référence (les images d'entraînement) et un dossier courant (les images récentes) de deux façons :

1. **Un classifieur de domaine** : une régression logistique essaie de distinguer les deux dossiers à partir des plongements. AUC ≈ 0,5 : indiscernables. AUC proche de 1 : les deux périodes se séparent facilement, il y a dérive. C'est le test le plus sensible, et il classe les images courantes de la plus « nouvelle » à la plus banale.
2. **Les statistiques simples** (test de Kolmogorov-Smirnov colonne par colonne) : moins sensibles, mais elles disent **quoi** a changé.

{{file:src/vtb/explore/drift.py|28-48}}

| Essai | AUC | Verdict | Ce qui a bougé |
|---|---|---|---|
| NEU-DET contre son dossier de validation (même acquisition) | 0,42 | pas de dérive | — |
| mêmes images assombries (gamma 1,4) et floutées (σ = 1,2) | 0,999 | dérive forte | netteté (médiane 378 → 13), luminosité (0,47 → 0,35) |

{{img:docs/assets/explore/drift_simulated_projection.jpg|Plongements de référence et courants après une dérive simulée : deux nuages séparés.}}

:::tip
En exploitation : lancer `drift.py` chaque semaine sur les images de la semaine, et regarder `most_novel.png` dès que l'AUC dépasse 0,6. C'est souvent là qu'on découvre un objectif encrassé, une lampe qui vieillit ou une nouvelle nuance d'acier, avant que le taux de défauts manqués ne monte.
:::
