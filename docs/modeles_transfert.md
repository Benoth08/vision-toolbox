# Choisir un modèle pré-entraîné et une stratégie de transfert

**Dans le fil : étape 8 (Construire le modèle)**, et étape 9 (Perte, optimiseur, hyperparamètres) pour les taux d'apprentissage par groupe de couches. Ce sont les mêmes numéros dans le cours, dans les notebooks et dans l'index `ETAPES.md`.

Sur des images industrielles, on part presque toujours d'un réseau pré-entraîné sur ImageNet : ses premières couches savent déjà détecter des bords, des textures et des motifs, et ces briques servent aussi bien pour une rayure sur une tôle que pour un chat. Deux décisions restent à prendre : **quel réseau**, et **combien de ce réseau on laisse apprendre**. La toolbox les sépare : `model.name` choisit le réseau, `model.transfer.strategy` choisit la stratégie. Les deux se changent dans le YAML, sans toucher au code.

## Le zoo de modèles

`python recipes/classification/scripts/list_models.py` affiche le catalogue (avec `--latency` pour mesurer la vitesse sur la machine). Chaque entrée du zoo (`src/vtb/models/zoo.py`) décrit où se trouve la tête, la première convolution et la liste des blocs, ce qui permet à une seule fonction de remplacer la tête, d'adapter l'entrée et de dégeler les N derniers blocs de n'importe quel réseau.

| Modèle | Paramètres | Top-1 ImageNet | Quand le choisir |
|---|---|---|---|
| `resnet18` | 11,7 M | 69,8 % | le choix par défaut : rapide, robuste, facile à exporter |
| `resnet50` | 25,6 M | 80,9 % | plus précis, 2 à 3 fois plus lent |
| `efficientnet_b0` | 5,3 M | 77,7 % | le meilleur rapport précision / calcul |
| `mobilenet_v3_small` / `_large` | 2,5 / 5,5 M | 67,7 / 75,3 % | PC industriel sans GPU, embarqué |
| `convnext_tiny` | 28,6 M | 82,5 % | architecture moderne, plus lente sur CPU |
| `regnet_y_800mf` | 6,4 M | 78,8 % | bon compromis sur CPU |
| `timm:<nom>` | — | — | n'importe lequel des centaines de modèles de timm |
| `simple_cnn`, `small_resnet` | 0,4 / 1,2 M | — | réseaux écrits de zéro (modèle custom) |

:::tip
La précision ImageNet ne prédit qu'en partie la précision sur des images d'acier. Le bon réflexe : essayer `resnet18` en stratégie `features` (vingt secondes), puis `partial`, et ne monter en taille que si l'erreur résiduelle le justifie et que la latence le permet.
:::

## Les cinq stratégies

| Stratégie | Ce qui apprend | Données | Domaine | Coût |
|---|---|---|---|---|
| `features` | rien dans le réseau : régression logistique sur les vecteurs du backbone gelé | quelques dizaines d'images par classe | proche d'ImageNet | secondes |
| `head` | la tête seule | peu | proche | minutes |
| `partial` | la tête et les N derniers blocs (`unfreeze_last`) | moyen | quelconque | le cas courant |
| `full` | tout, avec des taux par profondeur (`layer_decay`) | beaucoup | éloigné (rayons X, thermique, microscopie) | le plus long |
| `progressive` | un bloc de plus toutes les `progressive_every` époques | moyen | quelconque | stabilise les débuts |

Le code de la stratégie tient en une fonction : on gèle tout, puis on dégèle la tête et les blocs voulus.

{{file:src/vtb/models/transfer.py|145-176}}

Trois détails font la différence en pratique :

- **Taux discriminatifs** (`backbone_lr_factor`, `layer_decay`) : la tête neuve apprend avec le taux nominal, le backbone dix fois moins vite. Des pas trop grands sur des poids déjà bons les détruisent dans les premières époques.
- **BatchNorm des blocs gelés** (`freeze_batchnorm`) : un bloc gelé dont les statistiques de BatchNorm continuent de bouger n'est pas vraiment gelé. On les fige en mode évaluation.
- **La stratégie `features` reste un réseau** : la standardisation et la régression logistique sont repliées dans une couche linéaire, placée à la place de la tête. Le modèle s'évalue, prédit et s'exporte en ONNX comme les autres.

{{file:src/vtb/models/transfer.py|179-198}}

## Images grises et canaux multiples

Les images d'acier sont souvent en niveaux de gris, alors que les poids ImageNet attendent trois canaux. Deux options :

1. **Copier le gris sur trois canaux** (`data.color: rgb` sur une image grise) : le plus simple, aucune adaptation, mais un premier calcul trois fois plus lourd.
2. **Adapter la première convolution** (`model.in_channels: 1`) : on additionne les trois noyaux RGB. Sur une image grise, la réponse est celle du réseau RGB à la normalisation près (pour N canaux, la moyenne des noyaux est répartie sur les N canaux).

{{file:src/vtb/models/transfer.py|89-108}}

:::note
En segmentation, la toolbox va plus loin : l'image grise est copiée sur trois canaux **à l'intérieur** du modèle exporté, puis normalisée canal par canal. La réponse est alors exactement celle des poids ImageNet, et le programme de production envoie quand même une image à un seul canal.
:::

## Ce que donnent les stratégies sur NEU-CLS

Six défauts de laminage à chaud, 1 800 images, le même découpage pour toutes les expériences, test de 270 images lu une seule fois :

| Expérience | Stratégie | Paramètres entraînés | Temps (s) | F1 macro test |
|---|---|---|---|---|
| `resnet18_features` | features | 3 078 | 10 | 0,978 |
| `resnet18_head` | head | 3 078 | 131 | 0,959 |
| `resnet18_partial` | partial | 8,4 M | 478 | 1,000 |
| `resnet18_full` | full | 11,2 M | 191 | 1,000 |
| `resnet18_progressive` | progressive | 11,2 M | 599 | 1,000 |
| `resnet18_gray` (1 canal) | partial | 8,4 M | 122 | 1,000 |
| `simple_cnn` (de zéro) | tout | 0,4 M | 240 | 0,971 |
| `small_resnet` (de zéro) | tout | 1,2 M | 1 421 | 0,951 |

{{img:docs/assets/classification/resnet18_partial_confusion.jpg|Matrice de confusion du test pour resnet18 en stratégie partial : aucune erreur sur les 270 images.}}

Trois enseignements. La stratégie `features` atteint 97,8 % sans une seule rétropropagation : c'est le premier essai à faire sur un nouveau jeu, il dit en vingt secondes si le problème est facile. Dès qu'un bloc est dégelé, NEU-CLS est résolu : ce jeu est facile pour un réseau pré-entraîné, et il faut des jeux plus durs (défauts rares, classes proches, variations d'acquisition) pour départager les stratégies. Les réseaux de zéro plafonnent à 95-97 % avec bien plus de calcul : le pré-entraînement vaut de l'or quand les images sont rares. Les temps sont indicatifs (machine partagée pendant les essais).

## Un modèle maison

Un réseau écrit à la main s'ajoute au zoo en deux lignes : la classe dans `src/vtb/models/custom.py`, une entrée `ModelSpec` qui dit où est sa tête. Il profite alors de tout le reste (stratégies, export vérifié, carte du modèle).

{{file:src/vtb/models/custom.py}}
