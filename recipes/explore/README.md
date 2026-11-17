# Recette explore : regarder les donnees avant d'entrainer (et surveiller la ligne apres)

Trois analyses sans etiquette ni entrainement, a lancer sur tout nouveau jeu :

1. `check_dataset.py` : fichiers illisibles, types et tailles melanges, images vides ou
   saturees, doublons exacts et quasi-doublons (et ceux qui sont a cheval sur deux
   sous-ensembles ou portent deux etiquettes), raccourcis (une statistique simple qui predit la
   classe), images atypiques. Ecrit aussi `groups.csv` : les quasi-doublons regroupes, a donner a
   `data.groups_csv` avec `split.strategy: grouped`.
2. `embeddings.py` : un vecteur par image (reseau pre-entraine sans sa tete), projection 2D,
   groupes k-means, galerie par groupe, voisins d'une image.
3. `drift.py` : reference (images d'entrainement) contre courant (images recentes) : AUC d'un
   classifieur de domaine, statistiques qui ont bouge, images les plus "nouvelles".

## Commandes (depuis la racine du depot)

```
python recipes/explore/scripts/check_dataset.py --config recipes/explore/configs/neu_cls.yaml
python recipes/explore/scripts/check_dataset.py --config recipes/explore/configs/magnetic_tile.yaml
python recipes/explore/scripts/embeddings.py   --config recipes/explore/configs/neu_cls.yaml
python recipes/explore/scripts/drift.py        --config recipes/explore/configs/drift_none.yaml
python recipes/explore/scripts/drift.py        --config recipes/explore/configs/drift_simulated.yaml
python tools/notebooks.py recipes/explore/notebooks/explore.py --execute
```

## Les etapes du fil dans cette recette

Le fil en 13 etapes (le meme que dans le cours et les notebooks : `ETAPES.md`, a la racine) se
retrouve dans les scripts et dans le YAML, ou chaque cle de premier niveau est precedee d'une ligne
`# --- Etape N : ... ---`.

Toute la recette sert l'etape 2 (Importer et explorer) : regarder le jeu avant d'entrainer. Elle
alimente l'etape 4 (decoupage par groupe) et sert encore apres l'etape 13 (derive en production).

| script | etapes du fil couvertes | cles du YAML |
|---|---|---|
| `check_dataset.py` | 2 Importer et explorer : illisibles, tailles, doublons, raccourcis, images atypiques ; prepare l'etape 4 : ecrit `groups.csv` (a donner a `data.groups_csv`) et controle un `split.csv` existant | `data`, `split_file`, `max_hamming`, `min_correlation`, `runtime` |
| `embeddings.py` | 2 Importer et explorer : plongements d'un reseau gele, projection 2D, groupes, voisins | `data`, `preprocess`, `model`, `n_clusters`, `projection`, `gallery_per_cluster`, `runtime` |
| `drift.py` | apres l'etape 13 : les images de la ligne ressemblent-elles encore a celles de l'entrainement ? | `reference`, `current`, `simulate`, `preprocess`, `model`, `max_images`, `runtime` |

Le meme contenu, cellule par cellule : `notebooks/05_exploration_qualite.py`.

## Comment les doublons sont trouves

En deux temps, pour rester rapide et ne pas crier au loup sur les surfaces texturees :
empreinte perceptive dHash (64 bits, toutes les paires comparees par XOR et comptage de bits),
puis confirmation par la correlation des DETAILS de deux vignettes 64 x 64 (basses frequences
retirees). Sans la confirmation, l'eclairage commun de deux images d'acier differentes suffit a
rapprocher leurs empreintes : 9 724 faux candidats sur NEU-CLS, 2 vrais doublons.

## Resultats obtenus

| jeu | images | alertes | doublons confirmes |
|---|---|---|---|
| NEU-CLS | 1 800 (200 x 200) | 1 doublon a cheval sur train / test ; 47 images atypiques | 2 paires (dont 1 copie exacte) |
| Magnetic Tile | 1 344 (1 225 tailles differentes) | resize indispensable ; 4 doublons a deux etiquettes ; 4 images atypiques | 301 paires : **418 images en 161 groupes** |

Magnetic Tile contient la meme tuile photographiee sous plusieurs expositions (`exp1` a
`exp6`) : un decoupage aleatoire met des copies d'une meme tuile dans train et test et rend le
test optimiste. Les configurations completes de segmentation et de detection sur ce jeu
utilisent donc `groups_csv: outputs/explore/magnetic_tile/groups.csv` et un decoupage groupe.

Raccourcis sur NEU-CLS : la nettete seule predit la classe a 58,5 % (hasard : 16,7 %), sans
declencher l'alerte (seuil : 60 % du chemin entre hasard et perfection) ; c'est attendu, les
classes different par leur texture.

Plongements de NEU-CLS (resnet18 ImageNet, sans etiquette) : k-means a 6 groupes retrouve les
classes avec une purete de 0,913.

Derive (300 images de reference NEU-DET) :

| essai | images courantes | AUC du classifieur de domaine | verdict | statistique la plus changee |
|---|---|---|---|---|
| `drift_none` (dossier de validation livre a part) | 30 | 0.42 | pas de derive detectable | - |
| `drift_simulated` (gamma 1,4 + flou 1,2) | 150 | 0.999 | derive forte | nettete (mediane 378 -> 13), luminosite (0,47 -> 0,35) |

## Adapter

Changer `data` (meme schema que les autres recettes). Pour surveiller une ligne : `reference`
= le dossier d'entrainement, `current` = les images de la derniere semaine ; lancer `drift.py`
chaque semaine et regarder `most_novel.png` des que l'AUC depasse 0,6.
