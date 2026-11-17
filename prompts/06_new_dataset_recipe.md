# Tache : brancher un nouveau jeu de donnees sur une recette existante

Contexte : la toolbox change de donnees en changeant le YAML. Cette tache le fait pour le jeu
`<CHEMIN DU JEU>` (images et annotations), sans copier ces donnees dans le depot (`data/raw/`
est ignore par git) et sans ecrire de nom interne dans le depot.

## Etapes

1. `recipes/explore` : `check_dataset.py` sur le jeu ; me montrer les alertes (illisibles,
   doublons, fuites, raccourcis, tailles) AVANT d'aller plus loin.
2. Choisir la recette (classification, detection, segmentation, anomalie) d'apres la question
   posee et les annotations disponibles ; me proposer le choix avec la raison.
3. Ecrire `recipes/<recette>/configs/<jeu>_quick.yaml` et `<jeu>.yaml` (layout, couleur,
   `group_regex` si plusieurs images par piece, decoupage temporel si les donnees vieillissent).
4. Choisir le pretraitement avec `recipes/preprocessing/scripts/compare.py` (a la metrique).
5. Quick, puis nominale (GPU si besoin, regles RunPod), puis export et golden C#.
6. Rapport : chiffres, erreurs typiques (galerie), seuil propose sur la validation, limites.

## Livrable

Deux YAML, un rapport, aucun fichier de donnees ni nom interne dans le depot.
