# Recette classical : vision classique sans apprentissage

Seuillage, analyse de blobs, comparaison a une piece de reference, mesure au sous-pixel. Deux
usages : resoudre directement un probleme simple (contraste franc, pieces identiques et bien
positionnees, mesure dimensionnelle), et fournir la base de comparaison qu'un modele appris doit
battre pour justifier sa complexite.

## Commandes (depuis la racine du depot)

```
uv run python recipes/classical/scripts/baseline.py --config recipes/classical/configs/mvtec_bottle_reference.yaml
uv run python recipes/classical/scripts/baseline.py --config recipes/classical/configs/magnetic_tile.yaml
uv run python recipes/classical/scripts/measure_demo.py
uv run python tools/notebooks.py recipes/classical/notebooks/classical.py --execute
```

`baseline.py` choisit le seuil et l'aire minimale des blobs sur la validation (grille), avec un
critere au choix (`select: dice` ou `recall_at_max_false_alarm`), puis lit le test une seule
fois. Il produit split.csv, grid_val.csv (tous les reglages essayes), metrics.json et
examples.jpg (verite en vert, prediction en rouge).

## Les etapes du fil dans cette recette

Le fil en 13 etapes (le meme que dans le cours et les notebooks : `ETAPES.md`, a la racine) se
retrouve dans les scripts et dans le YAML, ou chaque cle de premier niveau est precedee d'une ligne
`# --- Etape N : ... ---`.

Pas de modele appris ici : les etapes 6 a 10 (augmenter, lots, modele, perte et optimiseur,
entrainer) sont remplacees par une methode classique, et rien n'est exporte (pas d'etape 12).

| script | etapes du fil couvertes | cles du YAML |
|---|---|---|
| `baseline.py` | 1 Reglages ; 2 Importer, 3 Cataloguer, 4 Decouper ; 5 Pretraiter (la carte de reponse) ; a la place des etapes 8 a 10 : methode `response` ou `reference` ; 11 Evaluer (seuil et aire regles sur la validation, test lu une fois) | `runtime` ; `data`, `max_per_class`, `split` ; `preprocess` ; `method`, `reference`, `invert` ; `thresholds`, `min_areas`, `select`, `max_false_alarm` |
| `measure_demo.py` | hors fil : demonstration de mesure en millimetres (calibration sur damier) | pas de YAML (`--output`, `--assets`) |

`assets_dir` (figures de la documentation) est hors fil. Le meme contenu, cellule par cellule :
`notebooks/16_vision_classique.py`.

## Deux methodes (cle `method`)

- `reference` : la piece de reference est la mediane d'images normales de train ; chaque image
  est alignee (ECC) puis soustraite ; les ecarts au-dessus du seuil deviennent des blobs. La
  methode reine quand les pieces sont identiques et bien positionnees.
- `response` : le pretraitement fabrique une carte ou les defauts ressortent (median, correction
  d'illumination, black-hat pour des defauts sombres), puis seuil et blobs.

## Resultats obtenus (test lu une fois)

| configuration | test | IoU pixel | Dice pixel | rappel image | fausses alarmes |
|---|---|---|---|---|---|
| `mvtec_bottle_reference.yaml` (reference, critere 0 fausse alarme) | 50 images (38 defauts) | 0.044 | 0.084 | 0.605 | 0.000 |
| `magnetic_tile.yaml` (reponse black-hat, critere Dice) | 99 images | 0.058 | 0.109 | 1.000 | 1.000 |

Lecture :
- Sur MVTec bottle, la comparaison a une reference detecte 60 % des bouteilles defectueuses
  sans aucune fausse alarme, sans le moindre apprentissage. A comparer (jeux de test differents,
  ordre de grandeur seulement) a l'auto-encodeur du cours (rappel 0.38 a 0 fausse alarme) et a
  PatchCore (1.00). La localisation au pixel reste grossiere (Dice 0.08).
- Sur Magnetic Tile (surface texturee), le seuillage echoue : la texture repond autant que les
  defauts, et le meilleur reglage au sens du Dice signale toutes les images. C'est la limite de la
  vision classique sur des surfaces naturelles, et le chiffre qu'un U-Net doit battre
  (U-Net du cours sur le meme jeu : IoU 0.57 a 0.63 sur le test).

Mesure (`measure_demo.py`, images synthetiques) : calibration sur 8 vues d'un damier, erreur de
reprojection 0.66 px, echelle 0.2500 mm / px ; largeur d'une bande de 60.25 mm mesuree a
60.255 mm +/- 0.032 (ecart-type sur 60 lignes).
