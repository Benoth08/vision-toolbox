# Recette anomaly : apprendre le normal, signaler l'ecart (sans exemple de defaut a l'entrainement)

Toutes les methodes derriere un seul protocole : train et val ne contiennent que des images
saines, le seuil est le quantile 0,99 des scores des saines de validation, le test (saines +
defauts) est lu une seule fois. Chaque methode produit une carte brute ; un module commun la
lisse, la ramene a la taille de l'image et en tire le score (`AnomalyModel`). Le modele exporte
prend une image dans [0, 1] et rend la carte et le score ; le seuil est dans la carte du modele.

## Commandes (depuis la racine du depot)

```
python tools/get_data/get_mvtec_bottle.py              # MVTec AD bottle (CC BY-NC-SA 4.0, usage non commercial)
python tools/get_data/get_magnetic_tile.py             # Magnetic Tile (surface texturee, proche de l'acier)
python recipes/anomaly/scripts/run.py --config recipes/anomaly/configs/quick.yaml
python recipes/anomaly/scripts/run.py --config recipes/anomaly/configs/patchcore_lite_bottle.yaml
python recipes/anomaly/scripts/predict.py --config recipes/anomaly/configs/patchcore_lite_bottle.yaml --source data/raw/mvtec_bottle/test/broken_large --csv scores.csv --maps outputs/anomaly/maps
python recipes/anomaly/scripts/benchmark.py --outputs outputs/anomaly --markdown docs/assets/anomaly/benchmark.md
python tools/notebooks.py recipes/anomaly/notebooks/anomaly.py --execute
```

Installation : le socle suffit pour `autoencoder` et `patchcore_lite` ; extra `anomaly` pour
les modeles anomalib (`method: anomalib`, `anomalib.model: Patchcore | Padim | Stfpm | Fastflow
| ReverseDistillation | Dfm | Supersimplenet | EfficientAd | Draem ...`).

## Les etapes du fil dans cette recette

Le fil en 13 etapes (le meme que dans le cours et les notebooks : `ETAPES.md`, a la racine) se
retrouve dans les scripts et dans le YAML, ou chaque cle de premier niveau est precedee d'une ligne
`# --- Etape N : ... ---`.

| script | etapes du fil couvertes | cles du YAML |
|---|---|---|
| tous | 1 Imports et reglages | `runtime` (dossier de sortie, graine, materiel) |
| `prepare.py` | 2 Importer, 3 Cataloguer (etiquettes good / defect), 4 Decouper (aucun defaut dans train ni val) | `data`, `normal_labels`, `split` |
| `train.py` | 5 Pretraiter, 6 Augmenter (auto-encodeur seulement), 7 Charger par lots, 8 Modele, 9 Perte et optimiseur, 10 Entrainer (ou remplir la memoire), puis le debut de l'etape 11 : seuils calibres sur les normales de validation ; sauvegarde le modele (debut de l'etape 12) | `preprocess`, `augment`, `method`, le bloc de la methode (`autoencoder`, `patchcore_lite` ou `anomalib`), `scoring` |
| `evaluate.py` | 11 Evaluer : AUROC image et pixel, rappel et fausses alarmes au seuil, sur le test | aucune cle propre (`split.file`) |
| `predict.py` | 13 Predire : score, decision, carte en option | `data.extensions` |
| `export.py` | 12 Sauvegarder et exporter : ONNX (image -> carte, score) verifie, carte du modele avec les seuils | `export` |

Reperes : `patchcore_lite` et PaDiM n'ont ni perte ni optimiseur (etape 9), ils memorisent des
statistiques des normales ; la taille des lots (etape 7) est dans le bloc de la methode. La meme
chaine, cellule par cellule : `notebooks/13_anomalies_autoencodeur.py`, `14_anomalies_patchcore.py`,
`15_anomalies_anomalib.py`.

Les docstrings des scripts numerotent les cinq verbes ("Etape 1" a "Etape 5") : ce sont les etapes
de la recette, pas celles du fil. `run.py` enchaine `prepare`, `train`, `evaluate`, `export`.

## Methodes et configurations

| fichier | methode | entrainement | remarque |
|---|---|---|---|
| `quick.yaml` | PaDiM (anomalib), 128 px | aucun | verification (une minute) |
| `patchcore_lite_bottle.yaml`, `patchcore_lite_tile.yaml` | PatchCore ecrit dans la toolbox | aucun | lisible, exporte d'un bloc ; memoire 2 % (bouteille), 0,5 % (tuiles) |
| `anomalib_patchcore_*.yaml` | PatchCore d'anomalib | aucun | la reference ; coreset 1 % |
| `anomalib_padim_*.yaml` | PaDiM | aucun | tres rapide, pieces alignees |
| `anomalib_dfm_bottle.yaml` | DFM | aucun | base de comparaison |
| `anomalib_stfpm_*.yaml` | STFPM (etudiant / professeur) | oui | bonne localisation |
| `anomalib_reverse_distillation_bottle.yaml` | Reverse Distillation | oui | tres bon sur MVTec |
| `anomalib_fastflow_bottle.yaml` | FastFlow | oui | inference rapide |
| `anomalib_supersimplenet_bottle.yaml` | SuperSimpleNet | oui | anomalies synthetiques |
| `anomalib_efficient_ad_bottle.yaml` | EfficientAd | oui | telecharge Imagenette : a lancer sur GPU (`prompts/04`) |
| `anomalib_draem_bottle.yaml` | DRAEM | oui | telecharge DTD : a lancer sur GPU (`prompts/04`) |
| `autoencoder_*.yaml` | auto-encodeur convolutif | oui | simple, explicable |

Toutes les configurations d'un meme jeu partagent le meme decoupage (`bottle_split.csv`,
`tile_split.csv`) : le banc compare des methodes, pas des tirages.

## Resultats obtenus

Conditions : CPU 2 coeurs partage entre plusieurs calculs (temps et latences indicatifs),
backbone resnet18 pour toutes les methodes (seuls ses poids etaient disponibles hors ligne ; les
articles utilisent souvent wide_resnet50, a lancer sur GPU avec `prompts/04`), images a 256 px.

MVTec AD bottle : 177 normales en train, 32 en validation (seuil), test de 20 normales + 63 defauts.

| experience | AUROC image | AUROC pixel | rappel au seuil | fausses alarmes | construction (s) | ONNX (ms / image) |
|---|---|---|---|---|---|---|
| patchcore_lite_bottle (toolbox, memoire 2 %) | 1.000 | 0.982 | 1.000 | 0.000 | 140 | 126 |
| anomalib_patchcore_bottle (coreset 1 %) | 1.000 | 0.982 | 1.000 | 0.000 | 461 | 37 (lot fixe de 1) |
| anomalib_padim_bottle | 0.995 | 0.982 | 0.984 | 0.050 | 90 | 151 |
| anomalib_dfm_bottle | 1.000 | 0.935 | 1.000 | 0.050 | 30 | 19 (lot fixe de 1) |
| anomalib_stfpm_bottle (15 epoques) | 1.000 | 0.977 | 1.000 | 0.050 | 273 | 27 |
| anomalib_reverse_distillation_bottle | 0.998 | 0.984 | 1.000 | 0.050 | 232 | 44 |
| anomalib_fastflow_bottle | 1.000 | 0.972 | 1.000 | 0.000 | 227 | 120 |
| anomalib_supersimplenet_bottle (10 epoques) | 0.948 | 0.658 | 0.746 | 0.000 | 774 | 40 |
| autoencoder_bottle | 0.788 | 0.720 | 0.460 | 0.000 | 1136 | 31 |
| quick (PaDiM, 128 px) | 0.997 | 0.975 | 0.984 | 0.050 | 24 | 10 |

Magnetic Tile : 646 normales en train, 115 en validation, test de 191 normales + 392 defauts
(cinq types). Surface texturee, beaucoup plus difficile.

| experience | AUROC image | AUROC pixel | rappel au seuil | fausses alarmes | rappel par type (Blowhole / Break / Crack / Fray / Uneven) |
|---|---|---|---|---|---|
| patchcore_lite_tile (memoire 0,5 %) | 0.965 | 0.749 | 0.651 | 0.010 | 0.76 / 0.87 / 0.42 / 0.91 / 0.40 |
| anomalib_padim_tile | 0.928 | 0.727 | 0.520 | 0.016 | 0.85 / 0.60 / 0.42 / 0.72 / 0.08 |
| anomalib_stfpm_tile (10 epoques) | 0.663 | 0.582 | 0.097 | 0.016 | 0.04 / 0.22 / 0.00 / 0.44 / 0.00 |
| autoencoder_tile | 0.541 | 0.595 | 0.028 | 0.005 | quasi nul |

Lecture :

- Sur la bouteille (objet, fond propre), presque toutes les methodes sont parfaites au niveau
  image : c'est la localisation (AUROC pixel) et les fausses alarmes qui les departagent.
  PatchCore (les deux versions) est en tete sans aucun entrainement.
- Sur les tuiles (texture), l'ecart se creuse : PatchCore reste le meilleur (AUROC image 0,965),
  les methodes entrainees sur CPU avec peu d'epoques decrochent, l'auto-encodeur ne voit rien
  (il reconstruit aussi bien la texture que les defauts). Le rappel au seuil 0,99 reste modeste
  (65 %) : sur ce jeu, il faut accepter plus de fausses alarmes ou passer a la supervision.
- Les "lots fixes de 1" : PatchCore et DFM d'anomalib figent la taille du lot dans leur graphe ;
  l'export le detecte, reexporte en lot de 1 et l'ecrit dans la carte (`dynamic_batch: false`).
- Non lances ici : EfficientAd et DRAEM (jeux auxiliaires a telecharger), PatchCore d'anomalib
  sur les tuiles (selection de coreset trop lente sur CPU) : `prompts/04`.
- Decoupage des tuiles : aleatoire parmi les normales ; le jeu contient des quasi-doublons
  (voir recipes/explore), ce qui rend les fausses alarmes un peu optimistes.

## Adapter a ses donnees

1. Des images saines en nombre (une centaine au minimum), quelques defauts pour evaluer ;
   `normal_labels` dit quelles etiquettes sont saines.
2. Commencer par `patchcore_lite` (sans entrainement) ; regarder les cartes des images saines
   les plus mal notees avant de toucher au seuil.
3. Le seuil vient de la validation (`scoring.image_quantile`) ; pour moins de fausses alarmes,
   monter le quantile, au prix du rappel.
