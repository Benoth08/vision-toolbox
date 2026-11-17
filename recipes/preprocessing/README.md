# Recette preprocessing : choisir, regler et verifier un pretraitement

Un atelier plutot qu'un modele : voir l'effet de chaque operation du catalogue sur ses images,
regler les parametres, puis juger les pipelines candidats a la METRIQUE (pas a l'oeil). Le
pipeline retenu se recopie ensuite dans la liste `preprocess:` du YAML de la recette de
classification, detection, segmentation ou anomalies.

## Commandes (depuis la racine du depot)

```
uv run python recipes/preprocessing/scripts/gallery.py --config recipes/preprocessing/configs/steel_gray.yaml
uv run python recipes/preprocessing/scripts/compare.py --config recipes/preprocessing/configs/steel_gray.yaml
uv run python recipes/preprocessing/scripts/compare.py --config recipes/preprocessing/configs/steel_vignetted.yaml
uv run python tools/notebooks.py recipes/preprocessing/notebooks/preprocessing.py --execute
```

- `gallery.py` : une planche avant / apres par famille d'operations (outputs/.../gallery/*.jpg,
  copiees dans docs/assets/preprocessing pour le site) et catalog.csv (operations et parametres).
- `compare.py` : pour chaque pipeline du YAML, caracteristiques d'un ResNet18 ImageNet gele,
  regression logistique, F1 macro en validation croisee 5 plis sur train + val (le test n'est
  jamais lu). Resultat : comparison.csv et comparison.md (moyenne ET ecart-type).
- Le notebook : catalogue, reglage (grille fixe ; curseurs ipywidgets avec `sliders()`),
  correction d'illumination sur vignettage artificiel et sur vraie image, coupe-bande FFT,
  tuiles, passage au YAML, resultats de compare.py.

## Les etapes du fil dans cette recette

Le fil en 13 etapes (le meme que dans le cours et les notebooks : `ETAPES.md`, a la racine) se
retrouve dans les scripts et dans le YAML, ou chaque cle de premier niveau est precedee d'une ligne
`# --- Etape N : ... ---`.

Toute la recette sert l'etape 5 (Pretraiter) : choisir la chaine a la metrique, avant d'entrainer.
La chaine retenue se recopie ensuite dans la cle `preprocess` de la recette d'entrainement.

| script | etapes du fil couvertes | cles du YAML |
|---|---|---|
| `gallery.py` | 5 Pretraiter : regarder l'effet de chaque operation (planches avant / apres) | `data`, `gallery`, `runtime` |
| `compare.py` | 2 a 4 (catalogue et decoupage, pour ne jamais lire le test), puis 5 Pretraiter : comparer les chaines par une sonde lineaire en validation croisee sur train + val | `data`, `split`, `pipelines`, `compare`, `runtime` |

Le meme contenu, cellule par cellule : `notebooks/02_opencv_pretraitements.py` (les operations) et
`notebooks/07_classification.py` (le choix a la metrique, etape 5).

## Configurations

| fichier | donnees | ce qu'il montre |
|---|---|---|
| `steel_gray.yaml` | NEU-CLS (bande laminee a chaud, gris) | 7 pipelines classiques compares a la metrique |
| `steel_vignetted.yaml` | NEU-CLS + vignettage aleatoire simule | les memes, quand l'eclairage est inegal |
| `microscope.yaml` | modele a remplir | shading_division, flat_field (lame vide), CLAHE |
| `linescan.yaml` | modele a remplir | column_profile, auto_roi en mode columns, fft_notch |
| `thermal16.yaml` | modele a remplir | window_level 16 -> 8 bits, fixe ou automatique, log |

Les trois modeles se valident au chargement (une operation ou un parametre mal ecrit est une
erreur immediate) ; changer `data.root` pour les faire tourner sur ses images.

## Resultats obtenus (CPU, 480 images NEU-CLS en validation croisee 5 plis)

| pipeline | F1 macro (eclairage d'origine) | F1 macro (vignettage simule) |
|---|---|---|
| aucun | 0.956 +/- 0.029 | 0.960 +/- 0.014 |
| percentile_stretch | 0.952 +/- 0.016 | 0.948 +/- 0.017 |
| clahe | 0.950 +/- 0.018 | 0.946 +/- 0.011 |
| shading_division | 0.947 +/- 0.018 | 0.956 +/- 0.019 |
| shading_division -> clahe | 0.947 +/- 0.026 | 0.950 +/- 0.020 |
| difference_of_gaussians | 0.946 +/- 0.016 | 0.939 +/- 0.020 |
| median_blur -> clahe | 0.945 +/- 0.025 | 0.952 +/- 0.015 |

Lecture : tous les ecarts sont plus petits que l'ecart-type. Pour classer NEU-CLS avec un
reseau pre-entraine, aucun pretraitement n'apporte de gain mesurable, meme avec un eclairage
inegal simule : les caracteristiques d'un ResNet pre-entraine y sont deja robustes. C'est le
resultat attendu d'un atelier honnete : un pretraitement n'est pas un gain gratuit. Il compte
surtout pour les methodes sensibles au niveau absolu des pixels (seuils classiques, cartes
d'anomalies, segmentation fine), pour les images 16 bits, les motifs periodiques, et quand
camera ou eclairage changent entre l'entrainement et la production.

Correction d'illumination sur un vignettage connu (notebook, ecart moyen centre / coins en
niveaux de gris) : image propre 1.1, vignettee 60.1 ; apres shading_division 5.9,
shading_subtraction 4.7, polynomial_background 5.7, morphological_background 19.9,
column_profile 34.3 (normal : le vignettage est radial, pas par colonne), flat_field 1.0.

Duree : gallery.py 25 s, compare.py 3 a 4 min par configuration (CPU 2 coeurs).
