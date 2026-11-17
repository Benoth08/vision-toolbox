# Outils de travail : afficher, diagnostiquer, juger un résultat

**Dans le fil : des outils pour plusieurs étapes.** Afficher les images et les lots : étapes 2 (Importer et explorer les données) et 7 (Charger par lots) ; diagnostiquer un entraînement : étapes 9 (Perte, optimiseur, hyperparamètres) et 10 (Entraîner) ; juger un résultat et choisir un seuil : étape 11 (Évaluer). Le tableau « Quand utiliser quoi », en fin de page, donne l'étape de chaque outil.

Tout projet de vision refait les mêmes gestes : regarder des images, superposer un masque, tracer une matrice de confusion, chercher un taux d'apprentissage, vérifier qu'un modèle peut apprendre, donner un intervalle de confiance, choisir un seuil. Deux modules les regroupent, écrits une fois et testés, utilisables dans les deux modes de la toolbox (notebook et production) :

| Module | Rôle | Dépendances |
|---|---|---|
| `vtb.viz` | **afficher** : images, superpositions, boîtes, métriques, courbes d'entraînement | numpy, matplotlib, OpenCV (pandas, scikit-learn, torch facultatifs) ; n'importe rien d'autre de la toolbox : le dossier se copie seul à côté d'un notebook |
| `vtb.toolkit` | **travailler** : inventaire et conversions de dossiers, diagnostics d'entraînement, statistiques pour juger un résultat | la toolbox (lecture d'images, tuiles) |
| `python -m vtb.toolkit` | les mêmes outils **en ligne de commande**, sans ouvrir Python | idem |

Le notebook `19_boite_a_outils` les montre tous sur un vrai résultat (un petit classifieur NEU-CLS entraîné en une minute) ; les figures de cette page en viennent.

## Afficher : `vtb.viz`

Trois principes. Toute image est acceptée telle quelle : grise ou couleur, 8 ou 16 bits, flottante, tenseur torch (C, H, W), normalisée ou non (`mean`, `std` annulent la normalisation pour l'affichage) ; le titre rappelle la forme, le type et la plage de valeurs, là où se voient la moitié des bugs. Une grille est assemblée en **une seule image** (une seule `imshow`), donc instantanée même pour des centaines de vignettes. Chaque fonction de figure rend la figure (ou l'axe) et accepte `save=chemin` ; dans un notebook, la figure s'affiche une seule fois, sans `plt.show()`.

```python
from vtb import viz

viz.setup_notebook(dpi=80, fmt="jpeg")        # figures légères dans le .ipynb
viz.show(image, "brute", colorbar=True)        # une image, quelle qu'elle soit
viz.grid(images, titles, ncols=8, borders=ok)  # cadre vert / rouge
viz.compare({"origine": image, "CLAHE": clahe, "division": flat})
```

### Images

| Fonction | Sert à |
|---|---|
| `show(image, title, bgr=False, mean=None, std=None, colorbar=False)` | une image, avec forme / type / plage dans le titre |
| `to_display(image, bgr, mean, std)` | la convertir en uint8 affichable (pour l'écrire ou l'envoyer ailleurs) |
| `read_image(path)` | lire une image pour l'affichage (chemins Windows avec accents, 16 bits) |
| `grid(images, titles, ncols, borders, max_side)` | beaucoup d'images d'un coup : tableaux, **chemins** ou lot NCHW ; `borders` : couleurs ou booléens juste / faux |
| `class_montage(paths, labels, per_class)` | le tour d'un jeu de données : n images tirées par classe |
| `compare(dict ou liste, diff=False)` | avant / après, plusieurs prétraitements côte à côte ; `diff=True` ajoute la carte de différence |
| `histogram(image, log)` | dynamique, saturation, image trop sombre |
| `profile(image, row, column)` | profil d'une ligne / colonne, ou moyennes des lignes et des colonnes : un éclairage non uniforme se voit immédiatement |
| `overlay_mask(image, mask, class_names, contours)` | masque de segmentation en couleurs, avec contours et légende |
| `overlay_heatmap(image, heatmap, alpha, cmap)` | carte d'anomalie, Grad-CAM |
| `draw_boxes(image, boxes, labels, scores, class_names, rotated, polygons)` | boîtes alignées (x1, y1, x2, y2), orientées (cx, cy, w, h, angle) ou polygones |
| `draw_detections(image, truth, pred, class_names, conf)` | vérité en trait vert, prédictions en couleur de classe avec score |
| `show_batch(batch, labels, class_names, mean, std)` | ce que reçoit vraiment le réseau (normalisation annulée) |
| `show_augmentations(image, transform, n)` | la même image augmentée n fois : est-ce encore la même classe ? |
| `pipeline_steps(image, [(nom, fonction), ...])` | chaque étape d'un prétraitement |

{{img:docs/assets/outils/viz_profile.jpg|viz.profile sur une tuile magnétique : moyennes des colonnes et des lignes ; l'éclairage varie du simple au double d'un bord à l'autre.}}

{{img:docs/assets/outils/viz_masks.jpg|viz.overlay_mask sur Magnetic Tile : une couleur par type de défaut, contours et légende.}}

{{img:docs/assets/outils/viz_detections.jpg|viz.draw_detections sur NEU-DET : vérité en vert, prédictions en couleur de classe avec leur score. Les prédictions viennent de l'essai rapide quick_yolo (2 époques sur CPU) : des boîtes médiocres sont attendues.}}

### Métriques

| Fonction | Sert à |
|---|---|
| `confusion_matrix(y_true, y_pred, class_names, normalize="true")` | nombres **et** pourcentages par ligne : quelle classe est prise pour quelle autre |
| `classification_report(y_true, y_pred, class_names)` | tableau précision / rappel / F1 / effectif, moyennes macro et pondérée |
| `per_class_bars(values, reference)` | AP, IoU ou F1 par classe, avec une référence éventuelle |
| `roc_curves`, `pr_curves` | binaire ou une-contre-toutes, AUC / AP dans la légende, point de fonctionnement marqué |
| `threshold_sweep(y_true, scores, chosen)` | rappel, précision, fausses alarmes et F1 pour chaque seuil ; rend aussi le tableau |
| `score_histogram(scores, is_defect, threshold, log)` | les deux populations (saines / défauts) et le seuil : se séparent-elles ? |
| `calibration(y_true, probs)` | diagramme de fiabilité, ECE dans le titre |
| `errors_gallery(images, y_true, y_pred, probs, class_names, k)` | les k erreurs les plus sûres d'elles (souvent des erreurs d'étiquette) |
| `metrics_table`, `compare_runs`, `latency_bars` | comparer des expériences : tableau, barres, latences |

{{img:docs/assets/outils/fig_confusion.jpg|Matrice de confusion (nombres et pourcentages) et F1 par classe du classifieur du notebook 19.}}

{{img:docs/assets/outils/viz_classification_metrics.jpg|La même planche pour le réseau simple_cnn de la recette classification, sur le test NEU-CLS (exactitude 0,970, ECE 0,027).}}

{{img:docs/assets/outils/fig_curves.jpg|Courbes ROC et précision-rappel une-contre-toutes, et diagramme de fiabilité.}}

### Entraînement

`plot_history(history, best="val_loss:min")` trace pertes et métriques (un `history.csv` de recette, un DataFrame, un dictionnaire de listes) et marque la meilleure époque. `LivePlot` redessine **une seule** figure à chaque époque dans un notebook (et imprime une ligne dans un script) :

```python
live = viz.LivePlot(("loss", "val_loss", "val_acc"))
for epoch in range(epochs):
    ...
    live.update(epoch, loss=train_loss, val_loss=val_loss, val_acc=val_acc)
live.close()
live.history.to_csv("history.csv", index=False)
```

{{img:docs/assets/outils/fig_history.jpg|Courbes d'entraînement de la tête linéaire (notebook 19), minimum brut de val_loss marqué ; l'arrêt anticipé avec min_delta = 0,001 a retenu l'époque 23 et arrêté à la 28e.}}

## Travailler sur les données : `vtb.toolkit.dataset`

| Fonction | Sert à |
|---|---|
| `folder_summary(root)` puis `summarize(files)` | inventaire d'un dossier reçu : nombre d'images par sous-dossier, tailles, canaux, types (8 / 16 bits), fichiers illisibles et leur cause |
| `mean_std(paths, size, max_images, color)` | moyenne et écart-type par canal (valeurs dans [0, 1]), **sur le train** ; utile pour un réseau appris de zéro |
| `class_weights(labels, mode)` | poids pour une perte pondérée : `inverse` (compense tout), `sqrt` (à moitié, souvent plus stable), `effective` (nombre effectif d'exemples) ; normalisés pour ne pas changer l'échelle de la perte |
| `sample_weights(labels)` | un poids par exemple, pour `WeightedRandomSampler` (l'autre façon de rééquilibrer) |
| `sample_per_class(frame, n)` | sous-ensemble équilibré et reproductible pour un essai rapide |
| `export_imagefolder(frame, out, mode="link")` | ranger un catalogue découpé en `<split>/<classe>/` (torchvision, Keras, Ultralytics-cls), par copie ou par lien |
| `resize_folder(src, dst, size, keep_aspect, pad)` | redimensionner tout un dossier en parallèle (arborescence et 16 bits conservés) |
| `to_8bit(image, method)`, `convert_folder_8bit` | passer du 16 au 8 bits : `scale` (sans contraste inventé), `minmax`, `percentile` (robuste) |
| `extract_frames(video, out, every, fps, start, end)` | images d'une vidéo de ligne ; les images sautées ne sont pas décodées |
| `make_patches(image, mask, size, stride)`, `patch_folder(...)` | découper de grandes images **et leurs masques** en patchs ; `keep_empty` ne garde qu'une fraction des patchs sans défaut |

:::warning
Après `patch_folder`, la colonne `source` doit servir de **groupe** au découpage (`split.strategy: grouped`) : deux patchs d'une même image de part et d'autre du découpage, c'est une fuite.
:::

## Diagnostiquer un entraînement : `vtb.toolkit.training`

Deux diagnostics à lancer **avant** un long entraînement, puis les petites briques d'une boucle.

**`overfit_one_batch(model, batch, loss_fn)`** : un modèle correct apprend par cœur un seul lot en quelques dizaines de pas. Si la perte ne tombe pas sous 10 % de sa valeur initiale, le problème est dans le code (cibles, perte, gel, taux), pas dans les données : inutile de lancer cinquante époques. Le verdict (`df.attrs["verdict"]`) liste ce qu'il faut vérifier ; les poids sont restaurés à la fin.

**`lr_find(model, loader, loss_fn)`** : le taux croît exponentiellement, un lot par pas ; la perte lissée descend, touche un minimum, puis explose. La suggestion est la pente la plus forte **avant** le minimum. En fine-tuning complet d'un réseau pré-entraîné, prendre plutôt 3 à 10 fois moins. Le modèle et l'optimiseur sont restaurés.

{{img:docs/assets/outils/fig_lr_find.jpg|lr_find sur resnet18 (layer4 et tête dégelés), NEU-CLS : perte brute, perte lissée et suggestion.}}

| Fonction | Sert à |
|---|---|
| `model_summary(model, (1, 3, 224, 224))` | paramètres totaux / entraînables / gelés, taille en Mo, GFLOPs d'une passe avant, forme de sortie de chaque bloc |
| `benchmark(model ou fonction, exemple)` | latence moyenne, médiane, p95, débit ; premiers passages exclus, synchronisation GPU |
| `max_batch_size(model, (3, 224, 224), device="cuda")` | plus grand lot qui tient sur le GPU (doublement puis dichotomie, marge de 10 %) |
| `EarlyStopping(patience, mode)` | `.step(valeur)` rend True quand il faut s'arrêter ; `.improved` pour sauvegarder le meilleur modèle |
| `AverageMeter`, `Timer`, `seed_everything` | moyenne pondérée d'une époque, chronomètre, graines |

Les lots sont `(entrées, cibles)` comme un DataLoader de classification ou de segmentation ; pour tout autre format (détection, dictionnaires), passer `compute_loss(model, batch) -> perte`.

## Juger un résultat : `vtb.toolkit.stats`

Un test de 270 images donne une exactitude à plus ou moins deux ou trois points près : deux modèles à 0,96 et 0,97 ne sont **pas** départagés par ce test. Les fonctions de ce module disent ce qu'un chiffre veut vraiment dire.

| Fonction | Question |
|---|---|
| `bootstrap_ci(metric, y_true, y_pred, groups=None)` | quelle est la précision de ce chiffre ? Intervalle par rééchantillonnage du test, **par pièce** si plusieurs images viennent de la même pièce |
| `wilson_ci(succès, total)` | l'intervalle d'une proportion, directement : un rappel de 48/79 vaut 0,61 [0,50 ; 0,71] |
| `mcnemar(y_true, pred_a, pred_b)` | A est-il vraiment meilleur que B sur ces images ? Seuls comptent les désaccords |
| `threshold_for(scores, is_defect, min_recall=, max_false_alarm=, min_precision=)` | quel seuil pour une contrainte métier ? Choisi sur la **validation** ; si la contrainte est impossible, le seuil le plus proche et un message qui le dit |
| `operating_points(scores, is_defect)` | le tableau complet seuil → rappel / fausses alarmes / précision |
| `expected_calibration_error(y_true, probs)` | la confiance annoncée correspond-elle à la justesse réelle ? |
| `worst_errors(y_true, probs, k)` | les erreurs à ouvrir en premier |

{{img:docs/assets/outils/fig_threshold.jpg|Balayage du seuil sur la validation (pitted contre le reste) : rappel, précision, fausses alarmes et F1.}}

## En ligne de commande : `python -m vtb.toolkit`

| Commande | Exemple |
|---|---|
| `summary` | `python -m vtb.toolkit summary data/raw/neu_cls_flat --csv inventaire.csv` (`--ext .jpg` quand des masques `.png` sont rangés à côté des images) |
| `mean-std` | `python -m vtb.toolkit mean-std data/train --size 224 --max-images 1000` |
| `class-weights` | `python -m vtb.toolkit class-weights data/raw/neu_cls_flat --mode sqrt` |
| `resize` | `python -m vtb.toolkit resize data/brut data/512 --size 512 --pad` |
| `to-8bit` | `python -m vtb.toolkit to-8bit data/16bits data/8bits --method percentile` |
| `frames` | `python -m vtb.toolkit frames ligne.mp4 data/frames --fps 2 --start 10 --end 70` |
| `patches` | `python -m vtb.toolkit patches data/raw/magnetic_tile/MT_Crack data/patches --size 128 --keep-empty 0.2` |
| `imagefolder` | `python -m vtb.toolkit imagefolder outputs/classification/split.csv data/imagefolder --mode link` |
| `threshold` | `python -m vtb.toolkit threshold val_scores.csv --min-recall 0.95` |
| `ci` | `python -m vtb.toolkit ci predictions.csv --true label --pred pred --group piece` |
| `mcnemar` | `python -m vtb.toolkit mcnemar predictions.csv --pred-a modele_a --pred-b modele_b` |

Chaque commande a son `--help`. Exemple de sortie :

{{text:docs/assets/outils/cli_summary.txt|0}}

## Quand utiliser quoi

| Étape | Situation | Outil |
|---|---|---|
| 2 | on vient de recevoir un dossier d'images | `summary`, puis `viz.class_montage`, puis le contrôle qualité (notebook `05_exploration_qualite`) |
| 2, 5 | images 16 bits, ou trop grandes | `to-8bit`, `resize`, `patches` |
| 2 | une vidéo de ligne à annoter | `frames --fps 1` |
| 7 | classes déséquilibrées | `class_weights` ou `sample_weights`, et regarder le rappel par classe, pas l'exactitude |
| 9, 10 | la perte ne descend pas | `overfit_one_batch`, puis `lr_find` |
| 8 | choisir un modèle pour la ligne | `model_summary` et `benchmark` sur la machine cible |
| 11 | un chiffre de test à présenter | `bootstrap_ci` ou `wilson_ci`, toujours avec l'intervalle |
| 11 | deux modèles proches | `mcnemar` sur les mêmes images |
| 11 | fixer le seuil de production | `threshold_sweep` puis `threshold_for` sur la validation ; vérification unique sur le test |
| 11 | comprendre les erreurs | `errors_gallery`, `worst_errors`, `confusion_matrix` |

Le guide `guide_autonomie` reprend ces outils dans l'ordre des 13 étapes d'un projet, de la réception des données au service en production, avec ce qu'il faut faire quand quelque chose ne marche pas.
