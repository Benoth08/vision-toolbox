"""Affichage : images, metriques, courbes d'entrainement (dossier autonome, copiable seul).

    from vtb import viz
    viz.setup_notebook(dpi=80, fmt="jpeg")
    viz.show(image); viz.grid(images, titles); viz.compare({"avant": a, "apres": b})
    viz.confusion_matrix(y_true, y_pred, names); viz.roc_curves(y, scores)
    live = viz.LivePlot(("train_loss", "val_loss")); live.update(epoch, train_loss=..., val_loss=...)

Dependances : numpy, matplotlib, opencv, pandas (+ scikit-learn pour ROC / PR, torch facultatif).
Aucun import du reste de vtb : copier le dossier `viz/` a cote d'un notebook suffit
(`import viz`).
"""

from .images import (  # fonctions d'affichage d'images (fichier images.py du meme dossier)
    as_numpy,  # tenseur, image PIL ou liste -> tableau numpy
    class_montage,  # une ligne d'exemples par classe
    compare,  # images cote a cote (avant / apres, difference)
    draw_boxes,  # dessine des boites et leurs etiquettes sur une image
    draw_detections,  # dessine verite terrain et predictions de detection
    grid,  # planche de vignettes avec titres et cadres
    histogram,  # histogramme des valeurs des pixels
    overlay_heatmap,  # carte de chaleur (anomalie, Grad-CAM) fondue sur l'image
    overlay_mask,  # masque de classes colore sur l'image
    pipeline_steps,  # image apres chaque etape d'un pretraitement
    profile,  # profil des niveaux (une ligne, une colonne, ou moyen)
    read_image,  # lecture d'une image pour l'affichage (RGB)
    show,  # affiche une image
    show_augmentations,  # plusieurs tirages d'une augmentation aleatoire
    show_batch,  # un lot tel que le voit le reseau
    to_display,  # image quelconque -> tableau uint8 affichable
)
from .metrics import (  # fonctions d'affichage de metriques (fichier metrics.py)
    calibration,  # diagramme de fiabilite : confiance annoncee vs exactitude reelle
    classification_report,  # precision, rappel, F1 par classe (tableau)
    compare_runs,  # barres groupees pour comparer plusieurs essais
    confusion_matrix,  # matrice de confusion (qui est predit quoi)
    errors_gallery,  # galerie des erreurs les plus sures
    latency_bars,  # barres de latence (ms par image)
    metrics_table,  # metriques mises en tableau
    per_class_bars,  # barres horizontales, une par classe
    pr_curves,  # courbes precision-rappel
    roc_curves,  # courbes ROC (rappel vs fausses alarmes)
    score_histogram,  # histogrammes des scores des sains et des defauts
    threshold_sweep,  # precision, rappel, F1 en fonction du seuil
)
from .style import TRUTH_COLOR, VizFigure, class_colors, in_notebook, render, save, setup_notebook  # couleurs, notebook, sauvegarde
from .training import LivePlot, plot_history  # courbes d'entrainement : en direct (LivePlot) ou apres coup (plot_history)

__all__ = [  # noms publics du module : ce que "from vtb.viz import *" importe
    "TRUTH_COLOR",  # couleur de la verite terrain (vert vif)
    "LivePlot",  # courbes d'entrainement en direct
    "VizFigure",  # figure qui s'affiche une seule fois dans un notebook
    "as_numpy",  # conversion en tableau numpy
    "calibration",  # diagramme de fiabilite
    "class_colors",  # couleurs stables par classe
    "class_montage",  # exemples par classe
    "classification_report",  # rapport par classe
    "compare",  # images cote a cote
    "compare_runs",  # comparaison d'essais
    "confusion_matrix",  # matrice de confusion
    "draw_boxes",  # dessin de boites
    "draw_detections",  # verite + predictions
    "errors_gallery",  # erreurs les plus sures
    "grid",  # planche de vignettes
    "histogram",  # histogramme des pixels
    "in_notebook",  # vrai dans un notebook Jupyter
    "latency_bars",  # barres de latence
    "metrics_table",  # tableau de metriques
    "overlay_heatmap",  # carte de chaleur sur image
    "overlay_mask",  # masque colore sur image
    "per_class_bars",  # barres par classe
    "pipeline_steps",  # etapes d'un pretraitement
    "plot_history",  # courbes d'entrainement apres coup
    "pr_curves",  # courbes precision-rappel
    "profile",  # profil des niveaux
    "read_image",  # lecture d'image
    "render",  # figure -> octets d'image (png / jpeg)
    "roc_curves",  # courbes ROC
    "save",  # enregistre une figure
    "score_histogram",  # histogrammes des scores
    "setup_notebook",  # reglage des figures du notebook
    "show",  # affiche une image
    "show_augmentations",  # tirages d'augmentation
    "show_batch",  # affiche un lot
    "threshold_sweep",  # balayage du seuil
    "to_display",  # image -> uint8 affichable
]
