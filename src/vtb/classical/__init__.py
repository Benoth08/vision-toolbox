"""Vision classique sans apprentissage : seuillage, blobs, mesure, comparaison a une reference.

A garder sous la main pour deux raisons : c'est souvent suffisant (contraste franc, pieces
identiques), et c'est toujours la base de comparaison qu'un modele appris doit battre.
"""

# blob = tache claire ou sombre isolee dans un masque ; detection, filtrage, tableau, dessin, retour en masque
from vtb.classical.blobs import blob_table, blobs_to_mask, detect_blobs, draw_blobs, filter_blobs
from vtb.classical.compare import compare_to_reference  # comparaison pixel a pixel a une piece de reference

# mesure : calibration pixels -> millimetres, largeur d'une bande, bord le plus net
from vtb.classical.measure import calibrate_chessboard, calibrate_from_distance, save_calibration, strip_width, strongest_edge
from vtb.classical.threshold import threshold, threshold_value  # seuillage : image -> masque binaire 0/1

__all__ = [  # noms exportes par "from vtb.classical import *" : l'API publique du module
    "blob_table",  # tableau des mesures de chaque blob
    "blobs_to_mask",  # blobs -> masque binaire
    "calibrate_chessboard",  # calibration avec une mire en damier
    "calibrate_from_distance",  # calibration avec une distance connue
    "compare_to_reference",  # comparaison a une piece de reference
    "detect_blobs",  # detection des blobs d'un masque
    "draw_blobs",  # dessin des blobs sur une image
    "filter_blobs",  # filtre des blobs par taille, forme...
    "save_calibration",  # sauvegarde d'une calibration
    "strip_width",  # largeur d'une bande (profil d'intensite)
    "strongest_edge",  # position du bord le plus net
    "threshold",  # seuillage d'une image
    "threshold_value",  # valeur de seuil choisie automatiquement
]
