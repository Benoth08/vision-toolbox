"""Tests de la vision classique sur images synthetiques : seuillage, blobs, mesure, calibration, comparaison."""

# Vision classique : traitements d'images sans apprentissage (seuils, contours, mesures geometriques).
# On fabrique des images dont on connait la reponse exacte (position, taille) et on verifie que les
# fonctions la retrouvent.

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[int], X | None) partout

import cv2  # OpenCV : lecture, dessin et traitement d'images
import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)
import pytest  # pytest : lance les tests ; pytest.approx compare des nombres a virgule avec une tolerance

from vtb.classical import (  # les fonctions de vision classique testees
    blob_table,  # tableau de mesures (aire, circularite, allongement...) de chaque tache claire ("blob")
    calibrate_chessboard,  # calibration de camera avec des photos d'un damier : donne les mm par pixel
    calibrate_from_distance,  # calibration simple : une distance connue en mm et sa longueur en pixels
    compare_to_reference,  # compare une image a une image de reference (piece "bonne") et isole les differences
    detect_blobs,  # trouve les taches et ne garde que celles qui passent les filtres
    draw_blobs,  # dessine les taches trouvees sur l'image
    filter_blobs,  # filtre un tableau de taches selon des seuils (allongement, aire...)
    strip_width,  # mesure la largeur d'une bande claire, ligne par ligne
    strongest_edge,  # position du bord le plus net d'un profil d'intensite, a une fraction de pixel pres
    threshold,  # seuillage : transforme une image en masque 0 / 1 (objet ou fond)
    threshold_value,  # valeur du seuil choisi automatiquement (methode d'Otsu)
)


# fabrique une image test et la renvoie : on sait exactement ou sont les deux objets
def shapes_image() -> np.ndarray:
    """Fond sombre, un disque (piqure) et une ligne fine (rayure), clairs."""
    image = np.full((120, 160), 30, np.uint8)  # image grise 120 x 160 remplie de 30 (fond sombre)
    cv2.circle(image, (40, 40), 10, 220, -1)  # disque plein (-1 = rempli) de rayon 10 centre en (40, 40), valeur 220
    cv2.line(image, (70, 90), (150, 80), 220, 2)  # ligne de (70, 90) a (150, 80), epaisseur 2 : la "rayure"
    return image  # renvoie l'image


@pytest.mark.parametrize("method", ["otsu", "triangle", "adaptive", "fixed"])  # le test tourne pour chaque methode de seuillage
# Verifie que chaque methode de seuillage separe les objets clairs du fond, avec un masque 0 / 1.
# Attraperait : un masque en 0 / 255 (incompatible avec la suite), ou un seuil inverse qui prend le fond pour l'objet.
def test_threshold_methods_find_the_bright_objects(method):
    # value : seuil de la methode "fixed" ; block_size et c : reglages de la methode "adaptive" (seuil local par zone)
    mask = threshold(shapes_image(), method=method, value=128, block_size=31, c=-20)
    assert mask.dtype == np.uint8 and set(np.unique(mask)) <= {0, 1}  # masque d'entiers 8 bits contenant seulement 0 et 1
    assert mask[40, 40] == 1 and mask[10, 10] == 0  # centre du disque = objet ; coin = fond
    assert threshold(shapes_image(), method="otsu", invert=True)[40, 40] == 0  # invert=True : objets sombres, donc le disque devient fond
    assert 30 <= threshold_value(shapes_image()) < 220  # deux niveaux seulement : Otsu coupe juste au-dessus du fond


# Verifie que les mesures de forme distinguent une piqure (ronde) d'une rayure (allongee).
# Attraperait : une formule de circularite ou d'allongement fausse, donc des defauts mal classes en inspection.
def test_blob_shape_measures_separate_a_pit_from_a_scratch():
    image = shapes_image()  # image test
    table = blob_table(threshold(image), image)  # une ligne de mesures par tache du masque (intensites lues dans l'image)
    assert len(table) == 2  # deux taches : le disque et la ligne
    disk, line = (  # range les deux lignes : la plus circulaire est le disque
        # ligne 0 puis 1 si la 0 est la plus circulaire, sinon l'inverse
        (table.iloc[0], table.iloc[1]) if table.iloc[0]["circularity"] > table.iloc[1]["circularity"] else (table.iloc[1], table.iloc[0])
    )
    assert disk["circularity"] > 0.8 and disk["elongation"] < 1.3  # disque : presque rond (1 = cercle parfait), peu allonge
    assert line["elongation"] > 10 and line["circularity"] < 0.4  # ligne : tres allongee, pas ronde
    assert disk["mean_intensity"] == pytest.approx(220, abs=1)  # intensite moyenne du disque = 220 (sa valeur de dessin)
    pits = filter_blobs(table, max_elongation=2.0)  # garde les taches peu allongees (les piqures)
    assert len(pits) == 1  # seul le disque passe
    kept, mask = detect_blobs(threshold(image), image, min_elongation=5.0)  # garde les taches tres allongees (les rayures)
    assert len(kept) == 1 and mask[40, 40] == 0 and mask.sum() > 50  # seule la ligne reste : le disque est efface du masque
    assert draw_blobs(image, table).shape == (120, 160, 3)  # le dessin renvoie une image couleur de la meme taille


# Verifie la mesure de bords a une fraction de pixel pres ("sous-pixel") et la largeur d'une bande.
# Attraperait : une mesure arrondie au pixel entier, trop grossiere pour du controle dimensionnel.
def test_subpixel_edge_and_strip_width():
    x = np.arange(60, dtype=np.float64)  # positions 0 a 59 le long d'une ligne de pixels
    edge = 20.3  # position exacte du bord, volontairement entre deux pixels
    profile = 50 + 150 / (1 + np.exp(-(x - edge) / 0.8))  # bord flou centre en 20.3
    assert strongest_edge(profile, "rising") == pytest.approx(edge, abs=0.1)  # bord montant retrouve a 0.1 pixel pres
    assert strongest_edge(profile[::-1], "falling") == pytest.approx(59 - edge, abs=0.1)  # profil retourne : bord descendant
    strip = np.zeros((20, 200), np.float32)  # image 20 lignes x 200 colonnes
    cols = np.arange(200)  # numeros de colonne 0 a 199
    # bande claire : bord montant en 40.25 et bord descendant en 160.75 (largeur exacte 120.5), identique sur chaque ligne
    strip[:] = 200 / (1 + np.exp(-(cols - 40.25) / 0.7)) * (1 - 1 / (1 + np.exp(-(cols - 160.75) / 0.7)))
    widths = strip_width(strip.astype(np.uint8))  # largeur mesuree ligne par ligne (image convertie en 8 bits)
    # largeur moyenne 120.5 a 0.3 pixel pres, et presque la meme sur toutes les lignes
    assert widths["width"].mean() == pytest.approx(120.5, abs=0.3) and widths["width"].std() < 0.05
    assert calibrate_from_distance(250, 50.0) == pytest.approx(0.2)  # 50 mm sur 250 pixels = 0.2 mm par pixel


# Verifie la calibration au damier sur des vues synthetiques en perspective. Attraperait : une conversion pixels -> mm
# fausse, qui rendrait fausses toutes les mesures en millimetres.
def test_chessboard_calibration_on_synthetic_views():
    square, pattern = 20, (7, 5)  # cases de 20 pixels ; 7 x 5 coins interieurs (ce que cherche OpenCV)
    # damier blanc avec une marge : (5 + 3) cases de haut, (7 + 3) cases de large
    board = np.full(((pattern[1] + 3) * square, (pattern[0] + 3) * square), 255, np.uint8)
    for r in range(pattern[1] + 1):  # pour chaque rangee de cases (6)
        for c in range(pattern[0] + 1):  # pour chaque colonne de cases (8)
            if (r + c) % 2 == 0:  # une case sur deux, en alternance
                y, x = (r + 1) * square, (c + 1) * square  # coin haut gauche de la case (decale d'une case de marge)
                board[y : y + square, x : x + square] = 0  # case noire
    h, w = board.shape  # hauteur et largeur du damier
    views = []  # les "photos" synthetiques du damier
    rng = np.random.default_rng(0)  # generateur aleatoire a graine fixe
    for k in range(8):  # 8 vues
        # decalage aleatoire des 4 coins (+/- 12 pixels) ; la premiere vue (k = 0) n'en a pas
        jitter = rng.uniform(-12, 12, (4, 2)).astype(np.float32) if k else np.zeros((4, 2), np.float32)
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])  # les 4 coins du damier
        matrix = cv2.getPerspectiveTransform(src, src + jitter + 20)  # transformation qui envoie ces coins sur les coins decales
        views.append(cv2.warpPerspective(board, matrix, (w + 40, h + 40), borderValue=255))  # vue en perspective, bord blanc
    calibration = calibrate_chessboard(views, pattern=pattern, square_mm=5.0)  # calibre : une case mesure 5 mm en vrai
    # au moins 6 vues ou le damier est trouve, et erreur de reprojection (ecart modele / image) sous 1 pixel
    assert calibration["n_views"] >= 6 and calibration["rms_px"] < 1.0
    assert calibration["mm_per_pixel"] == pytest.approx(5.0 / square, rel=0.05)  # 5 mm / 20 pixels = 0.25 mm par pixel (a 5 %)


# Verifie que la comparaison a une reference recale l'image puis isole le defaut ajoute, et lui seul.
# Attraperait : un recalage rate (toute la texture apparaitrait comme defaut) ou un defaut mal positionne.
def test_compare_to_reference_finds_the_inserted_defect():
    rng = np.random.default_rng(2)  # generateur aleatoire a graine fixe
    # texture aleatoire lissee (flou gaussien) : image de reference d'une piece "bonne"
    reference = cv2.GaussianBlur((rng.random((100, 140)) * 255).astype(np.uint8), (0, 0), 3)
    reference = cv2.normalize(reference, None, 0, 255, cv2.NORM_MINMAX)  # etire les valeurs sur toute la plage 0-255
    image = np.roll(reference, (1, 2), axis=(0, 1)).copy()  # image a controler : la reference decalee de 1 ligne et 2 colonnes
    cv2.circle(image, (90, 50), 5, 255, -1)  # on y ajoute un defaut : disque blanc de rayon 5 en (90, 50)
    # recalage ECC (methode qui aligne deux images), puis differences au-dessus de 0.2, taches d'au moins 10 pixels
    result = compare_to_reference(image, reference, align="ecc", threshold=0.2, min_area=10)
    blobs = result["blobs"]  # tableau des differences trouvees
    assert len(blobs) == 1  # une seule difference : le defaut
    # position du defaut dans le repere de la reference : (90 - 2, 50 - 1) = (88, 49), a 3 pixels pres
    assert abs(blobs.iloc[0]["cx"] - 88) < 3 and abs(blobs.iloc[0]["cy"] - 49) < 3  # defaut ramene dans le repere de la reference
