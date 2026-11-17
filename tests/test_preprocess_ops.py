"""Tests du catalogue de pretraitements : chaque operation en gris / couleur, 8 / 16 bits, puis comportements cibles."""

# Deux sortes de tests ici : un "contrat" commun verifie pour TOUTES les operations (type, nombre de dimensions,
# taille, valeurs finies), puis des tests cibles qui verifient sur une image fabriquee que l'operation fait bien
# ce qu'elle promet (augmenter le contraste, enlever des rayures periodiques, isoler de petits objets...).

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[int], X | None) partout

from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import cv2  # OpenCV : dessin et traitement d'images
import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)
import pytest  # pytest : lance les tests ; fixtures, parametrize, raises...

from vtb.io.images import write_image  # ecrit un tableau numpy en fichier image
from vtb.preprocess import REGISTRY, Pipeline  # REGISTRY : toutes les operations par nom ; Pipeline : suite d'operations
from vtb.preprocess.ops.geometry import letterbox_params  # calcule taille et bordures d'un letterbox

SHAPE = (96, 128)  # taille (hauteur, largeur) des images de test
SHAPE_CHANGING = {"resize", "crop", "pad", "letterbox", "auto_roi", "perspective"}  # operations qui changent la taille
TO_GRAY = {"to_gray", "extract_channel"}  # operations qui rendent une image a un seul canal (2 dimensions)
TO_UINT8 = {"to_uint8", "window_level"}  # operations qui rendent toujours du 8 bits


# fabrique une image de test et la renvoie (8 ou 16 bits, gris ou couleur)
def scene(dtype=np.uint8, color: bool = False, seed: int = 0) -> np.ndarray:
    """Scene structuree (gradient, rectangles, disques, leger bruit) : assez de texture pour ECC et ORB."""
    rng = np.random.default_rng(seed)  # generateur aleatoire a graine choisie
    yy, xx = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]]  # numero de ligne et de colonne de chaque pixel
    base = 0.35 + 0.2 * xx / SHAPE[1] + 0.05 * np.sin(yy / 6.0)  # degrade horizontal + ondulation verticale (valeurs 0-1)
    cv2.rectangle(base, (15, 15), (45, 40), 0.85, -1)  # rectangle plein clair
    cv2.rectangle(base, (80, 55), (115, 85), 0.1, -1)  # rectangle plein sombre
    cv2.circle(base, (70, 25), 9, 0.95, -1)  # disque plein tres clair
    cv2.circle(base, (30, 70), 6, 0.05, -1)  # disque plein tres sombre
    base = np.clip(base + rng.normal(0, 0.01, base.shape), 0, 1)  # leger bruit, borne a [0, 1]
    peak = 255 if dtype == np.uint8 else 65535  # valeur maximale du type (8 ou 16 bits)
    image = (base * peak).astype(dtype)  # passe de [0, 1] a la plage entiere du type
    if color:  # version couleur demandee
        # 3 canaux un peu differents (100 %, 90 %, 80 %) pour qu'ils ne soient pas identiques
        image = np.stack([image, (image * 0.9).astype(dtype), (image * 0.8).astype(dtype)], axis=-1)
    return image  # renvoie l'image


# scope="module" : la fixture est creee une seule fois pour tout ce fichier (et non pour chaque test)
@pytest.fixture(scope="module")
def references(tmp_path_factory) -> dict[str, str]:  # tmp_path_factory : cree des dossiers temporaires partages
    """Images de reference sur disque (flat-field, alignement), en 8 et 16 bits couleur (relues en gris si besoin)."""
    root = tmp_path_factory.mktemp("refs")  # dossier temporaire "refs"
    paths = {}  # nom -> chemin des fichiers de reference
    for name, dtype in (("8", np.uint8), ("16", np.uint16)):  # une version 8 bits et une 16 bits
        write_image(root / f"ref{name}.png", scene(dtype, color=True))  # image de reference (pour aligner / comparer)
        # flat-field : photo d'une surface uniforme, qui montre seulement les defauts d'eclairage a corriger
        flat = np.full((*SHAPE, 3), 200 if dtype == np.uint8 else 51000, dtype)
        write_image(root / f"flat{name}.png", flat)  # ecrit l'image de flat-field
        # memorise les deux chemins
        paths[f"ref{name}"], paths[f"flat{name}"] = str(root / f"ref{name}.png"), str(root / f"flat{name}.png")
    return paths  # renvoie le dictionnaire des chemins


# renvoie les parametres obligatoires d'une operation (ceux qui n'ont pas de valeur par defaut utilisable)
def params_for(name: str, bits: str, references: dict[str, str]) -> dict:
    """Parametres a fournir aux operations qui n'ont pas de valeur par defaut utilisable."""
    if name == "flat_field":  # correction par flat-field
        return {"flat_path": references[f"flat{bits}"]}  # chemin de l'image uniforme
    if name in ("align_ecc", "align_orb", "absdiff_reference"):  # alignement (ECC, ORB) ou difference avec une reference
        return {"reference_path": references[f"ref{bits}"]}  # chemin de l'image de reference
    if name == "perspective":  # redressement de perspective
        return {"points": [[10, 8], [118, 12], [120, 90], [6, 86]], "width": 100, "height": 80}  # 4 coins source, taille de sortie
    if name == "undistort":  # correction de la distorsion de l'objectif
        # matrice de la camera (focale 120, centre optique 64, 48) et coefficients de distorsion
        return {"camera_matrix": [[120.0, 0, 64.0], [0, 120.0, 48.0], [0, 0, 1]], "dist_coeffs": [-0.1, 0.01, 0, 0, 0]}
    if name == "resize":  # redimension
        return {"size": [48, 64]}  # vers 48 x 64
    if name == "crop":  # recadrage
        return {"x": 10, "y": 5, "width": 50, "height": 40}  # rectangle a garder
    if name == "pad":  # ajout de bordure
        return {"top": 2, "left": 3}  # 2 pixels en haut, 3 a gauche
    if name == "letterbox":  # redimension sans deformation + bordure
        return {"size": 64}  # carre de 64
    if name == "nl_means":  # debruitage "non-local means" (moyenne de zones qui se ressemblent)
        return {"search_window": 11}  # petite fenetre de recherche pour aller vite
    return {}  # les autres operations n'ont besoin de rien


@pytest.mark.parametrize("name", sorted(REGISTRY))  # chaque operation du catalogue
@pytest.mark.parametrize("bits,color", [("8", False), ("8", True), ("16", False), ("16", True)])  # 8 / 16 bits, gris / couleur
# Verifie le contrat commun : type conserve (sauf passage en 8 bits voulu), nombre de dimensions et taille conserves
# (sauf operations qui les changent par nature), valeurs finies. Attraperait : une operation qui passe en flottant,
# qui casse en 16 bits ou en couleur, ou qui produit des NaN (valeurs "pas un nombre").
def test_every_operation_keeps_the_image_contract(name, bits, color, references):
    image = scene(np.uint8 if bits == "8" else np.uint16, color)  # image de test au bon type
    if name == "extract_channel" and not color:  # extraire un canal d'une image grise n'a pas de sens
        with pytest.raises(ValueError):  # on attend donc une ValueError
            Pipeline([{"name": name}])(image)  # appel qui doit echouer
        return  # fin du test pour ce cas
    out = Pipeline([{"name": name, **params_for(name, bits, references)}])(image)  # applique l'operation seule
    assert out.dtype == (np.uint8 if name in TO_UINT8 else image.dtype), name  # meme type ; en cas d'echec pytest affiche le nom
    expected_ndim = 2 if name in TO_GRAY else image.ndim  # nombre de dimensions attendu (2 = gris, 3 = couleur)
    assert out.ndim == expected_ndim, name  # le nombre de dimensions est respecte
    if name not in SHAPE_CHANGING:  # pour les operations qui ne changent pas la taille
        assert out.shape[:2] == image.shape[:2], name  # meme hauteur et largeur
    assert np.isfinite(out.astype(np.float64)).all()  # aucune valeur infinie ou NaN


# mesure le contraste local : ecart moyen entre l'image et sa version floutee ; renvoie un nombre
def local_contrast(image: np.ndarray) -> float:
    x = image.astype(np.float32)  # en flottant pour eviter les debordements
    return float(np.abs(x - cv2.GaussianBlur(x, (0, 0), 4)).mean())  # moyenne des differences avec le flou gaussien


# Verifie que CLAHE (egalisation d'histogramme locale et bornee) et l'egalisation globale augmentent le contraste,
# aussi en 16 bits. Attraperait : une operation sans effet, ou qui ramene du 16 bits en 8 bits.
def test_clahe_and_equalize_increase_contrast():
    yy, xx = np.mgrid[0:128, 0:128]  # numeros de ligne et de colonne
    flat = (120 + 6 * np.sin(xx / 3.0) * np.sin(yy / 5.0)).astype(np.uint8)  # texture tres peu contrastee
    # le contraste local apres CLAHE est plus du double de celui d'avant
    assert local_contrast(Pipeline([{"name": "clahe", "clip_limit": 3.0}])(flat)) > 2 * local_contrast(flat)
    eq = Pipeline([{"name": "equalize_hist"}])(flat)  # egalisation globale de l'histogramme
    assert eq.min() == 0 and eq.max() == 255  # l'image occupe maintenant toute la plage 0-255
    eq16 = Pipeline([{"name": "equalize_hist"}])(flat.astype(np.uint16) * 100)  # meme chose sur une image 16 bits
    assert eq16.dtype == np.uint16 and eq16.max() > 60000  # reste en 16 bits et monte pres du maximum (65535)


# Verifie que le filtre FFT "notch" retire un motif periodique (stries) sans effacer le defaut.
# FFT : transformee de Fourier, decrit l'image comme une somme de frequences ; un motif regulier = un pic de frequence.
# Attraperait : un filtre qui laisse les stries, ou qui efface aussi le vrai defaut.
def test_fft_notch_removes_a_periodic_pattern_and_keeps_the_defect():
    yy, xx = np.mgrid[0:128, 0:160]  # numeros de ligne et de colonne
    clean = np.full((128, 160), 0.5)  # fond gris uniforme
    clean[60:70, 70:80] = 0.9  # le "defaut" a garder
    stripes = clean + 0.15 * np.sin(2 * np.pi * 12 * xx / 160)  # stries verticales : 12 periodes sur la largeur
    image = (np.clip(stripes, 0, 1) * 255).astype(np.uint8)  # image 8 bits
    out = Pipeline([{"name": "fft_notch"}])(image)  # filtre automatique (trouve lui-meme le pic)

    # amplitude du pic des stries dans le spectre (colonne de frequence 12) ; renvoie un nombre
    def amplitude(img: np.ndarray) -> float:
        spectrum = np.abs(np.fft.fft2(img.astype(np.float64) - img.mean()))  # spectre 2D (moyenne retiree)
        return float(spectrum[:, 12].max())  # la frequence 12 en x correspond aux stries

    assert amplitude(out) < 0.1 * amplitude(image)  # les stries ont perdu plus de 90 % de leur force
    assert out[62:68, 72:78].mean() > out[:, :40].mean() + 60  # le carre clair est toujours la
    manual = Pipeline([{"name": "fft_notch", "frequencies": [[0, 12]]}])(image)  # meme filtre, frequence donnee a la main
    assert amplitude(manual) < 0.1 * amplitude(image)  # meme resultat


# Verifie que le filtre median retire le bruit "poivre et sel" (pixels isoles tout noirs ou tout blancs).
# Attraperait : un filtre median inefficace, ou qui ne marche pas en 16 bits.
def test_median_removes_salt_and_pepper():
    rng = np.random.default_rng(1)  # generateur aleatoire a graine fixe
    image = np.full((64, 64), 128, np.uint8)  # image grise uniforme
    noisy = image.copy()  # copie a bruiter
    spots = rng.random(image.shape)  # un tirage entre 0 et 1 par pixel
    noisy[spots < 0.03], noisy[spots > 0.97] = 0, 255  # 3 % de pixels noirs, 3 % de pixels blancs
    out = Pipeline([{"name": "median_blur", "ksize": 3}])(noisy)  # median sur un voisinage 3 x 3
    # apres : moins de 0.5 % de pixels extremes ; avant : plus de 0.5 %
    assert ((out == 0) | (out == 255)).mean() < 0.005 < ((noisy == 0) | (noisy == 255)).mean()
    out16 = Pipeline([{"name": "median_blur", "ksize": 7}])(noisy.astype(np.uint16) * 257)  # meme chose en 16 bits, voisinage 7 x 7
    assert out16.dtype == np.uint16  # reste en 16 bits


# Verifie le top-hat (garde les petits objets clairs) et le black-hat (garde les petits objets sombres), operations
# morphologiques qui retirent le fond. Attraperait : un fond en degrade ou un grand objet pris pour un petit defaut.
def test_tophat_and_blackhat_isolate_small_objects():
    yy, xx = np.mgrid[0:96, 0:96]  # numeros de ligne et de colonne
    image = (60 + xx).astype(np.uint8)  # fond en degrade
    image[20:25, 20:25] = 250  # petit objet clair
    image[50:90, 50:90] = 250  # grand objet clair (doit disparaitre du top-hat)
    image[70:73, 10:13] = 5  # petite piqure sombre
    top = Pipeline([{"name": "tophat", "kernel": 11}])(image)  # top-hat avec un element de 11 pixels
    assert top[21:24, 21:24].mean() > 100 and top[60:80, 60:80].mean() < 10  # petit objet garde, grand objet efface
    black = Pipeline([{"name": "blackhat", "kernel": 11}])(image)  # black-hat avec le meme element
    assert black[70:73, 10:13].mean() > 40 and black[30:40, 60:70].mean() < 5  # piqure sombre visible, fond efface


# Verifie les operations d'intensite : etirement, gamma, fenetrage 16 -> 8 bits, inversion.
# Attraperait : une formule fausse (gamma qui assombrit au lieu d'eclaircir, fenetre mal centree...).
def test_intensity_operations():
    ramp = np.tile(np.arange(100, 151, dtype=np.uint8), (8, 1))  # rampe de 100 a 150, repetee sur 8 lignes
    stretched = Pipeline([{"name": "percentile_stretch", "low": 0, "high": 100}])(ramp)  # etire min-max sur 0-255
    assert stretched.min() == 0 and stretched.max() == 255  # toute la plage est utilisee
    assert (Pipeline([{"name": "gamma", "gamma": 0.5}])(ramp) >= ramp).all()  # gamma < 1 eclaircit
    raw = np.array([[29000, 30000, 31000, 40000]], np.uint16)  # 4 valeurs 16 bits
    # fenetrage (comme en radiographie) : la plage centre +/- largeur/2 = 29000-31000 est etalee sur 0-255
    window = Pipeline([{"name": "window_level", "center": 30000, "width": 2000}])(raw)
    assert window.tolist() == [[0, 128, 255, 255]]  # bas de fenetre = 0, centre = 128, haut et au-dela = 255
    assert Pipeline([{"name": "invert"}])(np.array([[0, 255]], np.uint8)).tolist() == [[255, 0]]  # inversion : noir <-> blanc


# Verifie letterbox, recadrage automatique sur l'objet et redressement d'une piece inclinee.
# Letterbox : redimension sans deformer, puis bordure pour obtenir un carre (utilise par YOLO).
# Attraperait : des bordures mal placees (boites decalees en detection), un recadrage qui coupe, un angle mal corrige.
def test_letterbox_auto_roi_and_deskew():
    image = np.zeros((60, 100), np.uint8)  # image noire 60 x 100
    boxed = Pipeline([{"name": "letterbox", "size": 80, "pad_value": 114}])(image)  # carre 80 x 80, bordure grise 114
    ratio, new_w, new_h, pad_x, pad_y = letterbox_params(60, 100, 80)  # calcul theorique : echelle, taille utile, bordures
    # carre de 80 ; coin = bordure (114) ; 100 -> 80 de large, 60 -> 48 de haut, bordure de 16 en haut (centree)
    assert boxed.shape == (80, 80) and boxed[0, 0] == 114 and (new_w, new_h, pad_x, pad_y) == (80, 48, 0, 16)
    scene_img = np.zeros((100, 120), np.uint8)  # image noire 100 x 120
    scene_img[30:60, 40:90] = 200  # objet clair de 30 x 50
    roi = Pipeline([{"name": "auto_roi", "margin": 2}])(scene_img)  # recadre automatiquement sur l'objet, marge 2
    assert roi.shape == (34, 54)  # 30 + 2 x 2 et 50 + 2 x 2
    strip = np.zeros((40, 120), np.uint8)  # image 40 x 120
    strip[:, 25:95] = 180  # bande claire verticale de 70 colonnes
    # mode "columns" : recadre seulement en largeur
    assert Pipeline([{"name": "auto_roi", "mode": "columns", "margin": 0}])(strip).shape == (40, 70)
    tilted = np.zeros((200, 200), np.uint8)  # image noire 200 x 200
    corners = cv2.boxPoints(((100, 100), (120, 40), 12.0)).astype(np.int32)  # coins d'un rectangle 120 x 40 incline de 12 degres
    cv2.fillPoly(tilted, [corners], 255)  # dessine ce rectangle en blanc
    straight = Pipeline([{"name": "deskew"}])(tilted)  # redresse l'image
    # angle du plus petit rectangle qui englobe les pixels blancs apres redressement
    _, _, angle = cv2.minAreaRect(cv2.findNonZero((straight > 127).astype(np.uint8)))
    assert min(abs(angle), abs(angle - 90), abs(angle + 90)) < 1.5  # droit a 1.5 degre pres (0 ou +/- 90 selon la convention)


# Verifie que l'alignement ECC compense un decalage, et que la difference avec la reference fait ressortir le defaut.
# Attraperait : un alignement rate, qui ferait apparaitre toute la scene comme un defaut.
def test_alignment_recovers_a_shift_and_absdiff_shows_the_defect(tmp_path: Path):
    reference = scene(np.uint8)  # image de reference (piece bonne)
    write_image(tmp_path / "ref.png", reference)  # ecrite sur disque (l'operation la lit depuis un fichier)
    shifted = np.roll(reference, (2, 3), axis=(0, 1))  # la meme decalee de 2 lignes et 3 colonnes
    # alignement ECC limite a une translation
    aligned = Pipeline([{"name": "align_ecc", "reference_path": str(tmp_path / "ref.png"), "motion": "translation"}])(shifted)
    inner = (slice(10, -10), slice(10, -10))  # zone interieure (on ignore 10 pixels de bord, abimes par le decalage)
    assert np.abs(aligned[inner].astype(int) - reference[inner].astype(int)).mean() < 2.0  # ecart moyen < 2 niveaux : bien aligne
    defective = shifted.copy()  # copie decalee...
    cv2.circle(defective, (100, 30), 4, 255, -1)  # ...avec un defaut : disque blanc de rayon 4
    # difference absolue avec la reference (apres alignement interne)
    diff = Pipeline([{"name": "absdiff_reference", "reference_path": str(tmp_path / "ref.png")}])(defective)
    assert diff[27:33, 97:103].mean() > 80 and diff[inner].mean() < 15  # le defaut ressort fort, le reste presque nul


# Verifie la balance des blancs "gray world" et l'extraction d'un canal HSV.
# Attraperait : une dominante de couleur non corrigee, ou un canal extrait dans le mauvais espace de couleur.
def test_color_operations():
    image = np.zeros((10, 10, 3), np.uint8)  # image couleur 10 x 10
    image[:, :, 0], image[:, :, 1], image[:, :, 2] = 200, 100, 50  # forte dominante rouge
    balanced = Pipeline([{"name": "gray_world"}])(image)  # "gray world" : suppose que la scene est grise en moyenne
    means = balanced.reshape(-1, 3).mean(axis=0)  # moyenne de chaque canal
    assert means.max() - means.min() < 2  # les 3 canaux ont maintenant presque la meme moyenne
    # HSV : teinte, saturation, valeur (luminosite) ; channel 2 = V
    value = Pipeline([{"name": "extract_channel", "space": "hsv", "channel": 2}])(image)
    assert value.shape == (10, 10) and abs(int(value[0, 0]) - 200) <= 1  # V = max(R, G, B)


# Verifie que les filtres de bords (Sobel, Canny) et passe-bande (difference de gaussiennes) ont une echelle fixe.
# Attraperait : une sortie renormalisee image par image, qui amplifierait le bruit d'une image sans aucun bord.
def test_edges_and_bandpass_have_a_fixed_scale():
    uniform = np.full((32, 32), 100, np.uint8)  # image uniforme : aucun detail
    assert (Pipeline([{"name": "difference_of_gaussians"}])(uniform) == 128).all()  # rien a signaler = gris moyen
    assert Pipeline([{"name": "sobel"}])(uniform).max() == 0  # Sobel (force des bords) : aucun bord, tout a 0
    step = np.zeros((32, 32), np.uint8)  # image noire...
    step[:, 16:] = 255  # ...dont la moitie droite est blanche : un bord vertical en colonne 16
    edges = Pipeline([{"name": "canny"}])(step)  # detecteur de contours Canny
    assert set(np.unique(edges)) <= {0, 255} and edges[:, 14:18].any()  # sortie binaire 0 / 255, contour trouve pres de la colonne 16
