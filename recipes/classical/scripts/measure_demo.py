"""Demonstration de mesure : calibration sur damier, puis largeur d'une bande en millimetres au sous-pixel.

    python recipes/classical/scripts/measure_demo.py --output outputs/classical/measure

Tout est synthetique (pas besoin de camera) : un damier de cases de 5 mm photographie sous huit
angles sert a calibrer, puis une bande de largeur connue (60,25 mm) est mesuree ligne par ligne.
Produit calibration.json, widths.csv et measure.log ; copie dans docs/assets/classical.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[str], X | None) partout

import argparse  # argparse : lit les options passees en ligne de commande (--output, --assets)
import logging  # logging : affiche des messages de suivi (mieux que print dans un script)
import shutil  # shutil : copie de fichiers (le journal vers la documentation)
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import cv2  # OpenCV : traitement d'images (ici : deformation en perspective du damier)
import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)

# calibration de camera sur damier, enregistrement en JSON, mesure de largeur d'une bande ligne par ligne
from vtb.classical.measure import calibrate_chessboard, save_calibration, strip_width
from vtb.core.runtime import setup_logging  # configure le journal (console + fichier)

logger = logging.getLogger(__name__)  # "journal" propre a ce fichier : logger.info(...) ecrit un message de suivi
# une case du damier = 20 pixels = 5 mm ; PATTERN = nombre de coins interieurs (colonnes, lignes) cherches par OpenCV
SQUARE_PX, SQUARE_MM, PATTERN = 20, 5.0, (7, 5)


def chessboard_views(n_views: int = 8, seed: int = 0) -> list[np.ndarray]:  # renvoie n_views images de damier
    """Un damier (PATTERN coins interieurs) vu sous plusieurs perspectives legeres."""
    rows, cols = PATTERN[1] + 1, PATTERN[0] + 1  # n coins interieurs = n + 1 cases dans chaque direction
    board = np.full(((rows + 2) * SQUARE_PX, (cols + 2) * SQUARE_PX), 255, np.uint8)  # feuille blanche avec une marge d'une case
    for r in range(rows):  # pour chaque rangee de cases
        for c in range(cols):  # pour chaque colonne de cases
            if (r + c) % 2 == 0:  # une case sur deux, en alternance (motif de damier)
                # peint la case en noir (decalee d'une case pour la marge) ; a:b = tranche de lignes ou de colonnes
                board[(r + 1) * SQUARE_PX : (r + 2) * SQUARE_PX, (c + 1) * SQUARE_PX : (c + 2) * SQUARE_PX] = 0
    h, w = board.shape  # hauteur et largeur du damier en pixels
    rng, views = np.random.default_rng(seed), []  # generateur de hasard reproductible ; liste des vues produites
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])  # les 4 coins du damier (haut-gauche, haut-droit, bas-droit, bas-gauche)
    for k in range(n_views):  # pour chaque vue a fabriquer
        # deplace chaque coin de -12 a +12 pixels au hasard (simule une camera un peu inclinee) ; la 1re vue reste droite
        jitter = rng.uniform(-12, 12, (4, 2)).astype(np.float32) if k else np.zeros((4, 2), np.float32)
        # deforme le damier pour que ses coins aillent aux positions deplacees (+20 : marge) ; le fond ajoute reste blanc
        views.append(cv2.warpPerspective(board, cv2.getPerspectiveTransform(src, src + jitter + 20), (w + 40, h + 40), borderValue=255))
    return views  # renvoie la liste d'images


def strip_image(width_mm: float, mm_per_pixel: float, noise: float = 4.0, seed: int = 1) -> np.ndarray:  # renvoie une image 60 x 400
    """Bande claire sur fond sombre, bords flous (optique), un peu de bruit, largeur width_mm."""
    columns = np.arange(400, dtype=np.float64)  # numeros de colonne 0, 1, ..., 399 (image de 400 pixels de large)
    left = 80.3  # bord gauche de la bande, volontairement entre deux pixels (mesure au sous-pixel)
    right = left + width_mm / mm_per_pixel  # bord droit : largeur en mm convertie en pixels
    # profil d'intensite d'une ligne : fond a 40, bande a 200 ; chaque bord est une marche adoucie (sigmoide) comme en optique
    profile = 40 + 160 / (1 + np.exp(-(columns - left) / 0.8)) - 160 / (1 + np.exp(-(columns - right) / 0.8))
    # repete le profil sur 60 lignes et ajoute un bruit aleatoire (ecart-type noise) comme sur un vrai capteur
    image = np.tile(profile, (60, 1)) + np.random.default_rng(seed).normal(0, noise, (60, 400))
    return np.clip(image, 0, 255).astype(np.uint8)  # borne a 0-255 et passe en entiers 8 bits (image standard)


def main() -> None:  # point d'entree : calibre, mesure la bande, ecrit les resultats ; ne renvoie rien
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # l'aide reprend la 1re ligne de la docstring
    parser.add_argument("--output", type=Path, default=Path("outputs/classical/measure"))  # dossier des resultats
    parser.add_argument("--assets", type=Path, default=Path("docs/assets/classical"))  # dossier de la documentation
    args = parser.parse_args()  # lit la ligne de commande ; args.output et args.assets contiennent les valeurs
    setup_logging(args.output / "measure.log")  # journal sur la console et dans measure.log (cree le dossier au besoin)

    # calibration : retrouve les reglages de la camera (matrice, distorsion) et l'echelle mm / pixel a partir des damiers
    calibration = calibrate_chessboard(chessboard_views(), pattern=PATTERN, square_mm=SQUARE_MM)
    save_calibration(calibration, args.output / "calibration.json")  # enregistre la calibration en JSON
    scale = calibration["mm_per_pixel"]  # echelle mesuree : combien de millimetres represente un pixel
    logger.info(  # affiche le resultat de la calibration
        # erreur de reprojection = ecart moyen (pixels) entre coins detectes et coins recalcules par le modele de camera
        "calibration : %d vues, erreur de reprojection %.3f px, echelle %.4f mm / px (attendu %.4f)",
        calibration["n_views"],  # nombre de vues ou le damier a ete trouve
        calibration["rms_px"],  # erreur de reprojection (plus petit = meilleure calibration)
        scale,  # echelle mesuree
        SQUARE_MM / SQUARE_PX,  # echelle attendue (5 mm / 20 px = 0,25 mm / px) pour comparer
    )

    true_mm = 60.25  # vraie largeur de la bande, connue puisque l'image est synthetique
    widths = strip_width(strip_image(true_mm, SQUARE_MM / SQUARE_PX))  # mesure ligne par ligne : bords et largeur en pixels
    widths["width_mm"] = widths["width"] * scale  # largeur convertie en millimetres avec l'echelle de la calibration
    widths.to_csv(args.output / "widths.csv", index=False)  # enregistre toutes les mesures
    mean, std = widths["width_mm"].mean(), widths["width_mm"].std()  # moyenne et ecart-type (dispersion) des largeurs
    logger.info(  # affiche le bilan de la mesure
        "largeur mesuree : %.3f mm +/- %.3f (ecart-type sur %d lignes) ; vraie largeur %.2f mm ; erreur %.3f mm",  # modele
        mean,  # largeur moyenne mesuree
        std,  # dispersion d'une ligne a l'autre (effet du bruit)
        len(widths),  # nombre de lignes mesurees
        true_mm,  # vraie largeur
        mean - true_mm,  # erreur de mesure (mesure - vraie valeur)
    )
    args.assets.mkdir(parents=True, exist_ok=True)  # cree le dossier de la documentation s'il n'existe pas
    shutil.copy2(args.output / "measure.log", args.assets / "measure.log")  # copie le journal pour la documentation


if __name__ == "__main__":  # vrai seulement quand on lance le fichier directement (pas quand on l'importe)
    main()  # lance le script
