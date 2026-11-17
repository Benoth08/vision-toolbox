"""Analyse de blobs : les composantes connexes d'un masque, mesurees et filtrees.

C'est l'outil de base de la vision industrielle classique : apres un seuillage, chaque tache est
mesuree (aire, boite, elongation, circularite, intensite) et on ne garde que celles qui
ressemblent a un defaut (une piqure est petite et ronde, une rayure longue et fine). Le tableau
pandas se lit, se filtre et s'exporte directement.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[str], X | None) partout

import cv2  # OpenCV : composantes connexes, contours, moments, dessin
import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)
import pandas as pd  # pandas : tableaux a lignes et colonnes nommees (DataFrame)

COLUMNS = [  # colonnes du tableau des blobs (blob = tache isolee du masque), dans l'ordre d'affichage
    "id",  # numero du blob (1, 2, 3... ; 0 est le fond)
    "x",  # abscisse du coin haut gauche de sa boite englobante
    "y",  # ordonnee du coin haut gauche
    "width",  # largeur de la boite, en pixels
    "height",  # hauteur de la boite, en pixels
    "area",  # aire : nombre de pixels du blob
    "cx",  # abscisse du centre de gravite
    "cy",  # ordonnee du centre de gravite
    "perimeter",  # longueur du contour, en pixels
    "circularity",  # 1 pour un disque, proche de 0 pour une ligne
    "elongation",  # 1 pour un disque, grand pour une rayure
    "angle",  # orientation de l'axe principal, en degres
    "mean_intensity",  # niveau de gris moyen sous le blob (si l'image est donnee)
    "max_intensity",  # niveau de gris maximal sous le blob
]


# numerote les taches d'un masque ; renvoie (carte des numeros, statistiques, centres)
def label_components(mask: np.ndarray, connectivity: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(etiquettes (H, W) int32, statistiques OpenCV, centroides) ; l'etiquette 0 est le fond.

    OpenCvSharp : Cv2.ConnectedComponentsWithStats(mask, labels, stats, centroids, PixelConnectivity.Connectivity8).
    """
    # composante connexe = groupe de pixels non nuls qui se touchent (8 = diagonales comprises, 4 = cotes seulement)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=connectivity)
    return labels, stats, centroids  # count n'est pas renvoye : c'est stats.shape[0]


# mesure chaque blob du masque ; renvoie un DataFrame, une ligne par blob
def blob_table(mask: np.ndarray, image: np.ndarray | None = None, connectivity: int = 8) -> pd.DataFrame:
    """Une ligne par blob : position, taille, forme, intensite (si l'image est fournie).

    circularity = 4 pi aire / perimetre^2 (1 pour un disque, proche de 0 pour une ligne) ;
    elongation = rapport des axes de l'ellipse d'inertie (1 pour un disque, grand pour une rayure) ;
    angle = orientation de l'axe principal en degres.
    """
    labels, stats, centroids = label_components(mask, connectivity)  # numerote les blobs
    # image en niveaux de gris flottants pour les intensites ; None si aucune image n'est donnee
    gray = None if image is None else (image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)).astype(np.float32)
    rows = []  # une ligne (dictionnaire) par blob
    for blob_id in range(1, stats.shape[0]):  # pour chaque blob (on saute 0, le fond)
        x, y, w, h, area = (int(v) for v in stats[blob_id])  # boite englobante (x, y, largeur, hauteur) et aire
        component = (labels[y : y + h, x : x + w] == blob_id).astype(np.uint8)  # petit masque 0/1 du seul blob, dans sa boite
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # contour exterieur, tous les points
        perimeter = float(sum(cv2.arcLength(c, True) for c in contours))  # longueur du contour ferme (True)
        moments = cv2.moments(component, binaryImage=True)  # moments : sommes ponderees des positions, decrivent la forme
        mu20, mu02, mu11 = moments["mu20"], moments["mu02"], moments["mu11"]  # moments centres d'ordre 2 (dispersion en x, en y, croisee)
        common = np.sqrt(4 * mu11**2 + (mu20 - mu02) ** 2)  # terme commun aux deux axes de l'ellipse d'inertie
        major, minor = (mu20 + mu02 + common) / 2, (mu20 + mu02 - common) / 2  # dispersion le long du grand et du petit axe
        # elongation = rapport grand axe / petit axe ; blob d'un pixel d'epaisseur (petit axe nul) : on prend le plus grand cote
        elongation = float(np.sqrt(major / minor)) if minor > 1e-9 else float(max(w, h))
        angle = float(np.degrees(0.5 * np.arctan2(2 * mu11, mu20 - mu02)))  # orientation du grand axe, en degres
        # niveaux de gris des pixels du blob ; sans image : un seul NaN ("pas un nombre") pour garder les colonnes
        values = gray[y : y + h, x : x + w][component > 0] if gray is not None else np.array([np.nan])
        rows.append(  # ajoute la ligne du blob
            {  # dictionnaire : nom de colonne -> valeur pour ce blob
                "id": blob_id,  # numero du blob
                "x": x,  # coin haut gauche, x
                "y": y,  # coin haut gauche, y
                "width": w,  # largeur de la boite
                "height": h,  # hauteur de la boite
                "area": area,  # nombre de pixels
                "cx": float(centroids[blob_id][0]),  # centre de gravite, x
                "cy": float(centroids[blob_id][1]),  # centre de gravite, y
                "perimeter": perimeter,  # longueur du contour
                "circularity": float(4 * np.pi * area / perimeter**2) if perimeter > 0 else 1.0,  # 4 pi aire / perimetre^2
                "elongation": elongation,  # rapport des axes
                "angle": angle,  # orientation
                "mean_intensity": float(values.mean()),  # intensite moyenne
                "max_intensity": float(values.max()),  # intensite maximale
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)  # tableau final (colonnes dans l'ordre de COLUMNS, meme s'il n'y a aucun blob)


# garde les blobs dont la taille et la forme sont dans les bornes ; renvoie le tableau filtre
def filter_blobs(
    table: pd.DataFrame,  # tableau produit par blob_table
    min_area: float = 0,  # aire minimale en pixels
    max_area: float = float("inf"),  # aire maximale (inf = infini, pas de limite)
    min_circularity: float = 0.0,  # circularite minimale
    max_circularity: float = float("inf"),  # circularite maximale
    min_elongation: float = 0.0,  # elongation minimale (ex. 4 pour ne garder que les rayures)
    max_elongation: float = float("inf"),  # elongation maximale
) -> pd.DataFrame:  # renvoie un DataFrame
    """Garde les blobs dont la taille et la forme tombent dans les intervalles demandes."""
    keep = (  # colonne de True/False : True si le blob passe les trois filtres
        table["area"].between(min_area, max_area)  # between : valeur dans [min, max], bornes comprises
        & table["circularity"].between(min_circularity, max_circularity)  # & = ET logique, case par case
        & table["elongation"].between(min_elongation, max_elongation)  # troisieme condition
    )
    return table[keep].reset_index(drop=True)  # garde les lignes True et renumerote 0, 1, 2...


def blobs_to_mask(mask: np.ndarray, table: pd.DataFrame, connectivity: int = 8) -> np.ndarray:  # renvoie un masque des seuls blobs gardes
    """Masque uint8 {0, 1} ne contenant que les blobs du tableau (apres filtrage)."""
    labels, _, _ = label_components(mask, connectivity)  # renumerote le masque (memes numeros que dans blob_table)
    return np.isin(labels, table["id"].to_numpy()).astype(np.uint8)  # 1 si le numero du pixel est dans le tableau, sinon 0


# mesure puis filtre en un appel ; renvoie (tableau filtre, masque filtre)
def detect_blobs(mask: np.ndarray, image: np.ndarray | None = None, **filters: float) -> tuple[pd.DataFrame, np.ndarray]:
    """Raccourci : mesurer, filtrer, et rendre (tableau filtre, masque filtre)."""
    table = filter_blobs(blob_table(mask, image), **filters)  # **filters : min_area=..., max_elongation=... passes tels quels
    return table, blobs_to_mask(mask, table)  # tableau et masque coherents


# dessine la boite et le numero de chaque blob ; renvoie une image RGB 8 bits
def draw_blobs(image: np.ndarray, table: pd.DataFrame, color: tuple[int, int, int] = (255, 60, 60)) -> np.ndarray:
    """Image RGB 8 bits avec la boite et le numero de chaque blob (pour regarder le resultat)."""
    from vtb.io.images import to_uint8  # import a la demande : conversion en octets 0-255

    canvas = to_uint8(image)  # image en 8 bits
    canvas = np.stack([canvas] * 3, axis=-1) if canvas.ndim == 2 else canvas.copy()  # gris -> 3 canaux ; couleur -> copie
    for row in table.itertuples():  # pour chaque blob (itertuples : ligne lisible comme row.x, row.y...)
        # rectangle de la boite ; -1 car OpenCV inclut le pixel du coin bas droit
        cv2.rectangle(canvas, (row.x, row.y), (row.x + row.width - 1, row.y + row.height - 1), color, 1)
        # numero du blob juste au-dessus de sa boite (au moins a y = 8 pour rester visible)
        cv2.putText(canvas, str(row.id), (row.x, max(row.y - 2, 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return canvas  # renvoie l'image dessinee
