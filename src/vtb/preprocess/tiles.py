"""Tuiles : traiter une tres grande image (camera lineaire, tole entiere) morceau par morceau.

Principe : decouper en tuiles qui se recouvrent, traiter chaque tuile a la resolution native,
puis recoller (cartes d'anomalie, masques) ou ramener les objets dans l'image complete (boites).
Le recouvrement evite qu'un defaut a cheval sur deux tuiles soit coupe ; les doublons de boites
qu'il cree se retirent ensuite par une NMS (torchvision.ops.batched_nms), laissee a l'appelant.

Regle absolue : si le modele infere sur des tuiles, il s'entraine sur des tuiles decoupees de la
meme facon (meme taille, meme echelle).
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (ex. int | tuple[int, int])

from dataclasses import dataclass  # @dataclass : cree une classe de donnees sans ecrire __init__ a la main

import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)


@dataclass(frozen=True)  # frozen=True : une tuile ne peut plus etre modifiee apres creation
class Tile:  # un morceau rectangulaire de la grande image
    """Une tuile : coin haut gauche (x, y) et taille, en pixels de l'image complete."""

    x: int  # colonne du coin haut gauche
    y: int  # ligne du coin haut gauche
    width: int  # largeur de la tuile en pixels
    height: int  # hauteur de la tuile en pixels

    def slice(self) -> tuple[slice, slice]:  # renvoie (lignes, colonnes) a utiliser comme image[tuile.slice()]
        return slice(self.y, self.y + self.height), slice(self.x, self.x + self.width)  # plage de lignes puis plage de colonnes


def _starts(length: int, size: int, step: int) -> list[int]:  # renvoie les positions de depart des tuiles sur un axe
    """Positions de depart le long d'un axe ; la derniere tuile est collee au bord (pas de tuile partielle)."""
    if length <= size:  # l'image est plus petite que la tuile sur cet axe
        return [0]  # une seule tuile, au debut
    starts = list(range(0, length - size + 1, step))  # departs tous les "step" pixels sans depasser le bord
    if starts[-1] != length - size:  # si la derniere tuile n'atteint pas exactement le bord
        starts.append(length - size)  # ajoute une tuile collee au bord (elle recouvre un peu plus la precedente)
    return starts  # liste des positions de depart


# grille de tuiles couvrant toute l'image ; renvoie la liste des tuiles
def tile_grid(height: int, width: int, tile_size: int | tuple[int, int] = 512, overlap: float | int = 0.25) -> list[Tile]:
    """Grille de tuiles couvrant toute l'image ; overlap en fraction (< 1) ou en pixels (>= 1).

    Une image plus petite que la tuile donne une seule tuile de la taille de l'image.
    """
    # taille de tuile : un entier = carre ; sinon (hauteur, largeur)
    tile_h, tile_w = (tile_size, tile_size) if isinstance(tile_size, int) else (int(tile_size[0]), int(tile_size[1]))
    over_h = int(round(overlap * tile_h)) if overlap < 1 else int(overlap)  # recouvrement vertical en pixels (fraction ou pixels)
    over_w = int(round(overlap * tile_w)) if overlap < 1 else int(overlap)  # recouvrement horizontal en pixels
    if over_h >= tile_h or over_w >= tile_w:  # un recouvrement plus grand que la tuile ferait avancer de 0 pixel ou reculer
        raise ValueError("le recouvrement doit etre plus petit que la tuile")  # erreur claire
    tiles = []  # liste des tuiles
    for y in _starts(height, tile_h, tile_h - over_h):  # pour chaque position de depart verticale (pas = tuile - recouvrement)
        for x in _starts(width, tile_w, tile_w - over_w):  # pour chaque position de depart horizontale
            tiles.append(Tile(x, y, min(tile_w, width), min(tile_h, height)))  # tuile ; jamais plus grande que l'image
    return tiles  # renvoie toutes les tuiles


def extract_tiles(image: np.ndarray, tiles: list[Tile]) -> list[np.ndarray]:  # decoupe l'image ; renvoie un morceau par tuile
    """Les morceaux d'image correspondant aux tuiles (copies : on peut les modifier sans toucher l'image)."""
    return [image[t.slice()].copy() for t in tiles]  # pour chaque tuile, copie de la zone correspondante


# recolle les cartes de chaque tuile en une carte de l'image entiere ; renvoie cette carte
def stitch_maps(maps: list[np.ndarray], tiles: list[Tile], height: int, width: int, mode: str = "max") -> np.ndarray:
    """Recolle des cartes par tuile (score d'anomalie, probabilite de defaut) en une carte de l'image complete.

    mode max : la valeur la plus forte l'emporte dans les recouvrements (prudent pour un defaut) ;
    mode mean : moyenne des tuiles qui se recouvrent (carte plus lisse, sans couture visible).
    Les cartes doivent avoir la taille de leur tuile (redimensionner avant si le modele a reduit).
    """
    if mode not in ("max", "mean"):  # seuls deux modes existent
        raise ValueError("mode : max ou mean")  # erreur claire
    shape = (height, width, *maps[0].shape[2:]) if maps else (height, width)  # taille de la carte finale (+ canaux eventuels)
    if mode == "max":  # mode maximum
        full = np.full(shape, -np.inf, dtype=np.float32)  # carte remplie de moins l'infini (tout max le remplacera)
        for tile_map, tile in zip(maps, tiles):  # pour chaque carte de tuile et sa tuile
            region = full[tile.slice()]  # zone de la carte finale couverte par la tuile (vue, pas copie)
            np.maximum(region, tile_map.astype(np.float32), out=region)  # garde le plus grand des deux, ecrit directement dans la carte
        full[np.isinf(full)] = 0.0  # pixels jamais couverts : 0 au lieu de moins l'infini
        return full  # renvoie la carte complete
    total = np.zeros(shape, dtype=np.float32)  # mode moyenne : somme des valeurs par pixel
    count = np.zeros((height, width) + (1,) * (len(shape) - 2), dtype=np.float32)  # nombre de tuiles qui couvrent chaque pixel
    for tile_map, tile in zip(maps, tiles):  # pour chaque carte de tuile et sa tuile
        total[tile.slice()] += tile_map.astype(np.float32)  # ajoute ses valeurs dans la zone
        count[tile.slice()] += 1.0  # compte une tuile de plus sur cette zone
    return total / np.maximum(count, 1.0)  # moyenne = somme / nombre (1 minimum pour ne pas diviser par 0)


def offset_boxes(boxes: np.ndarray, tile: Tile) -> np.ndarray:  # deplace des boites de la tuile vers l'image complete
    """Boites (N, 4 + k) en coordonnees de la tuile (x1, y1, x2, y2, ...) -> coordonnees de l'image complete."""
    array = np.asarray(boxes, dtype=np.float32)  # convertit en tableau de flottants
    if array.size == 0:  # aucune boite
        return array.reshape(0, 4)  # tableau vide de forme (0, 4)
    out = array.reshape(-1, array.shape[-1]).copy()  # copie en 2D : une ligne par boite
    out[:, [0, 2]] += tile.x  # decale x1 et x2 de la position horizontale de la tuile
    out[:, [1, 3]] += tile.y  # decale y1 et y2 de la position verticale de la tuile
    return out  # boites en coordonnees de l'image complete


def tile_image(  # raccourci : renvoie (tuiles, morceaux d'image)
    image: np.ndarray,  # image complete a decouper
    tile_size: int | tuple[int, int] = 512,  # taille des tuiles en pixels
    overlap: float | int = 0.25,  # recouvrement (fraction ou pixels)
) -> tuple[list[Tile], list[np.ndarray]]:  # renvoie la liste des tuiles et la liste des morceaux
    """Raccourci : la grille et les morceaux d'une image."""
    tiles = tile_grid(image.shape[0], image.shape[1], tile_size, overlap)  # grille calculee a partir de hauteur et largeur
    return tiles, extract_tiles(image, tiles)  # renvoie la grille et les morceaux
