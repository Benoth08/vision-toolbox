"""Afficher des images : une seule, une grille, avant / apres, masques, cartes, boites, lots.

    from vtb import viz
    viz.show(image, "brute")                                   # gris, BGR, 16 bits, float, tenseur...
    viz.grid(images, titles, ncols=8, borders=[True, False])   # vert = juste, rouge = faux
    viz.compare({"avant": raw, "apres": corrected}, diff=True)
    rgb = viz.overlay_mask(image, mask, class_names=["fond", "crack"])
    rgb = viz.draw_detections(image, truth={"boxes": gt, "labels": gl}, pred={"boxes": b, "labels": l, "scores": s})
    viz.show_batch(next(iter(loader)), class_names=names, mean=MEAN, std=STD)

Toutes les entrees passent par `to_display` : HxW, HxWx1, HxWx3 (RGB, ou BGR avec bgr=True),
CHW, tenseur torch (meme sur GPU, avec gradient), uint8 / uint16 / float. Les grilles sont
assemblees en UNE image (une seule imshow : rapide meme pour 100 vignettes).
Les fonctions qui dessinent (overlay_*, draw_*) renvoient un tableau RGB uint8, a passer a
`show` ou `grid` ; celles qui tracent renvoient la Figure ou l'Axes (`save=` pour l'ecrire).
"""

from __future__ import annotations  # permet les annotations de type modernes (int | None, list[...])

import math  # fonctions mathematiques (ceil = arrondi au-dessus)
import random  # tirages au hasard (choisir des images a montrer)
import zlib  # crc32 : somme de controle d'un texte, stable d'une execution a l'autre (couleur des classes)
from collections.abc import Callable, Mapping, Sequence  # types : fonction, dictionnaire, liste ou tuple
from pathlib import Path  # Path : chemin de fichier independant du systeme
from typing import Any  # Any : n'importe quel type

import cv2  # OpenCV : lecture, redimensionnement, dessin sur des images
import matplotlib.pyplot as plt  # pyplot : interface de dessin de matplotlib (figures, axes)
import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)
from matplotlib.figure import Figure  # Figure : la page qui contient les graphiques

# couleurs (juste, faux, verite terrain), figure speciale notebook, couleur de classe, outils de figure, conversion de couleur
from .style import KO_COLOR, OK_COLOR, TRUTH_COLOR, VizFigure, _color, finish, new_axes, new_figure, to_rgb255

# ---------------------------------------------------------------------------------------------
# conversion vers un tableau affichable


def _is_tensor(value: Any) -> bool:  # vrai si value est un tenseur PyTorch (sans importer torch, qui est facultatif)
    return type(value).__module__.split(".")[0] == "torch" and hasattr(value, "detach")  # le type vient du module torch


def as_numpy(value: Any) -> np.ndarray:  # convertit tenseur, image PIL ou liste en tableau numpy ; le renvoie
    """Tenseur torch (CPU / GPU, avec gradient, demi-precision), image PIL ou liste -> numpy."""
    if _is_tensor(value):  # si c'est un tenseur PyTorch
        value = value.detach().cpu()  # detach : sans suivi du gradient ; cpu : copie depuis le GPU si besoin
        if value.is_floating_point():  # nombres a virgule (eventuellement en demi-precision 16 bits)
            value = value.float()  # passage en 32 bits (numpy gere mal certains formats 16 bits)
        return value.numpy()  # tableau numpy
    return np.asarray(value)  # autre chose : numpy fait la conversion


def _hwc(array: np.ndarray) -> np.ndarray:  # ramene un tableau a la forme image (H, W) ou (H, W, 3) ; le renvoie
    """Ramene a HxW ou HxWx3 : CHW -> HWC, lot de 1 -> image, alpha et canal unique retires."""
    while array.ndim > 3 and array.shape[0] == 1:  # tant qu'il y a une dimension de lot de taille 1 en tete
        array = array[0]  # on la retire
    # forme CHW (canaux en premier, comme PyTorch) : peu de canaux devant, beaucoup derriere
    if array.ndim == 3 and array.shape[0] in (1, 2, 3, 4) and array.shape[-1] not in (1, 2, 3, 4):
        array = np.moveaxis(array, 0, -1)  # deplace les canaux a la fin : HWC (forme attendue par matplotlib)
    if array.ndim == 3:  # image avec une dimension canal
        if array.shape[-1] == 1:  # un seul canal
            array = array[..., 0]  # retire la dimension canal : image grise (H, W)
        elif array.shape[-1] == 2:  # deux canaux (gris + transparence)
            array = array[..., 0]  # garde le gris
        elif array.shape[-1] >= 4:  # 4 canaux ou plus (RGB + transparence...)
            array = array[..., :3]  # garde les 3 premiers
    if array.ndim != 2 and not (array.ndim == 3 and array.shape[-1] == 3):  # si la forme n'est toujours pas une image
        raise ValueError(f"image attendue HxW, HxWxC ou CxHxW, recu la forme {array.shape}")  # erreur claire avec la forme recue
    return array  # image (H, W) ou (H, W, 3)


# convertit des valeurs quelconques en entiers 0-255 (uint8) pour l'affichage ; renvoie le tableau converti
def _to_uint8(array: np.ndarray, stretch: bool | tuple[float, float] = False, unit_range: bool = False) -> np.ndarray:
    if array.dtype == np.uint8 and not stretch:  # deja en 0-255 et pas d'etirement demande
        return array  # rien a faire
    if array.dtype == bool:  # masque vrai / faux
        return array.astype(np.uint8) * 255  # vrai -> 255 (blanc), faux -> 0 (noir)
    floating = np.issubdtype(array.dtype, np.floating)  # vrai si nombres a virgule
    values = array.astype(np.float32, copy=True)  # copie en nombres a virgule 32 bits
    finite = np.isfinite(values)  # vrai la ou la valeur est un vrai nombre (ni nan ni infini)
    if not finite.all():  # s'il y a des valeurs non finies
        values[~finite] = float(values[finite].min()) if finite.any() else 0.0  # remplacees par le minimum (ou 0)
    if unit_range and not stretch:  # valeurs connues pour etre en 0-1 (apres annulation de la normalisation)
        return np.clip(np.rint(values * 255.0), 0, 255).astype(np.uint8)  # x 255, arrondi, borne a 0-255
    if isinstance(stretch, tuple):  # etirement entre deux percentiles, ex. (1, 99)
        low, high = np.percentile(values, stretch)  # valeurs basse et haute (ignore les extremes)
    else:  # sinon min et max
        low, high = float(values.min()), float(values.max())  # plus petite et plus grande valeur
        if floating and not stretch:  # nombres a virgule sans etirement force
            if low >= 0.0 and high <= 1.0:  # deja en 0-1
                return np.clip(np.rint(values * 255.0), 0, 255).astype(np.uint8)  # x 255
            if low >= 0.0 and high <= 255.0:  # deja en 0-255
                return np.clip(np.rint(values), 0, 255).astype(np.uint8)  # simple arrondi
    if high <= low:  # image constante (ou plage vide)
        return np.zeros(values.shape, np.uint8)  # image noire
    return np.clip(np.rint((values - low) * (255.0 / (high - low))), 0, 255).astype(np.uint8)  # etirement min-max : low -> 0, high -> 255


def to_display(  # convertit n'importe quelle image en uint8 affichable ; la renvoie
    image: Any,  # l'image (tableau, tenseur, liste...)
    *,  # les arguments suivants doivent etre nommes
    bgr: bool = False,  # vrai si l'image est en BGR (ordre d'OpenCV) : on la remet en RGB
    mean: Sequence[float] | float | None = None,  # moyenne de normalisation a annuler (tenseur d'entree de reseau)
    std: Sequence[float] | float | None = None,  # ecart-type de normalisation a annuler
    stretch: bool | tuple[float, float] = False,  # etirement du contraste (True : min-max ; (1, 99) : percentiles)
) -> np.ndarray:  # renvoie un tableau uint8 (H, W) ou (H, W, 3)
    """Image quelconque -> uint8 HxW (gris) ou HxWx3 (RGB).

    - float dans [0, 1] -> x 255 ; float dans [0, 255] -> tel quel ; autre plage -> min-max ;
    - uint16 et autres entiers -> etirement min-max (une camera 12 bits n'occupe pas 0-65535) ;
    - mean / std : annule la normalisation (tenseur d'entree du reseau) avant conversion ;
    - stretch=True : force le min-max ; stretch=(1, 99) : etirement entre percentiles.
    """
    array = _hwc(as_numpy(image))  # en numpy, a la forme (H, W) ou (H, W, 3)
    unit_range = False  # par defaut, plage de valeurs inconnue
    if mean is not None or std is not None:  # s'il faut annuler une normalisation
        channels = 3 if array.ndim == 3 else 1  # nombre de canaux de l'image
        m = np.asarray(0.0 if mean is None else mean, np.float32).reshape(-1)  # moyenne(s) en tableau (0 si absente)
        s = np.asarray(1.0 if std is None else std, np.float32).reshape(-1)  # ecart(s)-type(s) en tableau (1 si absent)
        m = m[:channels] if m.size >= channels else np.repeat(m, channels)  # une valeur par canal (recopiee si une seule)
        s = s[:channels] if s.size >= channels else np.repeat(s, channels)  # idem pour l'ecart-type
        if channels == 1:  # image grise
            m, s = m[0], s[0]  # un seul nombre suffit
        array = array.astype(np.float32) * s + m  # inverse de (x - m) / s : on retrouve l'image en 0-1
        unit_range = True  # on sait maintenant que les valeurs sont en 0-1
    out = _to_uint8(array, stretch, unit_range)  # conversion en 0-255
    if bgr and out.ndim == 3:  # image couleur en BGR
        out = out[..., ::-1]  # inverse l'ordre des canaux : BGR -> RGB
    return np.ascontiguousarray(out)  # tableau range d'un bloc en memoire (exige par OpenCV pour dessiner)


def _rgb(image: np.ndarray) -> np.ndarray:  # image grise -> 3 canaux identiques ; image couleur inchangee
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if image.ndim == 2 else image  # conversion OpenCV gris -> RGB


def _fit(image: np.ndarray, max_side: int | None) -> np.ndarray:  # redimensionne pour que le plus grand cote vaille max_side
    """Reduit (INTER_AREA) ou agrandit (plus proche voisin, pixels nets) au cote max demande."""
    if not max_side:  # pas de taille demandee
        return image  # inchangee
    h, w = image.shape[:2]  # hauteur et largeur
    scale = max_side / max(h, w)  # facteur d'echelle
    if abs(scale - 1.0) < 1e-3:  # deja a la bonne taille
        return image  # inchangee
    size = (max(1, round(w * scale)), max(1, round(h * scale)))  # nouvelle taille (largeur, hauteur), au moins 1 pixel
    # INTER_AREA pour reduire (moyenne, sans scintillement) ; INTER_NEAREST pour agrandir (pixels nets, pas de flou)
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST)


def read_image(path: str | Path) -> np.ndarray:  # lit une image pour l'afficher ; renvoie un tableau (RGB si couleur)
    """Lecture pour l'affichage : chemins Windows / accents, 8 ou 16 bits, BGR -> RGB."""
    data = np.fromfile(str(path), dtype=np.uint8)  # lit les octets bruts du fichier (accepte les chemins avec accents)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)  # decode les octets en image, sans changer la profondeur (8 ou 16 bits)
    if image is None:  # None = le fichier n'est pas une image lisible
        raise ValueError(f"image illisible : {path}")  # on arrete avec un message clair
    if image.ndim == 3 and image.shape[2] == 4:  # image avec transparence (BGRA)
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)  # vers RGB (transparence retiree)
    if image.ndim == 3:  # image couleur (OpenCV lit en BGR)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # vers RGB (ordre attendu par matplotlib)
    return image  # image grise : telle quelle


def _thickness(shape: tuple[int, ...]) -> int:  # epaisseur de trait adaptee a la taille de l'image ; renvoie un entier
    """Epaisseur proportionnelle au grand cote : reste visible une fois l'image reduite dans une grille."""
    return max(1, round(max(shape[:2]) / 350))  # 1 pixel par tranche de 350 pixels du grand cote, au moins 1


def _font_scale(shape: tuple[int, ...]) -> float:  # taille de texte OpenCV adaptee a l'image ; renvoie un nombre
    return max(0.35, max(shape[:2]) / 1000)  # proportionnelle au grand cote, au moins 0,35


def _describe(image: Any) -> str:  # courte description d'une image (type, forme, min et max) ; renvoie le texte
    array = as_numpy(image)  # en numpy
    if array.size == 0:  # tableau vide
        return f"{array.dtype} vide"  # on le dit
    shape = "x".join(str(s) for s in array.shape)  # forme ecrite "H x W x C"
    return f"{array.dtype} {shape} [{float(np.nanmin(array)):.3g}, {float(np.nanmax(array)):.3g}]"  # ex. "uint8 256x256 [0, 255]"


# ---------------------------------------------------------------------------------------------
# une image


def show(  # affiche une image ; renvoie l'axe
    image: Any,  # l'image (tableau, tenseur, ...)
    title: str | None = None,  # titre facultatif
    *,  # les arguments suivants doivent etre nommes
    bgr: bool = False,  # vrai si l'image est en BGR
    cmap: str = "gray",  # palette de couleurs pour une image a un canal
    colorbar: bool = False,  # ajoute une barre de couleur (echelle des valeurs)
    ax: Any = None,  # axe existant (None = nouvelle figure)
    size: float = 5,  # largeur de la figure en pouces
    mean: Sequence[float] | float | None = None,  # normalisation a annuler (moyenne)
    std: Sequence[float] | float | None = None,  # normalisation a annuler (ecart-type)
    stretch: bool | tuple[float, float] = False,  # etirement du contraste
    max_side: int | None = 2048,  # reduit les tres grandes images (affichage plus rapide)
    save: str | Path | None = None,  # fichier ou enregistrer la figure
) -> Any:  # renvoie l'axe
    """Affiche une image. colorbar=True sur une image a un canal : valeurs REELLES (carte, 16 bits)."""
    raw = _hwc(as_numpy(image))  # image en numpy, forme (H, W) ou (H, W, 3)
    if colorbar and raw.ndim == 2 and mean is None and std is None:  # barre de couleur sur une image a un canal
        data = raw.astype(np.float32)  # on garde les valeurs reelles (pas de conversion 0-255)
        kwargs: dict[str, Any] = {"cmap": cmap}  # matplotlib choisit l'echelle d'apres les valeurs
    else:  # cas general
        data = to_display(raw, bgr=bgr, mean=mean, std=std, stretch=stretch)  # conversion en 0-255
        kwargs = {"cmap": cmap, "vmin": 0, "vmax": 255} if data.ndim == 2 else {}  # echelle fixe 0-255 pour le gris
    if max_side and max(data.shape[:2]) > max_side:  # image trop grande
        data = _fit(data, max_side)  # reduite
    h, w = data.shape[:2]  # hauteur et largeur affichees
    fig, ax = new_axes(ax, figsize=(size, max(1.0, size * h / w) + (0.3 if title else 0)))  # figure aux proportions de l'image
    # affiche l'image ; pixels nets pour les petites images, lissage pour les grandes
    shown = ax.imshow(data, interpolation="nearest" if max(h, w) < 600 else "antialiased", **kwargs)
    if colorbar:  # si demande
        fig.colorbar(shown, ax=ax, fraction=0.046, pad=0.04)  # barre de couleur a droite
    ax.set_axis_off()  # pas d'axes gradues autour d'une image
    if title:  # si un titre est donne
        ax.set_title(title, fontsize=10)  # on l'affiche
    finish(fig, save)  # enregistre si demande
    return ax  # l'axe


# ---------------------------------------------------------------------------------------------
# grille assemblee en une seule image


def _montage(  # colle des vignettes dans une seule grande image (toile) ; renvoie la toile et la position de chaque vignette
    cells: Sequence[np.ndarray],  # les vignettes RGB
    ncols: int,  # nombre de vignettes par ligne
    band: int,  # hauteur en pixels de la bande reservee au titre au-dessus de chaque vignette
    gap: int,  # espace en pixels entre les vignettes
    left: int = 0,  # marge a gauche en pixels (pour des noms de lignes)
    background: int = 255,  # couleur du fond (255 = blanc)
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:  # renvoie (toile, liste de (x, y, largeur, hauteur))
    """Colle des vignettes RGB dans une toile ; renvoie la toile et (x, y, w, h) de chaque image."""
    nrows = math.ceil(len(cells) / ncols)  # nombre de lignes (arrondi au-dessus)
    cell_h = max(c.shape[0] for c in cells)  # hauteur d'une case = la plus haute vignette
    cell_w = max(c.shape[1] for c in cells)  # largeur d'une case = la plus large vignette
    height = nrows * (band + cell_h) + (nrows + 1) * gap  # hauteur totale : lignes + bandes de titre + espaces
    width = left + ncols * cell_w + (ncols + 1) * gap  # largeur totale : marge + colonnes + espaces
    canvas = np.full((height, width, 3), background, np.uint8)  # toile remplie de la couleur de fond
    boxes = []  # position de chaque vignette
    for i, cell in enumerate(cells):  # pour chaque vignette (i = numero)
        row, col = divmod(i, ncols)  # ligne et colonne : quotient et reste de la division par ncols
        y0 = gap + row * (band + cell_h + gap) + band  # haut de la case (sous la bande de titre)
        x0 = left + gap + col * (cell_w + gap)  # gauche de la case
        h, w = cell.shape[:2]  # taille de cette vignette
        y, x = y0 + (cell_h - h) // 2, x0 + (cell_w - w) // 2  # vignette centree dans sa case
        canvas[y : y + h, x : x + w] = cell  # copie les pixels de la vignette dans la toile
        boxes.append((x, y, w, h))  # retient sa position
    return canvas, boxes  # la toile et les positions


# couleur du cadre d'une vignette : booleen -> vert (juste) / rouge (faux), couleur -> RGB 0-255, None -> pas de cadre
def _border_color(value: Any) -> tuple[int, int, int] | None:
    if value is None:  # pas de cadre demande
        return None  # rien
    if isinstance(value, bool | np.bool_):  # un booleen (Python ou numpy)
        return OK_COLOR if value else KO_COLOR  # vert si vrai, rouge si faux
    return to_rgb255(value)  # sinon une couleur, convertie en 0-255


# cree une figure qui affiche la toile en entier, a la bonne proportion ; renvoie (figure, axe)
def _layout_figure(canvas: np.ndarray, width_in: float, suptitle: str | None) -> tuple[Figure, Any]:
    height, width = canvas.shape[:2]  # taille de la toile en pixels
    image_h_in = width_in * height / width  # hauteur en pouces qui respecte les proportions
    extra = 0.4 if suptitle else 0.0  # place en plus pour le titre general
    fig = new_figure((width_in, image_h_in + extra))  # la figure
    ax = fig.add_axes((0, 0, 1, image_h_in / (image_h_in + extra)))  # un axe qui occupe toute la largeur, sous le titre
    ax.imshow(canvas, interpolation="antialiased")  # une seule image affichee : rapide meme avec beaucoup de vignettes
    ax.set_xlim(-0.5, width - 0.5)  # cadrage exact sur la toile en largeur
    ax.set_ylim(height - 0.5, -0.5)  # et en hauteur (axe vertical vers le bas, comme une image)
    ax.set_axis_off()  # pas d'axes gradues
    if suptitle:  # si un titre general est donne
        fig.suptitle(suptitle, fontsize=11, y=1 - 0.08 / (image_h_in + extra))  # titre en haut de la figure
    return fig, ax  # la figure et son axe


def grid(  # planche de vignettes avec titres et cadres ; renvoie la figure
    images: Sequence[Any] | Any,  # images : liste, lot (N, C, H, W) ou (N, H, W), ou chemins de fichiers
    titles: Sequence[str] | None = None,  # titre de chaque vignette
    *,  # les arguments suivants doivent etre nommes
    ncols: int = 6,  # vignettes par ligne
    bgr: bool = False,  # vrai si les images sont en BGR
    size: float = 2.0,  # largeur d'une vignette en pouces
    max_side: int = 256,  # taille maximale d'une vignette en pixels
    borders: Sequence[Any] | None = None,  # cadre de chaque vignette (couleur, ou booleen juste / faux)
    suptitle: str | None = None,  # titre general
    fontsize: float = 8,  # taille du texte des titres
    mean: Sequence[float] | float | None = None,  # normalisation a annuler (moyenne)
    std: Sequence[float] | float | None = None,  # normalisation a annuler (ecart-type)
    save: str | Path | None = None,  # fichier ou enregistrer la figure
) -> Figure:  # renvoie la figure
    """Planche d'images (liste, lot NCHW / NHW, chemins). borders : couleurs ou booleens ok / ko."""
    if _is_tensor(images) or isinstance(images, np.ndarray):  # un lot sous forme de tenseur ou de tableau
        images = list(as_numpy(images))  # decoupe en liste d'images (une par element du lot)
    images = list(images)  # en liste dans tous les cas
    if not images:  # liste vide
        raise ValueError("grid : aucune image")  # erreur claire
    cells = []  # vignettes pretes
    for image in images:  # pour chaque image
        if isinstance(image, str | Path):  # si c'est un chemin
            image = read_image(image)  # on lit le fichier
        cells.append(_rgb(_fit(to_display(image, bgr=bgr, mean=mean, std=std), max_side)))  # affichable, reduite, en RGB
    ncols = max(1, min(ncols, len(cells)))  # pas plus de colonnes que d'images, au moins 1
    titles = [str(t) for t in titles] if titles is not None else None  # titres en texte
    lines = max((t.count("\n") + 1 for t in titles), default=0) if titles else 0  # nombre maximal de lignes d'un titre
    cell_w = max(c.shape[1] for c in cells)  # largeur d'une case
    gap = max(4, round(cell_w / 40))  # espace entre vignettes : proportionnel, au moins 4 pixels
    width_px = ncols * cell_w + (ncols + 1) * gap  # largeur de la toile en pixels
    width_in = ncols * size  # largeur de la figure en pouces
    px_per_in = width_px / width_in  # pixels de toile par pouce de figure
    # hauteur de bande pour les titres (72 points par pouce)
    band = math.ceil(fontsize * 1.3 * lines / 72 * px_per_in) + (2 if lines else 0)
    canvas, boxes = _montage(cells, ncols, band, gap)  # assemble la toile
    if borders is not None:  # si des cadres sont demandes
        thick = max(2, round(cell_w / 64))  # epaisseur du cadre, au moins 2 pixels
        for (x, y, w, h), value in zip(boxes, borders):  # pour chaque vignette et son reglage de cadre
            color = _border_color(value)  # couleur du cadre (ou None)
            if color is not None:  # si un cadre est voulu
                cv2.rectangle(canvas, (x, y), (x + w - 1, y + h - 1), color, thick)  # dessine le rectangle autour de la vignette
    fig, ax = _layout_figure(canvas, width_in, suptitle)  # figure qui affiche la toile
    if titles:  # s'il y a des titres
        cell_h = max(c.shape[0] for c in cells)  # hauteur d'une case
        for (x, y, w, h), title in zip(boxes, titles):  # pour chaque vignette et son titre
            top = y - (cell_h - h) // 2  # haut de la case (la vignette peut etre centree plus bas)
            # texte centre juste au-dessus de la case ; clip_on=False : peut deborder du cadre de l'axe
            ax.text(x + w / 2, top - 2, title, ha="center", va="bottom", fontsize=fontsize, clip_on=False)
    finish(fig, save)  # enregistre si demande
    return fig  # la figure


def compare(  # affiche des images cote a cote (avant / apres) ; renvoie la figure
    images: Mapping[str, Any] | Sequence[Any],  # dictionnaire titre -> image, ou liste d'images
    *,  # les arguments suivants doivent etre nommes
    bgr: bool = False,  # vrai si les images sont en BGR
    size: float = 4,  # largeur d'une image en pouces
    diff: bool = False,  # ajoute l'image de la difference absolue (pour deux images)
    cmap: str = "gray",  # palette pour les images a un canal
    save: str | Path | None = None,  # fichier ou enregistrer la figure
) -> Figure:  # renvoie la figure
    """Cote a cote (zoom lie si memes tailles) ; diff=True ajoute |a - b| pour deux images."""
    if isinstance(images, Mapping):  # dictionnaire titre -> image
        titles, values = [str(k) for k in images], list(images.values())  # titres et images
    else:  # liste d'images
        values = list(images)  # les images
        titles = [str(i) for i in range(len(values))]  # titres : numeros
    shown = [to_display(v, bgr=bgr) for v in values]  # images converties en 0-255
    if diff and len(shown) == 2 and shown[0].shape == shown[1].shape:  # difference demandee, deux images de meme forme
        # |a - b| calcule en int16 (sinon uint8 "tourne" : 3 - 5 donnerait 254)
        delta = np.abs(shown[0].astype(np.int16) - shown[1].astype(np.int16)).astype(np.uint8)
        shown.append(delta if delta.ndim == 2 else delta.max(axis=2))  # en couleur : plus grand ecart des 3 canaux
        titles.append(f"|difference| (max {int(delta.max())})")  # titre avec l'ecart maximal
    same = len({s.shape[:2] for s in shown}) == 1  # vrai si toutes les images ont la meme taille
    h, w = shown[0].shape[:2]  # taille de la premiere image
    fig, axes = plt.subplots(  # une ligne de graphiques
        1,  # une ligne
        len(shown),  # une colonne par image
        figsize=(size * len(shown), size * h / w + 0.4),  # taille de la figure
        squeeze=False,  # garde un tableau 2D d'axes meme avec une seule image
        sharex=same,  # zoom lie en largeur si memes tailles (zoomer une image zoome les autres)
        sharey=same,  # et en hauteur
        FigureClass=VizFigure,  # figure qui s'affiche une seule fois dans le notebook
    )
    for ax, image, title in zip(axes[0], shown, titles):  # pour chaque axe, image et titre
        # affiche en 0-255 ; pixels nets pour les petites images, lissage pour les grandes
        ax.imshow(image, cmap=cmap, vmin=0, vmax=255, interpolation="nearest" if max(image.shape[:2]) < 600 else "antialiased")
        ax.set_title(title, fontsize=10)  # titre
        ax.set_axis_off()  # pas d'axes gradues
    fig.tight_layout()  # ajuste les marges
    finish(fig, save)  # enregistre si demande
    return fig  # la figure


def histogram(  # histogramme des valeurs des pixels ; renvoie l'axe
    image: Any,  # l'image
    *,  # les arguments suivants doivent etre nommes
    bgr: bool = False,  # vrai si l'image est en BGR
    bins: int = 256,  # nombre d'intervalles
    log: bool = False,  # axe vertical en echelle logarithmique
    ax: Any = None,  # axe existant (None = nouvelle figure)
    save: str | Path | None = None,  # fichier ou enregistrer la figure
) -> Any:  # renvoie l'axe
    """Histogramme des valeurs REELLES (16 bits, float) ; une courbe par canal en couleur."""
    array = _hwc(as_numpy(image))  # image en numpy (valeurs reelles, pas de conversion 0-255)
    if bgr and array.ndim == 3:  # image couleur en BGR
        array = array[..., ::-1]  # remise en RGB
    fig, ax = new_axes(ax, figsize=(6, 3.2))  # figure et axe
    if array.dtype == np.uint8:  # image 8 bits
        value_range = (0, 256)  # plage fixe 0-255
    else:  # 16 bits ou nombres a virgule
        finite = array[np.isfinite(array)] if np.issubdtype(array.dtype, np.floating) else array  # ignore nan et infinis
        value_range = (float(finite.min()), float(finite.max()) + 1e-9)  # plage reelle (un epsilon pour inclure le max)
    # une courbe : grise pour une image a un canal, une par canal (rouge, vert, bleu) pour une image couleur
    channels = [(array, "gray", "valeurs")] if array.ndim == 2 else [(array[..., i], c, c) for i, c in enumerate(("red", "green", "blue"))]
    for values, color, label in channels:  # pour chaque canal (valeurs, couleur de trait, nom)
        counts, edges = np.histogram(values.ravel(), bins=bins, range=value_range)  # nombre de pixels par intervalle
        ax.plot(edges[:-1], counts, color=color, label=label, lw=1)  # courbe de l'histogramme
        ax.fill_between(edges[:-1], counts, color=color, alpha=0.15, step=None)  # remplissage leger sous la courbe
    if log:  # echelle logarithmique demandee
        ax.set_yscale("log")  # utile quand quelques valeurs dominent
    ax.set_xlabel("valeur")  # titre de l'axe horizontal
    ax.set_ylabel("pixels")  # titre de l'axe vertical
    ax.set_title(f"{array.dtype} min {array.min():.4g} max {array.max():.4g} moyenne {array.mean():.4g}", fontsize=9)  # resume chiffre
    ax.grid(alpha=0.3)  # quadrillage leger
    if array.ndim == 3:  # image couleur
        ax.legend(fontsize=8)  # legende des 3 canaux
    finish(fig, save)  # enregistre si demande
    return ax  # l'axe


def profile(  # profil des niveaux le long d'une ligne, d'une colonne, ou moyen ; renvoie la figure
    image: Any,  # l'image
    *,  # les arguments suivants doivent etre nommes
    row: int | None = None,  # numero de ligne a tracer
    column: int | None = None,  # numero de colonne a tracer
    bgr: bool = False,  # vrai si l'image est en BGR
    save: str | Path | None = None,  # fichier ou enregistrer la figure
) -> Figure:  # renvoie la figure
    """Profil de niveaux : une ligne, une colonne, ou (par defaut) moyenne des colonnes et des lignes.

    Le profil moyen des colonnes revele un eclairage non uniforme (pente, vignettage, bandes de
    camera lineaire) : c'est le controle avant / apres une correction d'illumination.
    """
    array = _hwc(as_numpy(image)).astype(np.float32)  # image en nombres a virgule (valeurs reelles)
    if bgr and array.ndim == 3:  # image couleur en BGR
        array = array[..., ::-1]  # remise en RGB
    gray = array if array.ndim == 2 else array.mean(axis=2)  # niveau de gris : moyenne des canaux
    h, w = gray.shape  # hauteur et largeur
    fig = new_figure((11, 3.4))  # figure large : image a gauche, courbe a droite
    ax_img = fig.add_subplot(1, 2, 1)  # graphique de gauche : l'image
    ax_img.imshow(to_display(_hwc(as_numpy(image)), bgr=bgr), cmap="gray", vmin=0, vmax=255)  # affiche l'image
    ax_img.set_axis_off()  # sans axes gradues
    ax_plot = fig.add_subplot(1, 2, 2)  # graphique de droite : le profil
    if row is not None:  # profil d'une ligne
        ax_img.axhline(row, color="red", lw=1)  # trait rouge horizontal sur l'image a cette ligne
        ax_plot.plot(gray[row], color="red", label=f"ligne {row}")  # niveaux le long de la ligne
        ax_plot.set_xlabel("x (colonne)")  # axe horizontal : position en x
    elif column is not None:  # profil d'une colonne
        ax_img.axvline(column, color="red", lw=1)  # trait rouge vertical sur l'image
        ax_plot.plot(gray[:, column], color="red", label=f"colonne {column}")  # niveaux le long de la colonne
        ax_plot.set_xlabel("y (ligne)")  # axe horizontal : position en y
    else:  # par defaut : profils moyens
        ax_plot.plot(np.arange(w), gray.mean(axis=0), label="moyenne de chaque colonne (en x)")  # variation gauche -> droite
        ax_plot.plot(np.arange(h), gray.mean(axis=1), label="moyenne de chaque ligne (en y)")  # variation haut -> bas
        ax_plot.set_xlabel("position (pixels)")  # titre de l'axe horizontal
    ax_plot.set_ylabel("niveau")  # titre de l'axe vertical
    ax_plot.grid(alpha=0.3)  # quadrillage leger
    ax_plot.legend(fontsize=8)  # legende
    fig.tight_layout()  # ajuste les marges
    finish(fig, save)  # enregistre si demande
    return fig  # la figure


# ---------------------------------------------------------------------------------------------
# superpositions et dessins (renvoient un tableau RGB uint8)


# met un masque sous forme (H, W) d'indices de classe entiers, a la taille shape ; renvoie le tableau
def _mask_array(mask: Any, shape: tuple[int, int]) -> np.ndarray:
    array = as_numpy(mask)  # en numpy
    while array.ndim > 2 and array.shape[0] == 1:  # dimensions de taille 1 en tete (lot, canal)
        array = array[0]  # retirees
    if array.ndim == 3 and array.shape[-1] == 1:  # canal unique a la fin
        array = array[..., 0]  # retire
    if array.ndim != 2:  # toujours pas un masque 2D
        raise ValueError(f"masque attendu HxW (indices de classe), recu {array.shape}")  # erreur claire
    if array.dtype == bool:  # masque vrai / faux
        array = array.astype(np.uint8)  # vrai -> 1, faux -> 0
    elif np.issubdtype(array.dtype, np.floating):  # nombres a virgule
        if array.max() <= 1.0 and not np.all(np.mod(array, 1) == 0):  # probabilites (0-1, pas que des entiers)
            array = (array >= 0.5).astype(np.uint8)  # seuil a 0,5 : 1 au-dessus, 0 en dessous
        else:  # indices de classe stockes en flottants
            array = np.rint(array).astype(np.int32)  # arrondis en entiers
    if array.shape != shape:  # taille differente de l'image
        # redimensionne au plus proche voisin (pas de classe intermediaire) ; shape[::-1] : OpenCV veut (largeur, hauteur)
        array = cv2.resize(array.astype(np.int32 if array.dtype != np.uint8 else np.uint8), shape[::-1], interpolation=cv2.INTER_NEAREST)
    return array.astype(np.int32)  # indices de classe en entiers 32 bits


# dessine une legende (carre de couleur + nom) en haut a gauche de l'image ; modifie canvas sur place
def _legend(canvas: np.ndarray, entries: Sequence[tuple[str, tuple[int, int, int]]]) -> None:
    scale = _font_scale(canvas.shape)  # taille du texte adaptee a l'image
    thick = max(1, round(scale * 1.5))  # epaisseur du texte
    (tw, th), _ = cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, scale, thick)  # hauteur d'une ligne de texte (th)
    y = 4  # position verticale de la premiere entree
    for name, color in entries:  # pour chaque classe (nom, couleur)
        (tw, _), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)  # largeur du nom (tw)
        cv2.rectangle(canvas, (2, y), (8 + th + tw + 4, y + th + 6), (0, 0, 0), -1)  # fond noir (-1 = rectangle plein)
        cv2.rectangle(canvas, (4, y + 3), (4 + th, y + th + 3), color, -1)  # carre plein de la couleur de la classe
        # nom en blanc a droite du carre ; LINE_AA : texte lisse
        cv2.putText(canvas, name, (8 + th, y + th + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)
        y += th + 8  # descend pour l'entree suivante


def overlay_mask(  # colore les classes d'un masque par-dessus l'image ; renvoie une image RGB uint8
    image: Any,  # l'image
    mask: Any,  # le masque : indice de classe par pixel (0 = fond)
    *,  # les arguments suivants doivent etre nommes
    class_names: Sequence[str] | None = None,  # noms des classes (fond en premier) : ajoute une legende
    alpha: float = 0.45,  # opacite de la couleur (0 = invisible, 1 = opaque)
    contours: bool = True,  # dessine le contour de chaque zone
    bgr: bool = False,  # vrai si l'image est en BGR
    colors: Sequence[Any] | None = None,  # couleurs imposees pour les classes 1, 2, ...
) -> np.ndarray:  # renvoie l'image superposee
    """Colore chaque classe du masque (0 = fond) ; contours nets ; legende si class_names."""
    base = _rgb(to_display(image, bgr=bgr)).copy()  # image en RGB 0-255 (copie : on va dessiner dessus)
    labels = _mask_array(mask, base.shape[:2])  # masque d'indices a la taille de l'image
    out = base.astype(np.float32)  # en nombres a virgule pour le melange
    present = [int(c) for c in np.unique(labels) if c > 0]  # classes presentes dans le masque, sauf le fond
    thick = _thickness(base.shape)  # epaisseur des contours
    # couleur de chaque classe presente : imposee si fournie, sinon couleur de palette
    palette = {c: (to_rgb255(colors[c - 1]) if colors is not None and c - 1 < len(colors) else _color(c - 1)) for c in present}
    for c in present:  # pour chaque classe presente
        region = labels == c  # pixels de cette classe
        out[region] = out[region] * (1 - alpha) + np.asarray(palette[c], np.float32) * alpha  # melange image / couleur
    out = np.clip(out, 0, 255).astype(np.uint8)  # retour en 0-255 entiers
    if contours:  # si les contours sont demandes
        for c in present:  # pour chaque classe presente
            # contours exterieurs des zones de cette classe (CHAIN_APPROX_SIMPLE : garde seulement les coins utiles)
            found, _ = cv2.findContours((labels == c).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, found, -1, palette[c], thick, cv2.LINE_AA)  # dessine tous les contours (-1) de la couleur de la classe
    if class_names is not None and present:  # noms fournis et au moins une classe presente
        # legende : nom de chaque classe presente (ou son numero si le nom manque) et sa couleur
        _legend(out, [(str(class_names[c]) if c < len(class_names) else str(c), palette[c]) for c in present])
    return out  # l'image superposee


def overlay_heatmap(  # fond une carte de chaleur coloree sur l'image ; renvoie une image RGB uint8
    image: Any,  # l'image
    heatmap: Any,  # la carte (anomalie, Grad-CAM...), de taille quelconque
    *,  # les arguments suivants doivent etre nommes
    alpha: float = 0.5,  # opacite de la carte
    cmap: str = "jet",  # palette de couleurs (jet : bleu = faible, rouge = fort)
    vmin: float | None = None,  # valeur affichee en bleu (None = minimum de la carte)
    vmax: float | None = None,  # valeur affichee en rouge (None = maximum de la carte)
    bgr: bool = False,  # vrai si l'image est en BGR
) -> np.ndarray:  # renvoie l'image superposee
    """Carte de chaleur (anomalie, Grad-CAM) remise a la taille de l'image et fondue dessus.

    Fixer vmin / vmax (par ex. vmax = seuil x 2) pour comparer plusieurs images a la meme echelle.
    """
    base = _rgb(to_display(image, bgr=bgr)).astype(np.float32)  # image RGB en nombres a virgule
    values = _hwc(as_numpy(heatmap)).astype(np.float32)  # carte en numpy
    if values.ndim == 3:  # carte a plusieurs canaux
        values = values.mean(axis=2)  # moyenne des canaux
    if values.shape != base.shape[:2]:  # taille differente de l'image
        values = cv2.resize(values, base.shape[1::-1], interpolation=cv2.INTER_LINEAR)  # agrandie en douceur (bilineaire)
    low = float(np.nanmin(values)) if vmin is None else vmin  # bas de l'echelle
    high = float(np.nanmax(values)) if vmax is None else vmax  # haut de l'echelle
    scaled = np.clip((values - low) / (high - low), 0, 1) if high > low else np.zeros_like(values)  # ramenee en 0-1
    colored = plt.get_cmap(cmap)(np.nan_to_num(scaled))[..., :3] * 255.0  # 0-1 -> couleurs RGB (nan -> 0), en 0-255
    return np.clip(base * (1 - alpha) + colored * alpha, 0, 255).astype(np.uint8)  # melange image / carte, en uint8


# texte d'etiquette d'une boite : nom de classe (si connu) et score ; renvoie le texte
def _label_text(label: Any, score: float | None, class_names: Sequence[str] | None) -> str:
    if label is None:  # pas d'etiquette
        name = ""  # texte vide
    elif class_names is not None and isinstance(label, int | np.integer) and 0 <= int(label) < len(class_names):  # indice valide
        name = str(class_names[int(label)])  # nom de la classe
    else:  # deja un nom, ou indice sans nom connu
        name = str(label)  # tel quel
    if score is not None:  # si un score est donne
        name = f"{name} {score:.2f}".strip()  # ajoute le score a 2 decimales (strip retire l'espace si le nom est vide)
    return name  # le texte


# couleur d'une boite selon sa classe (ou couleur imposee) ; renvoie (r, g, b) 0-255
def _label_color(label: Any, class_names: Sequence[str] | None, color: Any) -> tuple[int, int, int]:
    if color is not None:  # couleur imposee
        return to_rgb255(color)  # convertie en 0-255
    if label is None:  # pas d'etiquette
        return _color(0)  # premiere couleur de la palette
    if isinstance(label, int | np.integer):  # indice de classe
        return _color(int(label))  # couleur de cette classe
    if class_names is not None and str(label) in [str(c) for c in class_names]:  # nom connu
        return _color([str(c) for c in class_names].index(str(label)))  # couleur de son indice
    # nom inconnu : couleur tiree d'une somme de controle du nom (crc32), identique a chaque lancement
    # (hash() de Python change d'une execution a l'autre) ; une des 20 couleurs de la palette
    return _color(zlib.crc32(str(label).encode()) % 20)


# ecrit un texte sur un bandeau de couleur au-dessus du point (x, y) ; modifie canvas sur place
def _put_label(canvas: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    if not text:  # pas de texte
        return  # rien a dessiner
    scale = _font_scale(canvas.shape)  # taille du texte adaptee a l'image
    thick = max(1, round(scale * 1.5))  # epaisseur du texte
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)  # largeur, hauteur et descente du texte
    x = int(np.clip(x, 0, max(0, canvas.shape[1] - tw - 2)))  # decale vers la gauche si le texte sortirait a droite
    top = y - th - base - 2  # haut du bandeau : juste au-dessus du point
    if top < 0:  # le bandeau sortirait par le haut de l'image
        top = y + 1  # on le met en dessous du point
    cv2.rectangle(canvas, (x, top), (x + tw + 2, top + th + base + 2), color, -1)  # bandeau plein de la couleur de la boite
    luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]  # clarte percue de la couleur du bandeau
    ink = (0, 0, 0) if luminance > 150 else (255, 255, 255)  # texte noir sur fond clair, blanc sur fond fonce
    cv2.putText(canvas, text, (x + 1, top + th + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, ink, thick, cv2.LINE_AA)  # ecrit le texte


def draw_boxes(  # dessine des boites (et polygones) avec leur etiquette ; renvoie une image RGB uint8
    image: Any,  # l'image
    boxes: Any,  # boites (N, 4) en xyxy (coins haut-gauche et bas-droite) ou (N, 5) orientees (cx, cy, w, h, angle)
    *,  # les arguments suivants doivent etre nommes
    labels: Sequence[Any] | None = None,  # classe de chaque boite (indice ou nom)
    scores: Sequence[float] | None = None,  # score de confiance de chaque boite
    class_names: Sequence[str] | None = None,  # noms des classes (pour traduire les indices)
    color: Any = None,  # couleur imposee pour toutes les boites (None = couleur de classe)
    rotated: bool = False,  # vrai si les boites sont orientees
    polygons: Sequence[Any] | None = None,  # polygones a dessiner (contours de segmentation d'instances)
    bgr: bool = False,  # vrai si l'image est en BGR
    thickness: int | None = None,  # epaisseur du trait (None = automatique)
    min_side: int | None = None,  # agrandit d'abord les images dont le petit cote est plus petit
    show_text: bool = True,  # ecrit nom et score au-dessus des boites
) -> np.ndarray:  # renvoie l'image dessinee
    """Dessine des boites xyxy (ou (cx, cy, w, h, angle_deg) si rotated) et des polygones.

    labels : indices (couleur et nom via class_names) ou noms ; scores : affiches a 2 decimales.
    min_side : agrandit d'abord les petites images (NEU-DET 200 px) pour un texte lisible.
    """
    canvas = _rgb(to_display(image, bgr=bgr)).copy()  # image RGB 0-255 (copie : on dessine dessus)
    factor = 1.0  # facteur d'agrandissement (1 = aucun)
    if min_side and min(canvas.shape[:2]) < min_side:  # image trop petite pour un texte lisible
        factor = min_side / min(canvas.shape[:2])  # facteur pour atteindre min_side
        canvas = cv2.resize(canvas, None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)  # agrandie (pixels nets)
    thick = thickness or _thickness(canvas.shape)  # epaisseur donnee, sinon automatique
    array = _boxes_array(boxes)  # boites en tableau (N, 4) ou (N, 5)
    rotated = rotated or array.shape[1] == 5  # 5 valeurs par boite = boite orientee
    count = len(array) if len(array) else (len(polygons) if polygons is not None else 0)  # nombre d'objets a dessiner
    # etiquettes en liste Python (tenseur ou tableau convertis), ou None
    labels_list = None if labels is None else (list(labels) if isinstance(labels, list | tuple) else as_numpy(labels).reshape(-1).tolist())
    scores_list = None if scores is None else np.asarray(as_numpy(scores), np.float64).reshape(-1).tolist()  # scores en liste, ou None
    for i in range(count):  # pour chaque objet
        label = labels_list[i] if labels_list is not None and i < len(labels_list) else None  # son etiquette (si connue)
        score = scores_list[i] if scores_list is not None and i < len(scores_list) else None  # son score (si connu)
        col = _label_color(label, class_names, color)  # couleur de sa classe
        text = _label_text(label, score, class_names) if show_text else ""  # texte "nom score"
        if i < len(array):  # s'il a une boite
            if rotated:  # boite orientee
                cx, cy, w, h, angle = array[i]  # centre, largeur, hauteur, angle en degres
                corners = cv2.boxPoints(((cx * factor, cy * factor), (w * factor, h * factor), angle))  # 4 coins de la boite tournee
                pts = np.rint(corners).astype(np.int32)  # coins arrondis en pixels entiers
                cv2.polylines(canvas, [pts], True, col, thick, cv2.LINE_AA)  # trace le contour ferme (True)
                anchor = pts[np.argmin(pts[:, 1])]  # coin le plus haut : point d'accroche du texte
                _put_label(canvas, text, int(anchor[0]), int(anchor[1]), col)  # ecrit l'etiquette
            else:  # boite droite xyxy
                x1, y1, x2, y2 = np.rint(array[i] * factor).astype(int)  # coins en pixels entiers (apres agrandissement)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), col, thick, cv2.LINE_AA)  # trace le rectangle
                _put_label(canvas, text, x1, y1, col)  # ecrit l'etiquette au coin haut-gauche
        if polygons is not None and i < len(polygons) and polygons[i] is not None and len(polygons[i]):  # s'il a un polygone non vide
            pts = np.rint(np.asarray(polygons[i], np.float64).reshape(-1, 2) * factor).astype(np.int32)  # points (x, y) en pixels
            cv2.polylines(canvas, [pts], True, col, thick, cv2.LINE_AA)  # trace le polygone ferme
            if i >= len(array):  # polygone sans boite : l'etiquette n'a pas encore ete ecrite
                anchor = pts[np.argmin(pts[:, 1])]  # point le plus haut
                _put_label(canvas, text, int(anchor[0]), int(anchor[1]), col)  # ecrit l'etiquette
    return canvas  # l'image dessinee


def _as_target(target: Any) -> dict[str, Any]:  # cible de detection -> dictionnaire {"boxes": ..., "labels": ...}
    if target is None:  # pas de cible
        return {}  # dictionnaire vide
    if isinstance(target, Mapping):  # deja un dictionnaire
        return dict(target)  # copie
    return {"boxes": target}  # juste des boites : mises sous la cle "boxes"


def _boxes_array(boxes: Any) -> np.ndarray:  # boites -> tableau (N, 4) ou (N, 5) de nombres a virgule
    """Boites -> tableau float64 (N, 4) ou (N, 5) ; vide -> (0, 4)."""
    if boxes is None:  # pas de boites
        return np.zeros((0, 4))  # tableau vide a 4 colonnes
    array = np.asarray(as_numpy(boxes), np.float64)  # en numpy, nombres a virgule
    if array.size == 0:  # tableau vide
        return np.zeros((0, 4))  # forme vide standard
    return array.reshape(-1, array.shape[-1])  # une ligne par boite (aplatit les dimensions en trop)


def draw_detections(  # dessine verite terrain et predictions sur une image ; renvoie une image RGB uint8
    image: Any,  # l'image
    truth: Mapping[str, Any] | Any = None,  # verite : dict {"boxes", "labels"} ou tableau de boites
    pred: Mapping[str, Any] | Any = None,  # predictions : dict {"boxes", "labels", "scores"} ou tableau
    *,  # les arguments suivants doivent etre nommes
    class_names: Sequence[str] | None = None,  # noms des classes
    conf: float = 0.25,  # score minimal pour dessiner une prediction
    bgr: bool = False,  # vrai si l'image est en BGR
    min_side: int | None = None,  # agrandit les petites images avant de dessiner
) -> np.ndarray:  # renvoie l'image dessinee
    """Verite (trait vert vif, sans texte) et predictions (couleur de classe, nom + score >= conf).

    truth / pred : dict {"boxes": Nx4 xyxy ou Nx5 orientees, "labels": ..., "scores": ...}
    ou directement le tableau de boites. min_side : agrandit les petites images avant de dessiner.
    """
    truth_d, pred_d = _as_target(truth), _as_target(pred)  # verite et predictions en dictionnaires
    canvas = _rgb(to_display(image, bgr=bgr))  # image RGB 0-255
    factor = 1.0  # facteur d'agrandissement
    if min_side and min(canvas.shape[:2]) < min_side:  # image trop petite
        factor = min_side / min(canvas.shape[:2])  # facteur pour atteindre min_side
        canvas = cv2.resize(canvas, None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)  # agrandie

    def scaled(array: np.ndarray) -> np.ndarray:  # met les boites a l'echelle de l'image agrandie ; renvoie une copie
        out = array.copy()  # copie (on ne modifie pas l'original)
        out[:, :4] *= factor  # les 4 premieres valeurs (positions et tailles) multipliees ; l'angle ne change pas
        return out  # boites a l'echelle

    gt = _boxes_array(truth_d.get("boxes"))  # boites verite
    if len(gt):  # s'il y en a
        thick = _thickness(canvas.shape) + 1  # trait un peu plus epais que les predictions
        # verite en vert vif, sans texte (le texte des predictions reste lisible)
        canvas = draw_boxes(canvas, scaled(gt), color=TRUTH_COLOR, rotated=gt.shape[1] == 5, thickness=thick, show_text=False)
    boxes = _boxes_array(pred_d.get("boxes"))  # boites predites
    if len(boxes):  # s'il y en a
        scores = pred_d.get("scores")  # leurs scores (peut manquer)
        scores_np = None if scores is None else np.asarray(as_numpy(scores), np.float64).reshape(-1)  # scores en numpy
        keep = np.ones(len(boxes), bool) if scores_np is None else scores_np >= conf  # garde celles dont le score atteint conf
        labels = pred_d.get("labels")  # leurs classes
        if labels is not None:  # si les classes sont connues
            labels_list = list(labels) if isinstance(labels, list | tuple) else as_numpy(labels).reshape(-1).tolist()  # en liste
            labels = [v for v, k in zip(labels_list, keep) if k]  # garde les classes des boites retenues
        canvas = draw_boxes(  # dessine les predictions retenues
            canvas,  # sur l'image (deja dessinee avec la verite)
            scaled(boxes[keep]),  # boites retenues, a l'echelle
            labels=labels,  # leurs classes
            scores=None if scores_np is None else scores_np[keep],  # leurs scores
            class_names=class_names,  # noms des classes
            rotated=boxes.shape[1] == 5,  # orientees si 5 valeurs
        )
    return canvas  # l'image dessinee


# ---------------------------------------------------------------------------------------------
# jeux, lots, augmentations, pipelines


def class_montage(  # planche : une ligne par classe avec quelques images au hasard ; renvoie la figure
    paths: Sequence[Any],  # chemins des images
    labels: Sequence[Any],  # classe de chaque image
    *,  # les arguments suivants doivent etre nommes
    per_class: int = 6,  # images par classe
    size: float = 1.6,  # largeur d'une vignette en pouces
    read: Callable[[Any], np.ndarray] | None = None,  # fonction de lecture (None = read_image)
    seed: int = 0,  # graine du tirage (memes images a chaque appel)
    max_side: int = 160,  # taille maximale d'une vignette en pixels
    save: str | Path | None = None,  # fichier ou enregistrer la figure
) -> Figure:  # renvoie la figure
    """Une ligne par classe, `per_class` images tirees au hasard (graine fixe) ; effectif a gauche."""
    paths, labels = list(paths), [str(v) for v in labels]  # en listes ; etiquettes en texte
    if len(paths) != len(labels):  # chaque chemin doit avoir son etiquette
        raise ValueError("class_montage : autant de chemins que d'etiquettes")  # erreur claire sinon
    read = read or read_image  # fonction de lecture par defaut
    rng = random.Random(seed)  # generateur aleatoire a graine fixe (tirage reproductible)
    classes = sorted(set(labels))  # classes distinctes, triees
    cells: list[np.ndarray] = []  # toutes les vignettes, ligne apres ligne
    row_names = []  # nom affiche a gauche de chaque ligne
    blank = None  # case blanche pour completer les lignes incompletes
    for name in classes:  # pour chaque classe
        members = [p for p, lab in zip(paths, labels) if lab == name]  # chemins de cette classe
        picked = rng.sample(members, min(per_class, len(members)))  # tirage sans remise de per_class images (ou moins)
        row = [_rgb(_fit(to_display(read(p)), max_side)) for p in picked]  # vignettes lues, converties, reduites, en RGB
        if blank is None and row:  # premiere vignette disponible
            blank = np.full_like(row[0], 255)  # case blanche de meme taille
        # ajoute la ligne, completee par des cases blanches jusqu'a per_class
        cells.extend(row + [blank if blank is not None else np.full((max_side, max_side, 3), 255, np.uint8)] * (per_class - len(row)))
        row_names.append(f"{name} (n={len(members)})")  # nom de ligne avec l'effectif de la classe
    cell_w = max(c.shape[1] for c in cells)  # largeur d'une case
    gap = max(3, round(cell_w / 50))  # espace entre vignettes
    grid_w = per_class * cell_w + (per_class + 1) * gap  # largeur de la grille en pixels (sans la marge)
    fontsize = 9  # taille du texte des noms de lignes
    label_in = max(len(n) for n in row_names) * fontsize * 0.6 / 72 + 0.15  # largeur en pouces du plus long nom (estimation)
    px_per_in = grid_w / (per_class * size)  # pixels de toile par pouce
    left = math.ceil(label_in * px_per_in)  # marge gauche en pixels pour les noms
    canvas, boxes = _montage(cells, per_class, 0, gap, left=left)  # assemble la toile (pas de bande de titre : band = 0)
    fig, ax = _layout_figure(canvas, per_class * size + label_in, None)  # figure qui affiche la toile
    for r, name in enumerate(row_names):  # pour chaque ligne (r = numero)
        x, y, w, h = boxes[r * per_class]  # position de la premiere vignette de la ligne
        ax.text(left - 4, y + h / 2, name, ha="right", va="center", fontsize=fontsize)  # nom aligne a droite dans la marge
    finish(fig, save)  # enregistre si demande
    return fig  # la figure


def _label_title(label: Any, class_names: Sequence[str] | None) -> str:  # titre de vignette a partir d'une etiquette ; renvoie le texte
    array = as_numpy(label)  # etiquette en numpy
    if array.ndim >= 1 and array.size > 1:  # etiquette douce (mixup / cutmix) : deux classes principales
        order = np.argsort(array.ravel())[::-1][:2]  # indices des 2 plus grandes valeurs (tri decroissant)
        # noms de ces 2 classes (ou leurs numeros si les noms manquent)
        names = [class_names[i] if class_names is not None and i < len(class_names) else str(i) for i in order]
        return "\n".join(f"{n} {array.ravel()[i]:.2f}" for n, i in zip(names, order) if array.ravel()[i] > 0)  # "nom poids" par ligne
    value = array.item() if array.size == 1 else label  # valeur unique en nombre Python
    if (  # si on peut traduire la valeur en nom de classe :
        class_names is not None  # les noms sont connus
        and isinstance(value, int | np.integer | float)  # la valeur est un nombre
        and float(value).is_integer()  # entier (ex. 3 ou 3.0)
        and 0 <= int(value) < len(class_names)  # indice valide
    ):
        return str(class_names[int(value)])  # le nom de la classe
    return str(value)  # sinon la valeur telle quelle


def show_batch(  # affiche un lot tel que le voit le reseau ; renvoie la figure
    batch: Any,  # le lot : (images, etiquettes) d'un DataLoader, ou seulement les images
    labels: Any = None,  # etiquettes (si elles ne sont pas dans batch)
    class_names: Sequence[str] | None = None,  # noms des classes
    *,  # les arguments suivants doivent etre nommes
    mean: Sequence[float] | float | None = None,  # normalisation a annuler (moyenne)
    std: Sequence[float] | float | None = None,  # normalisation a annuler (ecart-type)
    ncols: int = 8,  # vignettes par ligne
    max_images: int = 32,  # nombre maximal d'images affichees
    bgr: bool = False,  # vrai si les images sont en BGR
    save: str | Path | None = None,  # fichier ou enregistrer la figure
) -> Figure:  # renvoie la figure
    """Un lot tel que le voit le reseau : (images, etiquettes) d'un DataLoader, NCHW normalise.

    Etiquettes : indices, noms, etiquettes douces (mixup) ou cibles de detection (dict "boxes").
    """
    if labels is None and isinstance(batch, list | tuple) and len(batch) >= 2:  # lot (images, etiquettes) d'un DataLoader
        batch, labels = batch[0], batch[1]  # separe images et etiquettes
    images = batch if isinstance(batch, list | tuple) else list(as_numpy(batch))  # liste d'images (un tenseur NCHW est decoupe)
    images = list(images)[:max_images]  # au plus max_images
    shown = [to_display(im, bgr=bgr, mean=mean, std=std) for im in images]  # images affichables (normalisation annulee)
    titles = None  # pas de titres par defaut
    if labels is not None:  # si on a des etiquettes
        labels = labels if isinstance(labels, list | tuple) else list(as_numpy(labels))  # en liste
        labels = list(labels)[: len(shown)]  # autant que d'images affichees
        if labels and isinstance(labels[0], Mapping) and "boxes" in labels[0]:  # cibles de detection (dict avec "boxes")
            shown = [  # chaque image avec ses boites dessinees
                draw_boxes(  # dessine les boites de la cible t sur l'image im
                    im,  # l'image
                    _boxes_array(t["boxes"]),  # ses boites
                    labels=t.get("labels"),  # leurs classes
                    class_names=class_names,  # noms des classes
                    rotated=_boxes_array(t["boxes"]).shape[1] == 5,  # orientees si 5 valeurs
                )
                for im, t in zip(shown, labels)  # pour chaque image et sa cible
            ]
        else:  # etiquettes de classification
            titles = [_label_title(v, class_names) for v in labels]  # un titre par image
    return grid(shown, titles, ncols=ncols, save=save)  # planche des images


def _seed_all(seed: int) -> None:  # fixe toutes les graines aleatoires (Python, numpy, et torch s'il est installe)
    random.seed(seed)  # hasard de Python
    np.random.seed(seed)  # hasard de numpy
    try:  # torch est facultatif
        import torch  # PyTorch

        torch.manual_seed(seed)  # hasard de PyTorch (transformations torchvision)
    except ImportError:  # torch absent
        pass  # rien a faire


def _to_tensor_input(image: np.ndarray) -> Any:  # image numpy -> tenseur PyTorch (C, H, W) en 0-1 ; le renvoie
    import torch  # PyTorch (import ici : facultatif pour le reste du module)

    array = image.astype(np.float32) / (255.0 if image.dtype == np.uint8 else 1.0)  # en 0-1 si l'image est en 0-255
    array = array[None] if array.ndim == 2 else np.moveaxis(array, -1, 0)  # gris -> (1, H, W) ; couleur HWC -> CHW
    return torch.from_numpy(np.ascontiguousarray(array))  # tenseur (memoire rangee d'un bloc)


# sortie d'une transformation (image seule, ou image + masque / boites) -> image dessinee ; renvoie un tableau uint8
def _render_output(output: Any, class_names: Sequence[str] | None) -> np.ndarray:
    target = None  # cible associee (masque, boites), si presente
    if isinstance(output, tuple | list) and len(output) == 2:  # sortie (image, cible)
        output, target = output  # separees
    if isinstance(output, Mapping) and "image" in output:  # sortie en dictionnaire {"image": ..., "mask": ..., "bboxes": ...}
        target = {k: v for k, v in output.items() if k != "image"} or None  # tout sauf l'image est la cible (None si vide)
        output = output["image"]  # l'image
    image = to_display(output)  # image affichable
    if target is None:  # pas de cible
        return image  # l'image seule
    if isinstance(target, Mapping):  # cible en dictionnaire
        if "mask" in target and target["mask"] is not None:  # un masque
            image = overlay_mask(image, target["mask"])  # superpose le masque
        boxes = target.get("boxes", target.get("bboxes"))  # des boites (cle "boxes", ou "bboxes" selon la bibliotheque)
        if boxes is not None:  # si des boites sont presentes
            boxes_np = _boxes_array(boxes)  # en tableau
            # dessine les boites avec leurs classes
            image = draw_boxes(image, boxes_np, labels=target.get("labels"), class_names=class_names, rotated=boxes_np.shape[1] == 5)
        return image  # l'image dessinee
    array = as_numpy(target)  # cible en numpy
    # un masque : ses 2 dernieres dimensions ont la taille de l'image
    if array.ndim >= 2 and array.shape[-2:] == image.shape[:2] or (array.ndim == 2 and array.shape == image.shape[:2]):
        return overlay_mask(image, array.reshape(image.shape[:2]))  # superpose le masque
    if array.ndim == 2 and array.shape[1] in (4, 5):  # un tableau de boites (N, 4) ou (N, 5)
        return draw_boxes(image, array, rotated=array.shape[1] == 5)  # dessine les boites
    return image  # cible non reconnue : l'image seule


def show_augmentations(  # montre l'original puis plusieurs tirages d'une augmentation ; renvoie la figure
    image: Any,  # l'image d'origine
    transform: Callable[..., Any],  # la transformation aleatoire a tester
    *,  # les arguments suivants doivent etre nommes
    n: int = 8,  # nombre de tirages
    target: Any = None,  # cible a transformer avec l'image (masque, boites)
    class_names: Sequence[str] | None = None,  # noms des classes
    seed: int | None = 0,  # graine (None = tirages differents a chaque appel)
    ncols: int = 4,  # vignettes par ligne
    max_side: int = 256,  # taille maximale d'une vignette
    save: str | Path | None = None,  # fichier ou enregistrer la figure
) -> Figure:  # renvoie la figure
    """L'original puis n tirages de la transformation (boites / masque dessines s'ils suivent).

    transform : torchvision v2 (tenseur CHW float) ou fonction numpy ; si l'appel echoue sur le
    tableau numpy, il est refait sur un tenseur float [0, 1]. target : masque, boites
    (tv_tensors) ou dict, passe en second argument.
    """
    if seed is not None:  # si une graine est donnee
        _seed_all(seed)  # tirages reproductibles
    source = image  # ce qu'on passe a la transformation (numpy d'abord)
    cells = [_render_output((image, target) if target is not None else image, class_names)]  # premiere vignette : l'original
    titles = ["original"]  # son titre
    use_tensor = False  # on essaie d'abord avec le tableau numpy
    for i in range(n):  # pour chaque tirage
        args = (source,) if target is None else (source, target)  # arguments : image seule, ou image et cible
        if not use_tensor:  # tant que numpy marche
            try:  # essai sur le tableau numpy
                out = transform(*args)  # applique la transformation
            except Exception:  # noqa: BLE001 - nouvel essai sur un tenseur
                use_tensor = True  # la transformation veut un tenseur : on bascule
        if use_tensor:  # mode tenseur
            if not _is_tensor(source):  # premiere fois : conversion
                source = _to_tensor_input(as_numpy(image))  # image -> tenseur (C, H, W) 0-1
            args = (source,) if target is None else (source, target)  # arguments refaits avec le tenseur
            out = transform(*args)  # applique la transformation
        cells.append(_render_output(out, class_names))  # vignette du resultat (cible dessinee si presente)
        titles.append(f"tirage {i + 1}")  # son titre
    return grid(cells, titles, ncols=ncols, max_side=max_side, save=save)  # planche des resultats


def pipeline_steps(  # montre l'image apres chaque etape d'un pretraitement ; renvoie la figure
    image: Any,  # l'image d'origine
    steps: Sequence[tuple[str, Callable[[np.ndarray], np.ndarray]]],  # etapes : (nom, fonction image -> image)
    *,  # les arguments suivants doivent etre nommes
    bgr: bool = False,  # vrai si l'image est en BGR
    cumulative: bool = True,  # vrai : chaque etape part du resultat precedent ; faux : chaque etape part de l'original
    ncols: int = 6,  # vignettes par ligne
    max_side: int = 256,  # taille maximale d'une vignette
    save: str | Path | None = None,  # fichier ou enregistrer la figure
) -> Figure:  # renvoie la figure
    """Etat de l'image apres chaque etape (cumulative) ou chaque etape seule sur l'original.

    Le titre donne type, forme et plage : on voit ou un uint8 devient float, ou un canal saute.
    """
    current = image  # image courante
    images, titles = [image], [f"original\n{_describe(image)}"]  # premiere vignette : l'original et sa description
    for name, func in steps:  # pour chaque etape (nom, fonction)
        current = func(current if cumulative else image)  # applique l'etape au resultat precedent (ou a l'original)
        images.append(current)  # garde le resultat
        titles.append(f"{name}\n{_describe(current)}")  # titre : nom de l'etape et description (type, forme, plage)
    return grid([to_display(im, bgr=bgr) for im in images], titles, ncols=ncols, max_side=max_side, save=save)  # planche
