"""Planches avant / apres : voir l'effet de chaque operation sur ses propres images.

Une planche par famille : premiere ligne = images d'origine, puis une ligne par operation. Les
images 16 bits sont etirees (percentiles 1-99) pour l'affichage seulement. Les figures sont des
objets matplotlib.figure.Figure (sans pyplot) : elles s'enregistrent dans un script sans ecran et
s'affichent dans un notebook (derniere expression d'une cellule, ou display(fig)).
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes

import logging  # journal : messages horodates sur la console et dans un fichier
from collections.abc import Mapping, Sequence  # types "dict en lecture" et "suite d'elements" pour les annotations
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux
from typing import Any  # Any = n'importe quel type

import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)
from matplotlib.backends.backend_agg import FigureCanvasAgg  # moteur de dessin de matplotlib qui marche sans ecran
from matplotlib.figure import Figure  # Figure : une figure matplotlib (contient les sous-graphiques)

from vtb.io.images import to_uint8  # conversion en 8 bits pour l'affichage
from vtb.preprocess.registry import REGISTRY  # registre de toutes les operations de pretraitement

logger = logging.getLogger(__name__)  # journal propre a ce module

# Operations qui ont besoin de fichiers ou de coordonnees propres a une installation : absentes
# des planches par defaut (les fournir dans params pour les y faire figurer).
NEEDS_SETUP = {"flat_field", "align_ecc", "align_orb", "absdiff_reference", "perspective", "undistort"}


def _display(image: np.ndarray) -> np.ndarray:  # prepare une image pour l'affichage (8 bits)
    return to_uint8(image, 1, 99) if image.dtype != np.uint8 else image  # 16 bits : etirement 1-99 % ; 8 bits : inchangee


# dessine une planche : originaux en haut, puis une ligne par operation ; renvoie la figure
def family_sheet(images: Sequence[np.ndarray], operations: Sequence[tuple[str, dict[str, Any]]], title: str, cell: float = 1.9) -> Figure:
    """Figure : ligne 0 = originaux, puis une ligne par (operation, parametres)."""
    rows, cols = len(operations) + 1, len(images)  # une ligne par operation + la ligne des originaux ; une colonne par image
    fig = Figure(figsize=(cell * cols + 1.6, cell * rows))  # taille en pouces : "cell" par case, + une marge pour les noms
    FigureCanvasAgg(fig)  # un canevas de rendu : savefig et l'affichage dans un notebook en ont besoin
    axes = fig.subplots(rows, cols, squeeze=False)  # grille de cases ; squeeze=False garde un tableau 2D meme avec 1 ligne
    for c, image in enumerate(images):  # pour chaque image d'origine (c = numero de colonne)
        axes[0][c].imshow(_display(image), cmap="gray", vmin=0, vmax=255)  # l'affiche en ligne 0 (echelle 0-255 fixe)
    axes[0][0].set_ylabel("origine", rotation=0, ha="right", va="center", fontsize=9)  # nom de la ligne 0
    for r, (name, params) in enumerate(operations, start=1):  # pour chaque operation (r = numero de ligne, a partir de 1)
        for c, image in enumerate(images):  # pour chaque image
            try:  # on essaie d'appliquer l'operation
                out = REGISTRY[name].func(image, **params)  # applique l'operation avec ses parametres
                axes[r][c].imshow(_display(out), cmap="gray", vmin=0, vmax=255)  # affiche le resultat
            except Exception as error:  # une planche ne doit pas tomber pour une image inadaptee
                axes[r][c].text(0.5, 0.5, type(error).__name__, ha="center", va="center", fontsize=7)  # ecrit le nom de l'erreur
        axes[r][0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9)  # nom de l'operation a gauche de la ligne
    for ax in axes.ravel():  # pour chaque case de la grille
        ax.set_xticks([])  # retire les graduations horizontales
        ax.set_yticks([])  # retire les graduations verticales
    fig.suptitle(title, fontsize=11)  # titre general de la planche
    fig.tight_layout()  # resserre la mise en page sans chevauchement
    return fig  # renvoie la figure


def figure_bytes(fig: Figure, fmt: str = "jpeg", dpi: int = 70) -> bytes:  # convertit une figure en octets d'image
    """Figure -> octets PNG ou JPEG (pour IPython.display.Image dans un notebook, ou une page web)."""
    import io  # io : fichiers en memoire (pas sur le disque)

    buffer = io.BytesIO()  # "fichier" en memoire qui recevra l'image
    # enregistre la figure dans ce tampon ; en JPEG on fixe la qualite a 85
    fig.savefig(buffer, format=fmt, dpi=dpi, **({"pil_kwargs": {"quality": 85}} if fmt == "jpeg" else {}))
    return buffer.getvalue()  # renvoie les octets de l'image


def save_gallery(  # ecrit une planche par famille d'operations ; renvoie les chemins ecrits
    images: Sequence[np.ndarray],  # images d'exemple a transformer
    out_dir: str | Path,  # dossier ou ecrire les planches
    params: Mapping[str, dict[str, Any]] | None = None,  # parametres par operation (nom -> parametres)
    families: Sequence[str] | None = None,  # familles a dessiner (None = toutes)
    skip: Sequence[str] = (),  # operations a ne pas dessiner
    dpi: int = 70,  # resolution des images ecrites (points par pouce)
) -> list[Path]:  # renvoie la liste des fichiers ecrits
    """Une planche JPEG par famille du registre dans out_dir ; renvoie les chemins ecrits.

    JPEG (qualite 85) plutot que PNG : une planche de photos pese 5 fois moins, ce qui compte pour
    les pages du site ou les images sont integrees.
    """
    params = dict(params or {})  # copie des parametres (dict vide si rien n'est donne)
    out_dir = Path(out_dir)  # convertit en objet chemin
    out_dir.mkdir(parents=True, exist_ok=True)  # cree le dossier s'il n'existe pas
    by_family: dict[str, list[tuple[str, dict[str, Any]]]] = {}  # famille -> liste de (operation, parametres)
    for spec in sorted(REGISTRY.values(), key=lambda s: (s.family, s.name)):  # pour chaque operation, triee par famille puis nom
        if spec.name in skip or (spec.name in NEEDS_SETUP and spec.name not in params):  # exclue, ou a besoin d'un reglage non fourni
            continue  # on la saute
        if families and spec.family not in families:  # si on a choisi des familles et que celle-ci n'en fait pas partie
            continue  # on la saute
        by_family.setdefault(spec.family, []).append((spec.name, params.get(spec.name, {})))  # range l'operation dans sa famille
    written = []  # chemins des planches ecrites
    for family, operations in by_family.items():  # pour chaque famille et ses operations
        fig = family_sheet(images, operations, f"famille : {family}")  # dessine la planche
        path = out_dir / f"{family}.jpg"  # nom du fichier : <famille>.jpg
        fig.savefig(path, dpi=dpi, pil_kwargs={"quality": 85})  # enregistre en JPEG qualite 85
        written.append(path)  # note le chemin
        logger.info("planche %-12s : %d operations -> %s", family, len(operations), path)  # journal
    return written  # renvoie les chemins ecrits
