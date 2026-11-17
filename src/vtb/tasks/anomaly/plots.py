"""Figures des anomalies : distribution des scores et cartes sur des exemples du test."""

from __future__ import annotations  # permet les annotations de type modernes (str | Path)

from pathlib import Path  # Path : chemin de fichier independant du systeme

import matplotlib  # matplotlib : bibliotheque de graphiques

matplotlib.use("Agg")  # moteur "Agg" : dessine dans des fichiers, sans fenetre (marche sur un serveur sans ecran)
# pyplot : l'interface de dessin de matplotlib (figures, axes) ; importe APRES le choix du moteur, d'ou le noqa E402
import matplotlib.pyplot as plt  # noqa: E402

# numpy : calcul sur des tableaux de nombres
import numpy as np  # noqa: E402

# pandas : tableaux de donnees (ici le decoupage du test)
import pandas as pd  # noqa: E402

# to_uint8 : convertit une image en entiers 0-255 pour l'afficher
from vtb.io.images import to_uint8  # noqa: E402


# histogramme des scores du test (normales vs defauts) avec le seuil ; enregistre la figure dans path, ne renvoie rien
def plot_score_histogram(scores: np.ndarray, labels: np.ndarray, threshold: float, path: str | Path) -> None:
    """Normales et defauts du test, et le seuil calibre : la separation se lit d'un coup d'oeil."""
    fig, ax = plt.subplots(figsize=(6.5, 3.2))  # une figure avec un seul graphique (taille en pouces)
    bins = np.linspace(scores.min(), scores.max(), 40)  # 40 intervalles communs aux deux histogrammes (comparables)
    ax.hist(scores[labels == 0], bins=bins, alpha=0.6, label="normales")  # histogramme des normales (etiquette 0), semi-transparent
    ax.hist(scores[labels == 1], bins=bins, alpha=0.6, label="defauts")  # histogramme des defauts (etiquette 1)
    ax.axvline(threshold, color="black", linestyle="--", label="seuil")  # ligne verticale pointillee au seuil de decision
    ax.set_xlabel("score image")  # titre de l'axe horizontal
    ax.legend()  # affiche la legende (normales, defauts, seuil)
    fig.tight_layout()  # ajuste les marges pour que rien ne depasse
    fig.savefig(path, dpi=110)  # enregistre l'image (dpi = resolution)
    plt.close(fig)  # ferme la figure pour liberer la memoire


# grille 3 lignes (image, masque verite, carte d'anomalie) x une colonne par exemple ; enregistre la figure dans path
def plot_maps(dataset, result: dict, frame: pd.DataFrame, threshold: float, path: str | Path, per_type: int = 1) -> None:
    """Image, masque verite, carte d'anomalie : une normale et un exemple par type de defaut, meme echelle de couleur."""
    picks = []  # indices des images a montrer
    types = frame["defect_type"].to_numpy()  # type de defaut de chaque image du test ("good" pour une normale)
    for t in ["good", *sorted(set(types) - {"good"})]:  # pour "good" puis chaque type de defaut, dans l'ordre alphabetique
        picks += list(np.where(types == t)[0][:per_type])  # garde les per_type premieres images de ce type
    # echelle commune a toutes les cartes ; certaines methodes (FastFlow) rendent des valeurs negatives
    vmin = float(np.percentile(result["maps"][picks], 0.5)) if picks else 0.0  # bas de l'echelle : percentile 0,5 (ignore les extremes)
    vmax = float(np.percentile(result["maps"][picks], 99.5)) if picks else 1.0  # haut de l'echelle : percentile 99,5
    vmin = min(vmin, 0.0) if vmin >= 0 else vmin  # si les valeurs sont positives, l'echelle part de 0 ; sinon on garde le negatif
    vmax = max(vmax, vmin + 1e-6)  # evite une echelle de largeur nulle (carte constante)
    # 3 lignes de graphiques, une colonne par exemple ; squeeze=False garde un tableau 2D meme avec une seule colonne
    fig, axes = plt.subplots(3, len(picks), figsize=(2.3 * len(picks), 7), squeeze=False)
    for col, i in enumerate(picks):  # pour chaque exemple choisi (col = numero de colonne, i = indice dans le test)
        image = to_uint8(dataset.load(i))  # image pretraitee, convertie en 0-255 pour l'affichage
        axes[0, col].imshow(image, cmap="gray" if image.ndim == 2 else None)  # ligne 1 : l'image (en gris si elle n'a qu'un canal)
        # titre : type et score ; " !" si le score depasse le seuil (piece signalee)
        axes[0, col].set_title(f"{types[i]}\nscore {result['scores'][i]:.3g}{' !' if result['scores'][i] > threshold else ''}", fontsize=8)
        axes[1, col].imshow(result["masks"][i], cmap="gray", vmin=0, vmax=1)  # ligne 2 : masque verite (blanc = defaut)
        axes[2, col].imshow(result["maps"][i], cmap="jet", vmin=vmin, vmax=vmax)  # ligne 3 : carte d'anomalie (rouge = anormal)
        for row in range(3):  # pour chacune des 3 lignes
            axes[row, col].axis("off")  # cache les axes gradues (inutiles sur une image)
    axes[1, 0].set_title("masque", fontsize=8, loc="left")  # nom de la ligne 2, au-dessus de la premiere colonne
    axes[2, 0].set_title("carte", fontsize=8, loc="left")  # nom de la ligne 3
    fig.tight_layout()  # ajuste les marges
    fig.savefig(path, dpi=100)  # enregistre la figure
    plt.close(fig)  # libere la memoire
