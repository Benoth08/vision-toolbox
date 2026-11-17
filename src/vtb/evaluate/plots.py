"""Figures communes : courbes d'apprentissage, matrice de confusion, planche d'images."""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes

from collections.abc import Sequence  # type "suite d'elements" (liste, tuple) pour les annotations
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import matplotlib  # matplotlib : bibliotheque de graphiques

matplotlib.use("Agg")  # pas de fenetre : les figures vont dans des fichiers (scripts, serveurs)
import matplotlib.pyplot as plt  # noqa: E402  # pyplot : interface de dessin (import apres le choix du moteur, d'ou le noqa)
import numpy as np  # noqa: E402  # numpy : calcul sur des tableaux de nombres
import pandas as pd  # noqa: E402  # pandas : tableaux de donnees (ici l'historique d'entrainement)


# trace les courbes d'apprentissage et les enregistre dans un fichier
def plot_history(history: pd.DataFrame, path: str | Path, metrics: Sequence[str] = ("loss", "acc")) -> None:
    """Une courbe train / val par metrique presente dans l'historique."""
    present = [m for m in metrics if f"train_{m}" in history and f"val_{m}" in history]  # metriques dont on a train ET val
    fig, axes = plt.subplots(1, len(present), figsize=(5 * len(present), 3.4), squeeze=False)  # un graphique par metrique, cote a cote
    for ax, metric in zip(axes[0], present):  # pour chaque graphique et sa metrique
        ax.plot(history["epoch"], history[f"train_{metric}"], label="train")  # courbe entrainement ; epoque = un passage sur tout le train
        ax.plot(history["epoch"], history[f"val_{metric}"], label="val")  # courbe validation
        ax.set_title(metric)  # titre = nom de la metrique
        ax.set_xlabel("epoque")  # axe horizontal
        ax.grid(alpha=0.3)  # quadrillage leger
        ax.legend()  # legende train / val
    fig.tight_layout()  # mise en page sans chevauchement
    fig.savefig(path, dpi=110)  # enregistre l'image
    plt.close(fig)  # libere la memoire de la figure


# dessine la matrice de confusion et l'enregistre dans un fichier
def plot_confusion(matrix: np.ndarray, class_names: Sequence[str], path: str | Path, title: str = "") -> None:
    """Lignes = verite, colonnes = prediction ; une case hors diagonale dit QUELLE confusion corriger."""
    fig, ax = plt.subplots(figsize=(1.1 * len(class_names) + 2.5, 1.0 * len(class_names) + 2.2))  # taille adaptee au nombre de classes
    ax.imshow(matrix, cmap="Blues")  # cases colorees : plus fonce = plus d'images
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")  # noms des classes en colonnes (inclines)
    ax.set_yticks(range(len(class_names)), class_names)  # noms des classes en lignes
    ax.set_xlabel("prediction")  # axe horizontal = classe predite
    ax.set_ylabel("verite")  # axe vertical = vraie classe
    for i in range(matrix.shape[0]):  # pour chaque ligne (vraie classe)
        for j in range(matrix.shape[1]):  # pour chaque colonne (classe predite)
            # ecrit le nombre dans la case : blanc sur fond fonce, noir sur fond clair
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    if title:  # si un titre est donne
        ax.set_title(title)  # l'affiche
    fig.tight_layout()  # mise en page sans chevauchement
    fig.savefig(path, dpi=110)  # enregistre l'image
    plt.close(fig)  # libere la memoire de la figure


# planche d'images avec un titre chacune, enregistree dans un fichier
def plot_gallery(images: Sequence[np.ndarray], titles: Sequence[str], path: str | Path, columns: int = 6, cmap: str = "gray") -> None:
    """Planche d'images avec un titre chacune (erreurs du modele, exemples par classe...)."""
    if not images:  # aucune image
        return  # rien a dessiner
    rows = int(np.ceil(len(images) / columns))  # nombre de lignes necessaires
    fig, axes = plt.subplots(rows, columns, figsize=(2.4 * columns, 2.6 * rows), squeeze=False)  # grille de cases
    for ax in axes.ravel():  # pour chaque case
        ax.axis("off")  # cache les axes (les cases vides restent blanches)
    for ax, image, title in zip(axes.ravel(), images, titles):  # pour chaque image et son titre
        ax.imshow(image, cmap=cmap if image.ndim == 2 else None)  # affiche l'image (palette grise seulement si elle est grise)
        ax.set_title(title, fontsize=8)  # petit titre au-dessus
    fig.tight_layout()  # mise en page sans chevauchement
    fig.savefig(path, dpi=110)  # enregistre l'image
    plt.close(fig)  # libere la memoire de la figure
