"""Courbes d'entrainement : apres coup (plot_history) ou en direct pendant la boucle (LivePlot).

    from vtb import viz
    viz.plot_history("outputs/classification/quick/history.csv", best="val_loss:min")

    live = viz.LivePlot(("train_loss", "val_loss", "val_acc"), title="resnet18")
    for epoch in range(epochs):
        ...
        live.update(epoch, train_loss=tl, val_loss=vl, val_acc=va)   # une seule figure redessinee
    history = live.history                                           # DataFrame

Les colonnes sont regroupees par metrique : `loss`, `train_loss` et `val_loss` vont dans le meme
panneau ; `lr` est trace en echelle log. La meilleure epoque (best="colonne:min|max", par
defaut val_loss:min si elle existe) est marquee sur tous les panneaux.
"""

from __future__ import annotations  # permet les annotations de type modernes

import time  # mesure du temps ecoule depuis le debut de l'entrainement
from collections.abc import Mapping, Sequence  # Mapping : type "dictionnaire" ; Sequence : type "liste ou tuple"
from pathlib import Path  # Path : chemin de fichier independant du systeme
from typing import Any  # Any : n'importe quel type

import numpy as np  # numpy : calcul sur des tableaux de nombres
import pandas as pd  # pandas : tableaux de donnees (l'historique : une ligne par epoque)
from matplotlib.figure import Figure  # Figure : la page qui contient les graphiques

from .style import _STATE, finish, in_notebook, new_figure, render  # outils communs de style et d'affichage (style.py)

_PREFIXES = ("train_", "val_", "valid_", "test_")  # prefixes retires pour regrouper les colonnes (val_loss -> loss)


def _as_frame(history: Any) -> pd.DataFrame:  # accepte un historique sous plusieurs formes ; renvoie un DataFrame avec une colonne epoch
    if isinstance(history, pd.DataFrame):  # deja un tableau pandas
        frame = history.copy()  # copie (on va peut-etre ajouter une colonne)
    elif isinstance(history, str | Path):  # un chemin de fichier
        frame = pd.read_csv(history)  # lecture du CSV
    elif isinstance(history, Mapping):  # un dictionnaire de listes (colonne -> valeurs)
        frame = pd.DataFrame({k: list(v) for k, v in history.items()})  # converti en tableau
    else:  # sinon : une liste de dictionnaires (une ligne par epoque)
        frame = pd.DataFrame(list(history))  # converti en tableau
    if "epoch" not in frame.columns:  # s'il n'y a pas de colonne epoch
        frame.insert(0, "epoch", np.arange(len(frame)))  # on la cree en premiere position : 0, 1, 2...
    return frame  # le tableau


def _base(column: str) -> str:  # nom de la metrique sans prefixe (val_loss -> loss) ; renvoie ce nom
    for prefix in _PREFIXES:  # pour chaque prefixe connu
        if column.startswith(prefix):  # si la colonne commence par ce prefixe
            return column[len(prefix) :]  # on renvoie la suite du nom
    return column  # sinon le nom tel quel


# regroupe les colonnes a tracer par metrique (un graphique par groupe) ; renvoie un dictionnaire metrique -> colonnes
def _groups(frame: pd.DataFrame, metrics: Sequence[str] | None) -> dict[str, list[str]]:
    # colonnes numeriques (sauf epoch) qui ont au moins une valeur
    columns = [c for c in frame.columns if c != "epoch" and pd.api.types.is_numeric_dtype(frame[c]) and frame[c].notna().any()]
    if metrics is not None:  # si on a demande certaines metriques seulement
        wanted = set(metrics)  # ensemble des noms demandes
        columns = [c for c in columns if c in wanted or _base(c) in wanted]  # garde les colonnes demandees (nom exact ou de base)
    groups: dict[str, list[str]] = {}  # dictionnaire resultat
    for column in columns:  # pour chaque colonne gardee
        groups.setdefault(_base(column), []).append(column)  # l'ajoute au groupe de sa metrique (cree le groupe au besoin)
    return groups  # les groupes


# decode best="colonne:min|max" ; renvoie (colonne, "min" ou "max"), ou None si rien d'utilisable
def _parse_best(frame: pd.DataFrame, best: str | None) -> tuple[str, str] | None:
    if best is None:  # si rien n'est precise
        for candidate in ("val_loss", "valid_loss"):  # on cherche une perte de validation
            if candidate in frame.columns and frame[candidate].notna().any():  # si cette colonne existe et a des valeurs
                return candidate, "min"  # meilleure epoque = perte de validation minimale
        return None  # aucune colonne candidate
    column, _, mode = best.partition(":")  # coupe "val_acc:max" en "val_acc" et "max"
    if column not in frame.columns or not frame[column].notna().any():  # colonne absente ou vide
        return None  # pas de meilleure epoque a marquer
    return column, (mode or ("min" if "loss" in column else "max"))  # mode donne, sinon min pour une perte, max pour le reste


# dessine les courbes dans une figure existante (un graphique par metrique) ; ne renvoie rien
def _draw_history(fig: Figure, frame: pd.DataFrame, metrics: Sequence[str] | None, best: str | None, title: str = "") -> None:
    groups = _groups(frame, metrics)  # colonnes regroupees par metrique
    if not groups:  # si rien a tracer
        ax = fig.add_subplot(1, 1, 1)  # un seul graphique
        ax.text(0.5, 0.5, "pas encore de valeur", ha="center", va="center")  # message au centre
        ax.set_axis_off()  # sans axes
        return  # fini
    ncols = min(3, len(groups))  # au plus 3 graphiques par ligne
    nrows = int(np.ceil(len(groups) / ncols))  # nombre de lignes necessaires (arrondi au-dessus)
    chosen = _parse_best(frame, best)  # colonne et sens de la meilleure epoque
    best_epoch = None  # pas de meilleure epoque par defaut
    if chosen is not None:  # si on sait quelle colonne regarder
        column, mode = chosen  # colonne et "min" / "max"
        values = frame[column]  # ses valeurs
        index = values.idxmin() if mode == "min" else values.idxmax()  # ligne du minimum (ou du maximum)
        best_epoch = frame.loc[index, "epoch"]  # numero d'epoque de cette ligne
        best_value = values.loc[index]  # valeur a cette epoque
    epochs = frame["epoch"].to_numpy()  # axe horizontal : les epoques
    for k, (name, columns) in enumerate(groups.items()):  # pour chaque groupe (k = numero, name = metrique)
        ax = fig.add_subplot(nrows, ncols, k + 1)  # graphique numero k + 1 dans la grille
        for column in columns:  # pour chaque colonne du groupe (ex. train_loss, val_loss)
            values = frame[column].to_numpy(dtype=float)  # valeurs en nombres a virgule (None -> nan)
            mask = np.isfinite(values)  # vrai la ou il y a une vraie valeur
            style = "--" if column.startswith(("val_", "valid_")) else "-"  # validation en tirets, entrainement en trait plein
            # trace la courbe ; points ronds seulement s'il y a moins de 30 valeurs (sinon illisible)
            ax.plot(epochs[mask], values[mask], style, marker="o" if mask.sum() < 30 else None, ms=3, lw=1.5, label=column)
        if name in ("lr", "learning_rate") and (frame[columns[0]] > 0).all():  # taux d'apprentissage strictement positif
            ax.set_yscale("log")  # echelle logarithmique : il varie souvent sur plusieurs ordres de grandeur
        if best_epoch is not None:  # si une meilleure epoque est connue
            ax.axvline(best_epoch, color="gray", ls=":", lw=1)  # ligne verticale pointillee a cette epoque, sur chaque graphique
            if chosen and chosen[0] in columns:  # sur le graphique qui contient la colonne de reference
                ax.annotate(  # petite etiquette de texte
                    f"meilleure {best_value:.4g}\nepoque {best_epoch}",  # texte : valeur et epoque
                    (best_epoch, best_value),  # point vise
                    xytext=(6, 6),  # decalage du texte
                    textcoords="offset points",  # decalage en points d'ecran
                    fontsize=7,  # petite police
                )
        ax.set_title(name, fontsize=10)  # titre : nom de la metrique
        ax.set_xlabel("epoque")  # titre de l'axe horizontal
        ax.grid(alpha=0.3)  # quadrillage leger
        ax.legend(fontsize=7)  # legende (noms des colonnes)
    if title:  # si un titre general est donne
        fig.suptitle(title, fontsize=10)  # titre au-dessus de toute la figure
    fig.tight_layout()  # ajuste les marges


def plot_history(  # trace l'historique d'entrainement apres coup ; renvoie la figure
    history: Any,  # l'historique (tableau, dictionnaire, liste ou chemin CSV)
    *,  # les arguments suivants doivent etre nommes (metrics=..., best=...)
    metrics: Sequence[str] | None = None,  # metriques a tracer (None = toutes)
    best: str | None = None,  # colonne et sens de la meilleure epoque ("val_loss:min")
    title: str = "",  # titre general
    save: str | Path | None = None,  # fichier ou enregistrer la figure (None = pas d'enregistrement)
) -> Figure:  # renvoie la figure
    """Courbes par metrique (train en trait plein, val en tirets) et meilleure epoque marquee.

    history : DataFrame, dict de listes, liste de dicts (une ligne par epoque) ou chemin CSV.
    """
    frame = _as_frame(history)  # historique converti en tableau
    groups = _groups(frame, metrics)  # groupes a tracer (pour dimensionner la figure)
    ncols = max(1, min(3, len(groups)))  # 1 a 3 colonnes de graphiques
    nrows = max(1, int(np.ceil(len(groups) / ncols)))  # au moins une ligne
    fig = new_figure((4.6 * ncols, 3.2 * nrows + (0.3 if title else 0)))  # taille adaptee, un peu plus haute s'il y a un titre
    _draw_history(fig, frame, metrics, best, title)  # dessine les courbes
    finish(fig, save)  # enregistre si demande
    return fig  # la figure


class LivePlot:  # courbes mises a jour pendant l'entrainement
    """Courbes mises a jour a chaque epoque, dans UNE seule sortie du notebook.

    Hors notebook (script, terminal), imprime une ligne par epoque. Rien n'est garde en memoire
    de pyplot : la figure est rendue en image puis remplacee a la mise a jour suivante.
    """

    # metrics : metriques a tracer ; title : titre ; every : redessine toutes les every epoques ; best : meilleure epoque
    def __init__(self, metrics: Sequence[str] = ("loss", "val_loss"), *, title: str = "", every: int = 1, best: str | None = None) -> None:
        self.metrics = list(metrics)  # metriques a tracer
        self.title = title  # titre
        self.every = max(1, every)  # frequence de redessin (au moins 1)
        self.best = best  # reglage de la meilleure epoque
        self._rows: list[dict[str, Any]] = []  # historique : une ligne (dictionnaire) par epoque
        self._handle: Any = None  # reference de la zone d'affichage du notebook (pour la remplacer)
        self._notebook = in_notebook()  # vrai si on est dans un notebook
        self._start = time.time()  # heure de depart (pour afficher le temps ecoule)
        self.figure: Figure | None = None  # derniere figure rendue

    @property  # s'utilise comme un attribut : live.history (sans parentheses)
    def history(self) -> pd.DataFrame:  # renvoie l'historique sous forme de tableau
        return pd.DataFrame(self._rows)  # une ligne par epoque

    def _render(self) -> Figure:  # dessine la figure a jour ; la renvoie
        frame = _as_frame(self._rows)  # historique en tableau
        groups = _groups(frame, self.metrics)  # groupes a tracer
        ncols = max(1, min(3, len(groups)))  # 1 a 3 colonnes
        nrows = max(1, int(np.ceil(len(groups) / ncols)))  # nombre de lignes
        fig = Figure(figsize=(4.6 * ncols, 3.2 * nrows + 0.3))  # Figure creee hors pyplot : rien a fermer, pas de fuite memoire
        last = self._rows[-1]  # derniere epoque enregistree
        # texte "nom valeur | nom valeur" des valeurs numeriques de la derniere epoque
        values = " | ".join(f"{k} {v:.4g}" for k, v in last.items() if k != "epoch" and isinstance(v, int | float | np.floating))
        # titre : titre donne (s'il existe) + numero d'epoque + dernieres valeurs
        title = f"{self.title} - epoque {last['epoch']} - {values}" if self.title else f"epoque {last['epoch']} - {values}"
        _draw_history(fig, frame, self.metrics, self.best, title)  # dessine les courbes
        return fig  # la figure

    def update(self, epoch: int, **values: Any) -> None:  # ajoute les valeurs d'une epoque et rafraichit l'affichage
        row = {"epoch": epoch}  # nouvelle ligne de l'historique
        # ajoute chaque valeur ; un tenseur PyTorch ou un nombre numpy est converti en nombre Python par .item()
        row.update({k: (float(v.item()) if hasattr(v, "item") else v) for k, v in values.items()})
        self._rows.append(row)  # enregistre la ligne
        if not self._notebook:  # hors notebook (script, terminal)
            # texte "nom valeur | ..." (nombres a 4 chiffres significatifs, le reste tel quel)
            text = " | ".join(f"{k} {v:.4g}" if isinstance(v, int | float) else f"{k} {v}" for k, v in row.items() if k != "epoch")
            # une ligne par epoque, affichee tout de suite
            print(f"epoque {epoch} | {text} | {time.time() - self._start:.0f} s", flush=True)
            return  # pas de figure hors notebook
        if (len(self._rows) - 1) % self.every and len(self._rows) > 1:  # pas une epoque a redessiner (sauf la premiere)
            return  # on attend la prochaine
        self._show(self._render())  # dessine et affiche

    def _show(self, fig: Figure) -> None:  # affiche la figure dans le notebook, a la place de la precedente
        from IPython.display import Image, display  # outils d'affichage de Jupyter

        self.figure = fig  # garde la derniere figure
        image = Image(data=render(fig), format=_STATE["fmt"])  # figure rendue en image (png ou jpeg)
        if self._handle is None:  # premier affichage
            self._handle = display(image, display_id=True)  # affiche et garde une reference pour la remplacer plus tard
        else:  # affichages suivants
            self._handle.update(image)  # remplace l'image dans la meme sortie (pas de nouvelle sortie)

    def close(self) -> Figure | None:  # dernier rendu a la fin de l'entrainement ; renvoie la figure finale (ou None)
        """Dernier rendu (si des epoques ont ete sautees par `every`) ; renvoie la figure finale."""
        if self._rows:  # s'il y a au moins une epoque
            fig = self._render()  # dessine la figure finale
            if self._notebook:  # dans un notebook
                self._show(fig)  # l'affiche
            self.figure = fig  # la garde
        return self.figure  # la figure finale (None si aucune epoque)
