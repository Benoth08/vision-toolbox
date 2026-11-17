"""Style commun des figures : couleurs de classe stables, reglage du notebook, sauvegarde.

    from vtb import viz
    viz.setup_notebook(dpi=80, fmt="jpeg")          # figures legeres dans le .ipynb
    colors = viz.class_colors(["crack", "hole"])     # {nom: (r, g, b)}, stable d'un appel a l'autre
    viz.save(fig, "outputs/figure.jpg")

Le module ne choisit pas de backend matplotlib (pas de matplotlib.use) : dans un notebook les
figures s'affichent a la fin de la cellule, dans un script on les ecrit avec `save=`.
Dans un notebook, une figure s'affiche une seule fois, que l'appel soit en derniere ligne de
cellule ou non (VizFigure / VizAxes._ipython_display_) ; pas besoin de plt.show().
"""

from __future__ import annotations  # permet les annotations de type modernes (int | None, list[...])

import colorsys  # conversions entre systemes de couleurs (ici teinte / saturation / valeur -> RGB)
from collections.abc import Sequence  # Sequence : type "liste ou tuple" pour les annotations
from pathlib import Path  # Path : chemin de fichier independant du systeme
from typing import Any  # Any : n'importe quel type

import matplotlib.pyplot as plt  # pyplot : l'interface de dessin de matplotlib (figures, axes)
import numpy as np  # numpy : calcul sur des tableaux de nombres
from matplotlib.axes import Axes  # Axes : une zone de dessin (un graphique) dans une figure
from matplotlib.figure import Figure  # Figure : la page entiere qui contient un ou plusieurs graphiques

# tab10 reordonne (pas de vert ni de gris dans les premieres classes : le vert vif est la verite
# terrain, le gris disparait sur l'acier), puis les teintes claires de tab20 ; au-dela, teintes au
# nombre d'or. RGB 0-255.
_PALETTE: list[tuple[int, int, int]] = [  # couleurs des classes, dans l'ordre : classe 0, classe 1, ...
    (31, 119, 180),  # bleu
    (255, 127, 14),  # orange
    (214, 39, 40),  # rouge
    (148, 103, 189),  # violet
    (23, 190, 207),  # cyan
    (227, 119, 194),  # rose
    (140, 86, 75),  # brun
    (188, 189, 34),  # olive
    (44, 160, 44),  # vert
    (127, 127, 127),  # gris
    (174, 199, 232),  # bleu clair
    (255, 187, 120),  # orange clair
    (255, 152, 150),  # rouge clair
    (197, 176, 213),  # violet clair
    (158, 218, 229),  # cyan clair
    (247, 182, 210),  # rose clair
    (196, 156, 148),  # brun clair
    (219, 219, 141),  # olive clair
    (152, 223, 138),  # vert clair
    (199, 199, 199),  # gris clair
]
TRUTH_COLOR = (0, 255, 0)  # verite terrain : vert vif, hors palette
OK_COLOR = (40, 170, 60)  # vert : prediction correcte
KO_COLOR = (220, 40, 40)  # rouge : prediction fausse

# reglages globaux d'affichage (format d'image, resolution), modifies par setup_notebook
_STATE: dict[str, Any] = {"fmt": "png", "dpi": None}


def _color(index: int) -> tuple[int, int, int]:  # couleur de la classe numero index ; renvoie (r, g, b) en 0-255
    if index < len(_PALETTE):  # si la palette a assez de couleurs
        return _PALETTE[index]  # on prend celle de la palette
    hue = (index * 0.618033988749895) % 1.0  # sinon teinte au nombre d'or : des teintes bien reparties, jamais identiques
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)  # teinte + saturation + luminosite fixes -> r, g, b entre 0 et 1
    return int(r * 255), int(g * 255), int(b * 255)  # passage en 0-255


# couleurs pour n classes (liste) ou pour des noms de classes (dictionnaire nom -> couleur)
def class_colors(n_or_names: int | Sequence[str]) -> list[tuple[int, int, int]] | dict[str, tuple[int, int, int]]:
    """Couleurs RGB 0-255 : une liste pour un entier, un dict nom -> couleur pour une liste de noms."""
    if isinstance(n_or_names, int | np.integer):  # si on a recu un nombre (entier Python ou numpy)
        return [_color(i) for i in range(int(n_or_names))]  # liste de n couleurs
    return {str(name): _color(i) for i, name in enumerate(n_or_names)}  # sinon une couleur par nom, dans l'ordre des noms


def to_mpl(color: Any) -> Any:  # convertit une couleur 0-255 au format de matplotlib (0-1) ; renvoie la couleur convertie
    """Couleur RGB 0-255 -> tuple 0-1 pour matplotlib (les noms 'red', '#ff0000' passent tels quels)."""
    if isinstance(color, tuple | list | np.ndarray) and len(color) in (3, 4) and max(color) > 1:  # suite de 3 ou 4 nombres en 0-255
        return tuple(float(c) / 255.0 for c in color)  # chaque composante divisee par 255
    return color  # deja au bon format (nom, code hexadecimal, ou 0-1)


def to_rgb255(color: Any) -> tuple[int, int, int]:  # convertit n'importe quelle couleur en (r, g, b) 0-255
    """Couleur quelconque (nom matplotlib, '#rrggbb', tuple 0-1 ou 0-255) -> tuple RGB 0-255."""
    if isinstance(color, tuple | list | np.ndarray) and len(color) in (3, 4):  # si c'est une suite de 3 ou 4 nombres
        values = [float(c) for c in color[:3]]  # garde r, g, b (ignore la transparence)
        if max(values) <= 1.0:  # si les valeurs sont en 0-1
            values = [v * 255 for v in values]  # passage en 0-255
        return int(round(values[0])), int(round(values[1])), int(round(values[2]))  # arrondi en entiers
    from matplotlib.colors import to_rgb  # convertit un nom ou un code '#rrggbb' en (r, g, b) 0-1

    r, g, b = to_rgb(color)  # conversion par matplotlib
    return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))  # passage en 0-255


def in_notebook() -> bool:  # renvoie vrai si le code tourne dans un notebook Jupyter
    """Vrai dans un noyau Jupyter (notebook, VS Code, nbclient)."""
    try:  # IPython n'est pas forcement installe
        from IPython import get_ipython  # donne l'interpreteur IPython en cours, ou None
    except ImportError:  # si IPython est absent
        return False  # on n'est certainement pas dans un notebook
    shell = get_ipython()  # l'interpreteur en cours
    return shell is not None and type(shell).__name__ == "ZMQInteractiveShell"  # ce type precis = noyau de notebook


def setup_notebook(dpi: int = 80, fmt: str = "png") -> None:  # regle resolution et format des figures ; ne renvoie rien
    """Regle la resolution et le format des figures du notebook ("jpeg" : .ipynb bien plus leger)."""
    _STATE["fmt"] = "jpeg" if fmt in ("jpg", "jpeg") else fmt  # "jpg" et "jpeg" donnent le meme format
    _STATE["dpi"] = dpi  # resolution memorisee pour render
    plt.rcParams["figure.dpi"] = dpi  # resolution par defaut de toutes les nouvelles figures
    if not in_notebook():  # hors notebook
        return  # rien d'autre a regler
    try:  # reglage du backend inline (affichage dans le notebook), qui peut manquer
        from matplotlib_inline.backend_inline import set_matplotlib_formats  # choisit le format des figures affichees

        set_matplotlib_formats(_STATE["fmt"])  # applique le format (png ou jpeg)
    except Exception:  # noqa: BLE001 - reglage de confort, jamais bloquant
        pass  # on ignore l'erreur : les figures s'affichent quand meme


def save(fig: Figure, path: str | Path, dpi: int = 120) -> Path:  # enregistre la figure dans un fichier ; renvoie son chemin
    """Ecrit la figure (dossier cree au besoin) ; jpeg en qualite 85."""
    path = Path(path)  # accepte un texte ou un Path
    path.parent.mkdir(parents=True, exist_ok=True)  # cree le dossier de destination s'il manque
    kwargs: dict[str, Any] = {"dpi": dpi, "bbox_inches": "tight"}  # resolution ; "tight" coupe les marges vides
    if path.suffix.lower() in (".jpg", ".jpeg"):  # si le fichier est un JPEG
        kwargs["pil_kwargs"] = {"quality": 85}  # qualite 85 : bon compromis taille / nettete
    fig.savefig(path, **kwargs)  # ecrit le fichier
    return path  # le chemin ecrit


class VizFigure(Figure):  # Figure qui sait s'afficher une seule fois dans un notebook
    """Figure affichee UNE fois dans un notebook, qu'elle soit renvoyee en derniere ligne ou non.

    A l'affichage (fin de cellule, plt.show, display ou derniere expression), elle est rendue puis
    retiree de pyplot : le backend inline ne la remontre pas une seconde fois.
    """

    def _ipython_display_(self) -> None:  # methode speciale appelee par Jupyter pour afficher l'objet
        from IPython.display import Image, display  # Image : image a afficher ; display : l'affiche dans la cellule

        display(Image(data=render(self), format=_STATE["fmt"]))  # rend la figure en image et l'affiche
        number = getattr(self, "number", None)  # numero de la figure dans pyplot (None si elle n'y est pas)
        if number is not None and plt.fignum_exists(number):  # si pyplot la connait encore
            plt.close(self)  # on la retire : elle ne sera pas affichee une deuxieme fois


class VizAxes(Axes):  # Axes qui affiche sa figure quand on le renvoie en derniere ligne d'une cellule
    """Axes dont l'affichage (derniere ligne d'une cellule) montre la figure au lieu de '<Axes: ...>'."""

    def _ipython_display_(self) -> None:  # appele par Jupyter pour afficher l'objet
        figure = getattr(self.figure, "figure", self.figure)  # sous-figure -> figure racine
        if isinstance(figure, VizFigure):  # si c'est une VizFigure
            figure._ipython_display_()  # elle sait s'afficher une seule fois
            return  # fini
        from IPython.display import Image, display  # outils d'affichage de Jupyter

        display(Image(data=render(figure), format=_STATE["fmt"]))  # sinon on rend et affiche la figure nous-memes


def render(fig: Figure, fmt: str | None = None, dpi: int | None = None) -> bytes:  # rend une figure en image ; renvoie les octets
    """Rend une figure en octets (png ou jpeg), sans pyplot."""
    import io  # io.BytesIO : un "fichier" en memoire

    buffer = io.BytesIO()  # fichier en memoire qui recevra l'image
    fmt = fmt or _STATE["fmt"]  # format demande, sinon celui du reglage global
    # format, resolution (demandee, sinon reglage global, sinon celle de la figure), marges coupees
    kwargs: dict[str, Any] = {"format": fmt, "dpi": dpi or _STATE["dpi"] or fig.dpi, "bbox_inches": "tight"}
    if fmt == "jpeg":  # si JPEG
        kwargs["pil_kwargs"] = {"quality": 85}  # qualite 85
    fig.savefig(buffer, **kwargs)  # ecrit l'image dans le fichier en memoire
    return buffer.getvalue()  # les octets de l'image


def new_figure(figsize: tuple[float, float], **kwargs: Any) -> Figure:  # cree une figure vide ; renvoie une VizFigure
    """plt.figure avec VizFigure."""
    return plt.figure(figsize=figsize, FigureClass=VizFigure, **kwargs)  # FigureClass : pyplot cree une VizFigure au lieu d'une Figure


def new_axes(ax: Any = None, figsize: tuple[float, float] = (5, 4)) -> tuple[Figure, Any]:  # renvoie (figure, axes) prets a dessiner
    """Renvoie (figure, axes) : l'axe donne, ou une nouvelle figure a un axe."""
    if ax is not None:  # si l'appelant fournit deja un axe (pour dessiner dans sa propre figure)
        return ax.figure, ax  # on l'utilise
    # sinon nouvelle figure avec un seul axe, des classes Viz pour un affichage unique dans le notebook
    fig, ax = plt.subplots(figsize=figsize, FigureClass=VizFigure, subplot_kw={"axes_class": VizAxes})
    return fig, ax  # la figure et son axe


def finish(fig: Figure, save_path: str | Path | None) -> None:  # fin commune des fonctions de dessin : sauvegarde eventuelle
    """Ecrit la figure si `save` est donne ; hors notebook, la libere de pyplot (pas de fuite)."""
    if save_path is None:  # si aucun fichier n'est demande
        return  # rien a faire
    save(fig, save_path)  # ecrit la figure
    number = getattr(fig, "number", None)  # numero de la figure dans pyplot
    if not in_notebook() and number is not None and plt.fignum_exists(number):  # hors notebook et toujours ouverte dans pyplot
        plt.close(fig)  # la ferme pour liberer la memoire (sinon les figures s'accumulent)
