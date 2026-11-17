"""Registre des operations de pretraitement : chaque operation est une fonction nommee.

Une operation suit toujours la meme signature : `func(image, **parametres) -> image`, ou image est
un ndarray (H, W) ou (H, W, 3) RGB, uint8 ou uint16, et la sortie garde le meme type (sauf mention
contraire dans la docstring). C'est ce qui permet de les enchainer librement depuis le YAML et de
les reproduire a l'identique en C# (OpenCvSharp expose les memes fonctions OpenCV).

    @register("clahe", family="contraste")
    def clahe(image, clip_limit=2.0, tile_grid=8): ...
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes

import inspect  # lit la signature d'une fonction (noms et valeurs par defaut de ses parametres)
from collections.abc import Callable  # type "fonction" pour les annotations
from dataclasses import dataclass  # @dataclass : cree une classe de donnees sans ecrire __init__ a la main
from typing import Any  # Any = n'importe quel type

import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)
import pandas as pd  # pandas : tableaux de donnees (ici le catalogue des operations)

Operation = Callable[..., np.ndarray]  # alias de type : une operation = une fonction qui renvoie une image


@dataclass(frozen=True)  # frozen=True : les champs ne peuvent plus etre modifies apres creation
class OpSpec:  # fiche d'une operation enregistree (nom, famille, fonction, resume)
    """Une operation enregistree : nom, famille (pour le catalogue), fonction, resume d'une ligne."""

    name: str  # nom utilise dans le YAML (ex. "clahe")
    family: str  # famille pour ranger le catalogue (ex. "contraste")
    func: Operation  # la fonction qui fait le travail
    summary: str  # premiere ligne de sa docstring (description courte)

    def defaults(self) -> dict[str, Any]:  # renvoie les parametres reglables et leur valeur par defaut
        """Parametres reglables et leur valeur par defaut (lus dans la signature : une seule source de verite)."""
        params = list(inspect.signature(self.func).parameters.values())[1:]  # parametres de la fonction, sauf le 1er (l'image)
        return {p.name: (None if p.default is inspect.Parameter.empty else p.default) for p in params}  # nom -> defaut (None : obligatoire)

    def check(self, params: dict[str, Any]) -> None:  # verifie les parametres ; leve une erreur s'ils ne conviennent pas
        """Refuse un parametre inconnu ou manquant avant de traiter la moindre image."""
        try:  # on essaie d'associer les parametres a la signature, sans executer la fonction
            inspect.signature(self.func).bind(np.zeros((1, 1), np.uint8), **params)  # image factice 1x1 + parametres donnes
        except TypeError as error:  # si un parametre est inconnu ou manquant
            # erreur claire qui liste les parametres acceptes ; "from error" garde la cause d'origine
            raise ValueError(f"parametres invalides pour {self.name} : {error} ; parametres acceptes : {self.defaults()}") from error


REGISTRY: dict[str, OpSpec] = {}  # le registre : nom de l'operation -> sa fiche (rempli par @register a l'import)


def register(name: str, family: str) -> Callable[[Operation], Operation]:  # cree un decorateur qui inscrit une fonction dans le registre
    """Decorateur : inscrit la fonction dans le registre sous ce nom."""

    def decorator(func: Operation) -> Operation:  # recoit la fonction decoree, l'inscrit et la renvoie inchangee
        if name in REGISTRY:  # si ce nom est deja pris
            raise ValueError(f"operation deja enregistree : {name}")  # erreur : deux operations ne peuvent pas avoir le meme nom
        summary = (inspect.getdoc(func) or "").splitlines()[0] if func.__doc__ else ""  # premiere ligne de la docstring (ou vide)
        REGISTRY[name] = OpSpec(name, family, func, summary)  # inscrit la fiche dans le registre
        return func  # renvoie la fonction telle quelle (elle reste utilisable directement)

    return decorator  # renvoie le decorateur (utilise sous la forme @register("nom", family="..."))


def get_op(name: str) -> OpSpec:  # renvoie la fiche d'une operation a partir de son nom
    if name not in REGISTRY:  # si le nom n'existe pas (faute de frappe dans le YAML ?)
        raise ValueError(f"pretraitement inconnu : {name} ; disponibles : {sorted(REGISTRY)}")  # erreur avec la liste des noms valides
    return REGISTRY[name]  # renvoie la fiche trouvee


def list_ops() -> pd.DataFrame:  # renvoie le catalogue des operations sous forme de tableau
    """Le catalogue sous forme de tableau : famille, nom, resume, parametres par defaut."""
    # une ligne par operation : famille, nom, resume, parametres par defaut
    rows = [{"family": s.family, "name": s.name, "summary": s.summary, "defaults": s.defaults()} for s in REGISTRY.values()]
    return pd.DataFrame(rows).sort_values(["family", "name"]).reset_index(drop=True)  # tableau trie par famille puis nom
