"""Enchainement d'operations de pretraitement, construit depuis la liste `preprocess:` du YAML.

    preprocess:
      - {name: shading_division, sigma: 40}
      - {name: clahe, clip_limit: 2.0}
      - {name: resize, size: [256, 256]}

Le meme objet Pipeline sert a l'entrainement, a la prediction et a l'export : c'est la garantie
que l'image vue par le modele en production est preparee exactement comme a l'entrainement.
`describe()` donne la liste recopiee dans la carte du modele, que le programme C# reproduit.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes

from collections.abc import Mapping, Sequence  # types "dict en lecture" et "suite d'elements" pour les annotations
from typing import Any  # Any = n'importe quel type

import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)

from vtb.preprocess.registry import OpSpec, get_op  # fiche d'une operation ; recherche d'une operation par son nom


class Pipeline:  # suite d'operations appliquees l'une apres l'autre a une image
    """Une suite d'operations nommees, validees a la construction."""

    def __init__(self, steps: Sequence[Mapping[str, Any]] | None = None) -> None:  # construit le pipeline a partir des etapes du YAML
        self.steps: list[tuple[OpSpec, dict[str, Any]]] = []  # liste de (operation, parametres), dans l'ordre
        for step in steps or []:  # pour chaque etape demandee (aucune si None)
            step = dict(step.model_dump() if hasattr(step, "model_dump") else step)  # etape pydantic ou dict -> copie en dict simple
            spec = get_op(step.pop("name"))  # retire le nom du dict et cherche l'operation correspondante
            spec.check(step)  # verifie tout de suite les parametres restants (erreur avant de traiter une image)
            self.steps.append((spec, step))  # memorise l'operation et ses parametres

    def __call__(self, image: np.ndarray) -> np.ndarray:  # permet d'ecrire pipeline(image) ; renvoie l'image transformee
        for spec, params in self.steps:  # pour chaque operation, dans l'ordre
            image = spec.func(image, **params)  # applique l'operation ; sa sortie devient l'entree de la suivante
        return image  # renvoie l'image apres toutes les operations

    def __len__(self) -> int:  # permet d'ecrire len(pipeline) : nombre d'operations
        return len(self.steps)  # nombre d'etapes

    def describe(self) -> list[dict[str, Any]]:  # renvoie la liste des etapes avec tous leurs parametres
        """Liste [{name, parametres...}] avec les valeurs par defaut explicites : rien d'implicite dans la carte du modele."""
        # pour chaque etape : le nom, puis les valeurs par defaut ecrasees par les parametres donnes
        return [{"name": spec.name, **{**spec.defaults(), **params}} for spec, params in self.steps]

    def __repr__(self) -> str:  # texte affiche quand on imprime le pipeline
        return "Pipeline(" + " -> ".join(spec.name for spec, _ in self.steps) + ")"  # ex. Pipeline(clahe -> resize)
