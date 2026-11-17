"""Export ONNX : un lot fige dans un reshape doit etre detecte, meme quand le graphe tourne avec un autre lot."""

# ONNX : format de fichier standard pour un modele, lisible sans PyTorch (C#, ONNX Runtime, OpenVINO...).
# Lot (batch) : nombre d'images traitees ensemble. Un export "a lot dynamique" accepte n'importe quel nombre
# d'images ; mais si le code du modele ecrit ce nombre en dur, le fichier exporte donne des resultats faux.

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[int], X | None) partout

import torch  # PyTorch : tenseurs (tableaux de nombres) et reseaux de neurones
import torch.nn as nn  # nn : briques pour construire des reseaux (nn.Module est la classe de base d'un modele)

# onnx : taille de lot inscrite dans un fichier ONNX (None = dynamique), session d'execution ONNX Runtime, export
from vtb.export.onnx import batch_size_of, create_session, export_onnx


class FrozenReshape(nn.Module):  # modele piege : il ecrit la taille de lot (2) en dur dans un reshape
    """Le lot (2) est ecrit en dur : avec 3 images, le graphe tourne mais rend 2 sorties."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # calcul du modele : une somme par image (en apparence)
        return x.reshape(2, -1).sum(dim=1)  # remet en forme en "2 lignes" quel que soit le lot, puis somme chaque ligne


class Flexible(nn.Module):  # modele correct : il garde la taille de lot telle quelle
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # calcul du modele : la somme des pixels de chaque image
        return x.flatten(1).sum(dim=1)  # flatten(1) aplatit tout sauf la 1re dimension (le lot), puis somme par image


# Verifie que l'export detecte un lot fige et se replie sur un lot fixe de 1 (sur, a defaut d'etre souple).
# Attraperait : un fichier ONNX qui, avec 3 images, renverrait 2 resultats sans aucune erreur en production.
def test_frozen_batch_falls_back_to_fixed_batch(tmp_path):
    example = torch.rand(2, 1, 2, 3)  # exemple d'entree : 2 images, 1 canal, 2 x 3 pixels (sert a tracer le modele)
    path = export_onnx(FrozenReshape(), example, tmp_path / "frozen.onnx", ["out"])  # export ONNX avec une sortie nommee "out"
    assert batch_size_of(path) == 1  # le fichier a un lot fixe de 1 : le piege a ete detecte


# Verifie qu'un modele correct garde un lot dynamique, et qu'ONNX Runtime l'execute avec un autre nombre d'images.
# Attraperait : un controle trop severe qui figerait le lot de tous les modeles (perte de vitesse en traitement par lot).
def test_dynamic_batch_kept_when_outputs_follow(tmp_path):
    example = torch.rand(2, 1, 2, 3)  # exemple d'entree : 2 images
    path = export_onnx(Flexible(), example, tmp_path / "flex.onnx", ["out"])  # export du modele correct
    assert batch_size_of(path) is None  # None : taille de lot dynamique (n'importe quel nombre d'images)
    session = create_session(path)  # ouvre le fichier avec ONNX Runtime (le moteur qui execute les modeles ONNX)
    # execute avec 5 images (un autre lot qu'a l'export) ; l'entree est un tableau numpy ; [0] = premiere sortie
    out = session.run(None, {session.get_inputs()[0].name: torch.rand(5, 1, 2, 3).numpy()})[0]
    assert out.shape == (5,)  # une valeur par image : 5 sorties
