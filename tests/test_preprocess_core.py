"""Tests du registre, du pipeline, de la mise en tenseur et de la correction d'illumination."""

# Pretraitement : operations appliquees a chaque image AVANT le modele (redimension, correction d'eclairage...).
# Le meme pretraitement doit etre refait a l'identique en production, d'ou la fiche du modele (model card).

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[int], X | None) partout

import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)
import pytest  # pytest : lance les tests ; parametrize, approx, raises...
import torch  # PyTorch : tenseurs (tableaux de nombres, sur CPU ou GPU)

# card : fiche du modele (entree, sorties, pretraitement, classes) et sa lecture / ecriture en JSON
from vtb.export.card import InputSpec, ModelCard, OutputSpec, read_card, write_card
from vtb.preprocess import Pipeline, list_ops  # Pipeline : suite d'operations ; list_ops : tableau des operations disponibles
from vtb.preprocess.ops.illumination import apply_vignetting  # ajoute un vignettage artificiel (bords assombris)

# tensor : statistiques par canal, image -> entree du reseau, normalisation, image -> tenseur flottant (canaux, hauteur, largeur)
from vtb.preprocess.tensor import channel_stats, image_to_input, normalize, to_float_chw


# fabrique une image test et la renvoie : fond regulier, donc tout ecart d'eclairage se voit
def _textured(rng: np.random.Generator, shape=(128, 160)) -> np.ndarray:
    """Fond uniforme gris moyen avec un fin grain et quelques petits objets clairs."""
    image = np.clip(rng.normal(130, 6, shape), 0, 255).astype(np.uint8)  # gris 130 avec un leger grain, en 8 bits
    image[40:46, 50:56] = 230  # petit objet clair 6 x 6
    image[90:94, 120:126] = 230  # autre petit objet clair 4 x 6
    return image  # renvoie l'image


# Verifie que le pipeline refuse un nom d'operation ou un parametre inconnu, applique la bonne taille et se decrit lui-meme.
# Attraperait : une faute de frappe dans le YAML ignoree en silence (le pretraitement voulu ne serait pas fait).
def test_pipeline_validates_names_and_params():
    with pytest.raises(ValueError, match="inconnu"):  # on attend une erreur "inconnu"...
        Pipeline([{"name": "nope"}])  # ...pour une operation qui n'existe pas
    with pytest.raises(ValueError, match="parametres invalides"):  # on attend une erreur "parametres invalides"...
        Pipeline([{"name": "resize", "sise": 64}])  # faute de frappe dans un parametre
    pipe = Pipeline([{"name": "resize", "size": [32, 40]}])  # pipeline valide : redimension en 32 x 40 (hauteur, largeur)
    assert pipe(np.zeros((64, 80), np.uint8)).shape == (32, 40)  # l'image 64 x 80 sort en 32 x 40
    # describe() rend la liste complete des reglages, y compris les valeurs par defaut (interpolation "auto")
    assert pipe.describe() == [{"name": "resize", "size": [32, 40], "interpolation": "auto"}]
    assert {"resize", "shading_division"} <= set(list_ops()["name"])  # le registre contient au moins ces operations


# le test tourne pour les 3 methodes de correction d'eclairage
@pytest.mark.parametrize("op", ["shading_division", "shading_subtraction", "polynomial_background"])
# Verifie que chaque correction efface un vignettage (centre clair, coins sombres). Attraperait : une correction qui
# ne corrige rien, ou qui change le type ou la taille de l'image.
def test_illumination_corrections_flatten_a_vignetted_background(rng, op):  # rng : generateur aleatoire de conftest.py
    clean = _textured(rng)  # image test bien eclairee
    vignetted = apply_vignetting(clean, strength=0.6)  # la meme, avec des bords assombris
    params = {"sigma": 16} if op.startswith("shading") else {}  # sigma ~ 1/8 de la taille de cette petite image
    corrected = Pipeline([{"name": op, **params}])(vignetted)  # applique la correction
    assert corrected.dtype == np.uint8 and corrected.shape == clean.shape  # meme type 8 bits et meme taille

    # mesure l'ecart de luminosite entre le centre et les coins ; renvoie un nombre positif
    def border_center_gap(image: np.ndarray) -> float:
        center = image[48:80, 64:96].mean()  # luminosite moyenne d'un carre central
        # moyenne des 4 coins (carres de 16 x 16)
        corners = np.mean([image[:16, :16].mean(), image[-16:, -16:].mean(), image[:16, -16:].mean(), image[-16:, :16].mean()])
        return abs(float(center) - float(corners))  # ecart absolu centre / coins

    # Le vignettage cree un fort ecart centre / coins ; la correction doit le reduire nettement.
    assert border_center_gap(vignetted) > 30  # le vignettage est bien visible
    assert border_center_gap(corrected) < 0.35 * border_center_gap(vignetted)  # apres correction : ecart reduit de plus de 65 %


# Verifie que les corrections gardent les 16 bits et fonctionnent sur une image couleur.
# Attraperait : une image 16 bits ecrasee en 8 bits (perte de precision) ou un plantage sur 3 canaux.
def test_illumination_works_in_16_bits_and_color(rng, image16):  # image16 : image 16 bits de conftest.py
    out16 = Pipeline([{"name": "shading_division", "sigma": 20}])(image16)  # correction sur l'image 16 bits
    assert out16.dtype == np.uint16  # le resultat reste en 16 bits
    color = np.stack([_textured(rng)] * 3, axis=-1)  # image couleur : la meme image grise empilee sur 3 canaux
    # correction de profil de colonnes puis fond morphologique (noyau 15) : la taille de l'image ne change pas
    assert Pipeline([{"name": "column_profile"}, {"name": "morphological_background", "kernel": 15}])(color).shape == color.shape


# Verifie la mise en tenseur (image -> nombres entre 0 et 1, canaux en premier), la normalisation et les statistiques.
# Attraperait : des canaux dans le mauvais ordre ou des valeurs mal mises a l'echelle (le modele recevrait n'importe quoi).
def test_tensor_conversion_and_stats():
    image = np.full((4, 5, 3), 255, np.uint8)  # image couleur blanche 4 x 5 (hauteur, largeur, canaux)
    tensor = to_float_chw(image)  # tenseur flottant (canaux, hauteur, largeur), valeurs dans [0, 1]
    assert tensor.shape == (3, 4, 5) and torch.allclose(tensor, torch.ones_like(tensor))  # canaux en premier ; 255 devient 1.0
    assert to_float_chw(np.zeros((4, 5), np.uint16)).shape == (1, 4, 5)  # image grise 16 bits : 1 canal ajoute
    # normalisation : (valeur - moyenne) / ecart-type = (1 - 0.5) / 0.5 = 1
    assert normalize(tensor, [0.5] * 3, [0.5] * 3).max() == pytest.approx(1.0)
    assert image_to_input(image, [0, 0, 0], [1, 1, 1]).shape == (1, 3, 4, 5)  # entree du reseau : dimension de lot (1) ajoutee
    # moyenne et ecart-type par canal sur deux images : une noire, une blanche
    mean, std = channel_stats([np.zeros((2, 2), np.uint8), np.full((2, 2), 255, np.uint8)])
    assert mean == pytest.approx([0.5]) and std == pytest.approx([0.5])  # moitie noir / moitie blanc : moyenne 0.5, ecart 0.5


# Verifie que la fiche du modele s'ecrit puis se relit sans rien perdre. Attraperait : une information de deploiement
# perdue (normalisation, classes...), donc un modele mal utilise en production.
def test_model_card_roundtrip(tmp_path):
    card = ModelCard(  # fiche complete d'un modele de classification
        task="classification",  # type de tache
        model_name="resnet18",  # architecture
        # entree : 128 x 128, 3 canaux RGB, moyenne et ecart-type de normalisation par canal
        input=InputSpec(height=128, width=128, channels=3, color="RGB", mean=[0.5] * 3, std=[0.2] * 3),
        preprocess=Pipeline([{"name": "resize", "size": 128}]).describe(),  # pretraitement a refaire en production
        # sortie : logits = scores bruts par classe, avant softmax (la fonction qui les change en probabilites)
        outputs=[OutputSpec(name="logits", shape=["batch", 6], description="scores avant softmax")],
        class_names=list("abcdef"),  # 6 classes nommees a, b, c, d, e, f
    )
    write_card(card, tmp_path / "model_card.json")  # ecrit la fiche en JSON
    assert read_card(tmp_path / "model_card.json") == card  # relue, elle est identique
