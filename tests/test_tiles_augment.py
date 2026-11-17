"""Tests des tuiles, des augmentations (image, boites, masques) et des defauts synthetiques."""

# Tuiles : on decoupe une grande image en carres qui se chevauchent, on traite chaque carre, puis on recolle.
# Utile quand les defauts sont minuscules par rapport a l'image (une reduction globale les ferait disparaitre).

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[int], X | None) partout

import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)
import pytest  # pytest : lance les tests ; parametrize, approx, raises...
import torch  # PyTorch : tenseurs (tableaux de nombres) et reseaux de neurones
from torchvision import tv_tensors  # types de torchvision qui marquent un tenseur comme boites ou masque

# augment : liste des augmentations, construction d'une augmentation, nettoyage des boites devenues trop petites
from vtb.augment import available_augmentations, build_augmentation, sanitize_boxes

# synthetic : defauts artificiels (tache de Perlin, morceau deplace, rayure) et bruit de Perlin (bruit lisse et organique)
from vtb.augment.synthetic import cutpaste, perlin_anomaly, perlin_noise, synthetic_scratch
from vtb.core.config import AugmentStep  # une etape d'augmentation sous forme de configuration validee

# tiles : une tuile (x, y, largeur, hauteur), decalage des boites, recollage des cartes, grille de tuiles, decoupage
from vtb.preprocess.tiles import Tile, offset_boxes, stitch_maps, tile_grid, tile_image


# Verifie que la grille de tuiles couvre toute l'image, sans tuile coupee, et refuse un chevauchement impossible.
# Attraperait : une bande de l'image jamais analysee (un defaut au bord ne serait jamais vu).
def test_tile_grid_covers_the_image_with_overlap_and_no_partial_tile():
    tiles = tile_grid(300, 1000, tile_size=256, overlap=0.25)  # image 300 x 1000, tuiles de 256, chevauchement 25 %
    covered = np.zeros((300, 1000), bool)  # carte des pixels couverts (tout a faux au depart)
    for tile in tiles:  # pour chaque tuile
        assert tile.width == 256 and tile.height == 256  # jamais de tuile partielle
        covered[tile.slice()] = True  # marque ses pixels comme couverts (slice() donne la zone de la tuile)
    assert covered.all()  # tous les pixels sont couverts
    assert max(t.x + t.width for t in tiles) == 1000 and max(t.y + t.height for t in tiles) == 300  # les tuiles touchent les bords
    assert tile_grid(100, 80, 256) == [Tile(0, 0, 80, 100)]  # image plus petite que la tuile
    with pytest.raises(ValueError):  # on attend une erreur...
        tile_grid(100, 100, 64, overlap=64)  # ...si le chevauchement (64 pixels) egale la taille de la tuile


@pytest.mark.parametrize("mode", ["max", "mean"])  # le test tourne pour les deux facons de fusionner les zones communes
# Verifie que decouper puis recoller une carte redonne exactement la carte d'origine.
# Attraperait : un decalage au recollage, donc des defauts places au mauvais endroit dans l'image entiere.
def test_stitch_rebuilds_the_full_map(mode):
    full = np.random.default_rng(0).random((200, 330)).astype(np.float32)  # carte aleatoire 200 x 330
    tiles, pieces = tile_image(full, 96, overlap=24)  # decoupe en tuiles de 96 avec 24 pixels de chevauchement
    assert np.allclose(stitch_maps(pieces, tiles, 200, 330, mode=mode), full)  # recollage identique a l'original


# Verifie que des boites trouvees dans une tuile sont ramenees dans le repere de l'image entiere.
# Attraperait : des boites dessinees au mauvais endroit (oubli du decalage de la tuile).
def test_offset_boxes_moves_tile_coordinates_to_the_image():
    boxes = offset_boxes(np.array([[1, 2, 11, 12, 0.9]]), Tile(100, 50, 64, 64))  # boite + score dans une tuile placee en (100, 50)
    assert boxes.tolist() == [[101, 52, 111, 62, pytest.approx(0.9)]]  # coins decales de (100, 50) ; score inchange
    assert offset_boxes(np.zeros((0, 4)), Tile(0, 0, 8, 8)).shape == (0, 4)  # aucune boite : tableau vide, sans plantage


# Verifie la construction des augmentations : liste vide, nom ou parametre inconnus, melange config / dictionnaire.
# Attraperait : une faute de frappe dans le YAML ignoree en silence, ou des valeurs de pixels qui sortent de [0, 1].
def test_augmentation_builder_validates_and_returns_none_when_empty():
    assert build_augmentation([]) is None and build_augmentation(None) is None  # rien a faire : None (pas d'augmentation)
    with pytest.raises(ValueError, match="inconnue"):  # on attend une erreur "inconnue"...
        build_augmentation([{"name": "flip"}])  # ...car "flip" n'existe pas (c'est hflip ou vflip)
    with pytest.raises(ValueError, match="parametres invalides"):  # on attend une erreur "parametres invalides"...
        build_augmentation([{"name": "hflip", "prob": 0.5}])  # ...car le parametre s'appelle p, pas prob
    # le catalogue contient au moins ces augmentations courantes
    assert set(available_augmentations()) >= {"hflip", "rot90", "random_resized_crop", "gaussian_noise", "random_erasing"}
    aug = build_augmentation(  # augmentation construite a partir d'une etape de config ET d'un dictionnaire
        # luminosite / contraste tres forts et bruit tres fort, toujours appliques (p = 1)
        [AugmentStep(name="brightness_contrast", brightness=0.9, contrast=0.9, p=1.0), {"name": "gaussian_noise", "sigma": 0.5, "p": 1.0}]
    )
    out = aug(torch.rand(1, 32, 32))  # applique a une image grise aleatoire
    assert out.min() >= 0 and out.max() <= 1  # les augmentations photometriques ne sortent jamais de [0, 1]


# Verifie que les augmentations geometriques deplacent boites et masques exactement comme l'image.
# Attraperait : une boite ou un masque qui ne suit plus l'objet (le modele apprendrait de fausses positions).
def test_geometric_augmentations_move_boxes_and_masks_with_the_image():
    image = torch.zeros(1, 40, 60)  # image grise noire 40 x 60
    image[0, 10:30, 5:20] = 1.0  # rectangle blanc : lignes 10 a 29, colonnes 5 a 19
    # sa boite (x1, y1, x2, y2) dans une image 40 x 60
    boxes = tv_tensors.BoundingBoxes(torch.tensor([[5.0, 10.0, 20.0, 30.0]]), format="XYXY", canvas_size=(40, 60))
    mask = tv_tensors.Mask((image[0] > 0).to(torch.uint8))  # son masque : 1 sur le rectangle, 0 ailleurs
    # retournement horizontal puis rotation de 90 degres, toujours appliques, en detection
    aug = build_augmentation([{"name": "hflip", "p": 1.0}, {"name": "rot90", "p": 1.0}], task="detection")
    torch.manual_seed(0)  # graine fixe
    out_image, out_boxes = aug(image, boxes)  # applique a l'image et a la boite
    ys, xs = torch.nonzero(out_image[0] > 0.5, as_tuple=True)  # coordonnees des pixels blancs apres transformation
    # la boite transformee entoure exactement les pixels blancs
    assert out_boxes[0].tolist() == [xs.min().item(), ys.min().item(), xs.max().item() + 1, ys.max().item() + 1]
    # retournement vertical puis translation (jusqu'a 10 %), en segmentation
    seg = build_augmentation([{"name": "vflip", "p": 1.0}, {"name": "affine", "translate": 0.1, "p": 1.0}], task="segmentation")
    out_image, out_mask = seg(image, mask)  # applique a l'image et au masque
    assert out_mask.dtype == torch.uint8 and set(torch.unique(out_mask).tolist()) <= {0, 1}  # masque toujours en 0 / 1
    assert torch.equal(out_mask.bool(), out_image[0] > 0.5)  # le masque couvre exactement les pixels blancs


# Verifie que les boites devenues trop petites sont retirees AVEC leur etiquette.
# Attraperait : des listes boites / etiquettes desynchronisees (chaque boite recevrait l'etiquette de la suivante).
def test_sanitize_removes_degenerate_boxes_and_their_labels():
    # 2 boites : une normale 10 x 10, et une de 0.5 pixel de large (degeneree)
    boxes = tv_tensors.BoundingBoxes(torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 5.5, 20.0]]), format="XYXY", canvas_size=(32, 32))
    kept, labels = sanitize_boxes(boxes, torch.tensor([3, 4]), min_size=1.0)  # retire les boites de moins de 1 pixel de cote
    assert kept.shape == (1, 4) and labels.tolist() == [3]  # reste la 1re boite, avec son etiquette 3


@pytest.mark.parametrize("make", [perlin_anomaly, cutpaste, synthetic_scratch])  # chaque fabrique de defaut
@pytest.mark.parametrize("dtype,color", [(np.uint8, False), (np.uint8, True), (np.uint16, False)])  # 8 bits gris / couleur, 16 bits gris
# Verifie que chaque defaut synthetique rend une image de meme type et taille, et un masque non vide qui correspond
# aux pixels modifies. Attraperait : un defaut dessine sans masque (etiquette fausse) ou une image passee en 8 bits.
def test_synthetic_defects_return_image_and_mask(make, dtype, color):
    rng = np.random.default_rng(3)  # generateur aleatoire a graine fixe
    # image grise de fond (0.5 +/- 0.05) mise a l'echelle du type : 255 en 8 bits, 65535 en 16 bits
    base = (rng.normal(0.5, 0.05, (80, 100)).clip(0, 1) * (255 if dtype == np.uint8 else 65535)).astype(dtype)
    image = np.stack([base] * 3, axis=-1) if color else base  # version couleur (3 canaux identiques) si demandee
    out, mask = make(image, rng=7)  # fabrique le defaut : renvoie l'image modifiee et son masque
    assert out.dtype == image.dtype and out.shape == image.shape  # meme type, meme taille
    assert mask.shape == image.shape[:2] and mask.dtype == np.uint8 and mask.sum() > 0  # masque 2D en 8 bits, non vide
    changed = np.any(out != image, axis=-1) if color else out != image  # pixels modifies (sur au moins un canal en couleur)
    if make is not synthetic_scratch:  # la rayure est lissee : quelques pixels du bord changent hors du masque binaire
        assert not changed[mask == 0].any()  # aucun pixel modifie hors du masque
    assert changed[mask == 1].mean() > 0.3  # plus de 30 % des pixels du masque ont vraiment change


# Verifie que le bruit de Perlin est reproductible (meme graine = meme bruit) et lisse.
# Attraperait : des defauts synthetiques differents a chaque lancement, ou un bruit "neige" qui ne ressemble a aucun defaut.
def test_perlin_noise_is_smooth_and_reproducible():
    a, b = perlin_noise((64, 80), 4, rng=1), perlin_noise((64, 80), 4, rng=1)  # deux bruits 64 x 80, grille de 4, meme graine
    assert np.array_equal(a, b) and a.shape == (64, 80)  # strictement identiques, et a la bonne taille
    assert np.abs(np.diff(a, axis=1)).max() < 0.2  # lisse : pas de saut entre pixels voisins
