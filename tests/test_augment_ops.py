"""Tests du module d'augmentation : chaque operation (image seule, boites, masque), presets, melanges, outils."""

# Augmentation : transformations aleatoires (retournement, rotation, bruit, luminosite...) appliquees aux images
# d'entrainement pour que le modele voie plus de variete. Regle d'or : si l'image bouge, ses cibles (boites,
# masque) doivent bouger exactement pareil, sinon le modele apprend des etiquettes fausses.

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[int], X | None) partout

import inspect  # inspect : examiner une fonction (par exemple la liste de ses parametres)
import pickle  # pickle : transformer un objet Python en octets et inversement (utilise pour l'envoyer a un autre processus)
import re  # re : expressions regulieres, pour chercher des motifs dans du texte
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)
import pytest  # pytest : lance les tests ; parametrize, raises, approx...
import torch  # PyTorch : tenseurs (tableaux de nombres, sur CPU ou GPU) et reseaux de neurones
from torch.utils.data import DataLoader, TensorDataset  # DataLoader : decoupe un jeu en lots ; TensorDataset : jeu fait de tenseurs
from torchvision import tv_tensors  # types de torchvision qui marquent un tenseur comme boites ou masque (pour qu'il suive l'image)
from torchvision.transforms import v2  # transformations d'images de torchvision (version 2, qui gere aussi boites et masques)

import vtb.augment as augment  # le module teste, importe en entier (pour les fonctions appelees via augment.xxx)
from vtb.augment import (  # les elements du module utilises directement
    OPS,  # registre de toutes les operations d'augmentation : nom -> fabrique, famille, taches permises
    PRESETS,  # registre des presets : recettes toutes faites d'augmentations
    CopyPaste,  # colle des defauts de la banque sur une image, avec une probabilite p
    DefectBank,  # banque de defauts decoupes dans des images, a recoller ailleurs
    batch_augmentation,  # MixUp / CutMix : melange les images d'un lot entier
    benchmark_ops,  # mesure la vitesse de chaque operation
    build_augmentation,  # construit l'augmentation a partir d'une liste d'etapes (comme dans le YAML)
    check_alignment,  # verifie que boites et masque suivent l'image
    copy_paste,  # fonction de collage de defauts (version directe de CopyPaste)
    describe,  # tableau qui decrit les etapes d'une augmentation
    expand_steps,  # remplace un preset par la liste de ses etapes
    mosaic,  # assemble 4 images en une (mosaique facon YOLO)
    synthesize_defect,  # dessine un faux defaut sur une image saine
)
from vtb.augment.tools import alignment_pattern  # image test (rectangle, L, disque) avec ses boites et son masque
from vtb.core.config import AugmentStep  # une etape d'augmentation sous forme de configuration validee


# renvoie les parametres d'une etape qui s'applique a coup sur, pour pouvoir tester son effet
def _forced(name: str, **extra) -> dict:
    """Etape qui s'applique a coup sur (p = 1) ; taille de sortie fixee pour random_resized_crop."""
    params = {"name": name, **extra}  # nom de l'operation + parametres supplementaires eventuels
    if "p" in inspect.signature(OPS[name].builder).parameters:  # si l'operation a une probabilite p d'etre appliquee
        params["p"] = 1.0  # on la force a 1 : toujours appliquee
    if name == "random_resized_crop":  # ce recadrage a besoin d'une taille de sortie
        params["size"] = (64, 80)  # on garde la taille de l'image test (64 x 80)
    if name == "scale_jitter":  # idem pour le changement d'echelle
        params["target_size"] = (64, 80)  # taille de reference
    return params  # renvoie le dictionnaire de parametres


# renvoie les noms des operations permises pour une tache (classification, detection ou segmentation)
def _ops_for(task: str) -> list[str]:
    return [name for name, info in OPS.items() if task in info.tasks]  # parcourt le registre et garde celles de la tache


# deux parametrize empiles : le test tourne pour chaque operation ET pour 1 canal (gris) puis 3 canaux (couleur)
@pytest.mark.parametrize("channels", [1, 3])  # image grise puis couleur
@pytest.mark.parametrize("name", _ops_for("classification"))  # chaque operation permise en classification
# Verifie que chaque operation, appliquee a une image seule, garde le type, le nombre de canaux et les valeurs dans [0, 1].
# Attraperait : une operation qui transforme une image grise en couleur, ou qui sort des valeurs hors plage.
def test_every_operation_on_an_image_alone(name, channels):
    torch.manual_seed(0)  # graine fixe : tirages aleatoires reproductibles
    image = torch.rand(channels, 64, 80)  # image aleatoire : canaux x hauteur x largeur, valeurs entre 0 et 1
    out = build_augmentation([_forced(name)], "classification")(image)  # construit l'operation forcee et l'applique
    assert out.dtype == torch.float32 and out.shape[0] == channels  # toujours des flottants 32 bits, meme nombre de canaux
    assert out.min() >= 0 and out.max() <= 1  # valeurs restees entre 0 et 1


@pytest.mark.parametrize("name", _ops_for("detection"))  # chaque operation permise en detection
# Verifie que les boites restent des boites valides (meme nombre, meme ordre, x2 >= x1, dans l'image).
# Attraperait : une boite perdue (decalage etiquette / boite), inversee ou sortant de l'image.
def test_every_operation_with_boxes(name):
    torch.manual_seed(1)  # graine fixe
    image, boxes, _ = alignment_pattern((64, 80), 3)  # image test couleur 64 x 80 et ses 3 boites (masque ignore)
    out, out_boxes = build_augmentation([_forced(name)], "detection")(image, boxes)  # applique a l'image ET aux boites
    assert isinstance(out_boxes, tv_tensors.BoundingBoxes) and out_boxes.shape == (3, 4)  # meme nombre, meme ordre
    b = out_boxes.as_subclass(torch.Tensor)  # vue en tenseur simple (x1, y1, x2, y2 par ligne) pour les calculs
    height, width = out.shape[-2:]  # taille de l'image apres augmentation
    assert (b[:, 2] >= b[:, 0]).all() and (b[:, 3] >= b[:, 1]).all()  # x2 >= x1 et y2 >= y1 : boites dans le bon sens
    assert b.min() >= 0 and (b[:, [0, 2]] <= width).all() and (b[:, [1, 3]] <= height).all()  # ClampBoundingBoxes final
    assert out.min() >= 0 and out.max() <= 1  # image toujours dans [0, 1]


@pytest.mark.parametrize("name", _ops_for("segmentation"))  # chaque operation permise en segmentation
# Verifie que le masque garde son type entier, la taille de l'image et uniquement des numeros de classe existants.
# Attraperait : une interpolation qui cree des classes "1.5" (inexistantes) en melangeant deux classes voisines.
def test_every_operation_with_a_mask(name):
    torch.manual_seed(2)  # graine fixe
    image, _, mask = alignment_pattern((64, 80), 1)  # image test grise et son masque (classes 0 a 3)
    out, out_mask = build_augmentation([_forced(name)], "segmentation")(image, mask)  # applique a l'image ET au masque
    assert isinstance(out_mask, tv_tensors.Mask) and out_mask.dtype == torch.int64  # reste un masque d'entiers
    assert out_mask.shape[-2:] == out.shape[-2:]  # meme hauteur et largeur que l'image
    allowed = {0, 1, 2, 3}  # fond + les 3 classes de l'image test
    assert set(torch.unique(out_mask).tolist()) <= allowed  # jamais de classe intermediaire (plus proche voisin)


# operations geometriques (qui deplacent les pixels) utilisables en segmentation : ce sont elles qui risquent de desaligner
GEOMETRIC = [n for n, info in OPS.items() if info.family == "geometrie" and "segmentation" in info.tasks]


@pytest.mark.parametrize("name", GEOMETRIC)  # chaque operation geometrique
# Verifie, sur plusieurs tirages, que boites et masque restent superposes a l'image.
# Attraperait : une rotation ou un recadrage appliques a l'image mais pas (ou mal) aux cibles.
def test_geometric_operations_keep_boxes_and_masks_aligned(name):
    result = check_alignment([_forced(name)], "detection", size=(64, 80), trials=8)  # 8 tirages sur l'image test
    assert result["ok"], result["trials"]  # en cas d'echec, pytest affiche le tableau des tirages pour comprendre


# Verifie que le controleur d'alignement detecte bien une operation fautive (et accepte une operation correcte).
# Attraperait : un controleur trop indulgent, qui laisserait passer de vraies erreurs.
def test_check_alignment_catches_an_operation_that_forgets_the_targets():
    class ImageOnlyFlip(v2.Transform):  # erreur classique : on retourne l'image a la main, pas les cibles
        def transform(self, inpt, params):  # appelee par torchvision pour chaque element (image, boites, masque)
            # retourne gauche-droite seulement les tenseurs a virgule (l'image), laisse le reste tel quel
            return inpt.flip(-1) if isinstance(inpt, torch.Tensor) and inpt.is_floating_point() else inpt

    assert not check_alignment(ImageOnlyFlip(), trials=4)["ok"]  # l'operation fautive est rejetee
    assert check_alignment(v2.RandomHorizontalFlip(1.0), trials=4)["ok"]  # le vrai retournement de torchvision est accepte


# Verifie, sur 20 tirages, que le recadrage "sur" garde toujours les 3 objets entiers dans l'image.
# Attraperait : un recadrage qui coupe un defaut, donc une image "avec defaut" ou on ne voit plus le defaut.
def test_safe_crop_never_cuts_an_object():
    for seed in range(20):  # 20 tirages differents
        torch.manual_seed(seed)  # graine differente a chaque tirage
        image, boxes, mask = alignment_pattern((64, 80), 1)  # image test, boites, masque
        # recadrage qui garde 20 a 40 % de l'aire, applique a l'image, aux boites et au masque
        out, out_boxes, out_mask = build_augmentation([_forced("safe_crop", area=[0.2, 0.4])], "detection")(image, boxes, mask)
        assert out.shape == image.shape  # l'image recadree est remise a la taille d'origine
        for k in (1, 2, 3):  # chaque objet est encore entierement dans l'image
            assert (out_mask == k).any()  # la classe k apparait toujours dans le masque
        b = out_boxes.as_subclass(torch.Tensor)  # boites en tenseur simple
        assert (b[:, 0] >= 0).all() and (b[:, 2] <= 80).all()  # boites dans la largeur de l'image (0 a 80)


@pytest.mark.parametrize("preset", sorted(PRESETS))  # chaque preset
@pytest.mark.parametrize("task", ["classification", "detection", "segmentation"])  # pour chaque tache
# Verifie que chaque preset se construit pour chaque tache, ne contient que des operations permises, et garde les
# cibles alignees. Attraperait : un preset qui plante en detection, ou qui glisse une operation interdite.
def test_presets_build_for_every_task_and_keep_targets_aligned(preset, task):
    steps = expand_steps([{"name": "preset", "preset": preset}], task)  # liste des etapes du preset pour cette tache
    assert steps and all(task in OPS[s["name"]].tasks for s in steps)  # liste non vide, et chaque etape est permise
    aug = build_augmentation([AugmentStep(name="preset", preset=preset)], task)  # construction depuis un objet de config
    if task != "classification":  # en detection / segmentation, il y a des cibles a suivre
        assert check_alignment(aug, size=(64, 80), trials=6)["ok"]  # elles restent alignees
    else:  # en classification, seule l'image compte
        out = aug(torch.rand(1, 64, 80))  # applique a une image aleatoire grise
        assert out.min() >= 0 and out.max() <= 1  # valeurs toujours dans [0, 1]


# Verifie les regles des presets (filtre par tache, exclusion) et les messages d'erreur.
# Attraperait : un trou masquant un objet en detection, une rotation interdite sur une bande d'acier, ou une faute de
# frappe dans le YAML qui passerait en silence au lieu d'une erreur claire.
def test_preset_rules_task_filter_exclude_and_errors():
    # noms des etapes du preset "strong" en classification
    strong = [s["name"] for s in expand_steps([{"name": "preset", "preset": "strong"}], "classification")]
    assert "coarse_dropout" in strong  # en classification, "strong" contient des trous (coarse_dropout)...
    # ...mais pas en detection : un trou pourrait cacher entierement un objet dont la boite resterait
    assert "coarse_dropout" not in [s["name"] for s in expand_steps([{"name": "preset", "preset": "strong"}], "detection")]
    # exclude retire une operation precise du preset
    no_rot = expand_steps([{"name": "preset", "preset": "steel_texture", "exclude": ["rot90"]}], "classification")
    assert "rot90" not in [s["name"] for s in no_rot]  # rot90 a bien disparu
    # le preset bande d'acier n'a pas de rot90 : la direction de defilement doit rester verticale
    assert "rot90" not in [s["name"] for s in expand_steps([{"name": "preset", "preset": "steel_strip"}], "detection")]
    with pytest.raises(ValueError, match="preset inconnu"):  # le bloc doit lever une ValueError contenant ce texte
        build_augmentation([{"name": "preset", "preset": "extreme"}])  # preset qui n'existe pas
    with pytest.raises(ValueError, match="absent"):  # erreur attendue : operation exclue absente du preset
        build_augmentation([{"name": "preset", "preset": "light", "exclude": ["jpeg"]}])  # "light" n'a pas de jpeg
    with pytest.raises(ValueError, match="non prevue pour la tache detection"):  # erreur attendue : operation interdite
        build_augmentation([{"name": "randaugment"}], "detection")  # randaugment ne sait pas deplacer des boites
    with pytest.raises(ValueError, match="mode inconnu"):  # erreur attendue : valeur de parametre invalide
        build_augmentation([{"name": "illumination_gradient", "mode": "spiral"}])  # "spiral" n'existe pas


# Verifie que la meme graine donne exactement les memes augmentations. Attraperait : une source de hasard non
# controlee (numpy, random), qui rendrait les entrainements impossibles a reproduire.
def test_same_seed_same_augmentations():
    aug = build_augmentation([{"name": "preset", "preset": "strong"}], "classification")  # preset fort (beaucoup de hasard)
    image = torch.rand(3, 48, 48)  # image couleur aleatoire
    torch.manual_seed(5)  # graine 5...
    a = aug(image.clone())  # ...premier tirage (clone : copie, pour ne pas modifier l'original)
    torch.manual_seed(5)  # meme graine...
    b = aug(image.clone())  # ...second tirage
    assert torch.equal(a, b)  # resultats identiques au pixel pres


# Verifie que les operations photometriques (lumiere, capteur, occlusion) changent l'image sans toucher aux cibles.
# Attraperait : une operation de couleur qui deplacerait par erreur les boites ou le masque.
def test_photometric_operations_leave_targets_untouched():
    image, boxes, mask = alignment_pattern((64, 80), 3)  # image test couleur, boites, masque
    # toutes les operations qui changent les valeurs des pixels sans les deplacer
    photometric = [n for n, info in OPS.items() if info.family in ("photometrie", "capteur", "occlusion")]
    aug = build_augmentation([_forced(n) for n in photometric], "detection")  # toutes a la suite, forcees
    torch.manual_seed(3)  # graine fixe
    out, out_boxes, out_mask = aug(image, boxes, mask)  # applique a l'image, aux boites et au masque
    assert torch.equal(out_boxes.as_subclass(torch.Tensor), boxes.as_subclass(torch.Tensor))  # boites inchangees
    assert torch.equal(out_mask, mask) and not torch.equal(out, image)  # masque inchange, mais image bien modifiee


# Verifie MixUp / CutMix : etiquettes "douces" (melange de classes) qui somment a 1, pas de melange si p = 0, et un objet
# qui survit a pickle. Attraperait : des etiquettes mal ponderees, ou un plantage avec num_workers > 0 sous Windows.
def test_mixup_cutmix_collate_gives_soft_labels_and_survives_pickling():
    # 16 images grises 16 x 16 et leurs classes 0, 1, 2, 3, 0, 1... (% = reste de la division)
    data = TensorDataset(torch.rand(16, 1, 16, 16), torch.arange(16) % 4)
    collate = batch_augmentation(4, p=1.0)  # 4 classes ; melange applique a chaque lot (p = 1)
    collate = pickle.loads(pickle.dumps(collate))  # num_workers > 0 sous Windows : le collate est envoye aux workers
    # premier lot de 8 images ; collate_fn assemble les images en lot et applique le melange
    images, labels = next(iter(DataLoader(data, batch_size=8, collate_fn=collate)))
    assert images.shape == (8, 1, 16, 16) and labels.shape == (8, 4)  # etiquettes en vecteurs de 4 proportions (une par classe)
    assert torch.allclose(labels.sum(1), torch.ones(8))  # pour chaque image, les proportions somment a 1
    _, plain = next(iter(DataLoader(data, batch_size=8, collate_fn=batch_augmentation(4, p=0.0))))  # p = 0 : jamais de melange
    # etiquettes "one-hot" (un 1 pour la vraie classe, des 0 ailleurs) qui redonnent les classes d'origine
    assert torch.equal(plain.argmax(1), torch.arange(8) % 4) and set(plain.unique().tolist()) == {0.0, 1.0}
    loss = torch.nn.functional.cross_entropy(torch.randn(8, 4), labels)  # CrossEntropyLoss accepte les cibles douces
    assert torch.isfinite(loss)  # la perte est un nombre valide (ni infini ni NaN)


# fabrique une image de test avec un defaut connu ; renvoie (image, masque)
def _defect_scene(seed: int = 0):
    rng = np.random.default_rng(seed)  # generateur aleatoire a graine choisie
    # image grise 64 x 64 : gris moyen 0.5 avec un leger grain ; from_numpy la convertit en tenseur PyTorch
    image = torch.from_numpy(rng.normal(0.5, 0.02, (1, 64, 64)).astype(np.float32))
    mask = torch.zeros(64, 64, dtype=torch.int64)  # masque vide (classe 0 = fond)
    mask[20:30, 10:40] = 2  # zone du defaut, de classe 2 (10 lignes x 30 colonnes)
    image[0][mask == 2] = 0.1  # le defaut est sombre (0.1) sur l'image
    return image, mask  # renvoie l'image et son masque


# Verifie le copier-coller de defauts : masque, boites et etiquettes sont mis a jour, rien n'est colle hors du defaut,
# pas de collage sur un objet existant, et p = 0 ne touche a rien. Attraperait : un defaut colle sans etiquette
# (le modele apprendrait qu'un defaut visible est "normal") ou une boite qui ne correspond pas au defaut colle.
def test_defect_bank_and_copy_paste_update_mask_boxes_and_labels():
    image, mask = _defect_scene()  # image source avec un defaut de classe 2
    bank = DefectBank.from_masks([image], [mask])  # decoupe les defauts d'apres le masque
    assert len(bank) == 1 and bank.classes() == {2: 1}  # un seul defaut, de classe 2
    # image d'accueil saine (graine 1, on garde l'image) et masque vide
    target, empty = _defect_scene(1)[0], torch.zeros(64, 64, dtype=torch.int64)
    torch.manual_seed(0)  # graine fixe
    out = copy_paste(target, bank, mask=empty, count=(2, 2), scale=(1.0, 1.0))  # colle exactement 2 defauts, sans changer d'echelle
    # le masque contient maintenant plus de 100 pixels de classe 2, et seulement les classes 0 et 2
    assert (out["mask"] == 2).sum() > 100 and set(out["mask"].unique().tolist()) == {0, 2}
    assert out["labels"].tolist() == [2, 2] and out["boxes"].shape == (2, 4)  # 2 etiquettes et 2 boites ajoutees
    changed = (out["image"] - target).abs()[0] > 0.05  # pixels nettement modifies par le collage
    assert changed.any() and not changed[out["mask"] == 0].any()  # rien de colle hors du masque (bord adouci a part)
    x0, y0, x1, y1 = out["boxes"][0].int().tolist()  # coins de la premiere boite ajoutee
    assert (out["mask"][y0:y1, x0:x1] == 2).any()  # la boite entoure bien un defaut colle
    # en detection : un objet existant (boite 0,0 -> 30,30) sur une image 64 x 64
    boxes = tv_tensors.BoundingBoxes(torch.tensor([[0.0, 0.0, 30.0, 30.0]]), format="XYXY", canvas_size=(64, 64))
    det_bank = DefectBank.from_boxes([image], [[[10, 20, 40, 30]]], [[5]])  # banque depuis une boite, defaut de classe 5
    det = CopyPaste(det_bank, p=1.0, count=(1, 1))(target, boxes=boxes, labels=torch.tensor([1]))  # colle 1 defaut a coup sur
    # le resultat garde le type boites, et l'etiquette 5 s'ajoute apres l'objet existant (classe 1)
    assert isinstance(det["boxes"], tv_tensors.BoundingBoxes) and det["labels"].tolist() == [1, 5]
    new = det["boxes"][1].tolist()  # coordonnees de la boite collee
    assert not (new[0] < 30 and new[1] < 30)  # avoid_overlap : pas colle sur l'objet existant
    unchanged = CopyPaste(det_bank, p=0.0)(target, boxes=boxes, labels=torch.tensor([1]))  # p = 0 : jamais applique
    assert unchanged["image"] is target and unchanged["labels"].tolist() == [1]  # meme objet image, etiquettes inchangees


# Verifie la mosaique (4 images en une) : chaque boite tombe sur les pixels de sa propre image d'origine.
# Attraperait : des boites decalees ou attribuees a la mauvaise image, donc des etiquettes posees au mauvais endroit.
def test_mosaic_puts_each_box_on_pixels_of_its_own_image():
    samples = []  # les 4 echantillons a assembler
    for k in range(4):  # pour chaque image k = 0..3
        image = torch.full((1, 40, 40), (k + 1) / 5)  # image unie de valeur 0.2, 0.4, 0.6 ou 0.8 : on reconnait son origine
        # echantillon : image, une boite (5,5)-(35,35), classe k, masque uni de valeur k + 1 ;
        # "fmt: skip" demande a ruff format de ne pas remettre en forme ces deux lignes
        samples.append({"image": image, "boxes": torch.tensor([[5.0, 5.0, 35.0, 35.0]]), "labels": torch.tensor([k]),
                        "mask": torch.full((40, 40), k + 1)})  # fmt: skip
    for seed in range(5):  # 5 tirages (centre de la mosaique different)
        torch.manual_seed(seed)  # graine du tirage
        out = mosaic(samples, size=(80, 80))  # assemble les 4 images dans une image 80 x 80
        assert out["image"].shape == (1, 80, 80) and out["mask"].shape == (80, 80)  # tailles attendues
        # pour chaque boite et sa classe ; strict=True : erreur si les deux listes n'ont pas la meme longueur
        for box, label in zip(out["boxes"].tolist(), out["labels"].tolist(), strict=True):
            x0, y0, x1, y1 = (int(round(v)) for v in box)  # coins arrondis en pixels entiers
            patch = out["image"][0, y0:y1, x0:x1]  # pixels de la mosaique sous la boite
            assert torch.allclose(patch, torch.full_like(patch, (label + 1) / 5))  # ils viennent tous de l'image de cette classe
            assert (out["mask"][y0:y1, x0:x1] == label + 1).all()  # et le masque dit la meme chose


# Verifie le defaut synthetique : il ecrit sa classe dans le masque, exige un masque, est refuse en classification, et
# synthesize_defect renvoie image + masque. Attraperait : un faux defaut dessine sans etiquette dans le masque.
def test_synthetic_defect_writes_the_mask_and_needs_one():
    image, _, mask = alignment_pattern((64, 80), 1)  # image test grise et son masque
    empty = tv_tensors.Mask(torch.zeros_like(mask))  # masque vide de meme taille
    # rayure synthetique de classe 4, toujours appliquee
    aug = build_augmentation([{"name": "synthetic_defect", "kinds": ["scratch"], "class_index": 4, "p": 1.0}], "segmentation")
    torch.manual_seed(0)  # graine fixe
    out, out_mask = aug(image, empty)  # applique a l'image et au masque vide
    assert (out_mask == 4).sum() > 10 and not torch.equal(out, image)  # la rayure est dans le masque (classe 4) et sur l'image
    with pytest.raises(ValueError, match="masque"):  # erreur attendue si aucun masque n'est fourni
        augment.SyntheticDefect(p=1.0)(image)  # appel avec l'image seule
    with pytest.raises(ValueError, match="non prevue"):  # erreur attendue : pas de masque en classification
        build_augmentation([{"name": "synthetic_defect"}], "classification")  # construction pour la classification
    # tache organique (bruit de Perlin) sur une image grise unie couleur 32 x 32
    defect_image, defect_mask = synthesize_defect(torch.full((3, 32, 32), 0.5), "perlin")
    # bonnes tailles, et le masque n'est pas vide
    assert defect_image.shape == (3, 32, 32) and defect_mask.shape == (32, 32) and defect_mask.any()


# Verifie les outils d'inspection : description d'une augmentation, catalogue des operations, mesure de vitesse.
# Attraperait : un outil qui plante ou renvoie un tableau incomplet (colonnes ou lignes manquantes).
def test_tools_describe_catalog_and_benchmark():
    table = describe([{"name": "preset", "preset": "medium"}], "classification")  # tableau des etapes du preset medium
    assert len(table) >= 8 and {"transform", "p", "params"} <= set(table.columns)  # au moins 8 etapes et les bonnes colonnes
    assert describe(None).empty  # aucune augmentation : tableau vide (et pas de plantage)
    catalog = augment.augmentation_catalog()  # une ligne par operation disponible
    # autant de lignes que d'operations (au moins 40) et toutes les familles representees
    assert len(catalog) == len(OPS) >= 40 and set(catalog["family"]) >= {"geometrie", "photometrie", "capteur", "occlusion", "politique"}
    speed = benchmark_ops(["hflip", "jpeg"], size=(32, 32), repeats=2)  # chronometre 2 operations sur de petites images
    # colonnes attendues (operation, famille, millisecondes par image, images par seconde) et une ligne par operation
    assert list(speed.columns) == ["op", "family", "ms_per_image", "images_per_s"] and len(speed) == 2
    assert len(augment.presets_table()) == len(PRESETS)  # le tableau des presets a une ligne par preset


# Verifie que le dossier vtb/augment n'importe pas d'autres modules de la toolbox. Attraperait : une dependance cachee
# qui empecherait de copier ce dossier seul dans un autre projet.
def test_module_is_self_contained():
    """Le dossier vtb/augment se copie seul : aucun import d'un autre module vtb (sauf l'affichage optionnel)."""
    folder = Path(augment.__file__).parent  # dossier du module (__file__ = chemin de son __init__.py)
    for path in folder.glob("*.py"):  # pour chaque fichier Python du dossier
        for line in path.read_text(encoding="utf-8").splitlines():  # pour chaque ligne du fichier
            found = re.match(r"\s*(from|import)\s+(vtb\.[\w.]+)", line)  # cherche "from vtb.xxx" ou "import vtb.xxx" en debut de ligne
            if found:  # si la ligne importe un module vtb
                # seuls vtb.viz (affichage) et vtb.augment lui-meme sont permis ; sinon le message montre la ligne fautive
                assert found.group(2) in ("vtb.viz", "vtb.augment"), f"{path.name} : {line.strip()}"
