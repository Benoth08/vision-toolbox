"""Segmentation semantique de bout en bout sur un petit jeu image + masque synthetique (rien n'est telecharge)."""

# Segmentation semantique : le modele donne une classe a CHAQUE pixel (fond, rayure, tache...), et rend donc
# un masque de la taille de l'image au lieu d'une seule etiquette ou de boites.

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[int], X | None) partout

from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)
import pytest  # pytest : lance les tests ; fixtures, parametrize, approx, skip...
import torch  # PyTorch : tenseurs et reseaux de neurones

pytest.importorskip("segmentation_models_pytorch")  # sans cette bibliotheque, tout le fichier est saute

# imports places apres importorskip ; "noqa: E402" dit a ruff que ces imports tardifs sont voulus
# write_image : ecrit un tableau numpy en fichier image
from vtb.io.images import write_image  # noqa: E402

# workflow : les 5 etapes de la recette segmentation (prepare, train, evaluate, predict, export)
from vtb.tasks.segmentation import workflow  # noqa: E402

# configuration validee d'un projet de segmentation, et configuration du modele seul
from vtb.tasks.segmentation.config import SegmentationConfig, SegModelConfig  # noqa: E402

# encode_mask : transforme un masque brut (0 / 255, ou valeurs) en numeros de classe
from vtb.tasks.segmentation.data import encode_mask  # noqa: E402

# metrics : matrice de confusion par pixel, decisions par image, scores (IoU, Dice) par classe
from vtb.tasks.segmentation.metrics import confusion, image_decisions, scores_from_confusion  # noqa: E402

# build_segmenter : construit le reseau de segmentation (architecture + encodeur)
from vtb.tasks.segmentation.models import build_segmenter  # noqa: E402


@pytest.fixture  # fixture : jeu de tuiles avec masques, recree pour chaque test qui le demande
def tile_folder(tmp_path: Path) -> Path:  # renvoie le dossier racine (un sous-dossier par type de defaut)
    """Deux types de defaut (rayure, tache) et des images saines, rangees par dossier, masque .png a cote de l'image."""
    rng = np.random.default_rng(1)  # generateur aleatoire a graine fixe
    root = tmp_path / "tiles"  # dossier racine
    for label in ("Scratch", "Spot", "Free"):  # rayure, tache, sans defaut
        for i in range(8):  # 8 images par dossier
            image = (110 + rng.integers(0, 15, (64, 80))).astype(np.uint8)  # fond gris 110 + leger bruit
            mask = np.zeros((64, 80), np.uint8)  # masque vide (0 = fond)
            if label == "Scratch":  # rayure
                y = int(rng.integers(10, 50))  # hauteur aleatoire de la rayure
                image[y : y + 3, 10:70], mask[y : y + 3, 10:70] = 30, 255  # trait sombre de 3 x 60, et le meme dans le masque
            elif label == "Spot":  # tache
                x, y = int(rng.integers(10, 60)), int(rng.integers(10, 45))  # position aleatoire
                image[y : y + 8, x : x + 8], mask[y : y + 8, x : x + 8] = 240, 255  # carre clair 8 x 8, et le meme dans le masque
            write_image(root / label / f"{label}_{i}.jpg", image)  # image en .jpg
            write_image(root / label / f"{label}_{i}.png", mask)  # masque en .png de meme nom, a cote
    return root  # renvoie le dossier racine


# fabrique une configuration minimale et la renvoie ; mode : facon de lire les masques ("binary" ou "from_label")
def _config(root: Path, out: Path, mode: str = "binary") -> SegmentationConfig:
    return SegmentationConfig.model_validate(  # model_validate verifie le dictionnaire et le transforme en objet de configuration
        {  # la configuration ecrite comme un dictionnaire, section par section (comme dans un fichier YAML)
            # rangement image + masque ; seules les .jpg sont des images (les .png sont les masques) ; lecture en gris
            "data": {"root": str(root), "layout": "image_mask", "extensions": [".jpg"], "color": "gray"},
            "mask_mode": mode,  # binary : defaut / fond ; from_label : la classe du defaut vient du nom du dossier
            "normal_labels": ["Free"],  # le dossier "Free" contient les images saines
            "split": {"file": str(out / "split.csv"), "val_fraction": 0.25, "test_fraction": 0.25},  # 25 % validation, 25 % test
            "preprocess": [{"name": "resize", "size": 64}],  # redimension en 64
            "augment": [{"name": "hflip"}],  # augmentation : retournement horizontal (applique aussi au masque)
            # U-Net (architecture en "U" classique pour la segmentation) sur un encodeur ResNet18, sans poids pre-entraines
            "model": {"architecture": "unet", "encoder": "resnet18", "encoder_weights": "none"},
            "train": {"epochs": 2, "batch_size": 4, "lr": 1e-3},  # 2 epoques, lots de 4, taux d'apprentissage 0.001
            "evaluation": {"min_defect_area": 5},  # une image est declaree defectueuse si au moins 5 pixels de defaut sont predits
            "export": {"latency_runs": 1},  # mesure de vitesse a l'export : 1 passage
            "runtime": {"output_dir": str(out / mode), "num_workers": 0},  # dossier de sortie ; chargement dans le processus principal
        }
    )


@pytest.mark.parametrize("mode", ["binary", "from_label"])  # le test tourne pour les deux facons de lire les masques
# Verifie les 5 etapes de la recette et les classes produites selon le mode de masque.
# Attraperait : une etape cassee, des classes mal construites, des masques predits non ecrits, un export incoherent.
def test_segmentation_end_to_end(tile_folder: Path, tmp_path: Path, mode: str) -> None:
    config = _config(tile_folder, tmp_path / "out", mode)  # configuration de ce lancement
    workflow.prepare(config)  # etape 1 : catalogue + decoupage
    output = Path(config.runtime.output_dir)  # dossier de sortie
    output.mkdir(parents=True, exist_ok=True)  # le cree s'il n'existe pas
    device = torch.device("cpu")  # calcul sur le processeur
    summary = workflow.train(config, output, device)  # etape 2 : entrainement
    # binary : fond + "defect" ; from_label : fond + une classe par dossier de defaut
    expected = ["background", "defect"] if mode == "binary" else ["background", "Scratch", "Spot"]
    assert summary["class_names"] == expected  # les classes sont celles attendues, dans l'ordre
    metrics = workflow.evaluate(config, output, device)  # etape 3 : evaluation
    assert metrics["n_images"] > 0 and set(metrics["per_class"]) == set(expected)  # des images evaluees, une metrique par classe
    # etape 4 : prediction sur le dossier Spot, masques predits ecrits dans tmp_path/masks
    table = workflow.predict(config, output, device, tile_folder / "Spot", tmp_path / "masks")
    assert len(table) == 8 and (tmp_path / "masks" / "Spot_0_mask.png").exists()  # 8 lignes, et le masque de la 1re image existe
    card = workflow.export(config, output)  # etape 5 : export ONNX + fiche
    # la sortie a une carte par classe (dimension 1 = classes) ; l'entree a 1 canal (gris)
    assert card.outputs[0].shape[1] == len(expected) and card.input.channels == 1
    assert (output / "model.onnx").exists() and (output / "test_predictions.png").exists()  # modele et apercu des predictions ecrits


# Verifie les metriques sur un cas calcule a la main. Attraperait : une formule d'IoU, de Dice ou de decision par
# image fausse, donc des scores trompeurs.
def test_metrics_from_confusion() -> None:
    truth = np.array([[0, 0, 1, 1]])  # verite : 2 pixels de fond, 2 pixels de defaut
    pred = np.array([[0, 1, 1, 1]])  # prediction : 1 pixel de fond pris pour du defaut
    # matrice de confusion (qui est pris pour qui) puis scores par classe
    scores = scores_from_confusion(confusion(pred, truth, 2), ["background", "defect"])
    # IoU defaut = intersection 2 / union 3
    assert scores["per_class"]["defect"]["iou"] == pytest.approx(2 / 3, abs=1e-4)
    assert scores["per_class"]["defect"]["dice"] == pytest.approx(0.8)  # Dice = 2 x 2 / (3 + 2) = 0.8
    # 3 images avec 0, 30 et 50 pixels de defaut predits ; seule la 3e a vraiment un defaut ; seuil 20 pixels
    decisions = image_decisions([0, 30, 50], [False, False, True], min_area=20)
    # le vrai defaut est trouve ; une des 2 images saines (30 pixels) est signalee a tort
    assert decisions["image_recall"] == 1.0 and decisions["image_false_alarm_rate"] == 0.5


# Verifie les trois facons de transformer un masque brut en numeros de classe.
# Attraperait : une classe mal numerotee, ou une image saine dont le masque compterait comme defaut.
def test_encode_mask_modes() -> None:
    raw = np.array([[0, 255], [255, 0]], np.uint8)  # masque brut 2 x 2 : 255 = defaut
    assert encode_mask(raw, "binary", "x", ["background", "defect"]).tolist() == [[0, 1], [1, 0]]  # binary : 255 -> 1
    assert encode_mask(raw, "from_label", "Spot", ["background", "Scratch", "Spot"]).max() == 2  # dossier Spot : classe 2
    assert encode_mask(raw, "from_label", "Free", ["background", "Scratch", "Spot"]).max() == 0  # dossier sain : tout en fond
    values = np.array([[0, 2], [1, 0]], np.uint8)  # masque ou chaque pixel contient deja son numero de classe
    assert encode_mask(values, "from_values", "", ["background", "a", "b"]).tolist() == [[0, 2], [1, 0]]  # garde tel quel


# Verifie qu'en entree grise avec un encodeur pre-entraine ImageNet, le modele recopie l'image sur 3 canaux et donne
# exactement la meme sortie que le modele couleur. Attraperait : une adaptation au gris qui gacherait les poids ImageNet.
def test_gray_input_is_exactly_the_rgb_model_on_a_repeated_image() -> None:
    """Gris + encodeur ImageNet : l'image est repetee sur 3 canaux dans le modele, donc meme sortie que le modele RGB."""
    try:  # on essaie de charger les poids ImageNet...
        gray = build_segmenter(SegModelConfig(encoder_weights="imagenet"), 2, 1).eval()  # 2 classes, 1 canal d'entree, mode prediction
    except Exception as error:  # noqa: BLE001 - poids torchvision absents du cache et pas de reseau
        pytest.skip(f"poids ImageNet indisponibles : {error}")  # ...sinon le test est saute (pas d'echec)
    # meme modele en 3 canaux, sans charger de poids (ils vont etre copies)
    rgb = build_segmenter(SegModelConfig(encoder_weights="imagenet"), 2, 3, load_weights=False).eval()
    rgb.load_state_dict(gray.state_dict())  # copie tous les poids du modele gris dans le modele couleur
    x = torch.rand(1, 1, 64, 64)  # une image grise aleatoire 64 x 64
    with torch.no_grad():  # pas de gradient : on ne fait que predire
        assert torch.allclose(gray(x), rgb(x.repeat(1, 3, 1, 1)), atol=1e-6)  # gris == couleur sur l'image recopiee 3 fois
    # le modele gris repete l'image 3 fois, et son encodeur garde ses 3 canaux d'origine (poids ImageNet intacts)
    assert gray.repeat == 3 and gray.net.encoder.conv1.in_channels == 3
