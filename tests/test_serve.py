"""Le predicteur sans PyTorch (ONNX + carte du modele) rend les memes reponses que les workflows PyTorch.

C'est le test qui garantit que la carte du modele suffit a ecrire le programme de production.
"""

# Principe de chaque test : on entraine un petit modele, on predit avec PyTorch (la reference), puis on predit
# avec OnnxPredictor, qui n'utilise QUE le fichier ONNX et la fiche du modele (comme le ferait un programme C#
# en production). Les deux doivent donner la meme reponse ; sinon la fiche oublie une etape de pretraitement.

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[int], X | None) partout

from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)
import pytest  # pytest : lance les tests ; fixtures, parametrize, approx, importorskip...
import torch  # PyTorch : tenseurs et reseaux de neurones

from vtb.io.images import read_image, write_image  # lecture / ecriture d'images

# predictor : predicteur ONNX sans PyTorch, NMS en numpy, decodage des sorties brutes YOLOX
from vtb.serve.predictor import OnnxPredictor, nms, yolox_decode

CPU = torch.device("cpu")  # tous les calculs PyTorch de ce fichier se font sur le processeur


# enchaine preparation, entrainement et export pour une recette ; renvoie le dossier de sortie
def _run(workflow, config) -> Path:
    output = Path(config.runtime.output_dir)  # dossier de sortie lu dans la configuration
    output.mkdir(parents=True, exist_ok=True)  # le cree s'il n'existe pas
    workflow.prepare(config)  # catalogue + decoupage
    workflow.train(config, output, CPU)  # entrainement
    workflow.export(config, output)  # export ONNX + fiche du modele
    return output  # renvoie le dossier qui contient model.onnx et model_card.json


# Verifie que le predicteur ONNX donne la meme classe et la meme probabilite que PyTorch, avec un pretraitement
# en deux etapes (redimension + CLAHE). Attraperait : une etape de pretraitement absente de la fiche ou mal refaite.
def test_classification_predictor_matches_pytorch(class_folder: Path, tmp_path: Path) -> None:
    from vtb.tasks.classification import workflow  # recette classification (importee ici, seul ce test en a besoin)
    from vtb.tasks.classification.config import ClassificationConfig  # sa configuration

    config = ClassificationConfig.model_validate(  # configuration minimale validee
        {  # la configuration ecrite comme un dictionnaire, section par section (comme dans un fichier YAML)
            "data": {"root": str(class_folder), "color": "gray"},  # dossier des classes, images en gris
            "split": {"file": str(tmp_path / "split.csv"), "val_fraction": 0.25, "test_fraction": 0.25},  # 25 % val, 25 % test
            "preprocess": [{"name": "resize", "size": 32}, {"name": "clahe"}],  # redimension puis rehaussement de contraste
            # petit CNN sans poids pre-entraines, 1 canal d'entree, 2 couches de 8 et 16 filtres
            "model": {"name": "simple_cnn", "pretrained": False, "in_channels": 1, "extra": {"channels": [8, 16]}},
            "train": {"epochs": 2, "batch_size": 4},  # 2 epoques, lots de 4 images
            "export": {"latency_runs": 1},  # mesure de vitesse : 1 passage
            "runtime": {"output_dir": str(tmp_path / "cls"), "num_workers": 0},  # dossier de sortie ; chargement sans sous-processus
        }
    )
    output = _run(workflow, config)  # entraine et exporte
    reference = workflow.predict(config, output, CPU, class_folder / "beta")  # predictions PyTorch sur le dossier beta
    predictor = OnnxPredictor(output)  # predicteur qui ne lit que model.onnx et la fiche
    for _, row in reference.iterrows():  # pour chaque image predite par PyTorch
        result = predictor(read_image(row["path"], color=predictor.color))  # meme image, predite par ONNX (couleur lue dans la fiche)
        # meme classe, et meme probabilite a 0.001 pres
        assert result["label"] == row["label"] and result["probability"] == pytest.approx(row["probability"], abs=1e-3)


# Verifie que le predicteur ONNX d'anomalies donne le meme score et la meme decision que PyTorch.
# Attraperait : un seuil ou un lissage de carte different entre l'entrainement et la production.
def test_anomaly_predictor_matches_pytorch(mvtec_folder: Path, tmp_path: Path) -> None:
    from vtb.tasks.anomaly import workflow  # recette anomalies
    from vtb.tasks.anomaly.config import AnomalyConfig  # sa configuration

    config = AnomalyConfig.model_validate(  # configuration minimale validee
        {  # la configuration ecrite comme un dictionnaire, section par section
            "data": {"root": str(mvtec_folder), "layout": "mvtec"},  # donnees rangees a la facon MVTec
            "split": {"file": str(tmp_path / "split.csv"), "val_fraction": 0.3},  # 30 % des normales en validation
            "preprocess": [{"name": "resize", "size": 32}],  # redimension en 32
            "method": "patchcore_lite",  # PatchCore simplifie (memoire de vecteurs normaux, sans entrainement)
            "patchcore_lite": {"pretrained": False, "memory_fraction": 0.5},  # sans poids telecharges ; memoire reduite a 50 %
            "scoring": {"gaussian_sigma": 1.0},  # lissage de la carte d'anomalie
            "export": {"latency_runs": 1},  # mesure de vitesse : 1 passage
            "runtime": {"output_dir": str(tmp_path / "ano"), "num_workers": 0},  # dossier de sortie
        }
    )
    output = _run(workflow, config)  # entraine (memorise les normales), calibre et exporte
    reference = workflow.predict(config, output, CPU, mvtec_folder / "test" / "crack")  # predictions PyTorch sur les defauts
    predictor = OnnxPredictor(output)  # predicteur ONNX
    for _, row in reference.iterrows():  # pour chaque image
        result = predictor(read_image(row["path"], color=predictor.color))  # prediction ONNX
        # meme score (a 0.1 % pres) et meme decision anomalie oui / non
        assert result["score"] == pytest.approx(row["score"], rel=1e-3, abs=1e-5) and result["anomaly"] == row["anomaly"]


# Verifie que le predicteur ONNX de segmentation compte le meme nombre de pixels de defaut que PyTorch.
# Attraperait : un masque redimensionne differemment, ou une mauvaise classe retenue pour chaque pixel.
def test_segmentation_predictor_matches_pytorch(tmp_path: Path) -> None:
    pytest.importorskip("segmentation_models_pytorch")  # sans cette bibliotheque, le test est saute
    from vtb.tasks.segmentation import workflow  # recette segmentation
    from vtb.tasks.segmentation.config import SegmentationConfig  # sa configuration

    rng = np.random.default_rng(0)  # generateur aleatoire a graine fixe
    root = tmp_path / "tiles"  # dossier du petit jeu de donnees
    for label in ("Spot", "Free"):  # images avec tache, et images saines
        for i in range(8):  # 8 images par dossier
            # fond gris 110 + leger bruit (40 x 56), et masque vide
            image, mask = (110 + rng.integers(0, 15, (40, 56))).astype(np.uint8), np.zeros((40, 56), np.uint8)
            if label == "Spot":  # pour les images a defaut
                image[10:20, 20:30], mask[10:20, 20:30] = 240, 255  # carre clair 10 x 10, reporte dans le masque
            write_image(root / label / f"{label}_{i}.jpg", image)  # image .jpg
            write_image(root / label / f"{label}_{i}.png", mask)  # masque .png de meme nom
    config = SegmentationConfig.model_validate(  # configuration minimale validee
        {  # la configuration ecrite comme un dictionnaire, section par section
            # rangement image + masque ; seules les .jpg sont des images ; lecture en gris
            "data": {"root": str(root), "layout": "image_mask", "extensions": [".jpg"], "color": "gray"},
            "split": {"file": str(tmp_path / "split.csv"), "val_fraction": 0.25, "test_fraction": 0.25},  # 25 % val, 25 % test
            "preprocess": [{"name": "resize", "size": 32}],  # redimension en 32
            "model": {"architecture": "fpn", "encoder": "resnet18", "encoder_weights": "none"},  # FPN sur ResNet18, sans poids
            "train": {"epochs": 1, "batch_size": 4},  # 1 epoque, lots de 4
            "evaluation": {"min_defect_area": 3},  # image defectueuse a partir de 3 pixels de defaut predits
            "export": {"latency_runs": 1},  # mesure de vitesse : 1 passage
            "runtime": {"output_dir": str(tmp_path / "seg"), "num_workers": 0},  # dossier de sortie
        }
    )
    output = _run(workflow, config)  # entraine et exporte
    reference = workflow.predict(config, output, CPU, root / "Spot")  # predictions PyTorch sur les images a tache
    predictor = OnnxPredictor(output)  # predicteur ONNX
    for _, row in reference.iterrows():  # pour chaque image
        result = predictor(read_image(row["path"], color=predictor.color))  # prediction ONNX
        assert abs(result["defect_pixels"] - row["defect_pixels"]) <= 2  # pixels a la frontiere du seuil


# reglages minimaux de chaque moteur de detection teste ici
BACKENDS = {
    # YOLOX nano sans poids, entree 64, 1 epoque, lots de 4
    "yolox": {"yolox": {"variant": "nano", "weights": None, "input_size": 64, "epochs": 1, "batch_size": 4}},
    "rfdetr": {  # reglages de RF-DETR
        # RF-DETR nano sans poids, resolution 128, 1 epoque, lots de 4, sans accumulation de gradient ni arret anticipe
        "rfdetr": {"size": "nano", "weights": None, "resolution": 128, "epochs": 1, "batch_size": 4, "grad_accum_steps": 1, "patience": 0}
    },
}


@pytest.mark.parametrize("backend", sorted(BACKENDS))  # le test tourne pour chaque moteur
# Verifie que le predicteur ONNX de detection retrouve les memes boites que PyTorch (a 1 pixel pres), dans l'image
# d'origine. Attraperait : une erreur de letterbox ou de retour a l'echelle d'origine, ou un NMS different.
def test_detection_predictor_matches_pytorch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    pytest.importorskip(backend)  # si la bibliotheque du moteur n'est pas installee, le test est saute
    monkeypatch.chdir(tmp_path)  # jeux derives ecrits sous data/processed/ du dossier courant
    from vtb.annotate.schema import Box, ImageAnnotation  # boite et annotation d'image du format pivot
    from vtb.annotate.voc import write_voc_folder  # ecriture des annotations au format VOC
    from vtb.tasks.detection import workflow  # recette detection
    from vtb.tasks.detection.config import DetectionConfig  # sa configuration

    rng = np.random.default_rng(0)  # generateur aleatoire a graine fixe
    root, annotations = tmp_path / "det", []  # dossier du jeu, et liste des annotations
    for i in range(16):  # 16 images
        image = np.full((72, 96), 120, np.uint8)  # fond gris uniforme 72 x 96
        x, y = int(rng.integers(4, 60)), int(rng.integers(4, 40))  # position aleatoire du carre
        image[y : y + 20, x : x + 20] = 230  # carre clair 20 x 20
        # annotation de l'image : une boite "spot" autour du carre
        ann = ImageAnnotation(str(root / "images" / f"img_{i:02d}.png"), 96, 72, boxes=[Box("spot", x, y, x + 20, y + 20)])
        write_image(ann.image_path, image)  # ecrit l'image
        annotations.append(ann)  # garde l'annotation
    write_voc_folder(annotations, root / "annotations")  # ecrit les annotations VOC
    config = DetectionConfig.model_validate(  # configuration minimale validee
        {  # la configuration ecrite comme un dictionnaire, section par section
            # images, rangement "un fichier d'annotation par image", dossier des annotations, lecture en gris
            "data": {"root": str(root / "images"), "layout": "annotations", "annotations_dir": str(root / "annotations"), "color": "gray"},
            "split": {"file": str(tmp_path / "split.csv"), "val_fraction": 0.25, "test_fraction": 0.25},  # 25 % val, 25 % test
            "preprocess": [{"name": "resize", "size": [48, 64]}],  # le predicteur doit revenir a l'image d'origine
            "backend": backend,  # moteur teste
            **BACKENDS[backend],  # ses reglages
            "evaluation": {"conf_predict": 0.01},  # seuil de score tres bas : on compare aussi les boites peu sures
            "export": {"latency_runs": 1},  # mesure de vitesse : 1 passage
            "runtime": {"output_dir": str(tmp_path / backend), "num_workers": 0},  # dossier de sortie
        }
    )
    output = _run(workflow, config)  # entraine et exporte
    reference = workflow.predict(config, output, CPU, root / "images")  # predictions PyTorch (une ligne par boite)
    predictor = OnnxPredictor(output)  # predicteur ONNX
    for path, rows in reference.groupby("path"):  # pour chaque image et ses lignes de predictions PyTorch
        result = predictor(read_image(path, color=predictor.color))  # prediction ONNX de la meme image
        expected = rows.dropna(subset=["score"])  # retire la ligne vide d'une image sans detection (score manquant)
        assert result["count"] == len(expected)  # meme nombre de boites
        for obj, (_, row) in zip(  # parcourt en parallele les boites ONNX et les lignes PyTorch...
            sorted(result["objects"], key=lambda o: -o["score"]),  # ...boites ONNX triees par score decroissant
            expected.sort_values("score", ascending=False).iterrows(),  # ...lignes PyTorch triees de la meme facon
        ):
            assert obj["box"] == pytest.approx([row["x1"], row["y1"], row["x2"], row["y2"]], abs=1.0)  # memes coins a 1 pixel pres


# Verifie le service HTTP (FastAPI) : etat de sante, prediction sur une image envoyee, refus propre d'un fichier invalide.
# Attraperait : un service qui plante (erreur 500) sur un mauvais fichier au lieu de repondre "requete invalide" (400).
def test_service_endpoints(class_folder: Path, tmp_path: Path) -> None:
    pytest.importorskip("fastapi")  # sans FastAPI (le cadre du service web), le test est saute
    pytest.importorskip("httpx")  # idem sans httpx (utilise par le client de test)
    from fastapi.testclient import TestClient  # client qui appelle le service sans lancer de vrai serveur

    from vtb.serve.app import create_app  # fabrique l'application web a partir d'un dossier de modele exporte
    from vtb.tasks.classification import workflow  # recette classification
    from vtb.tasks.classification.config import ClassificationConfig  # sa configuration

    config = ClassificationConfig.model_validate(  # configuration minimale validee
        {  # la configuration ecrite comme un dictionnaire, section par section
            "data": {"root": str(class_folder), "color": "gray"},  # dossier des classes, images en gris
            "split": {"file": str(tmp_path / "split.csv"), "val_fraction": 0.25, "test_fraction": 0.25},  # 25 % val, 25 % test
            "preprocess": [{"name": "resize", "size": 32}],  # redimension en 32
            # tres petit CNN (une couche de 8 filtres), 1 canal d'entree, sans poids pre-entraines
            "model": {"name": "simple_cnn", "pretrained": False, "in_channels": 1, "extra": {"channels": [8]}},
            "train": {"epochs": 1, "batch_size": 4},  # 1 epoque, lots de 4
            "export": {"latency_runs": 1},  # mesure de vitesse : 1 passage
            "runtime": {"output_dir": str(tmp_path / "svc"), "num_workers": 0},  # dossier de sortie
        }
    )
    output = _run(workflow, config)  # entraine et exporte
    client = TestClient(create_app(output))  # client de test branche sur le service
    assert client.get("/health").json()["task"] == "classification"  # GET /health repond et annonce la tache du modele
    image = next((class_folder / "alpha").glob("*.png"))  # une image du dossier alpha
    # POST /predict avec le fichier image envoye comme un formulaire (nom, octets, type)
    response = client.post("/predict", files={"file": (image.name, image.read_bytes(), "image/png")})
    assert response.status_code == 200 and response.json()["label"] in {"alpha", "beta", "gamma"}  # 200 = succes, classe connue
    # un fichier qui n'est pas une image doit donner 400 (requete invalide), pas un plantage
    assert client.post("/predict", files={"file": ("x.png", b"pas une image", "image/png")}).status_code == 400


# Verifie le NMS en numpy et le decodage des sorties brutes YOLOX (utilises par le predicteur sans PyTorch).
# NMS (non-maximum suppression) : parmi des boites qui se recouvrent trop, on ne garde que celle de meilleur score.
# Attraperait : des doublons de boites en production, ou des boites YOLOX a la mauvaise position ou taille.
def test_numpy_nms_and_yolox_decode() -> None:
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)  # 2 boites presque identiques + 1 isolee
    # scores 0.9, 0.8, 0.7 ; seuil d'IoU 0.5 : la 2e (recouvre la 1re a ~68 %) est supprimee
    assert nms(boxes, np.array([0.9, 0.8, 0.7], np.float32), 0.5).tolist() == [0, 2]
    # sortie brute nulle pour une entree 64 x 64 : grilles 8 x 8, 4 x 4 et 2 x 2 (pas de 8, 16 et 32 pixels), 6 valeurs par case
    raw = np.zeros((8 * 8 + 4 * 4 + 2 * 2, 6), np.float32)
    decoded = yolox_decode(raw, 64)  # decode en (centre x, centre y, largeur, hauteur, ...) en pixels
    # 1re case : centre (0, 0), taille exp(0) x 8 = 8 ; derniere case (grille 2 x 2, pas 32) : centre (32, 32), taille 32
    assert decoded[0, :4].tolist() == [0.0, 0.0, 8.0, 8.0] and decoded[-1, :4].tolist() == [32.0, 32.0, 32.0, 32.0]
