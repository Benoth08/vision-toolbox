"""Detection de bout en bout sur un petit jeu synthetique (carres clairs, barres sombres), pour les quatre moteurs.

Rien n'est telecharge : Faster R-CNN et YOLOX partent de poids aleatoires ; le test Ultralytics
utilise weights/yolo11n.pt s'il est present (sinon il est saute). Les seuils de qualite sont bas :
on verifie la CHAINE (formats, reperes de coordonnees, export verifie), pas la precision.
"""

# Detection : le modele trouve des objets dans l'image et rend pour chacun une boite (x1, y1, x2, y2), une classe et
# un score de confiance. Moteur (backend) : la bibliotheque qui fournit le modele (torchvision, YOLOX, Ultralytics, RF-DETR).

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[int], X | None) partout

import json  # json : lire du texte JSON (format de donnees structurees)
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)
import pandas as pd  # pandas : tableaux de donnees avec colonnes nommees (DataFrame)
import pytest  # pytest : lance les tests ; fixtures, approx, importorskip, skip...
import torch  # PyTorch : tenseurs (tableaux de nombres, sur CPU ou GPU) et reseaux de neurones

from vtb.annotate.schema import Box, ImageAnnotation, Polygon, RotatedBox  # classes du format pivot d'annotation
from vtb.annotate.voc import write_voc_folder  # ecrit un fichier XML Pascal VOC par image

# detection : metriques COCO (mAP = precision moyenne, la note standard en detection) et point de fonctionnement a un seuil
from vtb.evaluate.detection import coco_metrics, operating_point
from vtb.io.images import write_image  # ecrit un tableau numpy en fichier image
from vtb.preprocess import Pipeline  # suite d'operations de pretraitement appliquee a chaque image
from vtb.tasks.detection import workflow  # les 5 etapes de la recette detection : prepare, train, evaluate, predict, export

# common : outils partages par les moteurs (annotations d'une partie, boites, comparaison, prediction par tuiles, echelle)
from vtb.tasks.detection.common import annotations_for, ensure_boxes, mean_best_iou, predict_tiled, scale_annotation
from vtb.tasks.detection.config import DetectionConfig  # configuration validee d'un projet de detection

REPO = Path(__file__).resolve().parents[1]  # racine du depot (le dossier parent de tests/), pour trouver weights/


@pytest.fixture  # fixture : jeu de detection synthetique, recree pour chaque test qui le demande
def det_folder(tmp_path: Path) -> Path:  # renvoie le dossier racine (images/ + annotations/)
    """24 images 96 x 96 en gris : un carre clair (spot) et une barre sombre (bar) par image, quelques images vides."""
    rng = np.random.default_rng(0)  # generateur aleatoire a graine fixe : meme jeu a chaque lancement
    root = tmp_path / "det"  # dossier racine dans le dossier temporaire du test
    annotations = []  # annotations de toutes les images
    for i in range(24):  # 24 images
        image = np.full((96, 96), 120, np.uint8) + rng.integers(0, 10, (96, 96), dtype=np.uint8)  # fond gris 120 + leger bruit
        ann = ImageAnnotation(str(root / "images" / f"img_{i:02d}.png"), 96, 96)  # annotation vide de l'image
        if i % 6 != 5:  # une image sur six sans defaut
            x, y = int(rng.integers(4, 50)), int(rng.integers(4, 50))  # position aleatoire du carre
            image[y : y + 20, x : x + 20] = 230  # carre clair 20 x 20
            ann.boxes.append(Box("spot", x, y, x + 20, y + 20))  # sa boite, classe "spot"
            bx, by = int(rng.integers(4, 30)), int(rng.integers(70, 86))  # position aleatoire de la barre (en bas)
            image[by : by + 6, bx : bx + 50] = 20  # barre sombre 50 x 6
            ann.boxes.append(Box("bar", bx, by, bx + 50, by + 6))  # sa boite, classe "bar"
        write_image(ann.image_path, image)  # ecrit l'image sur le disque
        annotations.append(ann)  # garde l'annotation
    write_voc_folder(annotations, root / "annotations")  # ecrit les annotations au format VOC
    return root  # renvoie le dossier racine


# fabrique une configuration minimale pour un moteur donne et la renvoie ; **section : reglages propres au moteur
def _config(root: Path, out: Path, backend: str, **section) -> DetectionConfig:
    return DetectionConfig.model_validate(  # model_validate verifie le dictionnaire et le transforme en objet de configuration
        {  # la configuration ecrite comme un dictionnaire, section par section (comme dans un fichier YAML)
            "data": {  # ou sont les donnees et comment les lire
                "root": str(root / "images"),  # dossier des images
                "layout": "annotations",  # rangement : un fichier d'annotation par image
                "annotations_dir": str(root / "annotations"),  # dossier des fichiers d'annotation
                "annotation_format": "voc",  # format de ces fichiers
                "color": "gray",  # images lues en niveaux de gris
            },
            "split": {"file": str(out / "split.csv"), "val_fraction": 0.25, "test_fraction": 0.25},  # 25 % validation, 25 % test
            "backend": backend,  # moteur de detection teste
            # conf_eval : seuil de score tres bas pour la mAP (elle a besoin de toutes les boites) ; conf_predict : seuil des predictions
            "evaluation": {"conf_eval": 0.001, "conf_predict": 0.05},
            "export": {"latency_runs": 1},  # mesure de vitesse a l'export : 1 passage seulement
            "runtime": {"output_dir": str(out / backend), "num_workers": 0},  # dossier de sortie ; chargement dans le processus principal
            **section,  # ajoute les reglages propres au moteur
        }
    )


# enchaine les 5 etapes de la recette ; renvoie (metriques, tableau des predictions, fiche du modele en dictionnaire)
def _run_all(config: DetectionConfig, source: Path) -> tuple[dict, pd.DataFrame, dict]:
    workflow.prepare(config)  # etape 1 : catalogue + decoupage + annotations pivot
    output = Path(config.runtime.output_dir)  # dossier de sortie
    output.mkdir(parents=True, exist_ok=True)  # le cree s'il n'existe pas
    device = torch.device("cpu")  # calcul sur le processeur
    workflow.train(config, output, device)  # etape 2 : entrainement
    metrics = workflow.evaluate(config, output, device)  # etape 3 : evaluation
    table = workflow.predict(config, output, device, source)  # etape 4 : prediction sur un dossier
    card = workflow.export(config, output)  # etape 5 : export ONNX + fiche
    return metrics, table, json.loads(card.model_dump_json())  # la fiche convertie en JSON puis en dictionnaire simple


# Verifie que la preparation ecrit le decoupage et le fichier d'annotations pivot, dans le meme ordre d'images.
# Attraperait : des annotations decalees d'une image (chaque image recevrait les boites d'une autre).
def test_prepare_writes_split_and_pivot(det_folder: Path, tmp_path: Path) -> None:
    config = _config(det_folder, tmp_path / "out", "fasterrcnn")  # configuration quelconque (le moteur n'importe pas ici)
    split = workflow.prepare(config)  # decoupage
    assert set(split["split"]) == {"train", "val", "test"}  # les 3 parties existent
    assert workflow.pivot_path(config.split.file).exists()  # le fichier pivot (split.vtb.json, a cote du split) est ecrit
    test, names = annotations_for(config.split.file, "test")  # annotations de la partie test et noms de classes
    assert names == ["bar", "spot"]  # classes triees par ordre alphabetique
    assert [a.image_path for a in test] == list(split.loc[split["split"] == "test", "path"])  # meme ordre que le split


# Verifie tout le parcours avec Faster R-CNN (torchvision) : metriques, une prediction par image, fiche et ONNX.
# Attraperait : une etape cassee, une image oubliee a la prediction, une fiche qui decrit mal les sorties.
def test_fasterrcnn_end_to_end(det_folder: Path, tmp_path: Path) -> None:
    section = {  # reglages minimaux
        # petit reseau resnet18 + FPN, sans poids pre-entraines, images entre 96 et 128 pixels, 2 epoques, lots de 4
        "fasterrcnn": {"backbone": "resnet18_fpn", "pretrained": "none", "min_size": 96, "max_size": 128, "epochs": 2, "batch_size": 4},
        "augment": [{"name": "hflip"}],  # augmentation : retournement horizontal
    }
    # lance les 5 etapes et predit sur toutes les images
    metrics, table, card = _run_all(_config(det_folder, tmp_path / "out", "fasterrcnn", **section), det_folder / "images")
    assert 0.0 <= metrics["map50"] <= 1.0 and "operating_point" in metrics  # mAP a IoU 0.5 dans [0, 1] ; point de fonctionnement present
    assert set(table["path"]) == {str(p) for p in sorted((det_folder / "images").glob("*.png"))}  # chaque image tracee
    # pas de redimensionnement cote client (le modele le fait lui-meme) ; sorties boites, scores, classes
    assert card["input"]["resize"] == "none" and [o["name"] for o in card["outputs"]] == ["boxes", "scores", "labels"]
    assert (tmp_path / "out" / "fasterrcnn" / "model.onnx").exists()  # le fichier ONNX est ecrit


# Verifie le parcours avec YOLOX et les particularites de son entree (BGR, letterbox colle en haut a gauche).
# Attraperait : une fiche fausse, et donc un deploiement (C#...) qui preparerait mal les images.
def test_yolox_end_to_end(det_folder: Path, tmp_path: Path) -> None:
    pytest.importorskip("yolox")  # si YOLOX n'est pas installe, le test est saute
    # modele nano sans poids pre-entraines, entree 64 x 64, 1 epoque, lots de 4, demi-epoque de mise en route
    section = {"yolox": {"variant": "nano", "weights": None, "input_size": 64, "epochs": 1, "batch_size": 4, "warmup_epochs": 0.5}}
    # lance les 5 etapes
    metrics, table, card = _run_all(_config(det_folder, tmp_path / "out", "yolox", **section), det_folder / "images")
    assert 0.0 <= metrics["map50_95"] <= 1.0  # mAP moyennee sur les seuils d'IoU 0.5 a 0.95, dans [0, 1]
    # YOLOX attend des images BGR (ordre OpenCV) ; letterbox = redimension sans deformer, completee par une bordure
    assert card["input"]["color"] == "BGR" and card["input"]["resize"] == "letterbox_top_left"
    assert "raw" in card["notes"]  # la fiche garde le rapport de verification des sorties brutes PyTorch / ONNX


# Verifie le parcours avec Ultralytics (YOLO11) quand les poids sont disponibles localement.
# Attraperait : un probleme de conversion vers le format YOLO ou une fiche d'entree fausse (letterbox, gris 114).
def test_ultralytics_end_to_end(det_folder: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("ultralytics")  # si Ultralytics n'est pas installe, le test est saute
    weights = REPO / "weights" / "yolo11n.pt"  # poids pre-entraines attendus dans le depot
    if not weights.exists():  # si le fichier de poids est absent
        pytest.skip("weights/yolo11n.pt absent (telecharge par Ultralytics en usage normal)")  # on saute le test (pas de telechargement)
    monkeypatch.chdir(tmp_path)  # le jeu YOLO est ecrit sous data/processed/ du dossier courant
    # images 96, 1 epoque, lots de 8, sans mise en route ; close_mosaic 0 : pas de phase speciale sans mosaique
    section = {"ultralytics": {"weights": str(weights), "imgsz": 96, "epochs": 1, "batch": 8, "warmup_epochs": 0, "close_mosaic": 0}}
    # lance les 5 etapes
    metrics, table, card = _run_all(_config(det_folder, tmp_path / "out", "ultralytics", **section), det_folder / "images")
    assert 0.0 <= metrics["map50"] <= 1.0  # mAP dans [0, 1]
    assert card["input"]["resize"] == "letterbox" and card["input"]["letterbox_pad_value"] == 114  # bordure grise de valeur 114


# Verifie le parcours avec RF-DETR (detecteur a base de transformer) et sa fiche (redimension etiree, sorties dets / labels).
# Attraperait : une fiche fausse ou un export dont les scores different de PyTorch.
def test_rfdetr_end_to_end(det_folder: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("rfdetr")  # si RF-DETR n'est pas installe, le test est saute
    monkeypatch.chdir(tmp_path)  # le jeu COCO est ecrit sous data/processed/ du dossier courant
    section = {  # reglages minimaux
        # modele nano sans poids, resolution 128, 1 epoque, lots de 4, sans accumulation de gradient ni arret anticipe
        "rfdetr": {"size": "nano", "weights": None, "resolution": 128, "epochs": 1, "batch_size": 4, "grad_accum_steps": 1, "patience": 0}
    }
    # lance les 5 etapes
    metrics, table, card = _run_all(_config(det_folder, tmp_path / "out", "rfdetr", **section), det_folder / "images")
    assert 0.0 <= metrics["map50"] <= 1.0  # mAP dans [0, 1]
    # "stretch" : image etiree au carre (deformee) ; deux sorties : boites (dets) et scores par classe (labels)
    assert card["input"]["resize"] == "stretch" and [o["name"] for o in card["outputs"]] == ["dets", "labels"]
    assert "max_score_gap" in card["notes"]  # la fiche note le plus grand ecart de score entre PyTorch et ONNX


# Verifie les metriques sur des cas connus : prediction parfaite, aucune prediction, et une fausse alarme.
# Attraperait : une mAP mal calculee, ou des vrais / faux positifs mal comptes.
def test_coco_metrics_perfect_and_empty() -> None:
    # verite terrain : image a avec un objet, image b vide
    truth = [ImageAnnotation("a.png", 100, 100, boxes=[Box("x", 10, 10, 50, 50)]), ImageAnnotation("b.png", 100, 100)]
    perfect = [[Box("x", 10, 10, 50, 50, 0.9)], []]  # predictions parfaites (meme boite, score 0.9 ; rien sur b)
    assert coco_metrics(truth, perfect, ["x"])["map50"] == pytest.approx(1.0)  # mAP parfaite = 1
    assert coco_metrics(truth, [[], []], ["x"])["map50"] == 0.0  # aucune prediction : mAP nulle
    # point de fonctionnement au seuil 0.5 : bonne boite sur a, plus une boite fausse sur b (image sans objet)
    op = operating_point(truth, [[Box("x", 10, 10, 50, 50, 0.9)], [Box("x", 0, 0, 5, 5, 0.8)]], ["x"], conf=0.5)
    # 1 vrai positif (tp), 1 faux positif (fp), 0 objet manque (fn)
    assert op["overall"]["tp"] == 1 and op["overall"]["fp"] == 1 and op["overall"]["fn"] == 0
    # par image : l'image a defaut est trouvee (rappel 1) et l'image propre est signalee a tort (fausse alarme 1)
    assert op["image_recall"] == 1.0 and op["image_false_alarm_rate"] == 1.0


# Verifie qu'une detection sous le seuil de confiance compte comme un objet manque.
# Attraperait : un seuil ignore, qui gonflerait artificiellement le rappel.
def test_operating_point_respects_threshold() -> None:
    truth = [ImageAnnotation("a.png", 100, 100, boxes=[Box("x", 10, 10, 50, 50)])]  # un objet
    low = [[Box("x", 10, 10, 50, 50, 0.1)]]  # bien place mais score 0.1
    assert operating_point(truth, low, ["x"], conf=0.5)["overall"]["fn"] == 1  # sous le seuil : manque


# Verifie le changement d'echelle des annotations et la conversion boite orientee -> boite droite.
# Attraperait : des predictions faites sur une image redimensionnee qui ne reviennent pas au bon endroit.
def test_scale_annotation_and_ensure_boxes() -> None:
    # annotation 50 x 25 avec une boite et un triangle
    ann = ImageAnnotation("", 50, 25, boxes=[Box("x", 1, 2, 3, 4, 0.5)], polygons=[Polygon("p", [(1, 1), (3, 1), (3, 2)])])
    out = scale_annotation(ann, 2.0, 4.0, "a.png", 100, 100)  # x multiplie par 2, y par 4, nouvelle image 100 x 100
    # x1 = 1 x 2 = 2 ; y2 = 4 x 4 = 16 ; dernier sommet (3, 2) -> (6, 8)
    assert (out.boxes[0].x1, out.boxes[0].y2) == (2.0, 16.0) and out.polygons[0].points[2] == (6.0, 8.0)
    # image avec seulement une boite orientee (angle 0) : centre (50, 50), 20 x 10
    only_rotated = [ImageAnnotation("r.png", 100, 100, rotated=[RotatedBox("c", 50, 50, 20, 10, 0.0)])]
    box = ensure_boxes(only_rotated)[0].boxes[0]  # boite droite qui l'englobe
    assert (box.x1, box.y1, box.x2, box.y2) == pytest.approx((40, 45, 60, 55))  # centre +/- demi-largeur et demi-hauteur


# Verifie la comparaison de deux jeux de detections (sert a verifier qu'ONNX donne les memes boites que PyTorch).
# Attraperait : un comparateur qui dirait "identique" alors que les boites sont decalees.
def test_mean_best_iou_detects_mismatch() -> None:
    a = [np.array([[0, 0, 10, 10, 0.9, 0]], np.float32)]  # une image, une detection : x1, y1, x2, y2, score, classe
    assert mean_best_iou(a, a)["mean_iou"] == pytest.approx(1.0)  # identiques : IoU (recouvrement) parfait = 1
    shifted = [np.array([[5, 0, 15, 10, 0.9, 0]], np.float32)]  # meme boite decalee de 5 pixels
    # intersection 5 x 10 = 50 ; union 100 + 100 - 50 = 150 ; IoU = 1/3
    assert mean_best_iou(a, shifted)["mean_iou"] == pytest.approx(1 / 3)


# Verifie la prediction par tuiles (grande image decoupee en morceaux qui se chevauchent) et la fusion des doublons.
# Attraperait : des boites hors de l'image, ou un plantage de l'assemblage des tuiles.
def test_predict_tiled_merges_duplicates() -> None:
    image = np.zeros((100, 300), np.uint8)  # grande image 100 x 300

    # faux modele : un objet au centre de chaque tuile ; renvoie (1, 6) = une boite, score 0.9, classe 0
    def fake(tile: np.ndarray) -> np.ndarray:  # un objet au centre de chaque tuile -> fusion par NMS entre tuiles voisines
        h, w = tile.shape[:2]  # taille de la tuile
        return np.array([[w / 2 - 5, h / 2 - 5, w / 2 + 5, h / 2 + 5, 0.9, 0]], np.float32)  # boite 10 x 10 centree

    # tuiles de 100 pixels, chevauchement 50 % ; NMS (suppression des boites qui se recouvrent trop) au-dela d'IoU 0.5
    rows = predict_tiled(fake, image, tile=100, overlap=0.5, iou_merge=0.5)
    assert len(rows) >= 1 and rows[:, 2].max() <= 300  # au moins une boite, et aucune ne depasse le bord droit (300)


# Verifie que le jeu de donnees YOLOX enregistre les bons facteurs d'echelle (pipeline puis letterbox).
# Attraperait : des cibles a la mauvaise echelle, donc un modele entraine sur des boites decalees.
def test_yolox_dataset_scales_targets_with_resize(det_folder: Path, tmp_path: Path) -> None:
    pytest.importorskip("yolox")  # si YOLOX n'est pas installe, le test est saute
    from vtb.tasks.detection.yolox_backend import YoloxDataset  # importe ici : ce module a besoin de YOLOX

    config = _config(det_folder, tmp_path / "out", "yolox")  # configuration YOLOX par defaut
    workflow.prepare(config)  # decoupage + annotations pivot
    train, names = annotations_for(config.split.file, "train")  # annotations d'entrainement et classes
    # pipeline 96 -> 48 pixels, images grises, entree reseau 64 x 64, au plus 10 objets par image
    dataset = YoloxDataset(train, names, Pipeline([{"name": "resize", "size": 48}]), "gray", 64, 10)
    _, _, scale = dataset[0]  # premier element : (image, cibles, echelles) ; on garde les echelles
    ratio, sx, sy = scale.tolist()  # ratio du letterbox, facteurs x et y du pipeline
    assert (sx, sy) == (2.0, 2.0) and ratio == pytest.approx(64 / 48)  # 96 / 48 = 2 ; 48 agrandi a 64
