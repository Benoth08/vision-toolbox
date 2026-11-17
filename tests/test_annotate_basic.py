"""Tests du format pivot et des formats VOC / YOLO / vtb.json (aller-retour sans perte)."""

# Format pivot : la representation interne commune (ImageAnnotation) vers et depuis laquelle on convertit
# tous les formats. Aller-retour : ecrire puis relire doit redonner exactement les memes annotations.

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[int], X | None) partout

from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import pytest  # pytest : lance les tests ; pytest.approx compare des nombres a virgule avec une tolerance

from vtb.annotate.formats import read_annotations, write_annotations  # lecture / ecriture d'annotations dans un format donne

# schema : les classes du format pivot (boite, boite orientee, polygone, annotation d'image) et la sauvegarde vtb.json
from vtb.annotate.schema import Box, ImageAnnotation, Polygon, RotatedBox, class_names_of, load_annotations, save_annotations

# yolo : conversion annotation <-> lignes texte YOLO, et ecriture d'un jeu de donnees complet au format YOLO
from vtb.annotate.yolo import annotation_to_yolo_lines, parse_yolo_lines, write_yolo_dataset
from vtb.core.config import DataConfig  # configuration qui decrit ou sont les donnees et comment elles sont rangees
from vtb.io.catalogs import build_catalog  # construit le catalogue : un tableau avec une ligne par image


# Verifie qu'un fichier VOC est bien lu, qu'une boite qui deborde est recadree dans l'image, et que le catalogue
# compte les objets. Attraperait : une boite hors image (plantage ou coordonnees fausses a l'entrainement).
def test_voc_reading_clips_boxes_and_catalog(voc_folder: Path):  # voc_folder : fixture de conftest.py (4 images VOC)
    annotations = read_annotations("voc", voc_folder / "images", voc_folder / "annotations")  # lit les 4 XML en format pivot
    assert len(annotations) == 4 and len(annotations[0].boxes) == 2  # 4 images lues, 2 objets dans la premiere
    assert annotations[0].boxes[1].x2 == 200  # xmax 205 ramene a la largeur de l'image
    catalog = build_catalog(  # catalogue construit depuis la configuration (comme le fait un vrai projet)
        # dossier des images, rangement "annotations" (un fichier d'annotation par image) et dossier des XML
        DataConfig(root=str(voc_folder / "images"), layout="annotations", annotations_dir=str(voc_folder / "annotations"))
    )
    assert catalog["n_objects"].tolist() == [2, 2, 2, 2]  # la colonne n_objects compte bien 2 objets par image


# Verifie la conversion vers les lignes texte YOLO et le retour, pour les trois taches : detection (boite droite),
# obb (boite orientee) et segment (polygone). Attraperait : une erreur de normalisation ou d'ordre des coordonnees.
def test_yolo_lines_roundtrip_detect_obb_segment():  # pas de fixture : les annotations sont construites a la main
    ann = ImageAnnotation(  # une image 200 x 100 avec un objet de chaque type
        "a.png",  # chemin de l'image (fictif : on ne lit pas l'image ici)
        200,  # largeur en pixels
        100,  # hauteur en pixels
        boxes=[Box("crack", 10, 20, 50, 60)],  # boite droite : coins (x1, y1) = (10, 20) et (x2, y2) = (50, 60)
        rotated=[RotatedBox("scratch", 100, 50, 80, 10, 30.0)],  # boite orientee : centre (100, 50), taille 80 x 10, angle 30 degres
        polygons=[Polygon("hole", [(10, 10), (30, 10), (30, 40), (10, 40)])],  # polygone : liste des sommets (x, y)
    )
    names = ["crack", "hole", "scratch"]  # liste des classes : la position dans la liste donne le numero de classe YOLO
    idx = {n: i for i, n in enumerate(names)}  # dictionnaire nom -> numero : {"crack": 0, "hole": 1, "scratch": 2}
    # format YOLO detection : "classe cx cy w h", centre et taille divises par la taille de l'image (valeurs entre 0 et 1) ;
    # ici cx = 30/200 = 0.15, cy = 40/100 = 0.4, w = 40/200 = 0.2, h = 40/100 = 0.4
    assert annotation_to_yolo_lines(ann, idx, "detect") == ["0 0.150000 0.400000 0.200000 0.400000"]
    # aller-retour detection : on ecrit les lignes puis on les relit en pixels (on garde seulement les boites)
    boxes, _, _ = parse_yolo_lines(annotation_to_yolo_lines(ann, idx, "detect"), 200, 100, names, "detect")
    # aller-retour obb (boites orientees) : on garde seulement les boites orientees relues
    _, rotated, _ = parse_yolo_lines(annotation_to_yolo_lines(ann, idx, "obb"), 200, 100, names, "obb")
    # aller-retour segment (polygones) : on garde seulement les polygones relus
    _, _, polygons = parse_yolo_lines(annotation_to_yolo_lines(ann, idx, "segment"), 200, 100, names, "segment")
    assert boxes[0].x1 == pytest.approx(10) and boxes[0].y2 == pytest.approx(60)  # la boite relue a les memes coins
    assert rotated[0].w * rotated[0].h == pytest.approx(800, rel=1e-3)  # meme aire, quelle que soit la convention d'angle
    assert polygons[0].bounding_box().x2 == pytest.approx(30)  # le polygone relu a le meme bord droit (x max = 30)


# Verifie l'ecriture d'un jeu complet YOLO (images/, labels/, data.yaml) puis sa relecture, la sauvegarde vtb.json
# et la reecriture VOC. Attraperait : des classes melangees, ou une perte d'information a la sauvegarde.
def test_yolo_dataset_and_vtb_json_roundtrip(voc_folder: Path, tmp_path: Path):  # tmp_path : dossier temporaire neuf
    annotations = read_annotations("voc", voc_folder / "images", voc_folder / "annotations")  # point de depart : les 4 images VOC
    names = class_names_of(annotations)  # classes presentes, triees : ["crack", "hole"]
    # repartition des images : 2 en train (entrainement), 1 en val (validation), 1 en test ; zip associe image et nom de split
    splits = {a.image_path: s for a, s in zip(annotations, ["train", "train", "val", "test"])}
    data_yaml = write_yolo_dataset(annotations, splits, names, tmp_path / "yolo")  # ecrit le jeu YOLO, renvoie le chemin de data.yaml
    assert "names" in data_yaml.read_text()  # data.yaml (le fichier lu par Ultralytics) contient bien la liste des classes
    # relit les etiquettes YOLO du split train
    back = read_annotations("yolo", tmp_path / "yolo" / "images" / "train", tmp_path / "yolo" / "labels" / "train")
    assert [b.label for b in back[0].boxes] == ["crack", "hole"]  # les numeros de classe redonnent les bons noms
    save_annotations(annotations, tmp_path / "all.vtb.json")  # sauvegarde au format natif de la toolbox (un fichier JSON)
    again, again_names = load_annotations(tmp_path / "all.vtb.json")  # relit ce fichier : annotations + noms de classes
    assert again == annotations and again_names == names  # identique a l'original : aucune perte
    write_annotations("voc", annotations, tmp_path / "voc_again")  # reecrit tout au format VOC
    assert len(list((tmp_path / "voc_again").glob("*.xml"))) == 4  # un fichier XML par image, donc 4
