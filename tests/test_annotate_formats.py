"""Aller-retour sans perte pour chaque format d'annotation, sur des annotations synthetiques connues."""

# Aller-retour : on ecrit des annotations connues dans un format, on les relit, et on doit retrouver exactement
# les memes objets. C'est le seul moyen sur de savoir qu'une conversion de format ne perd ni ne deforme rien.

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[int], X | None) partout

import json  # json : lire / ecrire des fichiers JSON (format texte de donnees structurees)
import math  # math : fonctions mathematiques de base (ici math.dist, distance entre deux points)
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)
import pytest  # pytest : lance les tests ; pytest.approx compare des nombres a virgule avec une tolerance

# check : compare deux jeux d'annotations et liste leurs differences (liste vide = identiques)
from vtb.annotate.check import annotation_differences, same_annotations

# formats : registres des lecteurs / ecrivains par format, conversion d'un format a l'autre, lecture et ecriture
from vtb.annotate.formats import READERS, WRITERS, convert, read_annotations, write_annotations

# geometry : outils pour les boites orientees (coins ordonnes, construction depuis des coins ou un coin d'origine, comparaison)
from vtb.annotate.geometry import ordered_corners, rotated_from_corners, rotated_from_origin, same_corners

# labelstudio : prefixe des URL "fichiers locaux", configuration XML d'etiquetage, et chemin local d'une image referencee
from vtb.annotate.labelstudio import LOCAL_FILES_PREFIX, labelstudio_config, resolve_image

# masks : conversions masque (image de classes) <-> objets (polygones, boites, boites orientees) et lecture d'un dossier de masques
from vtb.annotate.masks import mask_to_boxes, mask_to_polygons, mask_to_rotated_boxes, objects_to_mask, read_mask_folder
from vtb.annotate.schema import Box, ImageAnnotation, Polygon, RotatedBox  # classes du format pivot (representation commune)
from vtb.io.images import write_image  # ecrit un tableau numpy en fichier image (cree les dossiers au besoin)

NAMES = ["crack", "hole", "scratch"]  # les 3 classes utilisees partout dans ce fichier (l'ordre fixe leur numero)


@pytest.fixture  # fixture : donnees preparees pour les tests qui ont un parametre nomme "rich"
def rich(voc_folder: Path) -> list[ImageAnnotation]:  # renvoie 3 annotations d'image "riches" sur les vraies images de voc_folder
    """Trois images 200 x 100 avec etiquette d'image, boites, boite orientee et polygone (cas qui piegent les formats)."""
    images = sorted((voc_folder / "images").glob("*.jpg"))[:3]  # les 3 premieres images (triees pour un ordre stable)
    return [  # liste de 3 annotations
        ImageAnnotation(  # image 1 : un objet de chaque sorte
            str(images[0]),  # chemin de l'image
            200,  # largeur
            100,  # hauteur
            labels=["crack"],  # etiquette d'image entiere (classification)
            # deux boites dont une avec des coordonnees a virgule : verifie qu'aucun format n'arrondit en silence
            boxes=[Box("crack", 10, 20, 50, 60), Box("hole", 150.5, 0, 199, 40.25)],
            rotated=[RotatedBox("scratch", 100, 50, 80, 10, 30.0)],  # boite orientee : centre (100, 50), 80 x 10, angle 30 degres
            # polygone a 5 sommets irreguliers (pas un simple rectangle)
            polygons=[Polygon("hole", [(10.0, 10.0), (30.0, 12.0), (28.0, 40.0), (12.0, 38.0), (8.0, 20.0)])],
        ),
        # image 2 : une seule boite orientee avec un angle de 135 degres (au-dela de 90 : piege classique des conventions d'angle)
        ImageAnnotation(str(images[1]), 200, 100, rotated=[RotatedBox("scratch", 60, 40, 50, 6, 135.0)]),
        ImageAnnotation(str(images[2]), 200, 100),  # image sans objet : doit survivre a toutes les conversions
    ]


# parametrize : le meme test est lance une fois par ligne du tableau (format, fichier de sortie, tolerance en pixels)
@pytest.mark.parametrize(
    "fmt, target, tol",  # noms des parametres recus par le test
    [  # une ligne = un lancement du test
        ("coco", "set.json", 1e-6),  # COCO : un seul JSON pour tout le jeu ; tolerance quasi nulle
        ("labelme", "labelme", 1e-6),  # LabelMe : un JSON par image, dans un dossier
        ("labelstudio", "ls.json", 1e-6),  # Label Studio : coordonnees en pourcentage, d'ou de petits arrondis possibles
        ("cvat", "set.xml", 1e-2),  # CVAT : XML avec 2 decimales, d'ou une tolerance de 0.01 pixel
        ("vtb", "set.vtb.json", 1e-9),  # format natif : doit etre exact
    ],
)
# Verifie qu'ecrire puis relire dans ce format redonne les memes annotations. Attraperait : un objet perdu, un angle
# mal converti, une image vide oubliee, une coordonnee arrondie.
def test_roundtrip_without_loss(rich, voc_folder: Path, tmp_path: Path, fmt: str, target: str, tol: float):
    images_dir = voc_folder / "images"  # dossier des images (certains formats en ont besoin pour retrouver les fichiers)
    out = tmp_path / target  # fichier (ou dossier) de sortie dans le dossier temporaire
    # ecrit les annotations ; document_root sert a Label Studio pour construire les URL des images
    write_annotations(fmt, rich, out, class_names=NAMES, images_dir=images_dir, document_root=images_dir)
    back = read_annotations(fmt, images_dir, out, document_root=images_dir)  # relit ce qu'on vient d'ecrire
    assert same_annotations(rich, back, tol=tol) == []  # aucune difference a la tolerance pres


# Verifie que des predictions (boites avec score de confiance) exportees vers Label Studio gardent leur score et
# utilisent des URL "local-files". Attraperait : scores perdus (pre-annotation inutilisable) ou images introuvables.
def test_labelstudio_predictions_keep_scores_and_local_files_urls(rich, voc_folder: Path, tmp_path: Path):
    images_dir = voc_folder / "images"  # dossier des images
    scored = [  # copie des annotations ou chaque boite recoit un score de 0.8 (comme une prediction d'un modele)
        # pour chaque annotation a de rich : memes boites, avec score 0.8 en 6e argument
        ImageAnnotation(a.image_path, a.width, a.height, boxes=[Box(b.label, b.x1, b.y1, b.x2, b.y2, 0.8) for b in a.boxes])
        for a in rich  # pour chaque annotation des 3 images de la fixture rich
    ]
    # ecrit des taches Label Studio en tant que "predictions" (pre-annotations a corriger) et non annotations validees
    write_annotations("labelstudio", scored, tmp_path / "pre.json", document_root=images_dir.parent, as_predictions=True)
    tasks = json.loads((tmp_path / "pre.json").read_text())  # relit le JSON brut pour inspecter sa structure
    # l'image est referencee par une URL local-files relative a document_root : "images/<nom du fichier>"
    assert tasks[0]["data"]["image"] == LOCAL_FILES_PREFIX + "images/" + Path(rich[0].image_path).name
    assert tasks[0]["predictions"][0]["score"] == pytest.approx(0.8)  # le score est bien ecrit dans la prediction
    # relit les predictions (source="predictions") pour verifier l'aller-retour
    back = read_annotations("labelstudio", images_dir, tmp_path / "pre.json", document_root=images_dir.parent, source="predictions")
    # le score relu vaut 0.8 et l'URL est bien retraduite en chemin local de la bonne image
    assert back[0].boxes[0].score == pytest.approx(0.8) and Path(back[0].image_path) == Path(rich[0].image_path)


# Verifie qu'un fichier televerse dans Label Studio (renomme avec un prefixe aleatoire) est retrouve, et que la
# configuration XML d'etiquetage est correcte. Attraperait : images introuvables apres import, interface sans rotation.
def test_labelstudio_uploaded_file_names_and_config(voc_folder: Path):
    images_dir = voc_folder / "images"  # dossier des images
    # Label Studio renomme "img_0.jpg" en "1a2b3c4d-img_0.jpg" : le prefixe doit etre retire pour retrouver l'image locale
    assert resolve_image("/data/upload/3/1a2b3c4d-img_0.jpg", images_dir, None) == images_dir / "img_0.jpg"
    xml = labelstudio_config("rotated", NAMES)  # configuration pour des boites orientees avec nos 3 classes
    assert 'canRotate="true"' in xml and xml.count("<Label ") == 3  # rotation autorisee et une etiquette par classe


# Verifie que les differentes conventions de boite orientee (OpenCV, coins ordonnes, coin d'origine facon Label Studio)
# decrivent le meme rectangle. Attraperait : une boite tournee dans le mauvais sens ou decalee apres conversion.
def test_rotated_conventions_are_consistent():
    rbox = RotatedBox("scratch", 100, 50, 80, 10, 30.0)  # boite orientee de reference : centre (100, 50), 80 x 10, 30 degres
    corners = ordered_corners(rbox)  # ses 4 coins dans l'ordre p0, p0 + largeur, coin oppose, p0 + hauteur
    # cv2.boxPoints (convention OpenCV) donne les memes quatre coins, dans un autre ordre.
    # calcul : distances de chaque coin OpenCV a chaque coin ordonne, puis pour chacun la plus proche ; la pire doit etre ~0
    assert np.min(np.linalg.norm(np.array(rbox.corners())[:, None] - np.array(corners)[None], axis=2), axis=1).max() < 1e-4
    assert same_corners(rotated_from_corners("scratch", corners), rbox)  # reconstruire depuis les coins redonne la meme boite
    p0 = corners[0]  # coin d'origine (le point de depart de la convention Label Studio)
    assert same_corners(rotated_from_origin("scratch", p0[0], p0[1], 80, 10, 30.0), rbox)  # coin d'origine + taille + angle : idem
    assert math.dist(corners[0], corners[1]) == pytest.approx(80)  # le premier cote mesure bien la largeur (80)


# Verifie la conversion masque -> objets (polygone, boite, boite orientee) et objets -> masque, avec le filtre de taille.
# Attraperait : un decalage d'un pixel sur les bords, ou de petites taches de bruit prises pour des objets.
def test_masks_to_objects_and_back(tmp_path: Path):
    mask = np.zeros((60, 80), np.uint8)  # masque noir 60 lignes x 80 colonnes (0 = fond)
    mask[10:20, 5:25] = 255  # rectangle 20 x 10
    mask[40:42, 40:41] = 255  # objet de 2 pixels : sous min_area
    # contours -> polygones ; min_area=10 ignore les objets de moins de 10 pixels, epsilon=0 garde tous les points du contour
    polygons = mask_to_polygons(mask, ["defect"], min_area=10, epsilon=0.0)
    # un seul objet garde ; largeur 19 car le contour passe par les centres des pixels (x de 5 a 24)
    assert len(polygons) == 1 and polygons[0].bounding_box().width == pytest.approx(19)  # centres de pixels
    boxes = mask_to_boxes(mask, ["defect"], min_area=10)  # une boite droite par objet
    assert [(b.x1, b.y1, b.x2, b.y2) for b in boxes] == [(5.0, 10.0, 25.0, 20.0)]  # coins en bord de pixel : x2 = dernier x + 1
    rotated = mask_to_rotated_boxes(mask, ["defect"], min_area=10)  # une boite orientee minimale par objet
    assert sorted([rotated[0].w, rotated[0].h]) == pytest.approx([10, 20])  # memes dimensions, quel que soit le sens choisi
    redrawn = objects_to_mask(ImageAnnotation("a.png", 80, 60, boxes=boxes), ["defect"])  # redessine le masque a partir des boites
    # 200 pixels (20 x 10) a la valeur 1 (classe 0 + 1) et tout le rectangle d'origine est rempli
    assert int(redrawn.sum()) == 200 and redrawn[10:20, 5:25].all()


# Verifie la lecture d'un dossier de masques : multi-classes (valeur du pixel = numero de classe) et binaire avec un
# bord anti-crenele. Attraperait : des classes confondues, ou un objet fantome cree par le lissage des bords.
def test_mask_folder_reader_multiclass_and_anti_aliasing(tmp_path: Path):
    image_dir, mask_dir = tmp_path / "img", tmp_path / "masks"  # un dossier pour les images, un autre pour les masques
    write_image(image_dir / "a.jpg", np.zeros((40, 40, 3), np.uint8))  # image noire 40 x 40 (son contenu n'importe pas ici)
    classes = np.zeros((40, 40), np.uint8)  # masque de classes : 0 = fond
    classes[2:12, 2:12] = 1  # classe 1 = crack
    classes[20:30, 20:35] = 2  # classe 2 = hole
    write_image(mask_dir / "a.png", classes)  # le masque porte le meme nom que l'image (extension .png)
    anns = read_mask_folder(image_dir, mask_dir, ["crack", "hole"], kind="boxes")  # lit image + masque et en tire des boites
    assert sorted(b.label for b in anns[0].boxes) == ["crack", "hole"]  # une boite par classe, avec le bon nom
    binary = np.zeros((40, 40), np.uint8)  # nouveau masque, binaire cette fois (0 = fond, 255 = objet)
    binary[5:15, 5:15] = 255  # un carre d'objet
    binary[4, 5:15] = 3  # bord anti-crenele : ne doit pas creer un objet de plus
    write_image(mask_dir / "a.png", binary)  # remplace le masque precedent
    # lecture via le registre de formats ("masks"), en boites orientees : un seul objet attendu
    assert len(read_annotations("masks", image_dir, mask_dir, kind="rotated")[0].rotated) == 1


# Verifie que chaque format a un lecteur ET un ecrivain, et qu'une conversion directe COCO -> CVAT ne perd rien.
# Attraperait : un format ajoute a moitie, ou une conversion qui casse en passant d'un format a l'autre.
def test_convert_and_registries(rich, voc_folder: Path, tmp_path: Path):
    images_dir = voc_folder / "images"  # dossier des images
    # les deux registres (dictionnaires nom de format -> fonction) contiennent exactement les 8 formats attendus
    assert set(READERS) == set(WRITERS) == {"voc", "yolo", "coco", "labelme", "labelstudio", "cvat", "masks", "vtb"}
    write_annotations("coco", rich, tmp_path / "a.json", class_names=NAMES, images_dir=images_dir)  # ecrit en COCO
    # convertit le fichier COCO en fichier CVAT en une seule fonction
    convert("coco", images_dir, tmp_path / "a.json", "cvat", tmp_path / "a.xml", class_names=NAMES, images_dir=images_dir)
    # le CVAT obtenu redonne les annotations d'origine (tolerance 0.01 pixel a cause des 2 decimales de CVAT)
    assert same_annotations(rich, read_annotations("cvat", images_dir, tmp_path / "a.xml"), tol=1e-2) == []
    # compare une annotation complete avec la meme image vide : le comparateur doit signaler des differences
    differences = annotation_differences(rich[0], ImageAnnotation(rich[0].image_path, 200, 100))
    assert any("boxes" in d for d in differences)  # au moins une difference mentionne les boites (le comparateur n'est pas aveugle)
