"""vtb.viz : chaque fonction publique sur les entrees principales (backend Agg, donnees synthetiques)."""

# Ces tests ne regardent pas les images produites : ils verifient que chaque fonction d'affichage accepte les
# formats d'entree courants (numpy, tenseur, 8 / 16 bits, gris / couleur, chemin de fichier) sans planter,
# et renvoie le bon type d'objet (figure, axe, image) de la bonne taille.

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[int], X | None) partout

import ast  # ast : lit un fichier Python comme un arbre (pour lister ses imports sans l'executer)
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import matplotlib  # matplotlib : la bibliotheque de graphiques

# backend "Agg" : dessine en memoire sans ouvrir de fenetre (indispensable sans ecran) ; a choisir AVANT pyplot,
# d'ou les imports qui suivent places apres cette ligne ("noqa: E402" dit a ruff que c'est voulu)
matplotlib.use("Agg")
# OpenCV : lecture / ecriture et traitement d'images
import cv2  # noqa: E402

# pyplot : interface de matplotlib pour creer et fermer des figures
import matplotlib.pyplot as plt  # noqa: E402

# numpy : calcul sur des tableaux de nombres (une image = un tableau)
import numpy as np  # noqa: E402

# pandas : tableaux de donnees avec colonnes nommees (DataFrame)
import pandas as pd  # noqa: E402

# pytest : lance les tests ; fixtures, approx, raises, capsys...
import pytest  # noqa: E402

# PyTorch : tenseurs et reseaux de neurones
import torch  # noqa: E402

# Figure : le type d'objet "figure" de matplotlib (pour verifier ce que renvoient les fonctions)
from matplotlib.figure import Figure  # noqa: E402

# le module teste : fonctions d'affichage de la toolbox
from vtb import viz  # noqa: E402


# autouse=True : cette fixture s'applique automatiquement a tous les tests du fichier
@pytest.fixture(autouse=True)
def _close_figures():  # ferme toutes les figures apres chaque test (sinon elles s'accumulent en memoire)
    yield  # le test s'execute ici
    plt.close("all")  # apres le test : ferme toutes les figures ouvertes


@pytest.fixture  # fixture : image grise aleatoire
def gray(rng: np.random.Generator) -> np.ndarray:  # renvoie un tableau 60 x 80 en 8 bits
    return rng.integers(0, 256, (60, 80), dtype=np.uint8)  # valeurs aleatoires de 0 a 255


@pytest.fixture  # fixture : image couleur aleatoire
def color(rng: np.random.Generator) -> np.ndarray:  # renvoie un tableau 60 x 80 x 3 en 8 bits
    return rng.integers(0, 256, (60, 80, 3), dtype=np.uint8)  # valeurs aleatoires de 0 a 255 sur 3 canaux


# Verifie que le dossier viz/ n'importe aucun autre module de la toolbox. Attraperait : une dependance cachee qui
# empecherait de copier viz/ seul dans un autre projet.
def test_module_is_standalone():
    """Aucun import d'un autre module vtb : le dossier viz/ se copie seul."""
    for path in Path(viz.__file__).parent.glob("*.py"):  # pour chaque fichier Python du dossier viz/
        tree = ast.parse(path.read_text(encoding="utf-8"))  # lit le fichier comme un arbre syntaxique
        for node in ast.walk(tree):  # parcourt tous les elements de l'arbre
            # "from X import ..." absolu (level 0 = pas un import relatif "from . import")
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                assert not node.module.startswith("vtb"), f"{path.name} importe {node.module}"  # X ne doit pas etre un module vtb
            if isinstance(node, ast.Import):  # "import X"
                assert not any(a.name.startswith("vtb") for a in node.names), path.name  # aucun des X n'est un module vtb


# Verifie que to_display transforme toutes les sortes d'entrees en image affichable 8 bits (H x W ou H x W x 3).
# Attraperait : une image affichee noire, saturee, aux couleurs inversees ou un plantage selon le format d'entree.
def test_to_display_inputs(gray, color):
    assert viz.to_display(gray) is gray or np.array_equal(viz.to_display(gray), gray)  # image grise 8 bits : inchangee
    assert viz.to_display(gray[..., None]).shape == (60, 80)  # un canal unique en 3e dimension est retire
    assert np.array_equal(viz.to_display(color, bgr=True), color[..., ::-1])  # BGR (ordre OpenCV) remis en RGB
    rgba = np.dstack([color, np.full((60, 80), 255, np.uint8)])  # image RGBA : couleur + canal de transparence
    assert viz.to_display(rgba).shape == (60, 80, 3)  # le canal de transparence est retire
    chw = np.moveaxis(color, -1, 0)  # meme image au format canaux en premier (C, H, W), comme PyTorch
    assert np.array_equal(viz.to_display(chw), color)  # remise en (H, W, C)
    # uint16 12 bits : etirement min-max
    deep = (gray.astype(np.uint16) * 16).astype(np.uint16)  # valeurs jusqu'a 4080 (camera 12 bits)
    out = viz.to_display(deep)  # conversion pour l'affichage
    assert out.dtype == np.uint8 and out.min() == 0 and out.max() == 255  # 8 bits, toute la plage utilisee
    # float [0, 1] -> x255 ; float [0, 255] -> tel quel ; autre plage -> min-max
    assert np.array_equal(viz.to_display(gray / 255.0), gray)  # flottant 0-1 : multiplie par 255
    assert np.array_equal(viz.to_display(gray.astype(np.float32)), gray)  # flottant 0-255 : garde tel quel
    signed = viz.to_display(gray.astype(np.float32) - 1000)  # valeurs negatives (plage quelconque)
    assert signed.min() == 0 and signed.max() == 255  # etirees min-max
    # tenseur torch avec gradient, lot de 1, normalisation inverse
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # moyenne et ecart-type ImageNet par canal
    x = torch.from_numpy(color).permute(2, 0, 1).float() / 255  # image en tenseur (C, H, W) dans [0, 1]
    # image normalisee ((x - moyenne) / ecart-type) qui suit le gradient (comme dans un modele en cours d'entrainement)
    normalized = ((x - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]).requires_grad_()
    back = viz.to_display(normalized[None], mean=mean, std=std)  # lot de 1 image ; to_display annule la normalisation
    assert back.shape == (60, 80, 3) and np.abs(back.astype(int) - color.astype(int)).max() <= 1  # image d'origine a 1 niveau pres
    assert viz.to_display(np.zeros((5, 5), bool)).dtype == np.uint8  # masque booleen : converti en 8 bits
    assert viz.to_display(np.full((5, 5), np.nan)).dtype == np.uint8  # image pleine de NaN : pas de plantage
    with pytest.raises(ValueError):  # on attend une erreur...
        viz.to_display(np.zeros((2, 3, 4, 5)))  # ...pour un lot de plusieurs images (impossible a afficher en une)


# Verifie show (une image) et grid (planche de plusieurs images de formats differents).
# Attraperait : un plantage sur un melange numpy / tenseur / 16 bits / chemins, ou une figure non enregistree.
def test_show_and_grid(tmp_path, gray, color):
    ax = viz.show(gray, "gris", save=tmp_path / "show.jpg")  # affiche avec un titre et enregistre en JPEG
    assert (tmp_path / "show.jpg").exists() and ax.get_title() == "gris"  # fichier ecrit, titre en place
    ax = viz.show(gray.astype(np.float32) * 3.7, colorbar=True)  # image flottante avec barre de couleurs
    assert len(ax.figure.axes) == 2  # 2 axes : l'image et la barre de couleurs
    fig = viz.grid(  # planche de 4 images de formats differents
        [gray, color, torch.rand(3, 40, 40), gray.astype(np.uint16) * 200],  # gris, couleur, tenseur, 16 bits
        ["a", "b\nligne 2", "c", "d"],  # titres (dont un sur deux lignes)
        ncols=3,  # 3 images par ligne
        borders=[True, False, None, "orange"],  # cadre par defaut, aucun cadre, aucun cadre, cadre orange
        suptitle="planche",  # titre general
        save=tmp_path / "grid.png",  # fichier de sortie
    )
    assert isinstance(fig, Figure) and (tmp_path / "grid.png").exists()  # une figure, enregistree
    assert len(fig.axes) == 1  # une seule image assemblee
    fig = viz.grid(torch.rand(10, 1, 32, 32), ncols=5)  # un lot de 10 images grises en tenseur
    assert isinstance(fig, Figure)  # renvoie bien une figure
    path = tmp_path / "x.png"  # chemin d'une image sur disque
    cv2.imwrite(str(path), color)  # ecrit l'image couleur
    assert isinstance(viz.grid([path, str(path)]), Figure)  # accepte aussi des chemins (Path ou texte)
    with pytest.raises(ValueError):  # on attend une erreur...
        viz.grid([])  # ...pour une liste vide


# Verifie compare (cote a cote, avec difference), histogram (un trait par canal) et profile (intensite le long d'une ligne).
# Attraperait : un nombre d'axes ou de courbes faux, ou un plantage sur des images de tailles differentes.
def test_compare_histogram_profile(tmp_path, gray, color):
    fig = viz.compare({"avant": gray, "apres": 255 - gray}, diff=True, save=tmp_path / "cmp.png")  # 2 images + leur difference
    assert len(fig.axes) == 3 and (tmp_path / "cmp.png").exists()  # 3 axes, fichier ecrit
    assert len(viz.compare([gray, color[:30]]).axes) == 2  # liste de 2 images de tailles differentes : 2 axes
    ax = viz.histogram(color, save=tmp_path / "hist.png")  # histogramme d'une image couleur
    assert len(ax.lines) == 3  # une courbe par canal (R, G, B)
    assert len(viz.histogram(gray.astype(np.uint16) * 100, log=True).lines) == 1  # 16 bits gris, echelle log : une courbe
    for kwargs in ({}, {"row": 10}, {"column": 5}):  # profil par defaut, sur la ligne 10, sur la colonne 5
        fig = viz.profile(color, **kwargs)  # trace le profil d'intensite
        assert len(fig.axes) == 2  # 2 axes : l'image et la courbe
    viz.profile(gray, save=tmp_path / "profile.png")  # profil d'une image grise, enregistre
    assert (tmp_path / "profile.png").exists()  # fichier ecrit


# Verifie les superpositions (masque, carte de chaleur) et le dessin de boites (droites, orientees, polygones, detections).
# Attraperait : un masque de taille differente qui ferait planter, ou des boites dessinees au mauvais endroit.
def test_overlays_and_boxes(gray, color):
    mask = np.zeros((30, 40), np.int64)  # masque de classes plus petit que l'image (30 x 40 au lieu de 60 x 80)
    mask[5:15, 5:15] = 1  # zone de classe 1
    mask[20:25, 20:35] = 2  # zone de classe 2
    out = viz.overlay_mask(color, mask, class_names=["fond", "a", "b"])  # masque colore par classe sur l'image
    assert out.shape == (60, 80, 3) and out.dtype == np.uint8  # a la taille de l'image, en 8 bits
    assert viz.overlay_mask(gray, torch.from_numpy(mask > 0)).shape == (60, 80, 3)  # masque booleen en tenseur, image grise
    assert viz.overlay_mask(gray, np.zeros((60, 80))).shape == (60, 80, 3)  # masque vide : pas de plantage
    # carte de chaleur 15 x 20 agrandie et superposee ; echelle fixe 0-1
    heat = viz.overlay_heatmap(gray, np.random.default_rng(0).random((15, 20)), vmin=0, vmax=1)
    assert heat.shape == (60, 80, 3)  # image couleur de la taille de l'image
    boxes = np.array([[5, 5, 30, 30], [40, 10, 75, 50]], float)  # 2 boites (x1, y1, x2, y2)
    # dessine les boites avec classe et score
    drawn = viz.draw_boxes(gray, boxes, labels=[0, 1], scores=[0.9, 0.4], class_names=["crack", "hole"])
    assert drawn.shape == (60, 80, 3) and not np.array_equal(drawn[..., 0], gray)  # couleur, et l'image a bien ete modifiee
    # min_side=240 : l'image est agrandie pour que le texte soit lisible (60 x 80 -> 240 x 320)
    assert viz.draw_boxes(color, torch.tensor(boxes), labels=["x", "y"], min_side=240).shape == (240, 320, 3)
    rotated = viz.draw_boxes(gray, np.array([[40, 30, 30, 8, 30.0]]), rotated=True, labels=[0])  # boite orientee (cx, cy, w, h, angle)
    assert rotated.shape == (60, 80, 3)  # dessinee sans plantage
    poly = viz.draw_boxes(gray, None, polygons=[np.array([[5, 5], [30, 8], [20, 40]])], labels=[1])  # un triangle
    assert poly.shape == (60, 80, 3)  # dessine sans plantage
    assert np.array_equal(viz.draw_boxes(color, np.zeros((0, 4))), color)  # aucune boite : image rendue inchangee
    det = viz.draw_detections(  # verite terrain et predictions sur la meme image
        gray,  # image de fond
        truth={"boxes": boxes, "labels": [0, 1]},  # boites vraies
        pred={"boxes": boxes + 2, "labels": np.array([0, 1]), "scores": np.array([0.9, 0.1])},  # predictions decalees de 2 pixels
        class_names=["crack", "hole"],  # noms des classes
        conf=0.25,  # seuil : la prediction a 0.1 n'est pas dessinee
        min_side=120,  # agrandissement pour la lisibilite
    )
    assert det.shape == (120, 160, 3)  # image agrandie a 120 x 160
    assert viz.draw_detections(gray, truth=boxes).shape == (60, 80, 3)  # verite seule, donnee comme simple tableau


# Verifie les planches par classe, l'affichage d'un lot (classification, etiquettes douces, detection), l'apercu
# d'augmentations et les etapes d'un pipeline. Attraperait : un plantage sur l'un de ces formats de donnees.
def test_class_montage_batch_augmentations_pipeline(tmp_path, rng):
    paths, labels = [], []  # chemins des images et leurs classes
    for k, name in enumerate(("alpha", "beta", "gamma")):  # 3 classes
        for i in range(4 + k):  # 4, 5 puis 6 images selon la classe
            path = tmp_path / name / f"{i}.png"  # chemin de l'image
            path.parent.mkdir(parents=True, exist_ok=True)  # cree le dossier de la classe
            cv2.imwrite(str(path), rng.integers(0, 256, (40, 50), dtype=np.uint8))  # image grise aleatoire 40 x 50
            paths.append(path)  # memorise le chemin
            labels.append(name)  # et la classe
    fig = viz.class_montage(paths, labels, per_class=5, save=tmp_path / "montage.jpg")  # au plus 5 images par classe
    assert isinstance(fig, Figure) and (tmp_path / "montage.jpg").exists()  # figure renvoyee et enregistree
    batch = (torch.randn(6, 3, 32, 32), torch.tensor([0, 1, 2, 0, 1, 2]))  # lot de classification : 6 images normalisees + classes
    # affichage du lot en annulant la normalisation (moyenne 0.5, ecart-type 0.2)
    assert isinstance(viz.show_batch(batch, class_names=["a", "b", "c"], mean=(0.5,) * 3, std=(0.2,) * 3), Figure)
    soft = torch.softmax(torch.randn(4, 3), dim=1)  # etiquettes douces (probabilites par classe, comme apres MixUp)
    assert isinstance(viz.show_batch(torch.rand(4, 1, 20, 20), soft, ["a", "b", "c"]), Figure)  # images grises + etiquettes douces
    # cibles de detection : une boite de classe 1 par image, pour 2 images
    targets = [{"boxes": torch.tensor([[2.0, 2.0, 10.0, 12.0]]), "labels": torch.tensor([1])} for _ in range(2)]
    assert isinstance(viz.show_batch([torch.rand(3, 20, 20)] * 2, targets, ["a", "b"]), Figure)  # lot de detection

    from torchvision import tv_tensors  # types qui marquent un tenseur comme boites ou masque
    from torchvision.transforms import v2  # transformations d'images de torchvision

    image = rng.integers(0, 256, (40, 50), dtype=np.uint8)  # image grise aleatoire
    fig = viz.show_augmentations(image, lambda x: np.fliplr(x).copy(), n=3)  # augmentation = simple fonction numpy, 3 tirages
    assert isinstance(fig, Figure)  # figure renvoyee
    flip = v2.RandomHorizontalFlip(p=1.0)  # retournement horizontal toujours applique
    fig = viz.show_augmentations(image, flip, n=2)  # echoue sur numpy ? refait sur tenseur
    assert isinstance(fig, Figure)  # figure renvoyee
    tensor = torch.rand(3, 40, 50)  # image couleur en tenseur
    boxes = tv_tensors.BoundingBoxes([[2, 2, 20, 20]], format="XYXY", canvas_size=(40, 50))  # une boite sur cette image
    assert isinstance(  # apercu d'une rotation aleatoire (jusqu'a 10 degres) avec la boite qui suit
        # augmentation sur un tenseur avec une cible de detection (boites + classes), 2 tirages
        viz.show_augmentations(tensor, v2.RandomAffine(degrees=10), target={"boxes": boxes, "labels": torch.tensor([0])}, n=2),
        Figure,  # le resultat doit etre une figure
    )
    mask = tv_tensors.Mask(torch.zeros(40, 50, dtype=torch.int64))  # masque vide
    # apercu avec un masque comme cible, enregistre
    assert isinstance(viz.show_augmentations(tensor, flip, target=mask, n=2, save=tmp_path / "aug.png"), Figure)
    # etapes successives d'un pipeline : flou gaussien, puis passage en flottant 0-1
    fig = viz.pipeline_steps(image, [("flou", lambda x: cv2.GaussianBlur(x, (5, 5), 0)), ("float", lambda x: x.astype(np.float32) / 255)])
    assert isinstance(fig, Figure)  # figure renvoyee
    # etapes non cumulees (chacune appliquee a l'image d'origine), enregistre
    assert isinstance(viz.pipeline_steps(image, [("inverse", lambda x: 255 - x)], cumulative=False, save=tmp_path / "pipe.png"), Figure)


# Verifie les graphiques de classification multi-classes : matrice de confusion, rapport, barres, ROC, PR, calibration,
# galerie d'erreurs. Attraperait : un rapport aux chiffres faux, ou un plantage sur l'une des entrees acceptees.
def test_classification_metrics(tmp_path, rng):
    names = ["a", "b", "c"]  # 3 classes
    y_true = rng.integers(0, 3, 90)  # 90 vraies classes
    probs = rng.dirichlet((1, 1, 1), 90)  # 90 jeux de probabilites aleatoires qui somment a 1
    probs[np.arange(90), y_true] += 0.8  # favorise la vraie classe : modele plutot bon
    probs /= probs.sum(axis=1, keepdims=True)  # renormalise chaque ligne a 1
    y_pred = probs.argmax(axis=1)  # classe predite = la plus probable
    for normalize in ("true", "pred", "all", None):  # chaque facon de normaliser la matrice de confusion
        ax = viz.confusion_matrix(y_true, y_pred, names, normalize=normalize)  # trace la matrice
        assert ax.get_title().startswith("exactitude")  # le titre annonce l'exactitude
    # classes vraies en texte, predictions en tenseur : doit aussi fonctionner
    viz.confusion_matrix([names[i] for i in y_true], torch.tensor(y_pred), names, save=tmp_path / "cm.png")
    assert (tmp_path / "cm.png").exists()  # fichier ecrit
    report = viz.classification_report(y_true, y_pred, names)  # tableau precision / rappel / F1 / effectif
    assert list(report.columns) == ["precision", "recall", "f1", "support"]  # colonnes attendues
    assert report.loc["accuracy", "f1"] == pytest.approx(np.mean(y_true == y_pred), abs=1e-4)  # ligne accuracy = exactitude reelle
    assert report.loc["macro avg", "support"] == 90  # moyenne macro : effectif total 90
    ax = viz.per_class_bars(report.loc[names, "f1"], title="f1", reference=0.5)  # barres du F1 par classe, ligne de reference 0.5
    assert len(ax.patches) == 3  # une barre par classe
    # depuis un dictionnaire, avec une reference par classe, barres triees, enregistre
    viz.per_class_bars({"a": 0.2, "b": 0.8}, reference={"a": 0.3, "b": 0.7}, sort=True, save=tmp_path / "bars.png")
    ax = viz.roc_curves(y_true, probs, names, mark_threshold=0.5)  # courbes ROC (une par classe), seuil 0.5 marque
    assert "AUC macro" in ax.get_title()  # le titre donne l'AUC moyenne
    ax = viz.pr_curves(y_true, probs, names)  # courbes precision-rappel
    assert len(ax.get_legend().get_texts()) == 3  # une entree de legende par classe
    ax = viz.calibration(y_true, probs, save=tmp_path / "calib.png")  # diagramme de calibration
    assert "ECE" in ax.get_title()  # le titre donne l'erreur de calibration
    images = [rng.integers(0, 256, (20, 20), dtype=np.uint8) for _ in range(90)]  # une petite image par exemple
    fig = viz.errors_gallery(images, y_true, y_pred, probs, names, k=6)  # les 6 pires erreurs
    assert isinstance(fig, Figure)  # figure renvoyee
    assert isinstance(viz.errors_gallery(images, y_true, y_true, probs, names), Figure)  # aucune erreur


# Verifie les graphiques binaires (defaut / normal) : ROC, PR, balayage de seuil, histogramme des scores, calibration.
# Attraperait : un tableau de seuils incoherent (rappel qui remonte, comptes qui ne collent pas) ou un plantage a une classe.
def test_binary_metrics(tmp_path, rng):
    is_defect = rng.random(200) < 0.3  # 200 images, environ 30 % de defauts
    scores = rng.normal(0, 1, 200) + 2.0 * is_defect  # scores : les defauts ont en moyenne 2 de plus
    ax = viz.roc_curves(is_defect, scores, mark_threshold=1.0, save=tmp_path / "roc.png")  # ROC binaire, seuil 1.0 marque
    assert "AUC" in ax.get_legend().get_texts()[0].get_text()  # la legende donne l'AUC
    viz.pr_curves(torch.tensor(is_defect), torch.tensor(scores), mark_threshold=1.0)  # accepte aussi des tenseurs
    fig, table = viz.threshold_sweep(is_defect, scores, chosen=1.0, save=tmp_path / "sweep.png")  # metriques pour chaque seuil
    # figure renvoyee, et tableau avec toutes les colonnes attendues
    assert isinstance(fig, Figure) and {"threshold", "precision", "recall", "f1", "false_alarm", "tp", "fp", "fn", "tn"} <= set(
        table.columns  # colonnes du tableau
    )
    assert table["recall"].is_monotonic_decreasing  # plus le seuil monte, moins on trouve de defauts
    assert (table["tp"] + table["fn"] == is_defect.sum()).all()  # trouves + manques = nombre total de defauts, a chaque seuil
    ax = viz.score_histogram(scores, is_defect, threshold=1.0, log=True)  # histogrammes des scores normaux / defauts
    assert len(ax.get_legend().get_texts()) == 3  # legende : normaux, defauts, seuil
    probs = 1 / (1 + np.exp(-scores))  # scores changes en probabilites (fonction sigmoide)
    assert "ECE" in viz.calibration(is_defect, probs).get_title()  # calibration binaire
    one_class = viz.roc_curves(np.ones(10, bool), rng.random(10))  # une seule classe presente : ROC non definie
    assert one_class is not None  # mais pas de plantage


# Verifie les tableaux de metriques (un essai, plusieurs essais) et les graphiques de comparaison et de latence.
# Attraperait : des colonnes non numeriques qui feraient planter les graphiques, ou un arrondi mal applique.
def test_tables_and_runs(tmp_path):
    table = viz.metrics_table({"acc": 0.98765, "f1": 0.9, "name": "x", "per_class": [1, 2]})  # metriques d'un seul essai
    assert list(table.columns) == ["valeur"] and table.loc["acc", "valeur"] == 0.988  # une colonne "valeur", arrondi a 3 decimales
    runs = {"resnet18": {"acc": 0.99, "f1": 0.98, "curve": [1, 2]}, "cnn": {"acc": 0.95, "f1": 0.94}}  # deux essais
    table = viz.metrics_table(runs)  # une ligne par essai
    assert list(table.index) == ["resnet18", "cnn"] and "curve" not in table.columns  # les listes (courbes) sont ecartees
    table = viz.metrics_table([{"name": "a", "acc": 0.5}, {"name": "b", "acc": 0.75}])  # liste d'essais avec une cle name
    assert list(table.index) == ["a", "b"]  # name sert de nom de ligne
    fig = viz.compare_runs(runs, ["acc", "f1"], save=tmp_path / "runs.png")  # barres comparant les essais sur 2 metriques
    assert isinstance(fig, Figure) and (tmp_path / "runs.png").exists()  # figure renvoyee et enregistree
    assert isinstance(viz.compare_runs(pd.DataFrame(runs).T, "acc"), Figure)  # accepte un DataFrame et une seule metrique
    # temps d'inference par moteur (en millisecondes)
    ax = viz.latency_bars({"onnxruntime": 10.2, "openvino": 2.5, "torch": 25.0}, save=tmp_path / "lat.png")
    assert len(ax.patches) == 3  # une barre par moteur


# Verifie les courbes d'apprentissage (depuis un tableau, un CSV, un dictionnaire, une liste) et le suivi en direct.
# Attraperait : un historique d'entrainement illisible selon son format, ou un suivi en direct muet hors notebook.
def test_history_and_liveplot(tmp_path, capsys):  # capsys : capture ce qui est affiche
    history = pd.DataFrame(  # historique d'un entrainement de 6 epoques
        {  # une colonne par grandeur suivie
            "epoch": range(6),  # numero d'epoque 0 a 5
            "train_loss": [1.0, 0.7, 0.5, 0.4, 0.35, 0.3],  # perte d'entrainement
            "val_loss": [1.1, 0.8, 0.6, 0.55, 0.6, 0.65],  # perte de validation (remonte a la fin : sur-apprentissage)
            "train_acc": [0.5, 0.6, 0.7, 0.8, 0.85, 0.9],  # exactitude d'entrainement
            "val_acc": [0.45, 0.6, 0.7, 0.75, 0.74, 0.73],  # exactitude de validation
            "lr": [1e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5],  # taux d'apprentissage
        }
    )
    fig = viz.plot_history(history, save=tmp_path / "history.png")  # trace l'historique
    assert len(fig.axes) == 3 and (tmp_path / "history.png").exists()  # 3 graphiques (perte, exactitude, taux) ; fichier ecrit
    history.to_csv(tmp_path / "history.csv", index=False)  # meme historique en CSV
    # depuis le CSV, seulement la perte, avec le meilleur val_acc (maximum) marque : un seul graphique
    assert len(viz.plot_history(tmp_path / "history.csv", metrics=["loss"], best="val_acc:max").axes) == 1
    assert len(viz.plot_history({"loss": [3, 2, 1]}).axes) == 1  # depuis un dictionnaire
    # depuis une liste de dictionnaires (un par epoque)
    assert len(viz.plot_history([{"loss": 1.0, "val_loss": 1.2}, {"loss": 0.5, "val_loss": 0.9}]).axes) == 1

    live = viz.LivePlot(("train_loss", "val_loss"), title="essai")  # graphique mis a jour a chaque epoque
    for epoch in range(3):  # 3 epoques
        # valeurs en tenseur ou en nombre simple : les deux sont acceptes
        live.update(epoch, train_loss=torch.tensor(1.0 / (epoch + 1)), val_loss=1.2 / (epoch + 1))
    out = capsys.readouterr().out  # texte affiche pendant les mises a jour
    assert out.count("epoque") == 3  # hors notebook : une ligne par epoque
    # historique garde en tableau : une colonne par grandeur, une ligne par epoque
    assert list(live.history.columns) == ["epoch", "train_loss", "val_loss"] and len(live.history) == 3
    assert isinstance(live.close(), Figure)  # close renvoie la figure finale


# Verifie les outils de style : couleurs de classes distinctes et stables, reglage de matplotlib, rendu en JPEG / PNG,
# enregistrement. Attraperait : deux classes de meme couleur, ou un format d'image different de celui demande.
def test_style_helpers(tmp_path):
    colors = viz.class_colors(25)  # 25 couleurs
    assert len(colors) == 25 and len(set(colors)) == 25  # 25 couleurs toutes differentes
    named = viz.class_colors(["crack", "hole"])  # couleurs pour des classes nommees
    assert named == {"crack": colors[0], "hole": colors[1]}  # memes couleurs que la liste, dans l'ordre
    viz.setup_notebook(dpi=70, fmt="jpeg")  # hors notebook : ne fait que regler matplotlib
    assert plt.rcParams["figure.dpi"] == 70  # la resolution est bien reglee
    fig = viz.plot_history({"loss": [1, 0.5]})  # une figure quelconque
    assert viz.render(fig)[:2] == b"\xff\xd8"  # jpeg
    viz.setup_notebook(dpi=100, fmt="png")  # repasse en PNG
    assert viz.render(fig)[:4] == b"\x89PNG"  # les 4 premiers octets d'un fichier PNG
    path = viz.save(fig, tmp_path / "sub" / "fig.jpg")  # enregistre (en creant le sous-dossier)
    assert path.exists()  # fichier ecrit
    assert not viz.in_notebook()  # pytest ne tourne pas dans un notebook
