"""vtb.toolkit : outils donnees, entrainement, statistiques et ligne de commande, sur des donnees synthetiques."""

# vtb.toolkit regroupe de petits outils independants : inventaire d'un dossier d'images, poids de classes,
# redimension, extraction d'images d'une video, decoupage en patchs, controles d'entrainement (recherche du taux
# d'apprentissage, sur-apprentissage d'un lot), statistiques (intervalles de confiance, tests) et ligne de commande.

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[int], X | None) partout

import functools  # functools.partial : fixer certains arguments d'une fonction a l'avance
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import cv2  # OpenCV : traitement d'images et ecriture de videos
import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)
import pandas as pd  # pandas : tableaux de donnees avec colonnes nommees (DataFrame)
import pytest  # pytest : lance les tests ; fixtures, approx, raises, capsys...
import torch  # PyTorch : tenseurs (tableaux de nombres) et reseaux de neurones
from sklearn.metrics import accuracy_score, f1_score  # metriques de scikit-learn : exactitude et score F1
from torch import nn  # nn : briques de reseaux de neurones (couches, fonctions de perte)

from vtb.io.images import read_image, write_image  # lecture / ecriture d'images
from vtb.toolkit import dataset as ds  # outils donnees, abreges en ds
from vtb.toolkit import stats  # outils statistiques
from vtb.toolkit import training as tt  # outils d'entrainement, abreges en tt
from vtb.toolkit.__main__ import main as cli  # point d'entree de la ligne de commande, appele ici comme une fonction

# --------------------------------------------------------------------------- donnees


# Verifie l'inventaire d'un dossier d'images : nombre de fichiers, fichier illisible, dossiers, tableau resume.
# Attraperait : un fichier casse non signale (plantage plus tard en plein entrainement) ou des tailles mal comptees.
def test_folder_summary_and_summarize(class_folder: Path) -> None:
    (class_folder / "beta" / "broken.png").write_bytes(b"not an image")  # ajoute un faux fichier image dans beta
    files = ds.folder_summary(class_folder)  # une ligne par fichier : dossier, lisible ?, taille...
    assert len(files) == 37  # 36 images + le fichier casse
    assert int((~files["readable"]).sum()) == 1  # un seul fichier illisible (~ = "non")
    assert set(files["folder"]) == {"alpha", "beta", "gamma"}  # les trois dossiers de classes
    table = ds.summarize(files)  # resume par dossier
    assert list(table["folder"]) == ["alpha", "beta", "gamma", "TOTAL"]  # une ligne par dossier + une ligne de total
    assert table.loc[table["folder"] == "beta", "unreadable"].item() == 1  # le fichier illisible est compte dans beta
    assert table.loc[table["folder"] == "alpha", "sizes (HxW)"].item() == "48x48 (12)"  # alpha : 12 images de 48 x 48


# Verifie la moyenne et l'ecart-type par canal (servent a la normalisation) contre un calcul numpy direct.
# Attraperait : des statistiques fausses, donc une normalisation qui decale toutes les images.
def test_mean_std_matches_numpy(tmp_path: Path, rng: np.random.Generator) -> None:
    images = [rng.integers(0, 256, (20, 30, 3), dtype=np.uint8) for _ in range(4)]  # 4 images couleur aleatoires
    paths = []  # chemins des fichiers ecrits
    for i, image in enumerate(images):  # pour chaque image et son numero
        paths.append(tmp_path / f"{i}.png")  # chemin 0.png, 1.png...
        write_image(paths[-1], image)  # ecrit l'image ([-1] = dernier chemin ajoute)
    mean, std = ds.mean_std(paths)  # moyenne et ecart-type par canal, valeurs ramenees a [0, 1]
    stacked = np.stack(images).reshape(-1, 3) / 255.0  # tous les pixels de toutes les images, une colonne par canal
    np.testing.assert_allclose(mean, stacked.mean(0), atol=1e-4)  # meme moyenne qu'en numpy
    np.testing.assert_allclose(std, stacked.std(0), atol=1e-4)  # meme ecart-type qu'en numpy
    gray_mean, _ = ds.mean_std(paths, color="gray", size=10)  # en gris, sur des images reduites a 10 pixels
    assert len(gray_mean) == 1  # une seule valeur : un seul canal


# Verifie que les images 16 bits sont ramenees a [0, 1] avec leur vrai maximum (65535) et non 255.
# Attraperait : des moyennes absurdes (des centaines) sur des images de camera industrielle 16 bits.
def test_mean_std_16bit(tmp_path: Path) -> None:
    image = np.full((8, 8), 65535 // 2, dtype=np.uint16)  # image 16 bits uniforme a mi-hauteur
    write_image(tmp_path / "a.png", image)  # ecrite en PNG 16 bits
    mean, std = ds.mean_std([tmp_path / "a.png"], color="gray")  # statistiques en gris
    assert mean == pytest.approx([0.5], abs=1e-3) and std == pytest.approx([0.0], abs=1e-6)  # moyenne 0.5, ecart-type nul


# Verifie les poids de classes (donner plus d'importance aux classes rares) et leur normalisation.
# Attraperait : une classe rare ignoree par le modele, ou des poids qui changent l'echelle de la perte.
def test_class_weights_modes() -> None:
    labels = ["a"] * 90 + ["b"] * 10  # 90 % de "a", 10 % de "b" (classe rare)
    inverse = ds.class_weights(labels)  # poids par defaut : inverse de la frequence
    assert inverse["b"] == pytest.approx(9 * inverse["a"], rel=1e-3)  # "b" 9 fois plus rare : poids 9 fois plus grand
    for mode in ("inverse", "sqrt", "effective"):  # pour chaque facon de calculer les poids
        weights = ds.class_weights(labels, mode=mode)  # poids de ce mode
        assert 90 * weights["a"] + 10 * weights["b"] == pytest.approx(100, rel=1e-3)  # moyenne ponderee = 1
        assert weights["b"] > weights["a"]  # la classe rare pese toujours plus
    per_sample = ds.sample_weights(labels)  # un poids par exemple (pour un tirage equilibre)
    assert per_sample.shape == (100,) and per_sample[-1] > per_sample[0]  # 100 poids ; un "b" pese plus qu'un "a"
    with pytest.raises(ValueError):  # on attend une erreur...
        ds.class_weights(labels, mode="bad")  # ...pour un mode inconnu


# Verifie le tirage de quelques images par classe et l'export en arborescence "split/classe/image".
# Attraperait : un tirage different a chaque fois, ou deux images de meme nom qui s'ecrasent a l'export.
def test_sample_per_class_and_imagefolder(class_folder: Path, tmp_path: Path) -> None:
    files = ds.folder_summary(class_folder).rename(columns={"folder": "label"})  # inventaire ; la colonne folder devient label
    small = ds.sample_per_class(files, 5)  # 5 images par classe
    assert small["label"].value_counts().to_dict() == {"alpha": 5, "beta": 5, "gamma": 5}  # 5 de chaque
    assert small.equals(ds.sample_per_class(files, 5))  # reproductible
    small["split"] = ["train", "train", "val", "test", "test"] * 3  # attribue une partie a chaque image (meme motif par classe)
    duplicate = small.iloc[[0]].copy()  # copie de la premiere ligne...
    duplicate["path"] = str(tmp_path / "other" / Path(small["path"].iloc[0]).name)  # ...pointant vers un autre fichier de MEME nom
    write_image(duplicate["path"].item(), np.zeros((4, 4), np.uint8))  # cree ce fichier homonyme
    # export par liens (sans copier les octets) en dossier "split/classe/"
    out = ds.export_imagefolder(pd.concat([small, duplicate], ignore_index=True), tmp_path / "folder", mode="link")
    assert all(Path(p).exists() for p in out["new_path"])  # tous les fichiers exportes existent
    assert len(set(out["new_path"])) == len(out)  # le doublon de nom n'ecrase rien
    assert (tmp_path / "folder" / "val" / "alpha").is_dir()  # l'arborescence val/alpha existe


# Verifie la redimension d'un dossier (couleur et 16 bits) et d'une image (avec ou sans bordure, avec ou sans
# respect des proportions). Attraperait : des images deformees, ou des 16 bits ecrases en 8 bits.
def test_resize_folder_and_resize_image(tmp_path: Path, rng: np.random.Generator) -> None:
    src = tmp_path / "src"  # dossier source
    write_image(src / "a" / "x.png", rng.integers(0, 256, (100, 200, 3), dtype=np.uint8))  # image couleur 100 x 200
    write_image(src / "b" / "y.png", rng.integers(0, 65535, (50, 50), dtype=np.uint16))  # image grise 16 bits 50 x 50
    report = ds.resize_folder(src, tmp_path / "dst", 64, workers=2)  # plus grand cote ramene a 64, avec 2 processus en parallele
    assert report["ok"].all()  # toutes les images ont ete traitees
    assert read_image(tmp_path / "dst" / "a" / "x.png").shape == (32, 64, 3)  # 100 x 200 -> 32 x 64 : proportions gardees
    sixteen = read_image(tmp_path / "dst" / "b" / "y.png", color="unchanged")  # relit l'image 16 bits sans conversion
    assert sixteen.dtype == np.uint16 and sixteen.shape == (64, 64)  # toujours 16 bits, agrandie a 64 x 64
    padded = ds.resize_image(np.ones((100, 200), np.uint8), 64, pad=True)  # redimension + bordure noire pour faire un carre
    assert padded.shape == (64, 64) and padded[0, 0] == 0 and padded[32, 32] == 1  # carre ; coin = bordure (0) ; centre = image (1)
    # sans garder les proportions : taille donnee (largeur 30, hauteur 15) -> tableau 15 lignes x 30 colonnes
    assert ds.resize_image(np.ones((10, 20), np.uint8), (30, 15), keep_aspect=False).shape == (15, 30)


# Verifie les conversions 16 -> 8 bits : simple division (image sombre), etirement min-max, et conversion d'un dossier.
# Attraperait : une conversion qui rendrait des images presque noires sans prevenir, ou un dossier mal converti.
def test_to_8bit_methods(tmp_path: Path) -> None:
    image = np.linspace(1000, 3000, 100 * 100).reshape(100, 100).astype(np.uint16)  # degrade 16 bits de 1000 a 3000 seulement
    scaled = ds.to_8bit(image, method="scale")  # division simple (65535 -> 255)
    assert scaled.dtype == np.uint8 and scaled.max() < 20  # tres sombre : la dynamique reelle est petite
    stretched = ds.to_8bit(image, method="minmax")  # etirement : min -> 0, max -> 255
    assert stretched.min() == 0 and stretched.max() == 255  # toute la plage est utilisee
    assert ds.to_8bit(image).max() == 255  # la methode par defaut etire aussi
    write_image(tmp_path / "in" / "a.tif", image)  # ecrit l'image 16 bits en TIFF
    report = ds.convert_folder_8bit(tmp_path / "in", tmp_path / "out")  # convertit tout le dossier
    # tout est converti, et le resultat (en PNG) est bien en 8 bits
    assert report["ok"].all() and read_image(tmp_path / "out" / "a.png", color="unchanged").dtype == np.uint8


# fabrique une petite video de test (64 x 48) et renvoie son chemin ; frames : nombre d'images ; fps : images par seconde
def _write_video(path: Path, frames: int = 30, fps: float = 10.0) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))  # ecrivain video, codec MJPG
    assert writer.isOpened()  # le codec est disponible (sinon le test s'arrete ici)
    for i in range(frames):  # pour chaque image de la video
        frame = np.full((48, 64, 3), i * 8 % 256, np.uint8)  # image unie dont la valeur change a chaque image
        writer.write(frame)  # ajoute l'image a la video
    writer.release()  # ferme le fichier video
    return path  # renvoie le chemin de la video


# Verifie l'extraction d'images d'une video selon les differents criteres (toutes, une sur n, cadence, intervalle, maximum).
# Attraperait : un mauvais calcul du pas ou de l'intervalle de temps, ou des noms de fichiers mal numerotes.
def test_extract_frames(tmp_path: Path) -> None:
    video = _write_video(tmp_path / "clip.avi")  # video de 30 images a 10 images par seconde (3 secondes)
    assert ds.extract_frames(video, tmp_path / "all") == 30  # toutes les images
    assert ds.extract_frames(video, tmp_path / "every5", every=5) == 6  # une image sur 5
    assert ds.extract_frames(video, tmp_path / "fps2", fps=2) == 6  # 10 i/s -> une image sur 5
    assert ds.extract_frames(video, tmp_path / "window", start=1.0, end=1.9) == 10  # de 1.0 s a 1.9 s : images 10 a 19
    assert ds.extract_frames(video, tmp_path / "max", max_frames=3) == 3  # au plus 3 images
    names = sorted(p.name for p in (tmp_path / "every5").iterdir())  # noms des fichiers extraits une image sur 5
    assert names[1] == "clip_000005.png"  # le 2e fichier porte le numero de l'image dans la video (5)


# Verifie le decoupage d'une image en patchs : taille fixe, contenu exact, image entierement couverte, part de defaut.
# Attraperait : une bande d'image jamais decoupee (defaut rate), ou un pas plus grand que le patch (trous).
def test_make_patches_cover_image() -> None:
    image = np.arange(100 * 130, dtype=np.uint16).reshape(100, 130)  # image 100 x 130 dont chaque pixel a une valeur unique
    mask = np.zeros((100, 130), np.uint8)  # masque de defaut vide
    mask[90:, 120:] = 1  # petit defaut dans le coin bas droit
    patches = ds.make_patches(image, mask, size=64, stride=48)  # patchs de 64 x 64, decales de 48 pixels
    covered = np.zeros_like(mask)  # carte des pixels couverts
    for patch in patches:  # pour chaque patch
        assert patch["image"].shape == (64, 64)  # taille fixe
        # le contenu du patch est exactement la zone de l'image a sa position (x, y)
        np.testing.assert_array_equal(patch["image"], image[patch["y"] : patch["y"] + 64, patch["x"] : patch["x"] + 64])
        covered[patch["y"] : patch["y"] + 64, patch["x"] : patch["x"] + 64] = 1  # marque la zone comme couverte
    assert covered.all()  # toute l'image est couverte (y compris le coin du defaut)
    assert any(p["defect_fraction"] > 0 for p in patches)  # au moins un patch contient une part de defaut
    with pytest.raises(ValueError):  # on attend une erreur...
        ds.make_patches(image, size=32, stride=40)  # ...si le pas (40) depasse la taille (32) : il resterait des trous


# Verifie le decoupage d'un dossier dont les masques .png sont a cote des images, et l'option qui jette les patchs vides.
# Attraperait : des masques decoupes comme des images, ou des patchs sans defaut gardes alors qu'on les a exclus.
def test_patch_folder_with_side_masks(tmp_path: Path, rng: np.random.Generator) -> None:
    src = tmp_path / "tiles" / "MT_Crack"  # dossier d'images d'une classe de defaut
    for i in range(2):  # 2 images
        write_image(src / f"img{i}.jpg", rng.integers(0, 256, (64, 96), dtype=np.uint8))  # image aleatoire 64 x 96
        mask = np.zeros((64, 96), np.uint8)  # masque vide
        mask[5:10, 5:10] = 255 * (i == 0)  # defaut 5 x 5 dans la 1re image seulement (255 x vrai = 255, 255 x faux = 0)
        write_image(src / f"img{i}.png", mask)  # masque .png de meme nom, a cote de l'image
    table = ds.patch_folder(tmp_path / "tiles", tmp_path / "patches", size=32)  # patchs de 32 x 32
    assert len(table) == 12  # 2 images x (2 x 3) patchs, les .png masques ne sont pas decoupes
    assert int(table["has_defect"].sum()) == 1  # un seul patch contient le defaut
    assert all(Path(p).exists() for p in table["mask_path"])  # chaque patch a son masque ecrit
    # keep_empty=0.0 : on ne garde aucun patch sans defaut
    sparse = ds.patch_folder(tmp_path / "tiles", tmp_path / "sparse", size=32, keep_empty=0.0)
    assert len(sparse) == 1 and sparse["has_defect"].all()  # reste le seul patch avec defaut


# --------------------------------------------------------------------------- entrainement


# fabrique un petit probleme jouet ; renvoie (modele, chargeur de donnees)
def _toy_problem(seed: int = 0) -> tuple[nn.Module, torch.utils.data.DataLoader]:
    torch.manual_seed(seed)  # graine fixe : memes donnees et memes poids initiaux
    x = torch.randn(64, 2)  # 64 points en 2 dimensions
    y = (x[:, 0] + x[:, 1] > 0).long()  # classe 1 si x0 + x1 > 0, sinon 0 (separable par une droite)
    # DataLoader : decoupe les donnees en lots de 16, melanges a chaque epoque
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=16, shuffle=True)
    model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))  # petit reseau : 2 -> 16 -> 2 sorties (une par classe)
    return model, loader  # renvoie le modele et le chargeur


# Verifie la recherche du taux d'apprentissage (on l'augmente pas a pas en regardant la perte) : courbe, suggestion
# raisonnable, et poids du modele remis a l'identique. Attraperait : un modele deja abime par la recherche avant l'entrainement.
def test_lr_find_restores_weights_and_suggests() -> None:
    model, loader = _toy_problem()  # probleme jouet
    before = {k: v.clone() for k, v in model.state_dict().items()}  # copie de tous les poids avant la recherche
    # 60 pas, taux montant jusqu'a 10 ; CrossEntropyLoss : la perte standard en classification
    curve, lr = tt.lr_find(model, loader, nn.CrossEntropyLoss(), steps=60, end_lr=10.0)
    assert {"lr", "loss", "smoothed"} <= set(curve.columns) and len(curve) > 20  # courbe complete avec assez de points
    assert curve["lr"].is_monotonic_increasing  # le taux ne fait que monter
    assert 1e-6 < lr < 10.0  # taux suggere dans une plage raisonnable
    for key, value in model.state_dict().items():  # pour chaque tenseur de poids
        torch.testing.assert_close(value, before[key])  # identique a avant la recherche


# Verifie le controle "sur-apprendre un seul lot" : un modele sain doit memoriser un lot ; un modele gele ou des
# etiquettes melangees doivent etre signales. Attraperait : un bug d'entrainement (rien n'apprend) qu'on ne verrait
# qu'apres des heures de calcul.
def test_overfit_one_batch_ok_and_failure() -> None:
    model, loader = _toy_problem()  # probleme jouet
    batch = next(iter(loader))  # un seul lot (images, etiquettes)
    good = tt.overfit_one_batch(model, batch, nn.CrossEntropyLoss(), steps=200, lr=1e-2)  # 200 pas sur ce lot
    assert good.attrs["ok"], good.attrs["verdict"]  # le modele memorise le lot ; sinon pytest affiche le verdict
    assert "accuracy" in good.columns  # l'historique contient l'exactitude
    frozen = nn.Sequential(nn.Linear(2, 2))  # modele minimal...
    frozen[0].weight.requires_grad_(False)  # ...dont on gele les poids...
    frozen[0].bias.requires_grad_(False)  # ...et le biais : plus rien ne peut apprendre
    with pytest.raises(ValueError, match="gele"):  # on attend une erreur qui dit que tout est gele
        tt.overfit_one_batch(frozen, batch, nn.CrossEntropyLoss(), steps=5)  # le controle refuse de tourner
    shuffled = (batch[0], batch[1][torch.randperm(len(batch[1]))])  # memes entrees, etiquettes melangees au hasard
    # modele lineaire, taux minuscule, 30 pas : impossible de memoriser
    bad = tt.overfit_one_batch(nn.Sequential(nn.Linear(2, 2)), shuffled, nn.CrossEntropyLoss(), steps=30, lr=1e-4)
    assert not bad.attrs["ok"] and "ECHEC" in bad.attrs["verdict"]  # l'echec est bien signale


# Verifie le resume d'un modele : nombre de parametres, parametres geles, taille de sortie, cout de calcul.
# Attraperait : un comptage faux qui tromperait sur la taille ou la vitesse du modele.
def test_model_summary_counts() -> None:
    # convolution 3 -> 8 canaux (noyau 3 x 3), ReLU, moyenne globale, mise a plat, couche lineaire 8 -> 4
    model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 4))
    model[0].weight.requires_grad_(False)  # gele les poids de la convolution
    info, layers = tt.model_summary(model, (1, 3, 16, 16))  # resume pour une entree d'1 image couleur 16 x 16
    assert info["total_params"] == 8 * 3 * 9 + 8 + 8 * 4 + 4  # poids conv + biais conv + poids lineaire + biais lineaire
    assert info["frozen_params"] == 8 * 3 * 9  # seuls les poids de la convolution sont geles
    assert info["output"] == "1x4"  # sortie : 1 image, 4 valeurs
    # GFLOPs (milliards d'operations) : 2 operations par multiplication-addition, conv sur 16 x 16 positions + couche lineaire
    assert info["gflops"] == pytest.approx(2 * 8 * 3 * 9 * 16 * 16 / 1e9 + 2 * 8 * 4 / 1e9, rel=0.05)
    # une ligne par couche ; la 1re sort 1 x 8 x 16 x 16
    assert list(layers["module"]) == ["0", "1", "2", "3", "4"] and layers["output"].iloc[0] == "1x8x16x16"


# Verifie que la recherche de la plus grande taille de lot renvoie None sur processeur (elle n'a de sens que sur GPU).
# Attraperait : une recherche qui saturerait la memoire vive du PC en essayant des lots de plus en plus gros.
def test_max_batch_size_cpu_returns_none() -> None:
    assert tt.max_batch_size(nn.Linear(2, 2), (2,), device="cpu") is None  # sur CPU : None


# Verifie l'arret anticipe, la moyenne ponderee, le chronometre et la mesure de vitesse.
# Attraperait : un entrainement arrete trop tot ou trop tard, ou des moyennes de perte mal ponderees.
def test_early_stopping_meter_timer_benchmark() -> None:
    stopper = tt.EarlyStopping(patience=2, mode="max")  # arret apres 2 epoques sans amelioration ; plus grand = mieux
    decisions = [stopper.step(v) for v in (0.5, 0.6, 0.6, 0.59)]  # score de validation de 4 epoques ; True = il faut s'arreter
    # arret a la 4e epoque ; meilleur score 0.6 obtenu a l'epoque 1 (numerotee a partir de 0)
    assert decisions == [False, False, False, True] and stopper.best == 0.6 and stopper.best_epoch == 1
    meter = tt.AverageMeter()  # moyenne courante ponderee (par exemple la perte moyenne d'une epoque)
    meter.update(1.0, n=3)  # valeur 1.0 pour 3 exemples
    meter.update(3.0, n=1)  # valeur 3.0 pour 1 exemple
    assert meter.avg == pytest.approx(1.5)  # (1 x 3 + 3 x 1) / 4 = 1.5
    with tt.Timer() as timer:  # chronometre le bloc qui suit
        sum(range(1000))  # un petit calcul
    assert timer.seconds > 0  # une duree positive a ete mesuree
    result = tt.benchmark(nn.Linear(4, 2), torch.randn(8, 4), warmup=1, runs=5)  # vitesse d'un modele sur un lot de 8
    assert result["batch"] == 8 and result["mean_ms"] > 0 and result["images_per_s"] > 0  # taille de lot, temps et debit valides
    assert tt.benchmark(lambda x: x * 2, np.ones(3), warmup=0, runs=3)["mean_ms"] >= 0  # marche aussi avec une simple fonction


# --------------------------------------------------------------------------- statistiques


# Verifie l'intervalle de confiance par bootstrap (on retire au hasard des echantillons pour voir la variabilite d'un
# score), simple et par groupe. Attraperait : un intervalle different a chaque lancement, ou qui n'encadre pas le score.
def test_bootstrap_ci_reproducible_and_grouped(rng: np.random.Generator) -> None:
    y_true = rng.integers(0, 3, 300)  # 300 vraies classes (0, 1 ou 2)
    y_pred = np.where(rng.random(300) < 0.85, y_true, (y_true + 1) % 3)  # predictions justes 85 % du temps
    first = stats.bootstrap_ci(accuracy_score, y_true, y_pred, n=300)  # intervalle de l'exactitude, 300 tirages
    assert first == stats.bootstrap_ci(accuracy_score, y_true, y_pred, n=300)  # meme resultat au 2e appel : reproductible
    assert first.low < first.value < first.high and first.high - first.low < 0.15  # encadre le score, largeur raisonnable
    groups = np.repeat(np.arange(30), 10)  # 30 groupes (pieces) de 10 images
    # bootstrap par groupe sur le F1 macro (partial fixe average="macro")
    grouped = stats.bootstrap_ci(functools.partial(f1_score, average="macro"), y_true, y_pred, n=200, groups=groups)
    assert grouped.low < grouped.value < grouped.high  # encadre aussi le score
    assert "[" in str(first)  # l'affichage texte montre l'intervalle entre crochets


# Verifie l'intervalle de Wilson (intervalle de confiance d'une proportion, fiable meme avec peu d'exemples).
# Attraperait : un intervalle faux ou qui descend sous 0 quand il n'y a aucun succes.
def test_wilson_ci() -> None:
    interval = stats.wilson_ci(48, 79)  # 48 succes sur 79 essais
    assert interval.value == pytest.approx(48 / 79)  # la proportion observee
    assert 0.49 < interval.low < 0.51 and 0.70 < interval.high < 0.72  # bornes attendues (environ 0.50 et 0.71)
    assert stats.wilson_ci(0, 10).low == 0.0  # 0 succes : la borne basse vaut exactement 0


# Verifie le test de McNemar (deux modeles sur les memes images : la difference est-elle due au hasard ?).
# Attraperait : un "modele A meilleur" annonce sur une difference trop petite pour etre sure.
def test_mcnemar_cases() -> None:
    y = np.zeros(100, int)  # 100 vraies classes, toutes 0
    a = y.copy()  # modele A : toujours juste
    b = y.copy()  # modele B : copie...
    b[:30] = 1  # B se trompe 30 fois de plus
    result = stats.mcnemar(y, a, b)  # compare A et B
    # 30 cas ou A a raison et B tort ; p-value minuscule (difference certaine) ; verdict en faveur de A
    assert result["a_right_b_wrong"] == 30 and result["p_value"] < 1e-5 and "A est meilleur" in result["verdict"]
    same = stats.mcnemar(y, a, a)  # A compare a lui-meme
    assert same["p_value"] == 1.0  # aucune difference : p-value 1
    few = stats.mcnemar(y, a, np.r_[np.ones(3, int), np.zeros(97, int)])  # B ne se trompe que 3 fois
    assert few["method"] == "binomial exact" and few["p_value"] > 0.05  # peu de desaccords : test exact, difference non significative


# Verifie le choix d'un seuil de decision sous contraintes (rappel minimal, fausses alarmes maximales).
# Attraperait : un seuil qui ne respecte pas la contrainte demandee, ou une contrainte impossible non signalee.
def test_threshold_for_constraints() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])  # scores de 8 images
    defect = np.array([0, 0, 0, 1, 0, 1, 1, 1], bool)  # lesquelles ont vraiment un defaut (4 sur 8)
    recall = stats.threshold_for(scores, defect, min_recall=0.75)  # seuil qui trouve au moins 75 % des defauts
    # possible ; seuil 0.7 (3 defauts sur 4 au-dessus) ; aucune image saine au-dessus
    assert recall["feasible"] and recall["threshold"] == pytest.approx(0.7) and recall["false_alarm_rate"] == 0.0
    alarm = stats.threshold_for(scores, defect, max_false_alarm=0.25)  # seuil avec au plus 25 % de fausses alarmes
    assert alarm["threshold"] == pytest.approx(0.4) and alarm["recall"] == 1.0  # seuil 0.4 : tous les defauts trouves
    impossible = stats.threshold_for(scores, defect, min_recall=1.0, max_false_alarm=0.0)  # 100 % de rappel ET 0 fausse alarme
    assert not impossible["feasible"] and "IMPOSSIBLE" in impossible["message"]  # impossible ici, et c'est dit clairement
    with pytest.raises(ValueError):  # on attend une erreur...
        stats.threshold_for(scores, defect)  # ...si aucune contrainte n'est donnee
    table = stats.operating_points(scores, defect)  # tableau de tous les seuils possibles
    assert table["recall"].iloc[0] == 0.0 and table["recall"].iloc[-1] == 1.0  # le rappel va de 0 a 1 le long du tableau


# Verifie l'erreur de calibration (les probabilites annoncees correspondent-elles a la frequence reelle de reussite ?)
# et la liste des pires erreurs. Attraperait : un modele "sur de lui a tort" non detecte.
def test_calibration_and_worst_errors() -> None:
    y = np.array([0, 1, 1, 0])  # vraies classes de 4 images
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.95, 0.05], [0.6, 0.4]])  # probabilites predites (classe 0, classe 1)
    ece, table = stats.expected_calibration_error(y, probs, bins=5, return_table=True)  # ECE sur 5 tranches de confiance
    assert 0 <= ece <= 1 and table["count"].sum() == 4  # ECE entre 0 et 1 ; les 4 images sont reparties dans les tranches
    assert list(stats.worst_errors(y, probs)) == [2]  # seule erreur : l'image 2 (classe 1 predite 0 avec 95 % de confiance)
    # predictions parfaites et sures : calibration parfaite, ECE = 0
    assert stats.expected_calibration_error(np.array([1, 0]), np.array([1.0, 0.0])) == pytest.approx(0.0)


# --------------------------------------------------------------------------- ligne de commande


# Verifie chaque commande de la ligne de commande (code de retour et texte affiche), et les erreurs attendues.
# Attraperait : une commande cassee ou un argument obligatoire oublie qui passerait sans message.
def test_cli_commands(class_folder: Path, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:  # capsys : capture l'affichage
    assert cli(["summary", str(class_folder), "--csv", str(tmp_path / "files.csv")]) == 0  # inventaire + CSV ; 0 = succes
    assert "TOTAL" in capsys.readouterr().out and (tmp_path / "files.csv").exists()  # le tableau s'affiche et le CSV est ecrit
    with pytest.raises(FileNotFoundError):  # on attend une erreur "fichier introuvable"...
        cli(["summary", str(class_folder), "--ext", ".jpg"])  # aucune image .jpg dans ce dossier
    cli(["mean-std", str(class_folder), "--gray", "--max-images", "10"])  # moyenne / ecart-type en gris sur 10 images
    assert "mean:" in capsys.readouterr().out  # la moyenne est affichee
    cli(["class-weights", str(class_folder)])  # poids des classes du dossier
    assert "alpha" in capsys.readouterr().out  # les classes sont affichees
    cli(["resize", str(class_folder), str(tmp_path / "small"), "--size", "24"])  # redimension du dossier en 24
    assert "36 image(s) ecrites" in capsys.readouterr().out  # les 36 images sont ecrites
    # petit CSV de scores et de verites pour la commande threshold
    pd.DataFrame({"score": [0.1, 0.4, 0.8, 0.9], "is_defect": [0, 0, 1, 1]}).to_csv(tmp_path / "s.csv", index=False)
    cli(["threshold", str(tmp_path / "s.csv"), "--min-recall", "1.0"])  # seuil qui trouve 100 % des defauts
    assert "contraintes respectees" in capsys.readouterr().out  # contrainte atteignable et annoncee
    # CSV avec la verite (label) et deux jeux de predictions (pred et other) pour ci et mcnemar
    pd.DataFrame({"label": list("aabbccaabb"), "pred": list("aabbcbaabc"), "other": list("abbbccaabb")}).to_csv(
        tmp_path / "p.csv",  # fichier de sortie
        index=False,  # sans la colonne d'index de pandas
    )
    cli(["ci", str(tmp_path / "p.csv"), "--n", "100"])  # intervalles de confiance (100 tirages bootstrap)
    assert "f1_macro" in capsys.readouterr().out  # le F1 macro est affiche
    cli(["mcnemar", str(tmp_path / "p.csv"), "--pred-a", "pred", "--pred-b", "other"])  # compare les deux jeux de predictions
    assert "p_value" in capsys.readouterr().out  # la p-value est affichee
    video = _write_video(tmp_path / "v.avi", frames=10)  # video de 10 images
    cli(["frames", str(video), str(tmp_path / "frames"), "--every", "2"])  # extrait une image sur 2
    assert "5 image(s)" in capsys.readouterr().out  # 5 images extraites
    with pytest.raises(SystemExit):  # argparse arrete le programme (SystemExit)...
        cli(["resize", str(class_folder), str(tmp_path / "x")])  # ...car l'option obligatoire --size manque
