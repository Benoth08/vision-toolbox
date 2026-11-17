"""Tests du socle : configuration, lecture d'images, catalogues, decoupage sans fuite."""

# Fuite (leakage) : quand une meme piece (ou une quasi-copie d'image) se retrouve a la fois en entrainement et en
# test. Le modele "reconnait" alors la piece au lieu d'apprendre, et le score de test est faussement bon.

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[int], X | None) partout

from datetime import datetime, timedelta  # dates et durees (pour simuler des images prises jour apres jour)
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)
import pandas as pd  # pandas : tableaux de donnees avec colonnes nommees (DataFrame)
import pytest  # pytest : lance les tests ; parametrize, raises...
from pydantic import ValidationError  # erreur levee par pydantic quand une configuration est invalide

# config : modeles de configuration (donnees, decoupage), base "stricte" qui refuse les cles inconnues, lecture / ecriture YAML
from vtb.core.config import DataConfig, SplitConfig, StrictModel, load_config, save_config
from vtb.io.catalogs import add_groups, build_catalog  # construction du catalogue (une ligne par image) et ajout des groupes
from vtb.io.images import read_image, read_mask, to_uint8, write_image  # lecture / ecriture d'images et de masques, passage en 8 bits

# split : plis de validation croisee, controle de fuite, lecture / creation / ecriture du decoupage train / val / test
from vtb.split import assign_folds, check_no_leakage, load_split, make_split, save_split


# Verifie qu'une cle mal orthographiee dans le YAML provoque une erreur, et qu'une config valide se relit et se resauve.
# Attraperait : une faute de frappe ignoree en silence (le reglage voulu ne serait jamais applique).
def test_config_rejects_unknown_keys(tmp_path: Path):
    class Demo(StrictModel):  # petite configuration de test qui n'a qu'une section "data"
        data: DataConfig  # la section data suit le modele DataConfig

    path = tmp_path / "demo.yaml"  # fichier YAML temporaire
    path.write_text("data:\n  root: x\n  layuot: csv\n", encoding="utf-8")  # faute de frappe volontaire
    with pytest.raises(ValidationError):  # on attend une erreur de validation...
        load_config(path, Demo)  # ...a la lecture de ce fichier
    path.write_text("data:\n  root: x\n  layout: csv\n", encoding="utf-8")  # meme fichier, cle correctement ecrite
    config = load_config(path, Demo)  # cette fois la lecture reussit
    save_config(config, tmp_path / "out" / "config_used.yaml")  # sauvegarde la config reellement utilisee (tracabilite)
    assert "layout: csv" in (tmp_path / "out" / "config_used.yaml").read_text()  # le fichier sauve contient bien le reglage


# Verifie l'ecriture puis la lecture d'une image 16 bits dans un chemin avec accent, en gris / couleur / brut, et la
# lecture d'un masque. Attraperait : des 16 bits ecrases en 8 bits (perte de precision), un chemin avec accent illisible.
def test_images_roundtrip_gray_rgb_16bits_and_unicode_path(tmp_path: Path, image16: np.ndarray):
    path = tmp_path / "dossier_ecrou" / "piece_\u00e9.png"  # accent : cv2.imread echouerait sous Windows
    write_image(path, image16)  # ecrit l'image 16 bits de la fixture
    assert read_image(path, color="unchanged").dtype == np.uint16  # les 16 bits ne sont pas ecrases
    assert read_image(path, color="rgb").shape == (64, 64, 3)  # lecture forcee en couleur : 3 canaux
    assert read_image(path, color="gray").shape == (64, 64)  # lecture forcee en gris : 2 dimensions seulement
    stretched = to_uint8(image16, 1, 99)  # passage en 8 bits en etirant entre les percentiles 1 % et 99 %
    assert stretched.dtype == np.uint8 and stretched.min() == 0 and stretched.max() == 255  # toute la plage 0-255 est utilisee
    mask_path = tmp_path / "m.png"  # chemin d'un petit masque
    write_image(mask_path, np.array([[0, 100, 200]], np.uint8))  # masque 1 x 3 avec trois niveaux de gris
    assert read_mask(mask_path).tolist() == [[0, 0, 1]]  # relu en binaire : au-dessus de la moitie (127) = 1


# Verifie le catalogue d'un rangement "un dossier par classe" et l'extraction du groupe (numero de piece) du nom.
# Attraperait : des images oubliees, une classe manquante, ou des groupes mal extraits (donc un decoupage qui fuit).
def test_catalog_class_folders_and_groups(class_folder: Path):
    # catalogue du dossier ; group_regex : le motif "P" + chiffres en debut de nom donne le groupe (la piece)
    catalog = build_catalog(DataConfig(root=str(class_folder), group_regex=r"^(P\d+)_"))
    assert len(catalog) == 36 and sorted(catalog["label"].unique()) == ["alpha", "beta", "gamma"]  # 36 images, 3 classes
    assert catalog["group"].nunique() == 12  # 3 classes x 4 pieces


# Verifie le catalogue d'un rangement MVTec : images d'entrainement, defauts et chemins de masques.
# Attraperait : un defaut sans masque, ou un mauvais comptage des images d'entrainement.
def test_catalog_mvtec(mvtec_folder: Path):
    catalog = build_catalog(DataConfig(root=str(mvtec_folder), layout="mvtec"))  # catalogue au format MVTec
    assert (catalog["split"] == "train").sum() == 10  # les 10 normales du dossier train
    defects = catalog[catalog["label"] == "defect"]  # lignes des images avec defaut
    assert len(defects) == 3 and all(Path(p).exists() for p in defects["mask_path"])  # 3 defauts, chacun avec un masque existant


@pytest.mark.parametrize("strategy", ["stratified", "grouped", "random"])  # le test tourne pour chaque strategie de decoupage
# Verifie que chaque strategie donne les 3 parties, sans fuite, et le meme resultat avec la meme graine.
# Attraperait : un decoupage different a chaque lancement (resultats non comparables) ou une image dans deux parties.
def test_split_strategies_are_leak_free_and_reproducible(class_folder: Path, strategy: str):
    catalog = build_catalog(DataConfig(root=str(class_folder), group_regex=r"^(P\d+)_"))  # catalogue avec groupes
    config = SplitConfig(strategy=strategy, val_fraction=0.25, test_fraction=0.25)  # 25 % validation, 25 % test
    a, b = make_split(catalog, config, seed=1), make_split(catalog, config, seed=1)  # deux decoupages avec la meme graine
    assert a["split"].tolist() == b["split"].tolist()  # strictement identiques : reproductible
    if strategy == "grouped":  # decoupage par groupe : on controle les pieces
        check_no_leakage(a)  # aucune piece coupee entre deux sous-ensembles
    else:  # autres strategies : elles ignorent les pieces, on controle seulement que chaque image est dans une seule partie
        check_no_leakage(a.assign(group=a["path"]))  # groupe = chemin de l'image
    assert set(a["split"]) == {"train", "val", "test"}  # les 3 parties existent


# Verifie que le controle detecte la fuite quand on decoupe sans tenir compte des pieces.
# Attraperait : un controle de fuite muet, qui laisserait passer un score de test trop optimiste.
def test_ungrouped_split_of_grouped_data_is_detected_as_leak(class_folder: Path):
    catalog = build_catalog(DataConfig(root=str(class_folder), group_regex=r"^(P\d+)_"))  # catalogue avec groupes
    # decoupage "stratified" : il equilibre les classes mais disperse les vues d'une meme piece
    split = make_split(catalog, SplitConfig(strategy="stratified", val_fraction=0.25, test_fraction=0.25), seed=1)
    with pytest.raises(ValueError, match="fuite"):  # on attend une erreur dont le message contient "fuite"
        check_no_leakage(split)  # le controle doit voir des pieces coupees entre parties


# Verifie le decoupage temporel : le test contient les images les plus recentes. Attraperait : un modele evalue sur
# le passe alors qu'il servira sur des images futures (derive de production non vue).
def test_temporal_split_tests_on_the_most_recent_groups():
    start = datetime(2026, 1, 1)  # date de la premiere image
    catalog = pd.DataFrame(  # catalogue fictif de 40 images
        # chemins, classes a et b en alternance, une image par jour a partir du 1er janvier
        {"path": [f"img{i}.png" for i in range(40)], "label": ["a", "b"] * 20, "timestamp": [start + timedelta(days=i) for i in range(40)]}
    )
    catalog = add_groups(catalog, None)  # ajoute la colonne group (sans motif : chaque image est son propre groupe)
    # decoupage temporel : 20 % validation, 20 % test
    split = make_split(catalog, SplitConfig(strategy="temporal", val_fraction=0.2, test_fraction=0.2), seed=0)
    # la plus ancienne image de test est plus recente que la plus recente image d'entrainement
    assert split[split["split"] == "test"]["timestamp"].min() > split[split["split"] == "train"]["timestamp"].max()


# Verifie le decoupage "provided" (fourni par le rangement, comme MVTec) : la validation est prise dans le train, et
# train / val restent sans defaut. Attraperait : un defaut glisse dans l'entrainement d'un modele d'anomalies.
def test_provided_split_carves_val_and_normal_only_check(mvtec_folder: Path, tmp_path: Path):
    catalog = build_catalog(DataConfig(root=str(mvtec_folder), layout="mvtec"))  # catalogue MVTec (train / test deja fixes)
    split = make_split(catalog, SplitConfig(strategy="provided", val_fraction=0.2), seed=0)  # garde le decoupage fourni
    assert (split["split"] == "val").sum() == 2  # 20 % des 10 images de train passent en validation
    check_no_leakage(split, normal_only_train=True)  # controle aussi que train et val ne contiennent que des normales
    save_split(split, tmp_path / "split.csv")  # sauvegarde du decoupage en CSV
    # relecture en ne gardant que train et val
    assert set(load_split(tmp_path / "split.csv", ["train", "val"])["split"]) == {"train", "val"}
    dirty = split.copy()  # copie qu'on va volontairement abimer
    dirty.loc[dirty["label"] == "defect", "split"] = "train"  # on met les defauts en entrainement
    with pytest.raises(ValueError, match="non normales"):  # on attend une erreur qui parle d'images "non normales"
        check_no_leakage(dirty, normal_only_train=True)  # le controle doit refuser ce decoupage


# Verifie les plis de validation croisee (le train + val est coupe en k morceaux qui servent tour a tour de validation).
# Attraperait : des images de test glissees dans un pli, donc vues pendant la mise au point du modele.
def test_folds_only_on_train_and_val(class_folder: Path):
    catalog = build_catalog(DataConfig(root=str(class_folder), group_regex=r"^(P\d+)_"))  # catalogue avec groupes
    # decoupage par groupe puis repartition de train + val en 3 plis (numerotes 0, 1, 2)
    split = assign_folds(make_split(catalog, SplitConfig(strategy="grouped", test_fraction=0.25, val_fraction=0.25), seed=0), 3, seed=0)
    assert (split.loc[split["split"] == "test", "fold"] == -1).all()  # les images de test n'ont aucun pli (-1)
    assert set(split.loc[split["split"] != "test", "fold"]) == {0, 1, 2}  # les autres sont reparties dans les 3 plis
