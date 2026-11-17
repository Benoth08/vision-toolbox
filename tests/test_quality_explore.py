"""Controle du jeu (doublons, fuites, raccourcis), plongements et derive, sur des images synthetiques."""

# Raccourci (shortcut) : un indice trop simple (luminosite, taille...) qui suffit a deviner la classe ; le modele
# l'apprend au lieu du vrai defaut et echoue en production. Plongement (embedding) : vecteur de nombres qui resume
# une image ; deux images proches ont des vecteurs proches. Derive (drift) : les images de production ne
# ressemblent plus a celles de l'entrainement.

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[int], X | None) partout

import json  # json : convertir des donnees en texte JSON et inversement
import shutil  # shutil : copier des fichiers
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)
import pandas as pd  # pandas : tableaux de donnees avec colonnes nommees (DataFrame)
import pytest  # pytest : lance les tests ; fixtures, approx...
import torch  # PyTorch : tenseurs et reseaux de neurones

from vtb.explore.drift import domain_classifier, statistics_drift, verdict  # mesures de derive et verdict en clair
from vtb.explore.embeddings import cluster, cluster_purity, embed_images, nearest  # plongements, groupes, voisins
from vtb.io.images import write_image  # ecrit un tableau numpy en fichier image
from vtb.preprocess import Pipeline  # suite d'operations de pretraitement

# checks : empreinte d'image (dhash), doublons, images atypiques, analyse du jeu, detection de raccourcis, resume
from vtb.quality.checks import dhash, find_duplicates, flag_outliers, scan, shortcut_scores, summarize


@pytest.fixture  # fixture : petit catalogue d'images avec des problemes plantes expres
def dataset(tmp_path: Path) -> pd.DataFrame:  # renvoie le catalogue (une ligne par image : path, label, split)
    """Deux classes dont la luminosite differe (raccourci), un doublon exact a cheval sur train / test, un fichier casse."""
    rng = np.random.default_rng(0)  # generateur aleatoire a graine fixe
    rows = []  # lignes du catalogue
    for label, base in (("dark", 60), ("bright", 180)):  # classe sombre (60) et classe claire (180)
        for i in range(12):  # 12 images par classe
            image = (base + rng.integers(0, 40, (48, 48))).astype(np.uint8)  # niveau de base + bruit entre 0 et 39
            path = tmp_path / label / f"{label}_{i}.png"  # chemin de l'image
            write_image(path, image)  # ecrit l'image
            rows.append({"path": str(path), "label": label, "split": "train" if i < 8 else "test"})  # 8 en train, 4 en test
    copy = tmp_path / "bright" / "bright_copy.png"  # chemin de la copie
    shutil.copy(rows[12]["path"], copy)  # bright_0 (train) recopiee dans le test
    rows.append({"path": str(copy), "label": "bright", "split": "test"})  # la copie est declaree en test : fuite
    broken = tmp_path / "dark" / "broken.png"  # chemin d'un faux fichier image
    broken.write_bytes(b"pas une image")  # contenu qui n'est pas une image : fichier illisible
    rows.append({"path": str(broken), "label": "dark", "split": "train"})  # ajoute au catalogue
    return pd.DataFrame(rows)  # renvoie le catalogue sous forme de tableau


# Verifie que l'analyse du jeu trouve les 3 problemes plantes : fichier casse, doublon entre train et test, raccourci
# de luminosite. Attraperait : un controle qui laisse passer un probleme qui fausserait l'entrainement ou l'evaluation.
def test_scan_finds_broken_duplicates_and_shortcut(dataset: pd.DataFrame) -> None:
    stats = scan(dataset)  # statistiques de chaque image (lisible ?, moyenne, ecart-type, taille, empreinte...)
    assert (~stats["readable"].astype(bool)).sum() == 1  # exactement 1 image illisible (~ = "non")
    duplicates = find_duplicates(stats, max_hamming=0)  # paires de doublons ; 0 bit de difference tolere entre empreintes
    exact = duplicates[duplicates["exact"]]  # doublons au fichier pres
    assert len(exact) == 1 and bool(exact["cross_split"].iloc[0])  # un seul, et il est a cheval sur deux parties (fuite)
    shortcuts = shortcut_scores(stats)  # pour chaque statistique simple : peut-elle a elle seule predire la classe ?
    # la plus dangereuse est la moyenne (luminosite), qui predit la classe a plus de 90 %
    assert shortcuts.iloc[0]["statistic"] == "mean" and shortcuts.iloc[0]["balanced_accuracy"] > 0.9
    summary = summarize(stats, duplicates, shortcuts, flag_outliers(stats))  # resume + alertes (avec les images atypiques)
    text = " ".join(summary["alerts"])  # toutes les alertes en un seul texte
    assert "illisible" in text and "FUITE" in text and "raccourci" in text  # les 3 problemes sont signales
    assert json.loads(json.dumps(summary))["sizes"] == {"48x48": 25}  # le resume s'ecrit en JSON


# Verifie que l'empreinte dhash (64 bits qui resument l'image) change peu pour une image a peine eclaircie et beaucoup
# pour une autre image. Attraperait : des quasi-doublons non detectes, ou des images differentes confondues.
def test_dhash_is_robust_to_small_brightness_change() -> None:
    rng = np.random.default_rng(1)  # generateur aleatoire a graine fixe
    image = rng.integers(0, 255, (64, 64), dtype=np.uint8)  # image aleatoire
    shifted = np.clip(image.astype(int) + 3, 0, 255).astype(np.uint8)  # la meme, eclaircie de 3 niveaux
    other = rng.integers(0, 255, (64, 64), dtype=np.uint8)  # une autre image aleatoire
    # ^ (ou exclusif) marque les bits differents ; count("1") les compte (distance de Hamming)
    assert bin(dhash(image) ^ dhash(shifted)).count("1") <= 4  # au plus 4 bits differents : reconnue comme quasi-doublon
    assert bin(dhash(image) ^ dhash(other)).count("1") > 15  # plus de 15 bits differents : image differente


# Verifie le classifieur de domaine : s'il arrive a distinguer les vecteurs de reference des vecteurs actuels, il y a
# derive. Attraperait : une derive non signalee (modele utilise hors de son domaine) ou une fausse alerte.
def test_domain_classifier_separates_shifted_vectors() -> None:
    rng = np.random.default_rng(0)  # generateur aleatoire a graine fixe
    reference = rng.normal(size=(120, 16))  # 120 vecteurs de reference (taille 16)
    same = rng.normal(size=(120, 16))  # 120 vecteurs de meme distribution : pas de derive
    shifted = rng.normal(size=(120, 16)) + 1.0  # 120 vecteurs decales de 1 : derive nette
    auc_same, _ = domain_classifier(reference, same)  # AUC : 0.5 = indiscernables, 1 = parfaitement separes
    auc_shifted, novelty = domain_classifier(reference, shifted)  # idem, plus un score de nouveaute par vecteur courant
    assert auc_same < 0.65 and auc_shifted > 0.95 and len(novelty) == 120  # pas de derive / derive forte ; un score par vecteur
    # le verdict en clair correspond
    assert verdict(auc_same).startswith("pas de derive") and verdict(auc_shifted).startswith("derive forte")


# Verifie que la derive des statistiques simples designe la statistique qui a change (et dans quel sens).
# Attraperait : un rapport qui pointerait la mauvaise cause (par exemple le contraste au lieu de la luminosite).
def test_statistics_drift_reports_the_changed_statistic() -> None:
    rng = np.random.default_rng(0)  # generateur aleatoire a graine fixe
    # reference : luminosite moyenne ~0.5 et ecart-type ~0.1 sur 200 images
    reference = pd.DataFrame({"mean": rng.normal(0.5, 0.05, 200), "std": rng.normal(0.1, 0.01, 200)})
    # actuel : luminosite tombee a ~0.35, ecart-type inchange
    current = pd.DataFrame({"mean": rng.normal(0.35, 0.05, 200), "std": rng.normal(0.1, 0.01, 200)})
    table = statistics_drift(reference, current, ["mean", "std"])  # tableau trie de la statistique la plus changee a la moins changee
    # en tete : la moyenne, avec une mediane actuelle plus basse que celle de reference
    assert table.iloc[0]["statistic"] == "mean" and table.iloc[0]["current_median"] < table.iloc[0]["reference_median"]


# Verifie les plongements (vecteurs de longueur 1), la recherche des plus proches voisins et le regroupement.
# Attraperait : des vecteurs non normalises (similarites fausses) ou une recherche de voisins qui se trompe d'indice.
def test_embeddings_clusters_and_neighbours(dataset: pd.DataFrame) -> None:
    from vtb.explore.embeddings import build_embedder  # importe ici : seul ce test en a besoin

    paths = dataset[~dataset["path"].str.endswith("broken.png")]["path"].tolist()  # toutes les images sauf le fichier casse
    model = build_embedder("simple_cnn", pretrained=False)  # reseau qui transforme une image en vecteur
    # plongement de chaque image : redimension en 32, lecture en gris, lots de 8, sur le processeur
    vectors = embed_images(paths, Pipeline([{"name": "resize", "size": 32}]), "gray", model, batch_size=8, device=torch.device("cpu"))
    # un vecteur par image, chacun de longueur (norme) 1
    assert vectors.shape[0] == len(paths) and np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    order, sims = nearest(vectors, vectors[:2], k=3)  # 3 plus proches voisins des 2 premieres images (indices, similarites)
    assert order[0, 0] == 0 and sims[0, 0] == pytest.approx(1.0, abs=1e-5)  # une image est sa propre plus proche voisine
    labels = [Path(p).parent.name for p in paths]  # classe de chaque image = nom de son dossier
    groups = cluster(vectors, 2)  # regroupe les vecteurs en 2 groupes (k-means)
    assert 0.5 <= cluster_purity(groups, labels) <= 1.0  # purete (part d'images de la classe majoritaire du groupe) valide


# Verifie que les groupes de quasi-doublons alimentent le decoupage groupe (a-b-c = une meme piece, d seule).
# Attraperait : des quasi-doublons qui finiraient dans des parties differentes, donc une fuite.
def test_duplicate_groups_feed_a_grouped_split(tmp_path: Path) -> None:
    from vtb.io.catalogs import add_groups  # ajoute la colonne group au catalogue
    from vtb.quality.checks import duplicate_groups  # paires de doublons -> groupes

    paths = ["a.png", "b.png", "c.png", "d.png"]  # 4 images
    pairs = pd.DataFrame({"path_a": ["a.png", "b.png"], "path_b": ["b.png", "c.png"]})  # a-b-c : une meme piece
    groups = duplicate_groups(paths, pairs)  # tableau path -> group (a, b et c relies en chaine forment un seul groupe)
    # 2 groupes en tout ; d, sans doublon, est son propre groupe
    assert groups.set_index("path")["group"].nunique() == 2 and groups.set_index("path").loc["d.png", "group"] == "d.png"
    groups.to_csv(tmp_path / "groups.csv", index=False)  # ecrit ces groupes en CSV
    catalog = add_groups(pd.DataFrame({"path": paths}), None, str(tmp_path / "groups.csv"))  # le catalogue lit les groupes du CSV
    assert catalog["group"].iloc[0] == catalog["group"].iloc[2] != catalog["group"].iloc[3]  # a et c ensemble, d a part
