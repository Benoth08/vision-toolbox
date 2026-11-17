"""Les trois familles d'anomalies de bout en bout sur une arborescence MVTec synthetique, sans rien telecharger."""

# Detection d'anomalies : on n'apprend que sur des images normales, puis on signale ce qui s'en ecarte.
# Ces tests font tourner tout le parcours (preparer, entrainer, evaluer, predire, exporter) sur de toutes
# petites images, pour verifier que les etapes s'enchainent sans erreur et produisent les bons fichiers.

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[int], X | None) partout

import json  # json : lire des fichiers JSON (format texte de donnees structurees)
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import numpy as np  # numpy : calcul sur des tableaux de nombres
import pandas as pd  # pandas : tableaux de donnees avec colonnes nommees (DataFrame)
import pytest  # pytest : lance les tests ; pytest.approx compare des nombres a virgule avec une tolerance
import torch  # PyTorch : calcul sur des tenseurs (tableaux de nombres, sur CPU ou GPU) et reseaux de neurones

from vtb.tasks.anomaly import workflow  # les 5 etapes du parcours anomalies : prepare, train, evaluate, predict, export
from vtb.tasks.anomaly.config import AnomalyConfig  # configuration validee d'un projet d'anomalies
from vtb.tasks.anomaly.data import make_anomaly_split, to_good_defect  # decoupage train/val/test et etiquettes good / defect
from vtb.tasks.anomaly.patchcore_lite import greedy_coreset  # choix d'un petit sous-ensemble representatif de vecteurs

# scoring : carte d'anomalie -> score d'image, calcul des seuils de decision, metriques d'evaluation
from vtb.tasks.anomaly.scoring import MapToScore, anomaly_metrics, calibrate_thresholds


# fabrique une configuration minimale (petites images, peu d'iterations) pour une methode donnee et la renvoie
def _config(root: Path, out: Path, method: str, **section) -> AnomalyConfig:  # **section : reglages propres a la methode
    return AnomalyConfig.model_validate(  # model_validate verifie le dictionnaire et le transforme en objet de configuration
        {  # la configuration ecrite comme un dictionnaire, section par section (comme dans un fichier YAML)
            "data": {"root": str(root), "layout": "mvtec"},  # ou sont les images, et rangees a la facon MVTec
            "split": {"file": str(out / "split.csv"), "val_fraction": 0.3},  # fichier du decoupage ; 30 % des normales en validation
            "preprocess": [{"name": "resize", "size": 32}],  # pretraitement : redimensionner en 32 x 32
            "method": method,  # methode d'anomalie testee
            "scoring": {"gaussian_sigma": 1.0},  # lissage gaussien de la carte d'anomalie (enleve le bruit pixel a pixel)
            "export": {"latency_runs": 2},  # mesure de vitesse a l'export : 2 passages seulement pour aller vite
            # dossier de sortie propre a la methode ; num_workers 0 = images chargees dans le processus principal (plus simple)
            "runtime": {"output_dir": str(out / method), "num_workers": 0},
            **section,  # ajoute les reglages propres a la methode
        }
    )


# parametrize : le test est lance une fois par methode, avec ses reglages minimaux
@pytest.mark.parametrize(
    "method,section",  # noms des parametres recus par le test
    [  # une ligne = un lancement
        # PatchCore simplifie sans poids pre-entraines (pas de telechargement), memoire reduite a 50 %
        ("patchcore_lite", {"patchcore_lite": {"pretrained": False, "memory_fraction": 0.5, "coreset": "greedy"}}),
        # auto-encodeur minuscule entraine 2 epoques (une epoque = un passage complet sur les images d'entrainement)
        ("autoencoder", {"autoencoder": {"epochs": 2, "widths": [8, 16], "latent_channels": 4, "batch_size": 4}}),
    ],
)
# Verifie le parcours complet pour chaque methode maison. Attraperait : une etape qui casse, un defaut qui fuit dans
# l'entrainement, un fichier de calibration absent, un export dont les seuils ne correspondent pas a la calibration.
def test_anomaly_methods_end_to_end(mvtec_folder: Path, tmp_path: Path, method: str, section: dict):
    config = _config(mvtec_folder, tmp_path, method, **section)  # configuration de ce lancement
    output, device = Path(config.runtime.output_dir), torch.device("cpu")  # dossier de sortie ; calcul sur le processeur
    output.mkdir(parents=True)  # cree le dossier de sortie
    split = workflow.prepare(config)  # etape 1 : catalogue + decoupage train / val / test
    assert set(split.loc[split["split"].isin(["train", "val"]), "label"]) == {"good"}  # train et val ne contiennent que des normales

    summary = workflow.train(config, output, device)  # etape 2 : apprentissage sur les normales + calibration du seuil
    # 3 normales de validation (30 % de 10) servent a calibrer, et le fichier de calibration est ecrit
    assert summary["n_val_normals"] == 3 and (output / "calibration.json").exists()
    metrics = workflow.evaluate(config, output, device)  # etape 3 : evaluation sur le test
    # AUROC (qualite du classement normal / defaut, 1 = parfait) image et pixel, rappel, fausses alarmes, rappel par type
    assert {"auroc_image", "auroc_pixel", "recall", "false_alarm_rate", "recall_per_type"} <= set(metrics)
    # etape 4 : prediction sur un dossier d'images, en enregistrant les cartes d'anomalie en PNG
    table = workflow.predict(config, output, device, mvtec_folder / "test" / "crack", save_maps=tmp_path / "maps")
    assert len(table) == 3 and len(list((tmp_path / "maps").glob("*.png"))) == 3  # une ligne et une carte par image
    card = workflow.export(config, output)  # etape 5 : export du modele (ONNX) et de sa fiche de deploiement
    # le seuil inscrit dans la fiche est exactement celui de la calibration
    assert card.thresholds["image"] == pytest.approx(json.loads((output / "calibration.json").read_text())["image_threshold"])
    assert [o.name for o in card.outputs] == ["anomaly_map", "score"]  # le modele exporte sort la carte puis le score


# Verifie qu'un modele de la bibliotheque anomalib passe par les memes etapes que les methodes maison.
# Attraperait : un adaptateur anomalib qui ne respecte plus le contrat commun (evaluation ou export casses).
# monkeypatch : fixture pytest qui modifie l'environnement (ici le dossier courant) le temps du test seulement
def test_anomalib_model_through_the_same_contract(mvtec_folder: Path, tmp_path: Path, monkeypatch):
    pytest.importorskip("anomalib")  # si anomalib n'est pas installe, le test est saute (et non en echec)
    monkeypatch.chdir(tmp_path)  # anomalib ecrit sous data/processed et outputs : dans le dossier du test
    config = _config(  # configuration pour la methode "anomalib"
        mvtec_folder,  # dossier des donnees MVTec synthetiques
        tmp_path,  # dossier de sortie
        "anomalib",  # methode
        # modele PaDiM sur un ResNet18 non pre-entraine (pas de telechargement), 1 epoque, lots de 4 images
        anomalib={"model": "Padim", "params": {"backbone": "resnet18", "pre_trained": False}, "max_epochs": 1, "batch_size": 4},
    )
    output, device = Path(config.runtime.output_dir), torch.device("cpu")  # dossier de sortie ; calcul sur le processeur
    output.mkdir(parents=True)  # cree le dossier de sortie
    workflow.prepare(config)  # decoupage
    workflow.train(config, output, device)  # entrainement + calibration
    metrics = workflow.evaluate(config, output, device)  # evaluation
    assert "auroc_image" in metrics  # la metrique principale est bien calculee
    workflow.export(config, output)  # export
    assert (output / "model.onnx").exists()  # ONNX : format de modele standard, lisible sans PyTorch (C#, OpenVINO...)


# Verifie que le decoupage n'envoie jamais un defaut en train ou en validation. Attraperait : un modele qui apprend
# des defauts comme "normaux" (il ne les signalerait plus) ou un seuil calibre sur des defauts.
def test_split_keeps_every_defect_out_of_train_and_val():
    # catalogue fictif de 30 images : 20 normales (MT_Free) et 10 fissures (MT_Crack), noms a la facon du jeu Magnetic Tile
    catalog = pd.DataFrame({"path": [f"{i}.png" for i in range(30)], "label": ["MT_Free"] * 20 + ["MT_Crack"] * 10})
    # etiquettes ramenees a good / defect, puis decoupage 25 % val, 25 % test, graine 0
    split = make_anomaly_split(to_good_defect(catalog, ["MT_Free"]), val_fraction=0.25, test_fraction=0.25, seed=0)
    assert (split.loc[split["label"] == "defect", "split"] == "test").all()  # tous les defauts sont en test
    # les 3 parties existent, et le type d'origine (MT_Crack) est garde pour le rappel par type de defaut
    assert set(split["split"]) == {"train", "val", "test"} and split.loc[split["label"] == "defect", "defect_type"].eq("MT_Crack").all()


# Verifie la chaine carte -> score, le calcul du seuil et les metriques sur des cas dont on connait la reponse.
# Attraperait : un score qui ne monte pas avec l'anomalie, un seuil mal calcule, une metrique fausse.
def test_scoring_calibration_and_metrics():
    to_score = MapToScore((8, 8), sigma=1.0)  # agrandit la carte en 8 x 8, la lisse (sigma 1) et en tire un score par image
    maps, score = to_score(  # applique le module a un lot de 2 cartes brutes
        # 2 cartes 4 x 4 nulles ; index_put_ met 5.0 a la position (image 1, canal 0, ligne 1, colonne 1) : un point chaud
        torch.zeros(2, 1, 4, 4).index_put_((torch.tensor([1]), torch.tensor([0]), torch.tensor([1]), torch.tensor([1])), torch.tensor(5.0))
    )
    assert maps.shape == (2, 1, 8, 8) and score[1] > score[0]  # cartes a la taille demandee ; l'image avec point chaud a le plus haut score
    # seuil image = quantile 0.99 de 101 scores de normales regulierement espaces de 0 a 1 (cartes nulles pour les pixels)
    calibration = calibrate_thresholds(np.linspace(0, 1, 101), np.zeros((2, 4, 4)), 0.99, 0.999)
    assert calibration["image_threshold"] == pytest.approx(0.99)  # 99 % des normales sont sous le seuil : 1 % de fausses alarmes
    labels = np.array([0, 0, 1, 1])  # 2 normales (0) puis 2 defauts (1)
    metrics = anomaly_metrics(  # metriques avec des scores parfaitement separes par le seuil 1.0
        np.array([0.1, 0.2, 2.0, 3.0]),  # scores d'image : les 2 normales sous 1.0, les 2 defauts au-dessus
        np.zeros((4, 2, 2)),  # cartes d'anomalie (nulles : on ne teste pas les pixels ici)
        np.zeros((4, 2, 2)),  # masques de verite terrain (vides : pas d'AUROC pixel calculable)
        labels,  # etiquette de chaque image (0 normal, 1 defaut)
        1.0,  # seuil de decision image : au-dessus = defaut
        np.array(["good", "good", "a", "b"]),  # type de chaque image, pour le rappel par type de defaut
    )
    assert (  # separation parfaite, donc :
        metrics["auroc_image"] == 1.0  # classement parfait
        and metrics["recall"] == 1.0  # tous les defauts detectes
        and metrics["false_alarm_rate"] == 0.0  # aucune normale signalee a tort
        and metrics["recall_per_type"] == {"a": 1.0, "b": 1.0}  # chaque type de defaut detecte a 100 %
    )


# Verifie que le coreset glouton (PatchCore) garde un representant de chaque groupe de vecteurs, meme eloigne.
# Attraperait : une memoire qui oublie une zone de l'espace normal, donc de fausses alarmes sur des images normales.
def test_greedy_coreset_covers_distant_clusters():
    rng = torch.Generator().manual_seed(0)  # generateur aleatoire PyTorch a graine fixe : test reproductible
    # 3 amas de 50 vecteurs de dimension 8, tres serres (ecart 0.01) autour de 0, 10 et -10 ; cat les met bout a bout
    clusters = torch.cat([torch.randn(50, 8, generator=rng) * 0.01 + c for c in (0.0, 10.0, -10.0)])
    chosen = greedy_coreset(clusters, 3, seed=0)  # choisit 3 vecteurs : a chaque pas le plus loin de ceux deja pris
    assert len({int(i) // 50 for i in chosen}) == 3  # un representant par amas, la ou un tirage aleatoire en raterait
