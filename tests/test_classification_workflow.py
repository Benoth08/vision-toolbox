"""La recette classification de bout en bout sur un jeu synthetique : prepare, train, evaluate, predict, export."""

# Classification : le modele donne une classe a chaque image entiere. Ce test fait tourner les 5 etapes de la
# recette sur le petit jeu "un dossier par classe" de conftest.py, pour verifier qu'elles s'enchainent.

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[int], X | None) partout

import json  # json : lire des fichiers JSON (format texte de donnees structurees)
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import pytest  # pytest : lance les tests ; parametrize, raises...
import torch  # PyTorch : tenseurs (tableaux de nombres, sur CPU ou GPU) et reseaux de neurones

from vtb.export.card import read_card  # relit la fiche du modele exporte (model_card.json : entrees, classes, pretraitement)
from vtb.tasks.classification import workflow  # les 5 etapes de la recette : prepare, train, evaluate, predict, export
from vtb.tasks.classification.config import ClassificationConfig  # configuration validee d'un projet de classification


# fabrique une configuration minimale et rapide, et la renvoie ; **model : reglages du modele a ajouter ou remplacer
def _config(root: Path, out: Path, **model) -> ClassificationConfig:
    return ClassificationConfig.model_validate(  # model_validate verifie le dictionnaire et le transforme en objet de configuration
        {  # la configuration ecrite comme un dictionnaire, section par section (comme dans un fichier YAML)
            # images en niveaux de gris ; group_regex extrait le numero de piece (P000...) du nom de fichier
            "data": {"root": str(root), "color": "gray", "group_regex": r"^(P\d+)_"},
            # decoupage "grouped" : toutes les vues d'une meme piece vont dans la meme partie (sinon fuite : le modele
            # reverrait en test une piece deja vue en entrainement) ; 25 % validation, 25 % test
            "split": {"file": str(out / "split.csv"), "strategy": "grouped", "val_fraction": 0.25, "test_fraction": 0.25},
            "preprocess": [{"name": "resize", "size": 32}],  # pretraitement : redimensionner en 32 x 32
            # petit reseau convolutif maison, sans poids pre-entraines, 1 canal d'entree (gris), 2 couches de 8 et 16 filtres
            "model": {"name": "simple_cnn", "pretrained": False, "in_channels": 1, "extra": {"channels": [8, 16]}, **model},
            # 8 epoques (passages sur tout le jeu), lots de 4 images, pas d'apprentissage 0.01, patience = arret si pas de progres
            "train": {"epochs": 8, "batch_size": 4, "lr": 0.01, "patience": 8},  # BatchNorm : assez de pas pour stabiliser ses statistiques
            "export": {"latency_runs": 2},  # mesure de vitesse a l'export : 2 passages seulement
            "runtime": {"output_dir": str(out / "run"), "num_workers": 0},  # dossier de sortie ; chargement dans le processus principal
        }
    )


# "full" : on entraine tout le reseau ; "features" : le corps est gele, seule la derniere couche (la "tete") apprend
@pytest.mark.parametrize("strategy", ["full", "features"])
# Verifie les 5 etapes et les fichiers produits, pour les deux strategies d'apprentissage par transfert.
# Attraperait : une etape cassee, un modele qui n'apprend rien, une fiche d'export incoherente avec l'entrainement.
def test_end_to_end_on_synthetic_folders(class_folder: Path, tmp_path: Path, strategy: str):
    config = _config(class_folder, tmp_path, transfer={"strategy": strategy})  # configuration avec la strategie testee
    output, device = Path(config.runtime.output_dir), torch.device("cpu")  # dossier de sortie ; calcul sur le processeur
    output.mkdir(parents=True)  # cree le dossier de sortie
    split = workflow.prepare(config)  # etape 1 : catalogue + decoupage
    assert set(split["split"]) == {"train", "val", "test"}  # les trois parties existent

    summary = workflow.train(config, output, device)  # etape 2 : entrainement
    assert (output / "model.pt").exists() and summary["strategy"] == strategy  # poids sauvegardes ; bonne strategie appliquee

    metrics = workflow.evaluate(config, output, device)  # etape 3 : evaluation sur le test
    assert metrics["f1_macro"] > 0.6  # trois classes de luminosites tres differentes : facile
    assert (output / "confusion_matrix.png").exists()  # matrice de confusion (qui est pris pour qui) enregistree en image

    predictions = workflow.predict(config, output, device, class_folder / "alpha")  # etape 4 : prediction sur un dossier
    # une ligne par image (12), avec le chemin, la classe predite et sa probabilite
    assert len(predictions) == 12 and set(predictions.columns) == {"path", "label", "probability"}

    card = workflow.export(config, output)  # etape 5 : export ONNX + fiche du modele
    # fichier ONNX ecrit ; la fiche annonce 1 canal d'entree et les 3 classes dans l'ordre
    assert (output / "model.onnx").exists() and card.input.channels == 1 and card.class_names == ["alpha", "beta", "gamma"]
    # la fiche relue garde le pretraitement exact (avec la valeur par defaut d'interpolation) : le deploiement le refera a l'identique
    assert read_card(output / "model_card.json").preprocess == [{"name": "resize", "size": 32, "interpolation": "auto"}]
    # le fichier de metriques de test compte autant d'images que la partie test du decoupage
    assert json.loads((output / "test_metrics.json").read_text())["n_images"] == len(split[split["split"] == "test"])


# Verifie qu'un modele qui attend 3 canaux (couleur) sur des images grises est refuse avant l'entrainement.
# Attraperait : une erreur obscure en plein entrainement (ou pire, un entrainement inutile) au lieu d'un message clair.
def test_channel_mismatch_is_reported_before_training(class_folder: Path, tmp_path: Path):
    config = _config(class_folder, tmp_path, in_channels=3)  # modele a 3 canaux d'entree alors que les images sont grises
    output = Path(config.runtime.output_dir)  # dossier de sortie
    output.mkdir(parents=True)  # cree le dossier
    workflow.prepare(config)  # decoupage (necessaire avant l'entrainement)
    with pytest.raises(ValueError, match="in_channels"):  # on attend une ValueError dont le message cite in_channels
        workflow.train(config, output, torch.device("cpu"))  # l'entrainement doit refuser de demarrer
