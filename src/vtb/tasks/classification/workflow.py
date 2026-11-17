"""Les cinq etapes de la classification, appelees par les scripts de la recette et par le notebook.

prepare(config)          catalogue + decoupage -> split.csv
train(config)            train + val -> model.pt (poids + tout ce qu'il faut pour predire)
evaluate(config)         test, une seule fois -> metriques, matrice de confusion, erreurs
predict(config, source)  images -> classes + probabilites (CSV)
export(config)           model.pt -> model.onnx verifie + model_card.json (+ OpenVINO)
"""

from __future__ import annotations  # les annotations de type restent du texte : syntaxe moderne "X | None" partout

import json  # lecture / ecriture de fichiers JSON (metriques, resume)
import logging  # journal des messages (info, avertissements) au lieu de print
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import numpy as np  # numpy : calcul sur des tableaux de nombres
import pandas as pd  # pandas : tableaux de donnees (catalogue, decoupage, historique, predictions)
import torch  # PyTorch : tenseurs (tableaux de nombres, sur CPU ou GPU), sauvegarde des modeles
import torch.nn as nn  # nn : briques de reseau et fonctions de perte

from vtb.core.runtime import count_parameters, timer  # compter les poids d'un modele ; chronometrer un bloc de code

# carte du modele (model_card.json) : fiche qui decrit entree, sorties, classes, pour l'utiliser hors Python
from vtb.export.card import InputSpec, ModelCard, OutputSpec, framework_versions, write_card

# ONNX = format standard de modele, lisible par onnxruntime (C#, C++...) sans PyTorch : export, verification, latence
from vtb.export.onnx import batch_size_of, export_onnx, measure_latency, to_openvino, verify_onnx
from vtb.io.catalogs import build_catalog  # parcourt le dossier d'images et fabrique le catalogue (chemin, etiquette)
from vtb.io.images import list_images, read_image  # lister les images d'un dossier ; lire une image en tableau numpy
from vtb.models.transfer import (  # outils d'apprentissage par transfert (reutiliser un reseau deja entraine)
    ModelConfig,  # section "model" du YAML
    ProgressiveUnfreezer,  # strategie progressive : degele un bloc de plus toutes les k epoques
    apply_strategy,  # gele / degele les couches selon la strategie choisie
    build_classifier,  # construit le reseau avec une tete adaptee au nombre de classes
    default_normalization,  # moyenne / ecart-type ImageNet si le modele est pre-entraine
    freeze_frozen_batchnorm,  # garde en mode eval les BatchNorm des blocs geles
    param_groups,  # groupes de parametres avec un taux d'apprentissage different par bloc
)
from vtb.preprocess import Pipeline  # suite d'operations OpenCV appliquee a chaque image
from vtb.preprocess.tensor import normalize, to_float_chw  # image -> tenseur float ; normalisation (x - moyenne) / ecart-type
from vtb.split import check_no_leakage, load_split, make_split, save_split  # decoupage train / val / test et controle de fuite
from vtb.tasks.classification.config import ClassificationConfig  # le YAML de l'experience, valide

# Dataset, distributeur de lots, poids des classes, statistiques de normalisation
from vtb.tasks.classification.data import ClassificationDataset, build_loader, class_weights, compute_stats
from vtb.tasks.classification.engine import build_optimizer, fit  # fabrique l'optimiseur ; boucle d'entrainement complete
from vtb.tasks.classification.engine import evaluate as run_eval  # renomme : "evaluate" est deja le nom de l'etape 3 ci-dessous
from vtb.tasks.classification.features import extract_features, fit_logistic_head  # strategie "features" (regression logistique)

logger = logging.getLogger(__name__)  # journal propre a ce module


# --- etape 1 : catalogue et decoupage --------------------------------------------------------


def prepare(config: ClassificationConfig) -> pd.DataFrame:  # catalogue + decoupage ; renvoie le tableau avec la colonne subset
    # liste toutes les images et leur classe (avec la date si le decoupage est temporel)
    catalog = build_catalog(config.data, with_timestamps=config.split.strategy == "temporal")
    strategy = (  # strategie de decoupage effectivement utilisee
        "provided"  # garde le decoupage deja present sur le disque...
        # ...si les dossiers sont deja ranges en train / val / test et qu'une autre strategie etait demandee
        if "split" in catalog and config.split.strategy != "provided" and config.data.layout == "split_folders"
        else config.split.strategy  # sinon la strategie du YAML
    )
    # decoupe le catalogue (copie de la config avec la strategie choisie) ; graine fixe : decoupage reproductible
    split = make_split(catalog, config.split.model_copy(update={"strategy": strategy}), config.runtime.seed)
    check_no_leakage(split)  # verifie qu'aucune image (ou groupe) n'est a la fois dans train et test : sinon score triche
    save_split(split, config.split.file)  # enregistre split.csv : toutes les etapes suivantes relisent ce meme decoupage
    return split  # renvoie le catalogue decoupe


# --- etape 2 : entrainement --------------------------------------------------------------------


def augmentation_for(config: ClassificationConfig):  # renvoie la fonction d'augmentation du train, ou None
    """Le Compose d'augmentation du YAML (None si vide) : seulement pour le train."""
    if not config.augment:  # si aucune augmentation n'est listee dans le YAML
        return None  # pas d'augmentation
    from vtb.augment.transforms import build_augmentation  # import tardif : charge seulement si on en a besoin

    return build_augmentation(config.augment, task="classification")  # enchaine les augmentations demandees


def _normalization(  # choisit moyenne / ecart-type de normalisation ; renvoie (mean, std) ou (None, None)
    config: ClassificationConfig,  # la config de l'experience
    train_df: pd.DataFrame,  # les images du train (seules autorisees pour calculer des stats)
    pipeline: Pipeline,  # le pretraitement, applique avant de mesurer les stats
) -> tuple[list[float] | None, list[float] | None]:  # deux listes (une valeur par canal) ou deux None
    mode = config.train.normalization  # "auto", "imagenet", "dataset" ou "none"
    if mode == "none":  # si on ne veut pas normaliser
        return None, None  # pas de statistiques
    imagenet = default_normalization(config.model)  # stats ImageNet si le modele est pre-entraine, sinon None
    if mode == "imagenet" or (mode == "auto" and imagenet is not None):  # si on veut (ou doit) utiliser les stats ImageNet
        if imagenet is None:  # demande impossible : le modele n'a pas ete pre-entraine sur ImageNet
            raise ValueError("normalization imagenet demandee pour un modele sans poids pre-entraines")  # erreur claire
        return imagenet  # les stats avec lesquelles le reseau a ete pre-entraine
    # sinon : stats calculees sur un echantillon du train (jamais val / test, pour ne pas tricher)
    return compute_stats(train_df, pipeline, config.data.color, config.train.stats_max_images, config.runtime.seed)


def train(config: ClassificationConfig, output: Path, device: torch.device) -> dict:  # entraine ; renvoie un resume (dict)
    """Entraine selon config.model.transfer.strategy et ecrit model.pt, history.csv, curves.png."""
    from vtb.evaluate.plots import plot_history  # import tardif : matplotlib seulement quand on trace

    split_file = Path(config.split.file)  # chemin de split.csv ecrit par prepare
    train_df, val_df = load_split(split_file, "train"), load_split(split_file, "val")  # lignes du train et de la val
    class_names = sorted(load_split(split_file)["label"].astype(str).unique())  # noms des classes, tries : ordre stable
    class_to_idx = {name: i for i, name in enumerate(class_names)}  # nom de classe -> numero (0, 1, 2...)
    pipeline = Pipeline(config.preprocess)  # construit le pretraitement OpenCV decrit dans le YAML
    mean, std = _normalization(config, train_df, pipeline)  # statistiques de normalisation
    # reglages communs aux Dataset train et val (seule l'augmentation differe)
    common = {"class_to_idx": class_to_idx, "pipeline": pipeline, "color": config.data.color, "mean": mean, "std": std}
    train_set = ClassificationDataset(train_df, augment=augmentation_for(config), **common)  # train : avec augmentation
    val_set = ClassificationDataset(val_df, **common)  # val : sans augmentation (on mesure sur des images non modifiees)
    sample = train_set.load(0)  # premiere image pretraitee : sert a connaitre la taille et le nombre de canaux
    channels = 1 if sample.ndim == 2 else sample.shape[2]  # 2 dimensions = image grise (1 canal), sinon (H, W, C)
    if channels != config.model.in_channels:  # si les images ne correspondent pas a ce que le modele attend
        raise ValueError(  # on arrete avec un message qui dit quoi corriger
            # message : nombre de canaux des images contre nombre attendu par le modele
            f"les images ont {channels} canal(aux) (data.color={config.data.color}) mais model.in_channels={config.model.in_channels}"
        )
    rt, tc = config.runtime, config.train  # raccourcis vers les sections runtime et train
    train_loader = build_loader(train_set, tc.batch_size, True, rt.num_workers, rt.seed)  # lots du train, melanges
    val_loader = build_loader(val_set, tc.batch_size, False, rt.num_workers, rt.seed)  # lots de la val, ordre fixe

    model, spec = build_classifier(config.model, len(class_names))  # le reseau (tete a nb classes sorties) et sa fiche
    model.to(device)  # deplace les poids sur CPU ou GPU
    transfer = config.model.transfer  # reglages de transfert (strategie, blocs degeles...)
    weights = class_weights(train_df, class_names).to(device) if tc.class_weighting else None  # poids des classes, si demandes
    # perte d'entropie croisee : la mesure standard d'erreur en classification (penalise une mauvaise probabilite)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=tc.label_smoothing)
    # resume des donnees dans le journal : classes, tailles, forme d'entree, moyenne
    logger.info("classes %s | train %d | val %d | entree %s | mean %s", class_names, len(train_set), len(val_set), sample.shape, mean)

    with timer("entrainement") as elapsed:  # chronometre tout l'entrainement (duree lue ensuite dans elapsed)
        if transfer.strategy == "features" and spec.source != "custom":  # strategie "features" sur un modele pre-entraine
            apply_strategy(model, spec, transfer)  # gele tout le reseau
            feats, targets = extract_features(  # calcule un vecteur par image du train
                model,  # le reseau gele
                spec,  # sa fiche (nom de la tete)
                # un loader du train SANS augmentation et sans melange : vecteurs des vraies images
                build_loader(ClassificationDataset(train_df, **common), tc.batch_size, False, rt.num_workers, rt.seed),
                device,  # CPU ou GPU
            )
            fit_logistic_head(model, spec, feats, targets, transfer.logistic_c, rt.seed)  # regression logistique -> nouvelle tete
            model.to(device)  # la nouvelle tete a ete creee sur CPU : on remet tout sur le device
            history = pd.DataFrame(  # historique d'une seule "epoque" (pas de boucle ici)
                # perte du train inconnue (NaN = pas un nombre) ; perte de val mesuree tout de suite
                [{"epoch": 1, "train_loss": float("nan"), "val_loss": run_eval(model, val_loader, criterion, device).loss}]
            )
            # sauve best.pt au meme format que fit, pour que la suite du code soit identique
            torch.save({"model_state": model.state_dict(), "epoch": 1, "val_loss": float(history["val_loss"].iloc[0])}, output / "best.pt")
        else:  # toutes les autres strategies : entrainement par descente de gradient
            # progressive : un objet qui degele les blocs au fil des epoques ; sinon rien
            unfreezer = ProgressiveUnfreezer(model, spec, transfer) if transfer.strategy == "progressive" else None
            if unfreezer is None:  # si la strategie n'est pas progressive
                apply_strategy(model, spec, transfer)  # gele / degele une fois pour toutes
            logger.info(  # affiche combien de poids vont vraiment apprendre
                "strategie %s | parametres entrainables %d sur %d",  # gabarit du message
                transfer.strategy,  # nom de la strategie
                count_parameters(model),  # poids entrainables (non geles)
                count_parameters(model, False),  # poids au total
            )

            def make_optimizer():  # fabrique un optimiseur neuf avec les parametres actuellement degeles
                return build_optimizer(param_groups(model, spec, tc.lr, transfer), tc.optimizer, tc.weight_decay)  # un taux par bloc

            history = fit(  # lance la boucle d'entrainement ; renvoie l'historique
                model,  # le reseau
                train_loader,  # lots du train
                val_loader,  # lots de la val
                criterion,  # fonction de perte
                make_optimizer,  # fabrique d'optimiseur (rappelee si un bloc est degele)
                tc.scheduler,  # calendrier du taux d'apprentissage
                tc.epochs,  # nombre maximal d'epoques
                tc.patience,  # patience de l'arret anticipe
                device,  # CPU ou GPU
                output / "best.pt",  # ou sauver le meilleur modele
                warmup_epochs=tc.warmup_epochs,  # montee lineaire du taux
                amp=tc.amp,  # precision mixte
                grad_clip=tc.grad_clip,  # plafond du gradient
                on_epoch_start=unfreezer.update if unfreezer else None,  # crochet de degel progressif (ou rien)
                # garde les BatchNorm geles en mode eval, sauf en fine-tuning complet ("full") ou si desactive
                on_train_mode=freeze_frozen_batchnorm if transfer.freeze_batchnorm and transfer.strategy != "full" else None,
            )

    best = torch.load(output / "best.pt", map_location=device, weights_only=True)  # relit le meilleur point de sauvegarde
    model.load_state_dict(best["model_state"])  # remet dans le modele les poids de la meilleure epoque
    val = run_eval(model, val_loader, criterion, device)  # mesure finale sur la val avec ces poids
    bundle = {  # paquet complet sauve dans model.pt : tout ce qu'il faut pour predire plus tard
        "model_state": model.state_dict(),  # les poids
        "model_config": config.model.model_dump(mode="json"),  # comment reconstruire le reseau
        "class_names": class_names,  # noms des classes dans l'ordre des sorties
        "color": config.data.color,  # comment lire les images (rgb / gray)
        "preprocess": pipeline.describe(),  # le pretraitement exact, avec tous ses parametres
        "mean": mean,  # moyenne de normalisation
        "std": std,  # ecart-type de normalisation
        "input_hw": [int(sample.shape[0]), int(sample.shape[1])],  # hauteur et largeur d'entree
        "channels": channels,  # nombre de canaux d'entree
        "best_epoch": int(best["epoch"]),  # epoque du meilleur modele
    }
    torch.save(bundle, output / "model.pt")  # ecrit le paquet sur disque
    history.to_csv(output / "history.csv", index=False)  # ecrit l'historique en CSV
    if len(history) > 1:  # s'il y a plus d'une epoque (pas le cas de la strategie features)
        plot_history(history, output / "curves.png")  # trace les courbes de perte et d'accuracy
    summary = {  # resume chiffre de l'entrainement
        "strategy": transfer.strategy,  # strategie utilisee
        "best_epoch": int(best["epoch"]),  # meilleure epoque
        "val_loss": val.loss,  # perte de val du modele retenu
        "val_accuracy": val.accuracy,  # accuracy de val du modele retenu
        "train_seconds": round(elapsed["seconds"], 1),  # duree de l'entrainement, arrondie a 0.1 s
        "trainable_params": count_parameters(model),  # nombre de poids entrainables a la fin
    }
    (output / "val_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")  # ecrit le resume en JSON lisible
    # message final : meilleure epoque, scores de val, chemin du modele
    logger.info("meilleure epoque %d | val perte %.4f acc %.3f -> %s", best["epoch"], val.loss, val.accuracy, output / "model.pt")
    return summary  # renvoie le resume a l'appelant (script ou notebook)


# --- chargement du modele sauvegarde ------------------------------------------------------------


def load_trained(output: Path, device: torch.device) -> tuple[nn.Module, dict, Pipeline]:  # recharge model.pt ; renvoie 3 objets
    """Modele en mode eval + le paquet de reglages sauve avec lui + le pipeline reconstruit depuis le fichier."""
    # relit le paquet ; weights_only=True (lecture sure) suffit : il ne contient que des tenseurs, nombres, textes, listes et dicts
    bundle = torch.load(output / "model.pt", map_location=device, weights_only=True)
    # config du modele sans telechargement : les poids viennent du fichier, pas d'Internet
    model_config = ModelConfig.model_validate({**bundle["model_config"], "pretrained": False, "weights_path": None})
    model, _ = build_classifier(model_config, len(bundle["class_names"]))  # reconstruit un reseau vide de meme forme
    model.load_state_dict(bundle["model_state"])  # y charge les poids appris
    return model.to(device).eval(), bundle, Pipeline(bundle["preprocess"])  # modele pret a predire, reglages, pretraitement


def to_input_tensor(image: np.ndarray, bundle: dict) -> torch.Tensor:  # image pretraitee -> tenseur pret pour le modele
    """Image deja pretraitee -> tenseur normalise comme a l'entrainement (mean / std du paquet sauvegarde)."""
    tensor = to_float_chw(image)  # (H, W, C) uint8 -> (C, H, W) float dans [0, 1]
    return normalize(tensor, bundle["mean"], bundle["std"]) if bundle["mean"] is not None else tensor  # normalise si prevu


# --- etape 3 : evaluation sur le test ----------------------------------------------------------


def evaluate(config: ClassificationConfig, output: Path, device: torch.device) -> dict:  # mesure sur le test ; renvoie les metriques
    from vtb.evaluate.classification import classification_metrics, report_text  # calcul des metriques ; rapport texte par classe
    from vtb.evaluate.plots import plot_confusion, plot_gallery  # matrice de confusion ; planche d'images

    model, bundle, pipeline = load_trained(output, device)  # recharge le modele entraine
    names = bundle["class_names"]  # noms des classes dans l'ordre des sorties
    test_df = load_split(config.split.file, "test")  # images du test (jamais vues pendant l'entrainement)
    # Dataset du test avec le pretraitement et la normalisation SAUVES (pas ceux du YAML actuel)
    dataset = ClassificationDataset(test_df, {n: i for i, n in enumerate(names)}, pipeline, bundle["color"], bundle["mean"], bundle["std"])
    result = run_eval(  # une passe sur le test
        model,  # le modele
        build_loader(dataset, config.train.batch_size, False, config.runtime.num_workers, config.runtime.seed),  # lots du test
        nn.CrossEntropyLoss(),  # perte sans poids de classes (mesure brute)
        device,  # CPU ou GPU
    )
    metrics = classification_metrics(result.targets, result.preds, result.probs, names)  # accuracy, F1, matrice de confusion...
    (output / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")  # sauve les metriques en JSON
    # rapport texte : precision, rappel, F1 pour chaque classe
    (output / "classification_report.txt").write_text(report_text(result.targets, result.preds, names), encoding="utf-8")
    plot_confusion(  # dessine la matrice de confusion (lignes = verite, colonnes = prediction)
        np.array(metrics["confusion_matrix"]),  # la matrice : nombre d'images pour chaque couple (verite, prediction)
        names,  # noms des classes pour les axes
        output / "confusion_matrix.png",  # fichier image de sortie
        f"test : F1 macro {metrics['f1_macro']:.3f}",  # titre : F1 macro = moyenne des F1 des classes
    )
    wrong = np.where(result.preds != result.targets)[0][:18]  # indices des images mal classees (au plus 18)
    plot_gallery(  # planche des erreurs, pour voir ce qui trompe le modele
        [dataset.load(i) for i in wrong],  # les images mal classees, pretraitees
        # titre de chaque image : vraie classe, classe predite et probabilite donnee
        [f"vrai {names[result.targets[i]]}\npredit {names[result.preds[i]]} ({result.probs[i].max():.2f})" for i in wrong],
        output / "test_errors.png",  # fichier image de sortie
    )
    logger.info(  # resume du test dans le journal
        "test : accuracy %.3f | balanced %.3f | F1 macro %.3f | %d images\n%s",  # gabarit du message
        metrics["accuracy"],  # part d'images bien classees
        metrics["balanced_accuracy"],  # moyenne des rappels par classe (juste meme si classes desequilibrees)
        metrics["f1_macro"],  # F1 moyen des classes : la metrique a suivre
        metrics["n_images"],  # nombre d'images du test
        report_text(result.targets, result.preds, names),  # detail par classe
    )
    return metrics  # renvoie les metriques a l'appelant


# --- etape 4 : prediction ----------------------------------------------------------------------


@torch.no_grad()  # pas de gradient en prediction : plus rapide, moins de memoire
# predit la classe d'une image ou de toutes les images d'un dossier ; renvoie un tableau (chemin, classe, probabilite)
def predict(config: ClassificationConfig, output: Path, device: torch.device, source: Path, batch_size: int = 32) -> pd.DataFrame:
    """Classe et probabilite pour une image ou un dossier ; le pretraitement vient de model.pt."""
    model, bundle, pipeline = load_trained(output, device)  # recharge le modele et ses reglages
    paths = list_images(source, config.data.extensions)  # liste les images a predire (un fichier ou un dossier)
    rows = []  # une ligne de resultat par image
    with timer(f"prediction de {len(paths)} images") as elapsed:  # chronometre la prediction
        for start in range(0, len(paths), batch_size):  # parcourt les images par paquets de batch_size
            chunk = paths[start : start + batch_size]  # les chemins du paquet courant
            # lit, pretraite, normalise chaque image, empile en un lot (N, C, H, W) et l'envoie sur le device
            batch = torch.stack([to_input_tensor(pipeline(read_image(p, bundle["color"])), bundle) for p in chunk]).to(device)
            probs = torch.softmax(model(batch), 1).cpu().numpy()  # softmax : scores -> probabilites par classe
            for path, p in zip(chunk, probs):  # pour chaque image du paquet et ses probabilites
                # classe = celle de plus forte probabilite ; on garde aussi cette probabilite (4 decimales)
                rows.append({"path": str(path), "label": bundle["class_names"][int(p.argmax())], "probability": round(float(p.max()), 4)})
    logger.info("%.1f ms par image", 1000 * elapsed["seconds"] / max(len(paths), 1))  # temps moyen par image
    return pd.DataFrame(rows)  # tableau des predictions


# --- etape 5 : export --------------------------------------------------------------------------


def export(config: ClassificationConfig, output: Path) -> ModelCard:  # exporte en ONNX et ecrit la carte ; renvoie la carte
    """ONNX (lot dynamique) verifie sur de vraies images de validation, latences, carte du modele."""
    device = torch.device("cpu")  # export toujours sur CPU (resultat identique partout)
    model, bundle, pipeline = load_trained(output, device)  # recharge le modele entraine
    val_df = load_split(config.split.file, "val").head(3)  # lot de 3 : different du lot d'export (2)
    # les 3 images lues, pretraitees et normalisees comme a l'entrainement, empilees en un lot
    real = torch.stack([to_input_tensor(pipeline(read_image(p, bundle["color"])), bundle) for p in val_df["path"]])
    # exporte avec un exemple de 2 images a zero (seule la forme compte) ; sortie nommee "logits"
    onnx_path = export_onnx(model, torch.zeros(2, *real.shape[1:]), output / "model.onnx", ["logits"], opset=config.export.opset)
    report = verify_onnx(model, onnx_path, real, config.export.atol, config.export.rtol)  # compare PyTorch et ONNX sur les vraies images
    xml = to_openvino(onnx_path) if config.export.openvino else None  # conversion OpenVINO (CPU Intel) si demandee
    latency = measure_latency(model, onnx_path, real[:1], config.export.latency_runs, xml)  # temps par image, chaque moteur
    # relit les metriques du test si l'etape evaluate a deja tourne, sinon dict vide
    metrics = json.loads((output / "test_metrics.json").read_text()) if (output / "test_metrics.json").exists() else {}
    height, width = bundle["input_hw"]  # taille d'entree du modele
    card = ModelCard(  # carte du modele : tout ce que le programme de production (ex. C#) doit savoir
        task="classification",  # type de tache
        model_name=bundle["model_config"]["name"],  # nom du reseau (ex. resnet18)
        framework=framework_versions(),  # versions de torch, onnx... (tracabilite)
        input=InputSpec(  # description de l'entree attendue
            read_color=bundle["color"],  # comment lire le fichier image
            height=height,  # hauteur en pixels
            width=width,  # largeur en pixels
            channels=bundle["channels"],  # nombre de canaux
            color="GRAY" if bundle["channels"] == 1 else "RGB",  # ordre des couleurs
            mean=bundle["mean"] or [],  # moyenne de normalisation (liste vide = pas de normalisation)
            std=bundle["std"] or [],  # ecart-type de normalisation
        ),
        preprocess=bundle["preprocess"],  # pretraitement OpenCV a refaire a l'identique
        outputs=[  # description des sorties
            OutputSpec(  # une seule sortie
                name="logits",  # son nom dans le fichier ONNX
                shape=["batch", len(bundle["class_names"])],  # forme : (taille du lot, nombre de classes)
                description="scores avant softmax ; classe = argmax, probabilites = softmax",  # comment l'interpreter
            )
        ],
        class_names=bundle["class_names"],  # noms des classes dans l'ordre des sorties
        # metriques du test utiles en production (seulement celles qui existent)
        metrics={k: metrics[k] for k in ("accuracy", "balanced_accuracy", "f1_macro", "n_images") if k in metrics},
        latency_ms={k: v for k, v in latency.items() if k.endswith("_ms")},  # latences en millisecondes
        files={"onnx": "model.onnx", **({"openvino": "model.xml"} if xml else {})},  # fichiers livres (OpenVINO si present)
        notes=f"verification ONNX : {report}",  # ecarts mesures entre PyTorch et ONNX
    )
    card.input.dynamic_batch = batch_size_of(onnx_path) is None  # certains modeles figent le lot (anomalib PatchCore, DFM)
    write_card(card, output / "model_card.json")  # ecrit la carte en JSON a cote du modele
    logger.info("carte du modele -> %s", output / "model_card.json")  # indique ou elle a ete ecrite
    return card  # renvoie la carte a l'appelant
