"""Les cinq etapes de la detection d'anomalies, identiques pour toutes les methodes.

    prepare   catalogue -> good / defect -> decoupage (normales seulement dans train et val)
    train     autoencoder : entrainement ; patchcore_lite : memoire ; anomalib : fit du modele choisi
              puis, pour toutes : scores des normales de validation -> seuils (calibration.json)
    evaluate  test : AUROC image et pixel, rappel et fausses alarmes au seuil, rappel par type, figures
    predict   images -> score, decision, carte (PNG facultatif)
    export    ONNX : image [0, 1] -> (anomaly_map, score), verifie ; seuils dans la carte du modele

Toutes les methodes deviennent un meme nn.Module de production : AnomalyModel(raw_map, MapToScore),
ce qui rend les resultats comparables et les modeles interchangeables dans le programme final.
"""

from __future__ import annotations  # permet les annotations de type modernes

import json  # lecture / ecriture de fichiers JSON (seuils, metriques)
import logging  # journal des messages
from pathlib import Path  # Path : chemin de fichier independant du systeme

import numpy as np  # numpy : calcul sur des tableaux de nombres
import pandas as pd  # pandas : tableaux de donnees
import torch  # PyTorch : tenseurs et reseaux de neurones
import torch.nn as nn  # nn : modules de reseau

from vtb.core.runtime import timer  # chronometre utilisable avec "with" (mesure la duree d'un bloc)

# carte du modele (model card) : fichier JSON qui decrit l'entree, les sorties, les seuils et les metriques du modele exporte
from vtb.export.card import InputSpec, ModelCard, OutputSpec, framework_versions, write_card

# export ONNX (format standard de modele, lisible sans PyTorch), verification, mesure de vitesse, conversion OpenVINO (Intel)
from vtb.export.onnx import batch_size_of, export_onnx, measure_latency, to_openvino, verify_onnx
from vtb.io.catalogs import build_catalog  # construit le catalogue (une ligne par image) a partir du dossier de donnees
from vtb.io.images import list_images, read_image  # liste les images d'un dossier ; lit une image
from vtb.preprocess import Pipeline  # enchainement des etapes de pretraitement
from vtb.preprocess.tensor import to_float_chw  # image numpy (H, W, C) 0-255 -> tenseur (C, H, W) 0-1
from vtb.split import load_split, save_split  # lit / ecrit le fichier de decoupage train / val / test
from vtb.tasks.anomaly.config import AnomalyConfig  # la configuration de l'experience
from vtb.tasks.anomaly.data import AnomalyDataset, anomaly_loader, make_anomaly_split, to_good_defect  # donnees d'anomalies
from vtb.tasks.anomaly.scoring import MapToScore, anomaly_metrics, calibrate_thresholds  # score, seuils, metriques

logger = logging.getLogger(__name__)  # journal propre a ce module


class AnomalyModel(nn.Module):  # modele final commun a toutes les methodes : carte brute puis lissage et score
    """Le modele de production : image (N, C, S, S) dans [0, 1] -> (carte lissee (N, 1, S, S), score image (N,))."""

    def __init__(self, raw_map: nn.Module, to_score: MapToScore) -> None:  # assemble les deux morceaux
        super().__init__()  # initialise nn.Module
        self.raw_map, self.to_score = raw_map, to_score  # raw_map : methode choisie ; to_score : post-traitement commun

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # image -> (carte lissee, score)
        return self.to_score(self.raw_map(x))  # carte brute de la methode, puis lissage et score image


# --- etape 1 ------------------------------------------------------------------------------------


def prepare(config: AnomalyConfig) -> pd.DataFrame:  # etape 1 : catalogue + decoupage ; renvoie le decoupage
    catalog = to_good_defect(build_catalog(config.data), config.normal_labels)  # liste des images, etiquettes ramenees a good / defect
    # decoupage : normales en train / val / test, defauts tous en test
    split = make_anomaly_split(catalog, config.split.val_fraction, config.split.test_fraction, config.runtime.seed)
    save_split(split, config.split.file)  # enregistre le decoupage (les etapes suivantes le relisent)
    return split  # le tableau du decoupage


# --- etape 2 ------------------------------------------------------------------------------------


# cree les Datasets d'entrainement et de validation ; renvoie un dictionnaire nom -> Dataset
def _datasets(config: AnomalyConfig, pipeline: Pipeline) -> dict[str, AnomalyDataset]:
    frames = {s: load_split(config.split.file, s) for s in ("train", "val")}  # lignes du decoupage pour train et val
    augment = None  # pas d'augmentation par defaut
    if config.augment and config.method == "autoencoder":  # l'augmentation ne sert qu'a l'auto-encodeur (seule methode entrainee ici)
        from vtb.augment.transforms import build_augmentation  # import ici : seulement si on en a besoin

        augment = build_augmentation(config.augment, task="classification")  # transformations aleatoires decrites dans la config
    return {  # trois Datasets
        "train": AnomalyDataset(frames["train"], pipeline, config.data.color, augment),  # train avec augmentation (apprentissage)
        "train_eval": AnomalyDataset(frames["train"], pipeline, config.data.color),  # train sans augmentation (memoire PatchCore)
        "val": AnomalyDataset(frames["val"], pipeline, config.data.color),  # validation (calibration du seuil)
    }


def _image_size(dataset: AnomalyDataset) -> tuple[int, int, int]:  # renvoie (hauteur, largeur, canaux) des images pretraitees
    image = dataset.load(0)  # lit la premiere image (le pretraitement donne la meme taille a toutes)
    return image.shape[0], image.shape[1], 1 if image.ndim == 2 else image.shape[2]  # 1 canal si l'image est 2D (grise)


@torch.no_grad()  # pas de gradient : on ne fait que calculer des scores (plus rapide, moins de memoire)
# passe tout un Dataset dans le modele ; renvoie scores, cartes, masques et etiquettes en tableaux numpy
def score_dataset(model: AnomalyModel, dataset: AnomalyDataset, batch_size: int, device: torch.device) -> dict[str, np.ndarray]:
    """Scores (N,), cartes (N, S, S), masques (N, S, S) et etiquettes (N,) pour tout un Dataset."""
    model.eval().to(device)  # mode evaluation, sur CPU ou GPU
    scores, maps, masks, labels = [], [], [], []  # resultats accumules lot par lot
    for images, mask, label in anomaly_loader(dataset, batch_size, False, 0, 0):  # pour chaque lot, dans l'ordre (sans melange)
        anomaly_map, score = model(images.to(device))  # carte et score de chaque image du lot
        scores.append(score.cpu())  # ramene les scores sur le CPU (numpy ne lit pas la memoire du GPU)
        maps.append(anomaly_map[:, 0].cpu())  # cartes sans la dimension canal : (N, S, S)
        masks.append(mask[:, 0])  # masques sans la dimension canal
        labels.append(label)  # etiquettes du lot
    return {  # tout recolle en tableaux numpy
        "scores": torch.cat(scores).numpy(),  # (N,)
        "maps": torch.cat(maps).numpy(),  # (N, S, S)
        "masks": torch.cat(masks).numpy(),  # (N, S, S)
        "labels": torch.cat(labels).numpy(),  # (N,)
    }


# etape 2 : construit la methode et calibre ; renvoie un bilan
def train(config: AnomalyConfig, output: Path, device: torch.device) -> dict:
    """Construit la methode demandee, puis calibre les seuils sur les normales de validation."""
    pipeline = Pipeline(config.preprocess)  # pretraitement decrit dans la config
    sets = _datasets(config, pipeline)  # Datasets train, train_eval, val
    height, width, channels = _image_size(sets["train"])  # taille des images apres pretraitement
    rt = config.runtime  # raccourci vers les reglages d'execution (graine, processus de lecture)
    with timer(f"methode {config.method}") as elapsed:  # chronometre la construction de la methode
        if config.method == "autoencoder":  # methode 1 : auto-encodeur entraine
            from vtb.evaluate.plots import plot_history  # trace les courbes de perte
            from vtb.tasks.anomaly.autoencoder import ConvAutoencoder, ReconstructionMap, fit_autoencoder  # l'auto-encodeur

            ae = config.autoencoder  # raccourci vers ses reglages
            autoencoder = ConvAutoencoder(ae.latent_channels, ae.widths, channels).to(device)  # cree le reseau sur CPU / GPU
            history = fit_autoencoder(  # entrainement avec arret anticipe
                autoencoder,  # le reseau
                anomaly_loader(sets["train"], ae.batch_size, True, rt.num_workers, rt.seed),  # lots d'entrainement melanges
                anomaly_loader(sets["val"], ae.batch_size, False, rt.num_workers, rt.seed),  # lots de validation dans l'ordre
                ae.epochs,  # nombre maximal d'epoques
                ae.lr,  # taux d'apprentissage
                ae.weight_decay,  # penalite sur les poids
                ae.patience,  # patience de l'arret anticipe
                device,  # CPU ou GPU
                output / "best.pt",  # fichier du meilleur modele
            )
            history.to_csv(output / "history.csv", index=False)  # sauve l'historique des pertes
            plot_history(history, output / "curves.png", metrics=("loss",))  # trace les courbes de perte
            # recharge les poids du meilleur modele (pas ceux de la derniere epoque) ; weights_only : chargement sur
            autoencoder.load_state_dict(torch.load(output / "best.pt", map_location=device, weights_only=True)["model_state"])
            raw: nn.Module = ReconstructionMap(autoencoder)  # carte brute = erreur de reconstruction
        elif config.method == "patchcore_lite":  # methode 2 : PatchCore simplifie (pas d'entrainement, une memoire)
            from vtb.tasks.anomaly.patchcore_lite import PatchCoreLite  # la classe PatchCore de la toolbox

            pc = config.patchcore_lite  # raccourci vers ses reglages
            raw = PatchCoreLite(pc.layers, pc.pretrained)  # cree l'extracteur et la memoire vide
            raw.fit(  # remplit la memoire avec les vecteurs des normales
                anomaly_loader(sets["train_eval"], pc.batch_size, False, rt.num_workers, rt.seed),  # normales du train, sans augmentation
                pc.memory_fraction,  # part des vecteurs a garder
                pc.coreset,  # methode de selection (greedy ou random)
                rt.seed,  # graine
                device,  # CPU ou GPU
            )
        else:  # methode 3 : un modele de la bibliotheque anomalib
            # fonctions de l'adaptateur anomalib
            from vtb.tasks.anomaly.anomalib_adapter import AnomalibRawMap, build_model, fit_anomalib, materialize_normals

            ac = config.anomalib  # raccourci vers ses reglages
            torch.manual_seed(rt.seed)  # PaDiM tire ses canaux au hasard a la construction : graine AVANT build_model
            model = build_model(ac.model, ac.params, (height, width))  # cree le modele anomalib par son nom
            train_dir = materialize_normals(  # ecrit les normales pretraitees sur disque pour anomalib
                load_split(config.split.file, "train"),  # lignes des normales d'entrainement
                pipeline,  # pretraitement a leur appliquer
                config.data.color,  # mode de lecture (rgb / gray)
                Path("data/processed/anomalib") / output.name,  # dossier propre a cette experience
            )
            # entrainement (ou memoire) du modele anomalib
            fit_anomalib(model, train_dir, ac.batch_size, ac.max_epochs, output, rt.seed, rt.num_workers, ac.use_datamodule)
            raw = AnomalibRawMap(model)  # enveloppe : renvoie la carte brute comme les autres methodes
    sc = config.scoring  # raccourci vers les reglages de score
    # modele de production : carte brute + lissage + score, sur CPU / GPU, en mode evaluation
    anomaly_model = AnomalyModel(raw, MapToScore((height, width), sc.gaussian_sigma, sc.image_score, sc.top_fraction)).to(device).eval()
    val = score_dataset(anomaly_model, sets["val"], 16, device)  # scores des normales de validation (16 images par lot)
    calibration = calibrate_thresholds(val["scores"], val["maps"], sc.image_quantile, sc.pixel_quantile)  # seuils = quantiles
    (output / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")  # sauve les seuils
    torch.save(  # sauve le modele complet et tout ce qu'il faut pour le reutiliser
        {  # dictionnaire enregistre dans model.pt
            "module": anomaly_model.cpu(),  # objet complet : chaque methode a sa propre structure (memoire, reseau anomalib)
            "method": config.method,  # nom de la methode
            "preprocess": pipeline.describe(),  # description du pretraitement (pour le refaire a l'identique)
            "color": config.data.color,  # mode de lecture des images (rgb / gray)
            "input_hw": [height, width],  # taille d'entree attendue
            "channels": channels,  # nombre de canaux
            **calibration,  # les seuils et leurs quantiles
        },
        output / "model.pt",  # fichier de destination
    )
    summary = {  # bilan de l'etape
        "method": config.method,  # methode
        "name": config.anomalib.model if config.method == "anomalib" else config.method,  # nom lisible (modele anomalib ou methode)
        "fit_seconds": round(elapsed["seconds"], 1),  # duree de construction en secondes
        **calibration,  # seuils
    }
    (output / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")  # sauve le bilan
    logger.info(  # message de fin
        "seuil image %.5f (quantile %.3f de %d normales de validation) -> %s",  # modele du message
        calibration["image_threshold"],  # seuil image
        sc.image_quantile,  # quantile utilise
        calibration["n_val_normals"],  # nombre de normales de validation
        output / "model.pt",  # fichier du modele
    )
    return summary  # le bilan


# recharge le modele sauve par train ; renvoie (modele, contenu du fichier, pretraitement)
def load_trained(output: Path, device: torch.device) -> tuple[AnomalyModel, dict, Pipeline]:
    # weights_only=False : le fichier contient un objet Python complet (a n'ouvrir que si on lui fait confiance)
    bundle = torch.load(output / "model.pt", map_location=device, weights_only=False)
    return bundle["module"].to(device).eval(), bundle, Pipeline(bundle["preprocess"])  # modele pret, infos, pretraitement reconstruit


# --- etape 3 ------------------------------------------------------------------------------------


def evaluate(config: AnomalyConfig, output: Path, device: torch.device) -> dict:  # etape 3 : mesures sur le test ; renvoie les metriques
    from vtb.tasks.anomaly.plots import plot_maps, plot_score_histogram  # figures (import ici : matplotlib est lent a charger)

    model, bundle, pipeline = load_trained(output, device)  # modele entraine et ses infos
    test_df = load_split(config.split.file, "test")  # images du test (normales mises de cote + tous les defauts)
    dataset = AnomalyDataset(test_df, pipeline, bundle["color"])  # Dataset du test
    with timer("scoring du test") as elapsed:  # chronometre le calcul des scores
        result = score_dataset(model, dataset, 16, device)  # scores, cartes, masques, etiquettes du test
    metrics = anomaly_metrics(  # AUROC, rappel, fausses alarmes, F1, rappel par type
        result["scores"],  # score de chaque image du test
        result["maps"],  # carte de chaque image
        result["masks"],  # masque verite de chaque image
        result["labels"],  # etiquette de chaque image (1 defaut, 0 normal)
        bundle["image_threshold"],  # seuil calibre sur la validation
        test_df["defect_type"].to_numpy(),  # type de defaut de chaque image (pour le rappel par type)
    )
    metrics.update(  # ajoute le seuil et la vitesse aux metriques
        # temps moyen par image en millisecondes (max evite une division par zero)
        {"image_threshold": bundle["image_threshold"], "ms_per_image": round(1000 * elapsed["seconds"] / max(len(dataset), 1), 1)}
    )
    (output / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")  # sauve les metriques
    plot_score_histogram(result["scores"], result["labels"], bundle["image_threshold"], output / "scores.png")  # figure des scores
    plot_maps(dataset, result, test_df, bundle["image_threshold"], output / "maps.png")  # figure des cartes sur des exemples
    logger.info(  # bilan du test
        "test : AUROC image %.3f | AUROC pixel %s | rappel %.3f | fausses alarmes %.3f | par type %s",  # modele du message
        metrics.get("auroc_image", float("nan")),  # AUROC image (nan si absente)
        f"{metrics['auroc_pixel']:.3f}" if "auroc_pixel" in metrics else "n/a",  # AUROC pixel ou "n/a" s'il n'y a pas de masques
        metrics["recall"],  # rappel
        metrics["false_alarm_rate"],  # taux de fausses alarmes
        metrics.get("recall_per_type"),  # rappel par type de defaut
    )
    return metrics  # les metriques


# --- etape 4 ------------------------------------------------------------------------------------


@torch.no_grad()  # pas de gradient : simple prediction
# etape 4 : score de chaque image d'un dossier, decision, carte en PNG si demande ; renvoie un tableau (une ligne par image)
def predict(config: AnomalyConfig, output: Path, device: torch.device, source: Path, save_maps: Path | None = None) -> pd.DataFrame:
    from vtb.explain.gradcam import overlay  # superpose une carte de chaleur coloree sur l'image
    from vtb.io.images import write_image  # ecriture d'image

    model, bundle, pipeline = load_trained(output, device)  # modele entraine et ses infos
    rows = []  # une ligne de resultat par image
    for path in list_images(source, config.data.extensions):  # pour chaque image du dossier source
        image = pipeline(read_image(path, color=bundle["color"]))  # lecture et pretraitement identiques a l'entrainement
        anomaly_map, score = model(to_float_chw(image)[None].to(device))  # [None] : lot d'une seule image ; renvoie carte et score
        value = float(score[0])  # score de cette image en nombre Python
        # ligne du tableau : chemin, score arrondi, decision (vrai si le score depasse le seuil)
        rows.append({"path": str(path), "score": round(value, 6), "anomaly": value > bundle["image_threshold"]})
        if save_maps is not None:  # si on a demande d'enregistrer les cartes
            heat = anomaly_map[0, 0].cpu().numpy()  # la carte de cette image en tableau numpy (S, S)
            heat = np.clip(heat / max(bundle["pixel_threshold"] * 2, 1e-9), 0, 1)  # meme echelle pour toutes les images
            write_image(  # ecrit l'image avec la carte superposee
                save_maps / f"{Path(path).stem}_map.png",  # nom : <nom de l'image>_map.png
                # image en 8 bits (une image 16 bits est divisee par 257 pour tenir dans 0-255) + carte coloree
                overlay(image if image.dtype == np.uint8 else (image / 257).astype(np.uint8), heat),
            )
    table = pd.DataFrame(rows)  # tableau des resultats
    logger.info(  # bilan
        "%d images | %d signalees (seuil %.5f)",  # modele du message
        len(table),  # nombre d'images traitees
        int(table["anomaly"].sum()) if len(table) else 0,  # nombre d'images signalees (0 si le tableau est vide)
        bundle["image_threshold"],  # seuil utilise
    )
    return table  # le tableau


# --- etape 5 ------------------------------------------------------------------------------------


def export(config: AnomalyConfig, output: Path) -> ModelCard:  # etape 5 : export ONNX verifie + carte du modele ; renvoie la carte
    device = torch.device("cpu")  # l'export se fait sur CPU (resultat portable)
    model, bundle, pipeline = load_trained(output, device)  # modele entraine et ses infos
    val_df = load_split(config.split.file, "val").head(3)  # 3 images de validation servent d'exemples reels
    # ces 3 images lues, pretraitees et empilees en un lot (3, C, S, S)
    real = torch.stack([to_float_chw(pipeline(read_image(p, color=bundle["color"]))) for p in val_df["path"]])
    # export ONNX a partir de 2 images d'exemple ; deux sorties nommees anomaly_map et score
    onnx_path = export_onnx(model, real[:2], output / "model.onnx", ["anomaly_map", "score"], opset=config.export.opset)
    report = verify_onnx(model, onnx_path, real, config.export.atol, config.export.rtol)  # compare ONNX et PyTorch (tolerances)
    xml = to_openvino(onnx_path) if config.export.openvino else None  # conversion OpenVINO si demandee
    latency = measure_latency(model, onnx_path, real[:1], config.export.latency_runs, xml)  # temps de calcul pour une image
    # metriques du test si l'etape evaluate a ete faite, sinon dictionnaire vide
    metrics = json.loads((output / "test_metrics.json").read_text()) if (output / "test_metrics.json").exists() else {}
    height, width = bundle["input_hw"]  # taille d'entree attendue
    card = ModelCard(  # la carte du modele : tout ce qu'un programme doit savoir pour utiliser le fichier ONNX
        task="anomaly",  # type de tache
        model_name=config.anomalib.model if config.method == "anomalib" else config.method,  # nom du modele
        framework=framework_versions(),  # versions des bibliotheques utilisees
        input=InputSpec(  # description de l'entree
            read_color=bundle["color"],  # mode de lecture de l'image
            height=height,  # hauteur attendue
            width=width,  # largeur attendue
            channels=bundle["channels"],  # nombre de canaux
            color="GRAY" if bundle["channels"] == 1 else "RGB",  # ordre des couleurs attendu
        ),
        preprocess=bundle["preprocess"],  # pretraitement a refaire avant le modele
        outputs=[  # description des sorties
            OutputSpec(  # sortie 1 : la carte
                name="anomaly_map",  # nom de la sortie dans le fichier ONNX
                shape=["batch", 1, height, width],  # forme : (lot, 1 canal, hauteur, largeur)
                description="carte d'anomalie lissee (plus haut = plus anormal)",  # sens de la sortie
            ),
            # sortie 2 : le score
            OutputSpec(name="score", shape=["batch"], description="score image ; anomalie si score > thresholds.image"),
        ],
        thresholds={"image": bundle["image_threshold"], "pixel": bundle["pixel_threshold"]},  # seuils calibres
        postprocess={  # regles de decision a appliquer apres le modele
            "decision": "score > thresholds.image",  # regle de decision
            "calibration": f"quantile {bundle['image_quantile']} des normales de validation",  # comment le seuil a ete fixe
        },
        # garde seulement les metriques principales presentes
        metrics={k: metrics[k] for k in ("auroc_image", "auroc_pixel", "recall", "false_alarm_rate", "f1", "n_images") if k in metrics},
        latency_ms={k: v for k, v in latency.items() if k.endswith("_ms")},  # temps mesures (cles finissant par _ms)
        files={"onnx": "model.onnx", **({"openvino": "model.xml"} if xml else {})},  # fichiers produits
        notes=f"methode {config.method} ; normalisation ImageNet eventuelle incluse dans le modele ; verification ONNX : {report}",  # notes
    )
    card.input.dynamic_batch = batch_size_of(onnx_path) is None  # certains modeles figent le lot (anomalib PatchCore, DFM)
    write_card(card, output / "model_card.json")  # ecrit la carte en JSON
    return card  # la carte
