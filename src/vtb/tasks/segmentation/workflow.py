"""Les cinq etapes de la segmentation semantique, appelees par les scripts de la recette et par le notebook.

prepare(config)          catalogue image + masque, decoupage -> split.csv
train(config)            train + val -> model.pt (modele complet : normalisation et softmax inclus)
evaluate(config)         test, une seule fois -> IoU / Dice par classe, decision par image, galerie
predict(config, source)  images -> masques PNG (+ superpositions) + CSV des surfaces
export(config)           model.pt -> model.onnx verifie + model_card.json (+ OpenVINO)
"""

from __future__ import annotations  # permet les annotations de type modernes

import json  # lecture / ecriture de fichiers JSON (metriques, bilans)
import logging  # journal des messages
import time  # mesure de la duree de chaque epoque
from pathlib import Path  # Path : chemin de fichier independant du systeme

import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)
import pandas as pd  # pandas : tableaux de donnees
import torch  # PyTorch : tenseurs (tableaux de nombres multi-dimensions) et reseaux de neurones
from torch import nn  # nn : briques des reseaux (pertes, outils sur les gradients)

from vtb.core.runtime import count_parameters, timer  # compte les poids d'un modele ; chronometre utilisable avec "with"

# carte du modele (model card) : fichier JSON qui decrit l'entree, les sorties, les seuils et les metriques du modele exporte
from vtb.export.card import InputSpec, ModelCard, OutputSpec, framework_versions, write_card

# export ONNX (format standard de modele, lisible sans PyTorch), verification, mesure de vitesse, conversion OpenVINO (Intel)
from vtb.export.onnx import batch_size_of, export_onnx, measure_latency, to_openvino, verify_onnx
from vtb.io.catalogs import build_catalog  # construit le catalogue (une ligne par image) a partir du dossier de donnees
from vtb.io.images import list_images, read_image, write_image  # lister, lire, ecrire des images
from vtb.preprocess import Pipeline  # enchainement des etapes de pretraitement
from vtb.preprocess.tensor import to_float_chw  # image numpy (H, W, C) 0-255 -> tenseur (C, H, W) 0-1

# fuite (meme image dans deux parties) ; lecture, creation, ecriture du decoupage train / val / test
from vtb.split import check_no_leakage, load_split, make_split, save_split
from vtb.tasks.segmentation.config import SegmentationConfig  # la configuration de l'experience
from vtb.tasks.segmentation.data import SegmentationDataset, build_loader, class_names_for  # donnees de segmentation
from vtb.tasks.segmentation.metrics import confusion, image_decisions, scores_from_confusion  # metriques (IoU, Dice, par image)
from vtb.tasks.segmentation.models import Segmenter, build_segmenter, param_groups  # modele et groupes de parametres

logger = logging.getLogger(__name__)  # journal propre a ce module


# --- etape 1 -------------------------------------------------------------------------------------


def prepare(config: SegmentationConfig) -> pd.DataFrame:  # etape 1 : catalogue + decoupage ; renvoie le decoupage
    """Catalogue (layout image_mask ou annotations au format masks) puis decoupage stratifie par etiquette d'image."""
    # liste des images avec leur masque ; dates de prise de vue lues seulement pour un decoupage temporel
    catalog = build_catalog(config.data, with_timestamps=config.split.strategy == "temporal")
    if "mask_path" not in catalog:  # si le catalogue n'a pas de colonne de masques
        raise ValueError("la segmentation a besoin d'un masque par image : layout image_mask (ou mvtec)")  # on arrete avec un message clair
    split = make_split(catalog, config.split, config.runtime.seed)  # decoupage train / val / test (stratifie : memes proportions)
    check_no_leakage(split)  # verifie qu'aucune image n'est dans deux parties
    save_split(split, config.split.file)  # enregistre le decoupage pour les etapes suivantes
    return split  # le tableau du decoupage


# noms des classes de pixels (fond en premier) selon la config ; renvoie la liste
def _class_names(config: SegmentationConfig, frame: pd.DataFrame) -> list[str]:
    return class_names_for(frame, config.mask_mode, config.class_names, config.normal_labels)  # delegue a la fonction de data.py


# cree les Datasets train / val / test (augmentation sur train seulement) ; renvoie un dictionnaire nom -> Dataset
def _datasets(config: SegmentationConfig, pipeline: Pipeline, class_names: list[str]) -> dict[str, SegmentationDataset]:
    from vtb.augment.transforms import build_augmentation  # construit les transformations aleatoires (import ici : chargement lent)

    augment = build_augmentation(config.augment, task="segmentation") if config.augment else None  # augmentation, ou rien
    out = {}  # dictionnaire resultat
    for subset in ("train", "val", "test"):  # pour chaque partie du decoupage
        frame = load_split(config.split.file, subset)  # lignes de cette partie
        out[subset] = SegmentationDataset(  # Dataset de cette partie
            frame,  # les lignes
            pipeline,  # le pretraitement
            config.data.color,  # mode de lecture (rgb / gray)
            config.mask_mode,  # facon de lire les masques
            class_names,  # noms des classes
            augment if subset == "train" else None,  # augmentation seulement pour l'entrainement
        )
    return out  # les trois Datasets


# --- etape 2 -------------------------------------------------------------------------------------


def _criterion(config: SegmentationConfig, num_classes: int, device: torch.device):  # choisit la fonction de perte ; la renvoie
    """Perte : entropie croisee (pixel par pixel) + Dice (recouvrement global, insensible a la domination du fond)."""
    import segmentation_models_pytorch as smp  # smp fournit les pertes Dice, Focal, Lovasz

    # poids par classe pour l'entropie croisee (donner plus d'importance aux defauts rares), sur le meme processeur que le modele
    weights = torch.tensor(config.train.class_weights, dtype=torch.float32, device=device) if config.train.class_weights else None
    ce = nn.CrossEntropyLoss(weight=weights)  # entropie croisee : penalise une faible probabilite donnee a la bonne classe
    name = config.train.loss  # nom de la perte choisie
    if name == "ce":  # entropie croisee seule
        return ce  # on la renvoie
    other = {  # les autres pertes, par nom ; "multiclass" : une classe par pixel parmi K
        "dice": smp.losses.DiceLoss("multiclass"),  # 1 - Dice : mesure directement le recouvrement
        "dice_ce": smp.losses.DiceLoss("multiclass"),  # la partie Dice de dice_ce
        "focal": smp.losses.FocalLoss("multiclass"),  # entropie croisee qui insiste sur les pixels difficiles
        "lovasz": smp.losses.LovaszLoss("multiclass"),  # optimise une approximation de l'IoU
    }[name]  # on prend celle demandee
    if name == "dice_ce":  # combinaison des deux
        return lambda logits, target: ce(logits, target) + other(logits, target)  # perte = entropie croisee + Dice
    return other  # la perte choisie


# probabilites -> classe de chaque pixel ; renvoie un tenseur (N, H, W) d'indices de classe
def predict_classes(probs: torch.Tensor, threshold: float) -> torch.Tensor:
    """Probabilites (N, K, H, W) -> classes (N, H, W) ; binaire : seuil sur la probabilite de defaut, sinon argmax."""
    if probs.shape[1] == 2:  # cas binaire (fond / defaut)
        return (probs[:, 1] > threshold).long()  # defaut si sa probabilite depasse le seuil (reglable, pas forcement 0,5)
    return probs.argmax(dim=1)  # plusieurs classes : la plus probable (argmax = indice du maximum)


@torch.no_grad()  # pas de gradient : simple evaluation (plus rapide, moins de memoire)
# matrice de confusion cumulee sur tous les lots d'un loader ; renvoie un tableau (K, K)
def _confusion_on(model: Segmenter, loader, num_classes: int, threshold: float, device: torch.device) -> np.ndarray:
    model.eval()  # mode evaluation (BatchNorm et dropout figes)
    total = np.zeros((num_classes, num_classes), np.int64)  # matrice vide a remplir
    for x, y in loader:  # pour chaque lot (images x, masques verite y)
        pred = predict_classes(model(x.to(device)), threshold).cpu().numpy()  # classes predites, ramenees en numpy
        total += confusion(pred, y.numpy(), num_classes)  # ajoute les comptes de ce lot
    return total  # matrice cumulee


def train(config: SegmentationConfig, output: Path, device: torch.device) -> dict:  # etape 2 : entrainement ; renvoie un bilan
    pipeline = Pipeline(config.preprocess)  # pretraitement decrit dans la config
    frame = load_split(config.split.file)  # tout le decoupage (pour connaitre toutes les etiquettes)
    class_names = _class_names(config, frame)  # noms des classes de pixels
    data = _datasets(config, pipeline, class_names)  # Datasets train / val / test
    sample, _ = data["train"].load(0)  # une image d'exemple (pour connaitre sa taille et ses canaux)
    in_channels = config.model.in_channels or (1 if sample.ndim == 2 else sample.shape[2])  # canaux : config, sinon deduits de l'image
    model = build_segmenter(config.model, len(class_names), in_channels).to(device)  # cree le modele et l'envoie sur CPU / GPU
    logger.info(  # resume du modele
        "%s / %s | %d classes %s | %d canal/canaux | %s parametres",  # modele du message
        config.model.architecture,  # architecture
        config.model.encoder,  # encodeur
        len(class_names),  # nombre de classes
        class_names,  # leurs noms
        in_channels,  # canaux d'entree
        f"{count_parameters(model):,}",  # nombre de poids, avec separateur de milliers
    )
    tc, seed, workers = config.train, config.runtime.seed, config.runtime.num_workers  # raccourcis
    train_loader = build_loader(data["train"], tc.batch_size, True, workers, seed)  # lots d'entrainement melanges
    val_loader = build_loader(data["val"], tc.batch_size, False, workers, seed)  # lots de validation dans l'ordre
    # optimiseur AdamW (regle de mise a jour des poids) avec deux groupes : encodeur a pas reduits, decodeur a pas normaux
    optimizer = torch.optim.AdamW(param_groups(model, tc.lr, tc.encoder_lr_factor, tc.weight_decay))
    # le taux d'apprentissage descend en demi-cosinus jusqu'a la derniere epoque (grands pas au debut, petits a la fin)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, tc.epochs))
    criterion = _criterion(config, len(class_names), device)  # la fonction de perte
    # GradScaler : evite que de tres petits gradients s'annulent en precision 16 bits ; actif seulement avec amp sur GPU
    scaler = torch.amp.GradScaler(enabled=tc.amp and device.type == "cuda")
    history, best, best_epoch, bad = [], -1.0, 0, 0  # historique, meilleur critere, son epoque, epoques sans progres
    with timer("entrainement") as elapsed:  # chronometre tout l'entrainement
        for epoch in range(1, tc.epochs + 1):  # pour chaque epoque (passage complet sur les donnees)
            start, total, n = time.time(), 0.0, 0  # debut, somme des pertes, images vues
            model.train()  # mode entrainement (BatchNorm apprend, dropout actif)
            for x, y in train_loader:  # pour chaque lot d'images et de masques
                x, y = x.to(device), y.to(device)  # envoie le lot sur CPU / GPU
                optimizer.zero_grad(set_to_none=True)  # efface les gradients du lot precedent
                with torch.autocast(device.type, enabled=tc.amp and device.type == "cuda"):  # calcul en 16 bits si amp sur GPU
                    loss = criterion(model.logits(x), y)  # perte calculee sur les logits (scores avant softmax)
                scaler.scale(loss).backward()  # gradients (directions de correction des poids), mis a l'echelle si amp
                if tc.grad_clip:  # si un plafond de gradient est demande
                    scaler.unscale_(optimizer)  # remet les gradients a leur vraie echelle avant de les mesurer
                    nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)  # plafonne leur norme (evite les pas geants)
                scaler.step(optimizer)  # l'optimiseur corrige les poids
                scaler.update()  # ajuste l'echelle pour le lot suivant
                total, n = total + loss.item() * len(x), n + len(x)  # cumule la perte (ponderee par la taille du lot) et le compte
            scheduler.step()  # fait descendre le taux d'apprentissage d'un cran
            val = scores_from_confusion(  # scores de validation (IoU, Dice) a partir de la matrice de confusion
                _confusion_on(model, val_loader, len(class_names), config.evaluation.threshold, device),  # matrice sur la validation
                class_names,  # noms des classes
            )
            # critere de selection : mIoU des defauts (le fond gonflerait le score), sinon mIoU global
            criterion_value = val["miou_defects"] if val["miou_defects"] is not None else val["miou"]
            improved = criterion_value is not None and criterion_value > best + 1e-4  # progres d'au moins 0,0001
            if improved:  # si le modele s'est ameliore
                best, best_epoch, bad = criterion_value, epoch, 0  # retient le record, son epoque, remet le compteur a zero
                torch.save(model.state_dict(), output / "best_state.pt")  # sauve les poids du meilleur modele
            else:  # sinon (pas de progres)
                bad += 1  # une epoque de plus sans amelioration
            history.append(  # ajoute une ligne a l'historique
                # epoque, perte moyenne d'entrainement, mIoU de validation global et des defauts
                {"epoch": epoch, "train_loss": total / max(n, 1), "val_miou": val["miou"], "val_miou_defects": val["miou_defects"]}
            )
            logger.info(  # bilan de l'epoque
                "epoque %2d | perte %.4f | val mIoU %.3f | mIoU defauts %s | %.0fs%s",  # modele du message
                epoch,  # numero d'epoque
                total / max(n, 1),  # perte moyenne
                val["miou"] or 0.0,  # mIoU global (0 s'il manque)
                val["miou_defects"],  # mIoU des defauts
                time.time() - start,  # duree de l'epoque
                " | meilleur" if improved else "",  # mention si nouveau meilleur modele
            )
            if bad >= tc.patience:  # si trop d'epoques sans progres
                logger.info("arret anticipe : %d epoques sans amelioration", tc.patience)  # on previent
                break  # arret anticipe
    if not (output / "best_state.pt").exists():  # aucune epoque retenue (validation vide ou score jamais calcule)
        logger.warning("aucun meilleur modele retenu en validation : on garde les poids de la derniere epoque")  # on previent
        torch.save(model.state_dict(), output / "best_state.pt")  # sauve les poids actuels pour que la suite fonctionne
    # recharge les poids du meilleur modele (weights_only : chargement sur, que des tenseurs)
    model.load_state_dict(torch.load(output / "best_state.pt", map_location=device, weights_only=True))
    bundle = {  # tout ce qu'il faut pour reconstruire et utiliser le modele
        "model_state": model.state_dict(),  # les poids
        "model_config": config.model.model_dump(),  # les reglages du reseau (pour le reconstruire)
        "class_names": class_names,  # noms des classes
        "in_channels": in_channels,  # canaux d'entree
        "color": config.data.color,  # mode de lecture des images
        "preprocess": pipeline.describe(),  # description du pretraitement
        "input_hw": list(sample.shape[:2]),  # taille d'entree (hauteur, largeur)
        "mask_mode": config.mask_mode,  # facon de lire les masques
        "best_epoch": best_epoch,  # epoque du meilleur modele
    }
    torch.save(bundle, output / "model.pt")  # sauve le tout dans model.pt
    (output / "best_state.pt").unlink(missing_ok=True)  # supprime le fichier intermediaire (sans erreur s'il manque)
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)  # sauve l'historique en CSV
    # bilan : meilleure epoque, meilleur critere, duree, classes
    summary = {"best_epoch": best_epoch, "val_criterion": best, "train_seconds": round(elapsed["seconds"], 1), "class_names": class_names}
    (output / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")  # sauve le bilan en JSON
    _plot_history(pd.DataFrame(history), output / "history.png")  # trace les courbes
    return summary  # le bilan


def _plot_history(history: pd.DataFrame, path: Path) -> None:  # courbes perte et IoU par epoque ; enregistre la figure dans path
    import matplotlib.pyplot as plt  # pyplot : interface de dessin de matplotlib (import ici : lent a charger)

    if history.empty:  # si aucune epoque n'a ete faite
        return  # rien a tracer
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))  # deux graphiques cote a cote
    axes[0].plot(history["epoch"], history["train_loss"])  # gauche : perte d'entrainement par epoque
    axes[0].set_title("perte (entrainement)")  # titre du graphique de gauche
    for c in ("val_miou", "val_miou_defects"):  # pour les deux mIoU de validation
        axes[1].plot(history["epoch"], history[c].astype(float), label=c)  # droite : courbe de ce mIoU (None -> nan, trou)
    axes[1].set_title("IoU (validation)")  # titre du graphique de droite
    axes[1].legend(fontsize=7)  # legende (noms des deux courbes)
    for ax in axes:  # pour chaque graphique
        ax.set_xlabel("epoque")  # titre de l'axe horizontal
        ax.grid(alpha=0.3)  # quadrillage leger
    fig.tight_layout()  # ajuste les marges
    fig.savefig(path, dpi=110)  # enregistre l'image
    plt.close(fig)  # libere la memoire


# recharge le modele sauve par train ; renvoie (modele, contenu du fichier, pretraitement)
def load_trained(output: Path, device: torch.device) -> tuple[Segmenter, dict, Pipeline]:
    from vtb.tasks.segmentation.config import SegModelConfig  # reglages du reseau (pour le reconstruire)

    # lit model.pt en lecture sure (weights_only=True) : le paquet ne contient que des tenseurs, nombres, textes, listes et dicts
    bundle = torch.load(output / "model.pt", map_location=device, weights_only=True)
    cfg = SegModelConfig.model_validate(bundle["model_config"])  # dictionnaire -> objet de config verifie
    # meme architecture, sans telecharger de poids pre-entraines (ils vont etre remplaces)
    model = build_segmenter(cfg, len(bundle["class_names"]), bundle["in_channels"], load_weights=False)
    model.load_state_dict(bundle["model_state"])  # charge les poids entraines
    return model.to(device).eval(), bundle, Pipeline(bundle["preprocess"])  # modele pret, infos, pretraitement reconstruit


# --- etape 3 -------------------------------------------------------------------------------------


def colorize(classes: np.ndarray, class_names: list[str]) -> np.ndarray:  # masque d'indices -> image couleur ; renvoie (H, W, 3)
    """Indices de classe -> image RGB (fond noir, une couleur stable par classe)."""
    from vtb.annotate.draw import color_of  # couleur attribuee a une classe (toujours la meme pour un meme nom)

    out = np.zeros((*classes.shape, 3), np.uint8)  # image noire de meme taille, 3 canaux
    for k, name in enumerate(class_names[1:], start=1):  # pour chaque classe sauf le fond (k = indice a partir de 1)
        out[classes == k] = color_of(name, class_names[1:])  # colore les pixels de cette classe
    return out  # l'image coloree


# superpose le masque colore sur l'image (transparence alpha) ; renvoie l'image melangee (H, W, 3) en 8 bits
def overlay(image: np.ndarray, classes: np.ndarray, class_names: list[str], alpha: float) -> np.ndarray:
    base = image if image.ndim == 3 else np.stack([image] * 3, -1)  # image grise -> 3 canaux identiques
    base = base if base.dtype == np.uint8 else (base / 257).astype(np.uint8)  # image 16 bits -> 8 bits (65535 / 257 = 255)
    colors = colorize(classes, class_names)  # masque en couleurs
    mask = classes > 0  # pixels qui ne sont pas du fond
    blended = base.copy()  # copie de l'image (on ne modifie pas l'originale)
    blended[mask] = (base[mask] * (1 - alpha) + colors[mask] * alpha).astype(np.uint8)  # melange image / couleur sur ces pixels
    return blended  # l'image superposee


# etape 3 : mesures sur le test ; renvoie les metriques
def evaluate(config: SegmentationConfig, output: Path, device: torch.device) -> dict:
    model, bundle, pipeline = load_trained(output, device)  # modele entraine et ses infos
    class_names = bundle["class_names"]  # noms des classes
    test = load_split(config.split.file, "test")  # lignes du test
    dataset = SegmentationDataset(test, pipeline, bundle["color"], bundle["mask_mode"], class_names)  # Dataset du test, sans augmentation
    total = np.zeros((len(class_names),) * 2, np.int64)  # matrice de confusion vide (K, K)
    areas, has_defect, gallery = [], [], []  # surfaces predites, presence reelle de defaut, exemples pour la figure
    with timer("prediction du test") as elapsed, torch.no_grad():  # chronometre, et pas de gradient
        for i in range(len(dataset)):  # pour chaque image du test, une par une
            image, mask = dataset.load(i)  # image pretraitee et masque verite
            # [None] : lot d'une image ; probabilites -> classes ; [0] : la seule image du lot ; retour en numpy
            pred = predict_classes(model(to_float_chw(image)[None].to(device)), config.evaluation.threshold)[0].cpu().numpy()
            total += confusion(pred, mask, len(class_names))  # ajoute les comptes de cette image
            areas.append(int((pred > 0).sum()))  # surface predite de defaut (pixels non fond)
            has_defect.append(bool(mask.any()))  # vrai si le masque verite contient un defaut
            if len(gallery) < 8 and (mask.any() or len(gallery) < 2):  # au plus 8 exemples : les 2 premiers, puis des images a defaut
                gallery.append((image, mask, pred, Path(dataset.paths[i]).name))  # garde l'exemple pour la figure
    metrics = scores_from_confusion(total, class_names)  # IoU, Dice par classe et moyennes
    metrics.update(image_decisions(areas, has_defect, config.evaluation.min_defect_area))  # ajoute rappel et fausses alarmes par image
    # nombre d'images et temps moyen par image en millisecondes
    metrics.update({"n_images": len(dataset), "ms_per_image": round(1000 * elapsed["seconds"] / max(1, len(dataset)), 1)})
    (output / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")  # sauve les metriques
    _gallery(gallery, class_names, config.evaluation.overlay_alpha, output / "test_predictions.png")  # figure des exemples
    logger.info(  # bilan du test
        "test (%d images) : mIoU defauts %s | Dice defauts %s | rappel image %s | fausses alarmes image %s",  # modele du message
        len(dataset),  # nombre d'images
        metrics["miou_defects"],  # mIoU des defauts
        metrics["mdice_defects"],  # Dice des defauts
        metrics["image_recall"],  # rappel image
        metrics["image_false_alarm_rate"],  # fausses alarmes image
    )
    return metrics  # les metriques


# figure : une ligne par exemple (image, verite superposee, prediction superposee) ; enregistre dans path
def _gallery(items: list, class_names: list[str], alpha: float, path: Path) -> None:
    import matplotlib.pyplot as plt  # interface de dessin de matplotlib

    if not items:  # si aucun exemple
        return  # rien a dessiner
    # une ligne par exemple, 3 colonnes ; squeeze=False garde un tableau 2D d'axes meme avec une seule ligne
    fig, axes = plt.subplots(len(items), 3, figsize=(9, 2.8 * len(items)), squeeze=False)
    for row, (image, mask, pred, name) in enumerate(items):  # pour chaque exemple (row = numero de ligne)
        shown = image if image.ndim == 3 else np.stack([image] * 3, -1)  # image grise -> 3 canaux pour l'affichage
        for col, (picture, title) in enumerate(  # pour chacune des 3 vignettes (col = numero de colonne)
            (  # les 3 couples (image a montrer, titre)
                (shown, name),  # l'image seule, titree par son nom de fichier
                (overlay(image, mask, class_names, alpha), "verite terrain"),  # la verite superposee
                (overlay(image, pred, class_names, alpha), "prediction"),  # la prediction superposee
            )
        ):
            axes[row, col].imshow(picture if picture.dtype == np.uint8 else (picture / 257).astype(np.uint8))  # affiche en 8 bits
            axes[row, col].set_title(title, fontsize=7)  # titre de la vignette
            axes[row, col].axis("off")  # cache les axes gradues
    fig.tight_layout()  # ajuste les marges
    fig.savefig(path, dpi=100)  # enregistre la figure
    plt.close(fig)  # libere la memoire


# --- etape 4 -------------------------------------------------------------------------------------


# etape 4 : segmente des images (un fichier ou un dossier) ; renvoie un tableau (une ligne par image : surfaces, decision)
def predict(config: SegmentationConfig, output: Path, device: torch.device, source: Path, save_masks: Path | None = None) -> pd.DataFrame:
    """Surface de chaque classe par image (pixels a la taille du modele) et decision ; masques et superpositions en option."""
    model, bundle, pipeline = load_trained(output, device)  # modele entraine et ses infos
    class_names = bundle["class_names"]  # noms des classes
    # un fichier, ou les images du dossier
    paths = [Path(source)] if Path(source).is_file() else list_images(source, config.data.extensions)
    rows = []  # une ligne de resultat par image
    with torch.no_grad():  # pas de gradient : simple prediction
        for path in paths:  # pour chaque image
            image = pipeline(read_image(path, color=bundle["color"]))  # lecture et pretraitement identiques a l'entrainement
            # classe de chaque pixel, en numpy (H, W)
            pred = predict_classes(model(to_float_chw(image)[None].to(device)), config.evaluation.threshold)[0].cpu().numpy()
            area = int((pred > 0).sum())  # surface de defaut en pixels (tout ce qui n'est pas fond)
            # chemin, surface, part de l'image
            row = {"path": str(path), "defect_pixels": area, "defect_fraction": round(area / pred.size, 5)}
            # surface de chaque classe de defaut
            row.update({f"pixels_{n}": int((pred == k).sum()) for k, n in enumerate(class_names) if k > 0})
            row["flagged"] = area >= config.evaluation.min_defect_area  # image signalee si la surface atteint le minimum
            rows.append(row)  # ajoute la ligne
            if save_masks is not None:  # si on veut enregistrer les masques
                write_image(Path(save_masks) / f"{path.stem}_mask.png", pred.astype(np.uint8))  # indices de classe
                write_image(  # et la superposition coloree
                    Path(save_masks) / f"{path.stem}_overlay.png",  # nom du fichier : <nom de l'image>_overlay.png
                    overlay(image, pred, class_names, config.evaluation.overlay_alpha),  # image + masque predit en couleur
                )
    table = pd.DataFrame(rows)  # tableau des resultats
    logger.info(  # bilan
        "%d images | %d signalees (>= %d pixels de defaut)",  # modele du message
        len(table),  # nombre d'images
        int(table["flagged"].sum()) if len(table) else 0,  # nombre d'images signalees (0 si tableau vide)
        config.evaluation.min_defect_area,  # surface minimale
    )
    return table  # le tableau


# --- etape 5 -------------------------------------------------------------------------------------


def export(config: SegmentationConfig, output: Path) -> ModelCard:  # etape 5 : export ONNX verifie + carte du modele ; renvoie la carte
    cpu = torch.device("cpu")  # l'export se fait sur CPU (resultat portable)
    model, bundle, pipeline = load_trained(output, cpu)  # modele entraine et ses infos
    val = load_split(config.split.file, "val").head(3)  # 3 images de validation servent d'exemples reels
    # ces 3 images lues, pretraitees, empilees en un lot (3, C, H, W)
    real = torch.stack([to_float_chw(pipeline(read_image(p, color=bundle["color"]))) for p in val["path"]])
    # export ONNX a partir de 2 images d'exemple ; une sortie nommee "probabilities"
    onnx_path = export_onnx(model, real[:2], output / "model.onnx", ["probabilities"], opset=config.export.opset)
    report = verify_onnx(model, onnx_path, real, config.export.atol, config.export.rtol)  # compare ONNX et PyTorch (tolerances)
    xml = to_openvino(onnx_path) if config.export.openvino else None  # conversion OpenVINO si demandee
    latency = measure_latency(model, onnx_path, real[:1], config.export.latency_runs, xml)  # temps de calcul pour une image
    # metriques du test si l'etape evaluate a ete faite, sinon dictionnaire vide
    metrics = json.loads((output / "test_metrics.json").read_text()) if (output / "test_metrics.json").exists() else {}
    height, width = bundle["input_hw"]  # taille d'entree attendue
    names = bundle["class_names"]  # noms des classes
    card = ModelCard(  # la carte du modele : tout ce qu'un programme doit savoir pour utiliser le fichier ONNX
        task="segmentation",  # type de tache
        model_name=f"{bundle['model_config']['architecture']}_{bundle['model_config']['encoder']}",  # ex. unet_resnet18
        framework=framework_versions(),  # versions des bibliotheques utilisees
        input=InputSpec(  # description de l'entree
            read_color=bundle["color"],  # mode de lecture de l'image
            height=height,  # hauteur attendue
            width=width,  # largeur attendue
            channels=bundle["in_channels"],  # nombre de canaux
            color="GRAY" if bundle["in_channels"] == 1 else "RGB",  # ordre des couleurs attendu
        ),
        preprocess=bundle["preprocess"],  # pretraitement a refaire avant le modele
        outputs=[  # description des sorties
            OutputSpec(  # une seule sortie
                name="probabilities",  # son nom dans le fichier ONNX
                shape=["batch", len(names), height, width],  # forme : (lot, classes, hauteur, largeur)
                description="probabilite de chaque classe par pixel (softmax)",  # son sens
            )
        ],
        class_names=names,  # noms des classes, dans l'ordre des canaux de sortie
        # seuils : probabilite de defaut et surface minimale pour signaler une image
        thresholds={"defect_probability": config.evaluation.threshold, "min_defect_area": float(config.evaluation.min_defect_area)},
        postprocess={  # ce que le programme doit faire apres le modele
            # classe par pixel
            "classes": "2 classes : pixel defaut si probabilities[1] > defect_probability ; sinon argmax sur les classes",
            "decision": "image signalee si le nombre de pixels de defaut >= min_defect_area",  # decision par image
            "normalization": "incluse dans le modele (entree dans [0, 1])",  # rien a normaliser cote programme
        },
        metrics={  # metriques principales du test
            k: metrics[k]  # valeur de la metrique k
            for k in ("miou", "miou_defects", "mdice_defects", "image_recall", "image_false_alarm_rate", "n_images")  # parmi celles-ci
            if k in metrics  # seulement si elle existe
        },
        latency_ms={k: v for k, v in latency.items() if k.endswith("_ms")},  # temps mesures (cles finissant par _ms)
        files={"onnx": "model.onnx", **({"openvino": "model.xml"} if xml else {})},  # fichiers produits
        notes=f"verification ONNX : {report}",  # resultat de la verification ONNX
    )
    # False si le lot est fige dans le graphe (l'export s'est alors replie sur un lot de 1)
    card.input.dynamic_batch = batch_size_of(onnx_path) is None
    write_card(card, output / "model_card.json")  # ecrit la carte en JSON
    return card  # la carte
