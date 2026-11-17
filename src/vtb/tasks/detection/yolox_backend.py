"""Moteur YOLOX (Megvii, Apache-2.0) : les briques officielles, une boucle ecrite ici qui tourne sur CPU comme sur GPU.

Le pretraitement de YOLOX differe d'Ultralytics sur trois points, et lui donner une image
preparee "a la Ultralytics" ne leve aucune erreur : les detections sont juste mauvaises.
  1. image redimensionnee sans deformer puis collee EN HAUT A GAUCHE d'un fond gris 114 ;
  2. canaux en BGR ;
  3. valeurs laissees dans [0, 255] (pas de division).
On appelle la fonction officielle `yolox.data.data_augment.preproc` : aucune divergence possible.

Reperes de coordonnees (source classique de bugs) :
  image d'origine --pretraitement OpenCV (resize eventuel)--> image "pipeline"
  image pipeline --preproc (ratio r)--> entree du reseau (S x S)
Les boites du reseau divisees par r sont dans l'image pipeline ; multipliees par
(largeur d'origine / largeur pipeline) elles reviennent dans l'image d'origine.

Installation : `pip install --no-build-isolation --no-deps yolox==0.3.0` puis l'extra yolox.
"""

from __future__ import annotations  # les annotations de type restent du texte : syntaxe moderne "X | None" partout

import logging  # journal des messages (info, avertissements) au lieu de print
import math  # fonctions mathematiques (cos, pi) pour le calendrier du taux d'apprentissage
import time  # mesure de la duree de chaque epoque
from functools import partial  # partial : fige certains arguments d'une fonction (donne une fonction a un seul argument)
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import cv2  # OpenCV : conversion de couleurs et traitement d'images
import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)
import pandas as pd  # pandas : tableaux de donnees ; ici l'historique epoque par epoque
import torch  # PyTorch : tenseurs (tableaux de nombres, sur CPU ou GPU) et reseaux de neurones
from torch import nn  # nn : briques de reseau (couches, BatchNorm, activations...)
from torch.utils.data import DataLoader, Dataset  # Dataset : donne une image par indice ; DataLoader : les regroupe en lots

from vtb.annotate.schema import ImageAnnotation  # format pivot : toutes les etiquettes d'une image
from vtb.evaluate.detection import coco_metrics  # evaluateur COCO commun : calcule le mAP (precision moyenne)
from vtb.io.images import read_image  # lit une image sur disque en tableau numpy
from vtb.preprocess import Pipeline  # suite d'operations OpenCV appliquee a chaque image
from vtb.tasks.detection.common import boxes_from_array, mean_best_iou, nms_per_class  # outils communs de detection
from vtb.tasks.detection.config import YoloxConfig  # section yolox du YAML

logger = logging.getLogger(__name__)  # journal propre a ce module

VARIANTS = {  # profondeur, largeur, convolutions separables (comme yolox.exp)
    "nano": (0.33, 0.25, True),  # le plus petit : convolutions separables (moins de calculs)
    "tiny": (0.33, 0.375, False),  # petit, bon compromis sur CPU
    "s": (0.33, 0.50, False),  # small
    "m": (0.67, 0.75, False),  # medium : deux fois plus de couches
    "l": (1.0, 1.0, False),  # large : le reseau de reference, le plus lent
}
# adresse des poids officiels pre-entraines sur COCO ; {variant} est remplace par nano, tiny...
WEIGHTS_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_{variant}.pth"


def build_model(variant: str, num_classes: int) -> nn.Module:  # construit un YOLOX vide ; renvoie le reseau
    """Reproduit yolox.exp.Exp.get_model : backbone CSPDarknet + PAFPN + tete decouplee sans ancres."""
    from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead  # briques officielles (import tardif : seulement pour ce moteur)

    depth, width, depthwise = VARIANTS[variant]  # multiplicateurs de taille de la variante choisie
    channels = [256, 512, 1024]  # nombres de canaux des trois niveaux de la pyramide (avant multiplication par width)
    # backbone (extrait les caracteristiques) + PAFPN (melange les echelles fine, moyenne, grossiere)
    backbone = YOLOPAFPN(depth, width, in_channels=channels, depthwise=depthwise)
    # assemble le detecteur : backbone + tete qui predit boites, "objectness" et classes
    model = YOLOX(backbone, YOLOXHead(num_classes, width, in_channels=channels, depthwise=depthwise))
    for module in model.modules():  # pour chaque couche du reseau
        if isinstance(module, nn.BatchNorm2d):  # si c'est une BatchNorm (normalise les activations par lot)
            module.eps, module.momentum = 1e-3, 0.03  # reglages officiels
    model.head.initialize_biases(1e-2)  # probabilite initiale ~1 % : presque aucune cellule ne contient d'objet
    return model  # renvoie le reseau pret a recevoir des poids


def load_pretrained(model: nn.Module, path: Path, variant: str) -> None:  # charge les poids COCO dans le modele (rien a renvoyer)
    """Poids COCO officiels ; la tete de classification (80 classes) est ignoree car sa forme ne correspond plus."""
    from yolox.utils import load_ckpt  # charge des poids en ignorant les couches de forme differente

    if not path.exists():  # si le fichier de poids n'est pas encore sur le disque
        path.parent.mkdir(parents=True, exist_ok=True)  # cree le dossier weights/ si besoin
        logger.info("telechargement des poids YOLOX-%s (GitHub) vers %s", variant, path)  # previent du telechargement
        torch.hub.download_url_to_file(WEIGHTS_URL.format(variant=variant), str(path))  # telecharge le fichier
    load_ckpt(model, torch.load(path, map_location="cpu", weights_only=False)["model"])  # lit le fichier et copie les poids
    logger.info("poids YOLOX-%s charges depuis %s (tete de classification reinitialisee)", variant, path)  # confirme


def _three_channels(image: np.ndarray) -> np.ndarray:  # renvoie l'image avec 3 canaux
    return image if image.ndim == 3 else np.stack([image] * 3, axis=-1)  # gris (H, W) -> (H, W, 3) en repetant le canal


def preprocess(image_rgb: np.ndarray, size: int) -> tuple[np.ndarray, float]:  # pretraitement YOLOX ; renvoie (tenseur, ratio)
    """RGB (ou gris) -> BGR -> preproc officiel : (3, S, S) float32 BGR [0, 255] et ratio r."""
    from yolox.data.data_augment import preproc  # la fonction officielle : aucune divergence possible

    bgr = cv2.cvtColor(_three_channels(image_rgb), cv2.COLOR_RGB2BGR)  # YOLOX attend du BGR (convention OpenCV)
    tensor, ratio = preproc(bgr, (size, size))  # redimension sans deformer, collee en haut a gauche, fond 114, (C, H, W)
    return np.ascontiguousarray(tensor, dtype=np.float32), float(ratio)  # tableau float32 contigu en memoire, et le ratio


class YoloxDataset(Dataset):  # jeu de donnees PyTorch pour YOLOX : image prete + cibles + echelles
    """(image (3, S, S) BGR [0, 255], cibles (max_labels, 5) [classe, cx, cy, w, h], echelle [r, sx, sy]).

    sx, sy ramenent l'image pipeline a l'image d'origine (1 sans redimensionnement).
    """

    def __init__(  # constructeur : garde les reglages et prepare les boites, sans lire les images
        self,  # l'objet en cours de creation
        annotations: list[ImageAnnotation],  # une annotation par image (chemin + boites)
        class_names: list[str],  # noms des classes
        pipeline: Pipeline,  # pretraitement OpenCV
        color: str,  # "rgb" ou "gray"
        size: int,  # cote de l'image carree en entree du reseau
        max_labels: int,  # nombre maximal de boites par image (les cibles ont une taille fixe)
        train_transform=None,  # augmentation YOLOX (train seulement) ; None pour val / test
    ) -> None:  # un constructeur ne renvoie rien
        index = {n: i for i, n in enumerate(class_names)}  # nom de classe -> numero
        # garde les reglages dans l'objet
        self.annotations, self.pipeline, self.color, self.size, self.max_labels = annotations, pipeline, color, size, max_labels
        self.train_transform = train_transform  # garde l'augmentation
        self.targets = [  # boites de chaque image, preparees une fois pour toutes
            # pour chaque image : tableau (n, 5) [x1, y1, x2, y2, classe] ; reshape garde 5 colonnes meme sans boite
            np.array([[b.x1, b.y1, b.x2, b.y2, index[b.label]] for b in a.boxes], np.float32).reshape(-1, 5)
            for a in annotations  # pour chaque image annotee
        ]

    def __len__(self) -> int:  # nombre d'images du jeu
        return len(self.annotations)  # une annotation par image

    def image(self, i: int) -> np.ndarray:  # lit et pretraite l'image numero i ; renvoie un tableau a 3 canaux
        return _three_channels(self.pipeline(read_image(self.annotations[i].image_path, color=self.color)))  # lit, pretraite

    def __getitem__(self, i: int):  # appele par le DataLoader : renvoie (tenseur image, cibles, echelles)
        ann, image = self.annotations[i], self.image(i)  # l'annotation et l'image pretraitee
        sx, sy = ann.width / image.shape[1], ann.height / image.shape[0]  # facteurs image d'origine / image pipeline
        targets = self.targets[i].copy()  # copie : on ne modifie pas les boites gardees en memoire
        targets[:, [0, 2]] /= sx  # boites annotees sur l'origine -> image pipeline
        targets[:, [1, 3]] /= sy  # meme chose pour les y
        if self.train_transform is not None:  # si on est sur le train (augmentation prevue)
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # YOLOX travaille en BGR
            # augmentation officielle : flip, couleurs, puis preproc ; labels = (max_labels, 5) [classe, cx, cy, w, h]
            tensor, labels = self.train_transform(bgr, targets, (self.size, self.size))
            ratio = min(self.size / image.shape[0], self.size / image.shape[1])  # meme ratio que preproc
        else:  # val / test : pas d'augmentation
            tensor, ratio = preprocess(image, self.size)  # pretraitement simple
            labels = np.zeros((self.max_labels, 5), np.float32)  # cibles vides (non utilisees en evaluation)
        scale = torch.tensor([ratio, sx, sy], dtype=torch.float32)  # tout ce qu'il faut pour revenir a l'image d'origine
        # tableaux numpy -> tenseurs PyTorch
        return torch.from_numpy(np.ascontiguousarray(tensor, dtype=np.float32)), torch.from_numpy(labels), scale


def _optimizer(model: nn.Module, cfg: YoloxConfig) -> torch.optim.Optimizer:  # cree l'optimiseur YOLOX ; le renvoie
    """SGD Nesterov, trois groupes : poids de BatchNorm et biais sans decroissance (les contraindre annulerait la normalisation)."""
    bn, weights, biases = [], [], []  # trois groupes de parametres
    for module in model.modules():  # pour chaque couche
        if hasattr(module, "bias") and isinstance(module.bias, nn.Parameter):  # si la couche a un biais entrainable
            biases.append(module.bias)  # groupe des biais
        if isinstance(module, nn.BatchNorm2d):  # si c'est une BatchNorm
            bn.append(module.weight)  # son poids va dans le groupe sans decroissance
        elif hasattr(module, "weight") and isinstance(module.weight, nn.Parameter):  # sinon, si elle a des poids entrainables
            weights.append(module.weight)  # groupe des poids avec decroissance
    # SGD Nesterov (descente de gradient avec elan) ; taux = taux par image x taille du lot ; premier groupe : BatchNorm
    optimizer = torch.optim.SGD(bn, lr=cfg.basic_lr_per_img * cfg.batch_size, momentum=0.9, nesterov=True)
    optimizer.add_param_group({"params": weights, "weight_decay": cfg.weight_decay})  # poids : avec penalite weight decay
    optimizer.add_param_group({"params": biases})  # biais : sans penalite
    return optimizer  # renvoie l'optimiseur


# multiplicateur du taux d'apprentissage a une iteration donnee ; renvoie un nombre entre ~0 et 1
def _warmup_cosine(iteration: int, total: int, warmup: int, min_ratio: float) -> float:
    """Montee quadratique (la tete neuve donne de grands gradients) puis cosinus : le scheduler yoloxwarmcos officiel."""
    if warmup > 0 and iteration < warmup:  # pendant la phase de montee
        return max((iteration / warmup) ** 2, 1e-3)  # monte comme un carre, jamais en dessous de 0.001
    progress = (iteration - warmup) / max(1, total - warmup)  # avancement apres la montee, de 0 a 1
    return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * min(progress, 1.0)))  # cosinus de 1 a min_ratio


def _rows(result: torch.Tensor | None) -> np.ndarray:  # sortie YOLOX d'une image -> tableau (n, 6)
    """Une sortie de yolox.utils.postprocess (x1 y1 x2 y2 objet proba classe) -> (n, 6) [x1 y1 x2 y2 score classe]."""
    if result is None:  # None = aucune detection pour cette image
        return np.zeros((0, 6), np.float32)  # tableau vide a 6 colonnes
    r = result.detach().cpu().numpy()  # tenseur -> numpy (detache du calcul de gradient, ramene sur CPU)
    # coins, score = objectness (proba qu'il y ait un objet) x proba de la classe, numero de classe
    return np.column_stack([r[:, :4], r[:, 4] * r[:, 5], r[:, 6]]).astype(np.float32)


@torch.no_grad()  # pas de gradient : simple inference
# detecte sur toutes les images d'un loader ; renvoie une liste de tableaux (n, 6), un par image
def detect(model: nn.Module, loader: DataLoader, num_classes: int, conf: float, iou: float, device: torch.device) -> list[np.ndarray]:
    """Detections (n, 6) dans l'image d'ORIGINE pour toutes les images d'un loader non melange."""
    from yolox.utils import postprocess  # decodage officiel + filtrage par confiance + NMS

    model.eval()  # mode evaluation : BatchNorm figee, le reseau decode ses sorties
    out = []  # resultats, un par image, dans l'ordre du loader
    for images, _, scales in loader:  # pour chaque lot : images, cibles (ignorees), echelles
        results = postprocess(model(images.to(device)), num_classes, conf, iou, class_agnostic=False)  # NMS par classe
        for result, (ratio, sx, sy) in zip(results, scales.tolist()):  # pour chaque image du lot et ses echelles
            rows = _rows(result)  # detections en tableau (n, 6)
            rows[:, [0, 2]] *= sx / ratio  # x : entree du reseau -> image pipeline (/ratio) -> image d'origine (x sx)
            rows[:, [1, 3]] *= sy / ratio  # y : meme conversion
            out.append(rows)  # garde les detections de l'image
    return out  # renvoie toutes les detections


@torch.no_grad()  # pas de gradient : simple inference
def predict_images(  # predit sur des images en memoire ; renvoie un tableau (n, 6) par image
    model: nn.Module,  # le reseau YOLOX
    images: list[np.ndarray],  # images pretraitees (RGB ou gris)
    size: int,  # cote de l'entree du reseau
    num_classes: int,  # nombre de classes
    conf: float,  # confiance minimale
    iou: float,  # seuil de la NMS
    device: torch.device,  # CPU ou GPU
    batch_size: int = 16,  # images traitees ensemble
) -> list[np.ndarray]:  # une liste de tableaux (n, 6)
    """Images pipeline (RGB ou gris) -> (n, 6) dans le repere de ces images (production, tuiles)."""
    from yolox.utils import postprocess  # decodage officiel + NMS

    model.eval()  # mode evaluation
    out = []  # resultats, un par image
    for start in range(0, len(images), batch_size):  # parcourt les images par paquets
        prepared = [preprocess(im, size) for im in images[start : start + batch_size]]  # (tenseur, ratio) de chaque image
        batch = torch.from_numpy(np.stack([t for t, _ in prepared])).to(device)  # empile en un lot (N, 3, S, S)
        # pour chaque image : ses detections apres postprocess, et son ratio
        for result, (_, ratio) in zip(postprocess(model(batch), num_classes, conf, iou, class_agnostic=False), prepared):
            rows = _rows(result)  # tableau (n, 6)
            rows[:, :4] /= ratio  # entree du reseau -> image pipeline (pas de bordure : image en haut a gauche)
            out.append(rows)  # garde les detections
    return out  # renvoie toutes les detections


def train(  # boucle d'entrainement YOLOX ; renvoie l'historique (une ligne par epoque)
    train_set: YoloxDataset,  # images du train (avec augmentation)
    val_set: YoloxDataset,  # images de validation
    cfg: YoloxConfig,  # reglages yolox du YAML
    class_names: list[str],  # noms des classes
    conf_eval: float,  # seuil de confiance tres bas pour mesurer le mAP
    iou: float,  # seuil de la NMS
    device: torch.device,  # CPU ou GPU
    output: Path,  # dossier ou sauver best.pt
    seed: int,  # graine de melange
    num_workers: int,  # processus paralleles de lecture
) -> pd.DataFrame:  # renvoie l'historique
    """Boucle YOLOX : perte SimOTA calculee dans le modele, EMA des poids, mAP de validation a chaque epoque, arret anticipe."""
    from yolox.utils import ModelEMA  # EMA : moyenne glissante des poids (modele plus stable)

    model = build_model(cfg.variant, len(class_names))  # reseau de la variante choisie, tete au bon nombre de classes
    if cfg.weights:  # si des poids pre-entraines sont indiques
        load_pretrained(model, Path(cfg.weights), cfg.variant)  # on part des poids COCO
    model.to(device)  # deplace le modele sur CPU ou GPU
    loader = DataLoader(  # distributeur de lots du train
        train_set,  # le jeu d'entrainement
        batch_size=cfg.batch_size,  # images par lot
        shuffle=True,  # melange a chaque epoque
        num_workers=num_workers,  # processus de lecture
        drop_last=len(train_set) > cfg.batch_size,  # jette le dernier lot incomplet (s'il reste au moins un lot complet)
        generator=torch.Generator().manual_seed(seed),  # melange reproductible
    )
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=num_workers)  # val : ordre fixe
    optimizer = _optimizer(model, cfg)  # SGD avec trois groupes de parametres
    total = cfg.epochs * len(loader)  # nombre total d'iterations (pas d'optimisation)
    warmup = round(cfg.warmup_epochs * len(loader))  # nombre d'iterations de montee du taux
    schedule = partial(_warmup_cosine, total=total, warmup=warmup, min_ratio=cfg.min_lr_ratio)  # fonction iteration -> facteur
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)  # applique ce facteur au taux de depart
    ema = ModelEMA(model, 0.9998) if cfg.ema else None  # copie moyennee des poids (0.9998 = moyenne tres lente)
    history, best, bad = [], -1.0, 0  # historique, meilleur mAP (aucun au depart), epoques sans progres
    for epoch in range(1, cfg.epochs + 1):  # pour chaque epoque, numerotee a partir de 1
        start, sums = time.time(), {}  # heure de debut et cumul des pertes de l'epoque
        model.train()  # mode entrainement : le reseau calcule ses pertes
        for images, targets, _ in loader:  # pour chaque lot : images, cibles, echelles (ignorees)
            optimizer.zero_grad(set_to_none=True)  # efface les gradients du lot precedent
            losses = model(images.to(device), targets.to(device))  # dict : total_loss, iou_loss, conf_loss, cls_loss...
            losses["total_loss"].backward()  # retropropagation : calcule le gradient de chaque poids
            optimizer.step()  # corrige les poids dans le sens qui reduit la perte
            scheduler.step()  # le taux change a chaque iteration, pas a chaque epoque
            if ema is not None:  # si l'EMA est active
                ema.update(model)  # met a jour la moyenne glissante des poids
            for k, v in losses.items():  # pour chaque perte du dict
                sums[k] = sums.get(k, 0.0) + (v.item() if torch.is_tensor(v) else float(v))  # cumule sa valeur (nombre simple)
        evaluated = ema.ema if ema is not None else model  # on evalue le modele EMA s'il existe (plus stable)
        preds = detect(evaluated, val_loader, len(class_names), conf_eval, iou, device)  # detections sur la val
        # mAP de validation avec l'evaluateur COCO commun
        val = coco_metrics(val_set.annotations, [boxes_from_array(p, class_names) for p in preds], class_names)
        improved = val["map50_95"] > best + 1e-4  # progres reel (au-dela d'un petit ecart) du mAP50-95
        if improved:  # si le modele s'est ameliore
            best, bad = val["map50_95"], 0  # nouveau record, compteur d'echecs remis a 0
            # sauve les poids (state_dict = dictionnaire nom de couche -> valeurs), l'epoque et le score
            torch.save({"model_state": evaluated.state_dict(), "epoch": epoch, "val_map50_95": best}, output / "best.pt")
        else:  # sinon pas d'amelioration
            bad += 1  # une epoque de plus sans progres
        row = {  # ligne d'historique de l'epoque
            "epoch": epoch,  # numero d'epoque
            **{k: v / len(loader) for k, v in sums.items()},  # moyenne de chaque perte sur l'epoque
            "val_map50": val["map50"],  # mAP a IoU 0.5
            "val_map50_95": val["map50_95"],  # mAP moyen de IoU 0.5 a 0.95 (plus exigeant)
            "lr": optimizer.param_groups[0]["lr"],  # taux d'apprentissage actuel
        }
        history.append(row)  # ajoute la ligne
        logger.info(  # resume de l'epoque dans le journal
            "epoque %2d | perte %.3f | val mAP50 %.3f mAP50-95 %.3f | %.0fs%s",  # gabarit du message
            epoch,  # numero d'epoque
            row["total_loss"],  # perte totale moyenne
            val["map50"],  # mAP50 de validation
            val["map50_95"],  # mAP50-95 de validation
            time.time() - start,  # duree de l'epoque
            " | meilleur" if improved else "",  # marque le meilleur modele
        )
        if bad >= cfg.patience:  # si trop d'epoques sans progres
            logger.info("arret anticipe : %d epoques sans amelioration du mAP de validation", cfg.patience)  # previent
            break  # arrete la boucle
    return pd.DataFrame(history)  # historique en tableau


def load_trained(output: Path, variant: str, num_classes: int, device: torch.device) -> nn.Module:  # recharge best.pt ; renvoie le modele
    model = build_model(variant, num_classes)  # reseau vide de meme forme
    model.load_state_dict(torch.load(output / "best.pt", map_location=device, weights_only=True)["model_state"])  # y met les poids
    return model.to(device).eval()  # sur le bon device, en mode evaluation


# --- export de production ----------------------------------------------------------------------


# post-traitement de production d'une image : sortie ONNX brute -> boites finales (n, 6)
def postprocess_onnx(raw: np.ndarray, size: int, ratio: float, conf: float, iou: float) -> np.ndarray:
    """Sortie ONNX brute d'UNE image (A, 5 + C), non decodee -> (n, 6) dans l'image donnee au pretraitement.

    Decodage des grilles de pas 8, 16, 32 (demo_postprocess officiel) : cx = (x + colonne) x pas,
    w = exp(w) x pas ; puis score = objectness x probabilite de classe, filtrage, NMS par classe,
    division par le ratio (pas de bordure a retirer : l'image est collee en haut a gauche).
    C'est la fonction a reecrire dans le programme de production (C#, C++).
    """
    from yolox.utils.demo_utils import demo_postprocess  # decodage officiel des grilles

    decoded = demo_postprocess(raw[None].copy(), (size, size))[0]  # ajoute une dimension de lot, decode, la retire
    probs = decoded[:, 5:]  # probabilites par classe de chaque position
    classes = probs.argmax(axis=1)  # classe la plus probable de chaque position
    scores = decoded[:, 4] * probs[np.arange(len(classes)), classes]  # score = objectness x proba de cette classe
    keep = scores >= conf  # positions assez sures
    if not keep.any():  # si aucune ne passe le seuil
        return np.zeros((0, 6), np.float32)  # aucune boite
    c, s, k = decoded[keep, :4], scores[keep], classes[keep]  # centre-taille, scores et classes gardes
    # centre-taille -> coins (x1, y1, x2, y2), puis score et classe
    rows = np.column_stack([c[:, 0] - c[:, 2] / 2, c[:, 1] - c[:, 3] / 2, c[:, 0] + c[:, 2] / 2, c[:, 1] + c[:, 3] / 2, s, k])
    rows = nms_per_class(rows.astype(np.float32), iou)  # supprime les doublons classe par classe
    rows[:, :4] /= ratio  # entree du reseau -> image donnee au pretraitement
    return rows  # boites finales


def export(  # exporte YOLOX en ONNX et verifie ; renvoie un dict (rapport, modele, exemple)
    model: nn.Module,  # le reseau entraine
    samples: list[np.ndarray],  # quelques images reelles pour verifier
    size: int,  # cote de l'entree
    num_classes: int,  # nombre de classes
    path: Path,  # fichier ONNX a ecrire
    opset: int,  # version des operations ONNX
    conf: float,  # seuil de confiance pour la verification
    iou: float,  # seuil de NMS pour la verification
    atol: float,  # tolerance absolue sur l'ecart PyTorch / ONNX
    rtol: float,  # tolerance relative
) -> dict:  # renvoie un dict
    """Procedure officielle (SiLU d'export, sortie non decodee), puis deux verifications :
    sorties brutes torch == ONNX, et chaine ONNX + postprocess_onnx == detections PyTorch."""
    from yolox.models.network_blocks import SiLU  # version de l'activation SiLU compatible avec l'export ONNX
    from yolox.utils import replace_module  # remplace un type de couche par un autre dans tout le reseau

    from vtb.export.onnx import create_session, export_onnx, verify_onnx  # outils d'export generiques de la toolbox

    model = replace_module(model.cpu().eval(), nn.SiLU, SiLU)  # sur CPU, mode eval, SiLU exportable
    reference = predict_images(model, samples, size, num_classes, conf, iou, torch.device("cpu"))  # decodage PyTorch
    model.head.decode_in_inference = False  # la sortie exportee reste brute : le decodage se fera en production
    prepared = [preprocess(im, size) for im in samples]  # (tenseur, ratio) de chaque image d'exemple
    batch = torch.from_numpy(np.stack([t for t, _ in prepared]))  # empile en un lot
    export_onnx(model, batch[:2], path, ["output"], input_name="images", opset=opset)  # ecrit le fichier ONNX (exemple de 2 images)
    raw = verify_onnx(model, path, batch, atol, rtol)  # verification 1 : sorties brutes PyTorch == ONNX
    outputs = create_session(path).run(None, {"images": batch.numpy()})[0]  # sortie brute ONNX sur tout le lot
    # nos boites : sortie ONNX + post-traitement de production, image par image
    mine = [postprocess_onnx(outputs[i], size, ratio, conf, iou) for i, (_, ratio) in enumerate(prepared)]
    report = {"raw": raw, **mean_best_iou(mine, reference)}  # verification 2 : comparaison des boites
    logger.info("verification ONNX YOLOX : %s", report)  # affiche le rapport
    if report["mean_iou"] is not None and report["mean_iou"] < 0.98:  # si nos boites different de celles de PyTorch
        raise RuntimeError(f"le post-traitement ONNX ne reproduit pas PyTorch (IoU moyen {report['mean_iou']:.3f})")  # refuse
    return {"report": report, "model": model, "sample": batch[:1]}  # rapport, modele exporte, une entree d'exemple
