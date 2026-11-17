"""Moteur Faster R-CNN (torchvision, BSD) : 100 % PyTorch, le squelette du cours avec trois adaptations.

1. le Dataset renvoie (image, cible) ou la cible est un dict {boxes (n, 4) x1y1x2y2, labels (n,)},
   la classe 0 etant reservee au fond ;
2. collate_fn = tuple(zip(*batch)) : les images n'ont pas le meme nombre de boites ;
3. la perte est calculee DANS le modele : model(images, targets) renvoie un dict de pertes.
Les cinq lignes de la boucle ne changent pas.
"""

from __future__ import annotations  # les annotations de type restent du texte : syntaxe moderne "X | None" partout

import logging  # journal des messages (info, avertissements) au lieu de print
import time  # mesure de la duree de chaque epoque
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)
import pandas as pd  # pandas : tableaux de donnees ; ici l'historique epoque par epoque
import torch  # PyTorch : tenseurs (tableaux de nombres, sur CPU ou GPU) et reseaux de neurones
import torch.nn as nn  # nn : briques de reseau (Module, couches...)
from torch.utils.data import DataLoader, Dataset  # Dataset : donne une image par indice ; DataLoader : les regroupe en lots
from torchvision import tv_tensors  # tenseurs "etiquetes" (ex. BoundingBoxes) que les augmentations v2 savent deplacer

from vtb.annotate.schema import ImageAnnotation  # format pivot : toutes les etiquettes d'une image
from vtb.augment.transforms import sanitize_boxes  # retire les boites devenues trop petites apres augmentation
from vtb.evaluate.detection import coco_metrics  # evaluateur COCO commun : calcule le mAP (precision moyenne)
from vtb.io.images import read_image  # lit une image sur disque en tableau numpy
from vtb.preprocess import Pipeline  # suite d'operations OpenCV appliquee a chaque image
from vtb.preprocess.tensor import to_float_chw  # image numpy (H, W, C) -> tenseur float (C, H, W) dans [0, 1]
from vtb.tasks.detection.common import boxes_from_array  # tableau (n, 6) -> boites du format pivot
from vtb.tasks.detection.config import FasterRCNNConfig  # section fasterrcnn du YAML

logger = logging.getLogger(__name__)  # journal propre a ce module


def build_model(cfg: FasterRCNNConfig, num_classes: int) -> nn.Module:  # construit le detecteur ; renvoie le reseau
    """num_classes + 1 sorties (le fond). resnet18_fpn : backbone ImageNet seul ; *_coco : detecteur COCO complet, tete remplacee."""
    from torchvision.models import ResNet18_Weights, ResNet50_Weights  # references des poids ImageNet des ResNet

    # constructeurs officiels : Faster R-CNN generique et deux detecteurs complets pre-entraines sur COCO
    from torchvision.models.detection import FasterRCNN, fasterrcnn_mobilenet_v3_large_fpn, fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone  # ResNet + FPN (pyramide multi-echelle)
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor  # tete finale : classe + ajustement de boite

    sizes = {"min_size": cfg.min_size, "max_size": cfg.max_size}  # tailles de redimensionnement interne du modele
    if cfg.pretrained == "coco" and cfg.backbone != "resnet18_fpn":  # detecteur COCO complet (existe en resnet50 et mobilenet)
        # choisit le constructeur selon le backbone demande
        builder = fasterrcnn_resnet50_fpn_v2 if cfg.backbone == "resnet50_fpn" else fasterrcnn_mobilenet_v3_large_fpn
        model = builder(weights="DEFAULT", trainable_backbone_layers=cfg.trainable_layers, **sizes)  # avec poids COCO
        in_features = model.roi_heads.box_predictor.cls_score.in_features  # taille du vecteur qui entre dans la tete
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)  # nouvelle tete : nos classes + le fond
        return model  # renvoie le detecteur adapte
    if cfg.backbone == "mobilenet_v3_fpn":  # MobileNet (leger) sans poids COCO
        return fasterrcnn_mobilenet_v3_large_fpn(  # detecteur neuf
            weights=None,  # pas de poids COCO pour le detecteur complet
            weights_backbone="DEFAULT" if cfg.pretrained == "imagenet" else None,  # backbone ImageNet si demande, sinon zero
            num_classes=num_classes + 1,  # nos classes + 1 pour le fond
            **sizes,  # tailles de redimensionnement interne (min_size, max_size)
        )
    name = "resnet18" if cfg.backbone == "resnet18_fpn" else "resnet50"  # nom du ResNet a utiliser
    # poids ImageNet du ResNet choisi, ou None pour un depart de zero
    weights = (ResNet18_Weights.DEFAULT if name == "resnet18" else ResNet50_Weights.DEFAULT) if cfg.pretrained != "none" else None
    backbone = resnet_fpn_backbone(backbone_name=name, weights=weights, trainable_layers=cfg.trainable_layers)  # ResNet + FPN
    return FasterRCNN(backbone, num_classes=num_classes + 1, **sizes)  # detecteur complet autour de ce backbone


class DetectionDataset(Dataset):  # jeu de donnees PyTorch pour Faster R-CNN : (image, dict de cibles)
    """(image float (3, H, W) dans [0, 1], {boxes, labels}) ; labels decales de +1 (0 = fond)."""

    # constructeur : annotations, classes, pretraitement, couleur, augmentation (train seulement)
    def __init__(self, annotations: list[ImageAnnotation], class_names: list[str], pipeline: Pipeline, color: str, augment=None) -> None:
        self.annotations, self.pipeline, self.color, self.augment = annotations, pipeline, color, augment  # garde les reglages
        self.index = {n: i + 1 for i, n in enumerate(class_names)}  # nom de classe -> numero a partir de 1 (0 = fond)

    def __len__(self) -> int:  # nombre d'images du jeu
        return len(self.annotations)  # une annotation par image

    def image(self, i: int) -> np.ndarray:  # lit et pretraite l'image numero i ; renvoie un tableau a 3 canaux
        image = self.pipeline(read_image(self.annotations[i].image_path, color=self.color))  # lecture puis pretraitement
        return image if image.ndim == 3 else np.stack([image] * 3, -1)  # image grise -> 3 canaux identiques

    def __getitem__(self, i: int):  # appele par le DataLoader : renvoie (tenseur image, {boxes, labels})
        ann, image = self.annotations[i], self.image(i)  # l'annotation et l'image pretraitee
        sx, sy = image.shape[1] / ann.width, image.shape[0] / ann.height  # un resize du pretraitement met les boites a l'echelle
        # boites (n, 4) x1 y1 x2 y2 ramenees dans l'image pretraitee ; reshape garde 4 colonnes meme sans boite
        boxes = torch.tensor([[b.x1 * sx, b.y1 * sy, b.x2 * sx, b.y2 * sy] for b in ann.boxes], dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor([self.index[b.label] for b in ann.boxes], dtype=torch.int64)  # numero de classe de chaque boite
        tensor = to_float_chw(image)  # image -> tenseur float (3, H, W) dans [0, 1]
        if self.augment is not None:  # torchvision v2 deplace les boites avec l'image (retournement, rotation...)
            # enveloppe les boites pour que l'augmentation sache les transformer (format coins, taille de l'image)
            wrapped = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=tuple(tensor.shape[-2:]))
            tensor, wrapped = self.augment(tensor, wrapped)  # transforme image et boites ensemble
            wrapped, labels = sanitize_boxes(wrapped, labels)  # une boite sortie de l'image disparait avec son etiquette
            boxes = wrapped.as_subclass(torch.Tensor)  # redevient un tenseur ordinaire
        return tensor, {"boxes": boxes, "labels": labels}  # format de cible attendu par Faster R-CNN


def collate(batch):  # assemble un lot : renvoie (tuple d'images, tuple de cibles) sans les empiler
    return tuple(zip(*batch))  # images de tailles et nombres de boites differents : on garde des listes


@torch.no_grad()  # pas de gradient : simple inference
# predit sur des images en memoire ; renvoie un tableau (n, 6) par image
def predict_images(model: nn.Module, images: list[np.ndarray], conf: float, device: torch.device, batch_size: int = 4) -> list[np.ndarray]:
    """Images pipeline (RGB ou gris, uint8 ou uint16) -> (n, 6) [x1, y1, x2, y2, score, classe] dans le repere de ces images.

    Le modele redimensionne lui-meme (min_size, max_size) et rend les boites a l'echelle de
    l'image recue : aucun letterbox a defaire, contrairement aux YOLO.
    """
    model.eval()  # mode evaluation : le modele renvoie des detections au lieu de pertes
    model.roi_heads.score_thresh = conf  # confiance minimale d'une boite gardee
    out = []  # resultats, un par image
    for start in range(0, len(images), batch_size):  # parcourt les images par paquets
        # chaque image du paquet : 3 canaux, tenseur float [0, 1], envoyee sur le device (liste : tailles differentes permises)
        tensors = [to_float_chw(im if im.ndim == 3 else np.stack([im] * 3, -1)).to(device) for im in images[start : start + batch_size]]
        for result in model(tensors):  # pour chaque image : dict boxes, scores, labels
            # coins, scores, classes ramenees a partir de 0 (on retire le decalage du fond)
            rows = [result["boxes"].cpu().numpy(), result["scores"].cpu().numpy(), result["labels"].cpu().numpy() - 1]
            out.append(np.column_stack(rows).astype(np.float32).reshape(-1, 6))  # tableau (n, 6), vide si aucune boite
    return out  # renvoie toutes les detections


def detect(  # detections de tout un Dataset, ramenees dans l'image d'origine ; renvoie un tableau (n, 6) par image
    model: nn.Module,  # le detecteur
    dataset: DetectionDataset,  # le jeu d'images (val pendant l'entrainement)
    class_names: list[str],  # noms des classes
    conf: float,  # confiance minimale (tres basse pour le mAP)
    device: torch.device,  # CPU ou GPU
    batch_size: int = 4,  # images traitees ensemble
) -> list[np.ndarray]:  # une liste de tableaux (n, 6)
    """(n, 6) dans l'image d'ORIGINE pour toutes les images d'un dataset (validation pendant l'entrainement)."""
    out = []  # resultats, un par image
    for start in range(0, len(dataset), batch_size):  # parcourt le jeu par paquets
        indices = range(start, min(start + batch_size, len(dataset)))  # numeros des images du paquet
        images = [dataset.image(i) for i in indices]  # images pretraitees du paquet
        # pour chaque image : son numero, l'image et ses detections
        for i, image, rows in zip(indices, images, predict_images(model, images, conf, device, batch_size)):
            ann = dataset.annotations[i]  # annotation d'origine (donne la taille d'origine)
            rows[:, [0, 2]] *= ann.width / image.shape[1]  # x : image pretraitee -> image d'origine
            rows[:, [1, 3]] *= ann.height / image.shape[0]  # y : meme conversion
            out.append(rows)  # garde les detections
    return out  # renvoie toutes les detections


def train(  # boucle d'entrainement Faster R-CNN ; renvoie l'historique (une ligne par epoque)
    train_set: DetectionDataset,  # images du train (avec augmentation eventuelle)
    val_set: DetectionDataset,  # images de validation
    cfg: FasterRCNNConfig,  # reglages fasterrcnn du YAML
    class_names: list[str],  # noms des classes
    conf_eval: float,  # seuil de confiance tres bas pour mesurer le mAP
    device: torch.device,  # CPU ou GPU
    output: Path,  # dossier ou sauver best.pt
    seed: int,  # graine de melange
    num_workers: int,  # processus paralleles de lecture
) -> pd.DataFrame:  # renvoie l'historique
    model = build_model(cfg, len(class_names)).to(device)  # construit le detecteur et le place sur le device
    loader = DataLoader(  # distributeur de lots du train
        train_set,  # le jeu d'entrainement
        batch_size=cfg.batch_size,  # images par lot
        shuffle=True,  # melange a chaque epoque
        collate_fn=collate,  # assemblage en listes (nombre de boites variable)
        num_workers=num_workers,  # processus de lecture
        generator=torch.Generator().manual_seed(seed),  # melange reproductible
    )
    params = [p for p in model.parameters() if p.requires_grad]  # seulement les poids non geles
    # optimiseur SGD avec elan (momentum) et penalite sur les grands poids
    optimizer = torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    warmup = min(200, len(loader))  # nombre d'iterations de montee du taux (au plus une epoque)
    # montee lineaire du taux, iteration par iteration, jusqu'au taux plein
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda it: min(1.0, (it + 1) / warmup))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs))  # puis baisse en cosinus par epoque
    history, best, bad, iteration = [], -1.0, 0, 0  # historique, meilleur mAP, epoques sans progres, compteur d'iterations
    for epoch in range(1, cfg.epochs + 1):  # pour chaque epoque, numerotee a partir de 1
        start, total = time.time(), 0.0  # heure de debut et cumul des pertes
        model.train()  # mode entrainement : le modele renvoie ses pertes
        for images, targets in loader:  # pour chaque lot : liste d'images et liste de cibles
            images = [i.to(device) for i in images]  # envoie chaque image sur le device
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  # idem pour boxes et labels de chaque cible
            optimizer.zero_grad(set_to_none=True)  # efface les gradients du lot precedent
            losses = model(images, targets)  # la perte est calculee dans le modele (RPN + tete)
            loss = sum(losses.values())  # perte totale = somme des pertes partielles
            loss.backward()  # retropropagation : calcule le gradient de chaque poids
            optimizer.step()  # corrige les poids dans le sens qui reduit la perte
            if iteration < warmup:  # pendant la montee du taux
                scheduler.step()  # augmente le taux pour l'iteration suivante
            iteration += 1  # compte les iterations depuis le debut
            total += loss.item()  # cumule la perte (nombre simple, sans le graphe de calcul)
        cosine.step()  # fin d'epoque : baisse le taux selon le cosinus
        preds = detect(model, val_set, class_names, conf_eval, device)  # detections sur la validation
        # mAP de validation avec l'evaluateur COCO commun
        val = coco_metrics(val_set.annotations, [boxes_from_array(p, class_names) for p in preds], class_names)
        improved = val["map50_95"] > best + 1e-3  # progres reel du mAP50-95 (au-dela d'un petit ecart)
        if improved:  # si le modele s'est ameliore
            best, bad = val["map50_95"], 0  # nouveau record, compteur d'echecs remis a 0
            # sauve les poids (state_dict = dictionnaire nom de couche -> valeurs), l'epoque et le score
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_map50_95": best}, output / "best.pt")
        else:  # sinon pas d'amelioration
            bad += 1  # une epoque de plus sans progres
        # ligne d'historique : epoque, perte moyenne, mAP50 et mAP50-95 de validation
        history.append({"epoch": epoch, "train_loss": total / len(loader), "val_map50": val["map50"], "val_map50_95": val["map50_95"]})
        logger.info(  # resume de l'epoque dans le journal
            "epoque %2d | perte %.3f | val mAP50 %.3f mAP50-95 %.3f | %.0fs%s",  # gabarit du message
            epoch,  # numero d'epoque
            total / len(loader),  # perte moyenne de l'epoque
            val["map50"],  # mAP50 de validation
            val["map50_95"],  # mAP50-95 de validation
            time.time() - start,  # duree de l'epoque
            " | meilleur" if improved else "",  # marque le meilleur modele
        )
        if bad >= cfg.patience:  # si trop d'epoques sans progres
            logger.info("arret anticipe : %d epoques sans amelioration", cfg.patience)  # previent
            break  # arrete la boucle
    return pd.DataFrame(history)  # historique en tableau


# recharge best.pt ; renvoie le modele pret a predire
def load_trained(output: Path, cfg: FasterRCNNConfig, num_classes: int, device: torch.device) -> nn.Module:
    model = build_model(cfg.model_copy(update={"pretrained": "none"}), num_classes)  # reseau vide (pas de telechargement)
    model.load_state_dict(torch.load(output / "best.pt", map_location=device, weights_only=True)["model_state"])  # y met les poids
    return model.to(device).eval()  # sur le bon device, en mode evaluation


class ExportWrapper(nn.Module):  # enveloppe le modele pour lui donner une entree / sortie simple a exporter en ONNX
    """Une image (1, 3, H, W) float [0, 1] -> boxes (n, 4), scores (n,), labels (n,) : l'interface que l'ONNX expose.

    Redimensionnement, RPN, ROI pooling et NMS sont DANS le graphe : le programme de production
    n'a qu'a convertir l'image en float et lire trois tableaux.
    """

    def __init__(self, model: nn.Module) -> None:  # constructeur : garde le detecteur
        super().__init__()  # initialise la partie nn.Module (obligatoire)
        self.model = model  # le detecteur enveloppe

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # passe avant ; renvoie 3 tenseurs
        result = self.model([x[0]])[0]  # le detecteur attend une liste d'images (3, H, W) : on passe la seule image du lot
        return result["boxes"], result["scores"], result["labels"] - 1  # boites, scores, classes a partir de 0


# exporte en ONNX et verifie ; renvoie un dict (rapport, modele enveloppe, exemple)
def export(model: nn.Module, samples: list[np.ndarray], path: Path, opset: int, conf: float) -> dict:
    """Ancien exporteur (TorchScript) : celui que torchvision teste pour Faster R-CNN ; hauteur et largeur dynamiques.

    Verification : memes boites ONNX et PyTorch sur des images de tailles differentes (IoU moyen >= 0,98).
    """
    import warnings  # pour masquer des avertissements attendus

    import onnxruntime as ort  # onnxruntime : execute un modele ONNX sans PyTorch (comme en production)

    from vtb.tasks.detection.common import mean_best_iou  # compare deux listes de detections

    model = model.cpu().eval()  # export sur CPU, en mode evaluation
    model.roi_heads.score_thresh = conf  # le seuil de confiance est integre au graphe exporte
    wrapper = ExportWrapper(model).eval()  # entree tenseur unique, sortie trois tenseurs
    # chaque image d'exemple en tenseur (1, 3, H, W) : tailles differentes pour tester les axes dynamiques
    tensors = [to_float_chw(im if im.ndim == 3 else np.stack([im] * 3, -1))[None] for im in samples]
    path.parent.mkdir(parents=True, exist_ok=True)  # cree le dossier de sortie si besoin
    with warnings.catch_warnings():  # les reglages d'avertissements reviennent a la normale apres ce bloc
        warnings.simplefilter("ignore")  # avertissements de tracage attendus (boucles sur les images)
        torch.onnx.export(  # ecrit le fichier ONNX
            wrapper,  # le modele a exporter
            (tensors[0],),  # exemple d'entree (une image) qui sert a tracer le graphe
            str(path),  # fichier de sortie
            input_names=["input"],  # nom de l'entree dans le fichier ONNX
            output_names=["boxes", "scores", "labels"],  # noms des trois sorties
            # axes de taille variable : hauteur et largeur de l'image, nombre n de boites trouvees
            dynamic_axes={"input": {2: "height", 3: "width"}, "boxes": {0: "n"}, "scores": {0: "n"}, "labels": {0: "n"}},
            opset_version=max(opset, 11),  # version des operations ONNX (11 au minimum pour ce modele)
            dynamo=False,  # ancien exporteur (TorchScript), celui teste par torchvision
        )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])  # ouvre le modele ONNX sur CPU
    mine, reference = [], []  # detections ONNX et detections PyTorch
    with torch.no_grad():  # pas de gradient : simple inference
        for x in tensors:  # pour chaque image d'exemple
            boxes, scores, labels = session.run(None, {"input": x.numpy()})  # inference ONNX
            mine.append(np.column_stack([boxes, scores, labels]).astype(np.float32).reshape(-1, 6))  # en tableau (n, 6)
            b, s, k = wrapper(x)  # inference PyTorch sur la meme image
            # meme mise en forme (n, 6) pour comparer
            reference.append(np.column_stack([b.numpy(), s.numpy(), k.numpy()]).astype(np.float32).reshape(-1, 6))
    report = mean_best_iou(mine, reference)  # nombre de boites de chaque cote et IoU moyen
    logger.info("verification ONNX Faster R-CNN : %s", report)  # affiche le rapport
    # export refuse si le nombre de boites differe ou si les boites ne se recouvrent pas assez
    if report["boxes"] != report["reference_boxes"] or (report["mean_iou"] is not None and report["mean_iou"] < 0.98):
        raise RuntimeError(f"export ONNX Faster R-CNN incorrect : {report}")  # message avec le rapport complet
    return {"report": report, "model": wrapper, "sample": tensors[0]}  # rapport, modele enveloppe, une entree d'exemple
