"""Moteur Ultralytics (YOLO11) : la librairie fait Dataset, augmentations, boucle et validation.

Ce que la toolbox ajoute autour : le jeu au format YOLO ecrit depuis le pivot (avec le meme
pretraitement qu'en production), des predictions converties dans le pivot (pour l'evaluateur
COCO commun), et l'export ONNX verifie avec un post-traitement ecrit en clair (letterbox, NMS)
que le programme de production (C#, C++) reproduit sans Ultralytics.

Piege : Ultralytics suppose qu'un tableau numpy est en BGR (convention OpenCV). Les images de la
toolbox sont en RGB : on les convertit juste avant model.predict, et nulle part ailleurs.
"""

from __future__ import annotations  # les annotations de type restent du texte : syntaxe moderne "X | None" partout

import logging  # journal des messages (info, avertissements) au lieu de print
import shutil  # copie, deplacement et suppression de fichiers et dossiers
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import cv2  # OpenCV : lecture, conversion de couleurs et traitement d'images
import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)
import torch  # PyTorch : tenseurs (tableaux de nombres, sur CPU ou GPU) et reseaux de neurones

# format pivot : annotations d'une image, contour d'objet (polygone), boite tournee
from vtb.annotate.schema import ImageAnnotation, Polygon, RotatedBox
from vtb.annotate.yolo import write_yolo_dataset  # ecrit un jeu au format YOLO (images + fichiers texte d'etiquettes)
from vtb.core.runtime import quiet_libraries  # fait taire les messages trop bavards des bibliotheques
from vtb.preprocess import Pipeline  # suite d'operations OpenCV appliquee a chaque image

# letterbox = redimensionner sans deformer dans un carre, en completant les bords par du gris ; ses parametres (ratio, marges)
from vtb.preprocess.ops.geometry import letterbox, letterbox_params

# outils communs : conversions boites <-> tableaux, comparaison de detections, NMS par classe, boite tournee -> droite
from vtb.tasks.detection.common import boxes_from_array, boxes_to_array, mean_best_iou, nms_per_class, rotated_to_box
from vtb.tasks.detection.config import DetectionConfig  # le YAML de l'experience de detection, valide

logger = logging.getLogger(__name__)  # journal propre a ce module

# seuil de la verification d'export : assez bas pour qu'il y ait des boites a comparer, meme avec un modele
# peu entraine (un seuil de production a 0.25 peut ne rien laisser passer sur 4 images)
VERIFY_CONF = 0.05  # confiance minimale d'une boite pendant la verification de l'export


def write_dataset(  # ecrit le jeu au format YOLO dans out_dir ; renvoie le chemin du fichier data.yaml
    annotations: dict[str, list[ImageAnnotation]],  # annotations par sous-ensemble (train / val / test)
    class_names: list[str],  # noms des classes, dans l'ordre des numeros
    pipeline: Pipeline,  # pretraitement a appliquer aux images copiees
    color: str,  # "rgb" ou "gray"
    out_dir: Path,  # dossier ou ecrire le jeu YOLO
    task: str,  # "detect", "obb" ou "segment" : format des etiquettes
) -> Path:  # renvoie le chemin de data.yaml (le fichier que lit Ultralytics)
    """images/<split> + labels/<split> + data.yaml ; les images sont ecrites APRES pretraitement (la meme fonction qu'en production).

    Les coordonnees YOLO etant normalisees, un redimensionnement ne les change pas ; un recadrage
    les fausserait : seul resize est compatible ici (comme pour les masques).
    """
    flat = [a for subset in annotations.values() for a in subset]  # toutes les annotations en une seule liste
    splits = {a.image_path: subset for subset, items in annotations.items() for a in items}  # chemin d'image -> sous-ensemble

    def transform(image: np.ndarray) -> np.ndarray:  # pretraite une image avant de l'ecrire ; renvoie une image a 3 canaux
        out = pipeline(image if color == "rgb" else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))  # passe en gris si demande, pretraite
        return out if out.ndim == 3 else np.stack([out] * 3, axis=-1)  # image grise -> 3 canaux identiques (YOLO attend 3 canaux)

    if out_dir.exists():  # si le dossier existe deja (run precedent)
        shutil.rmtree(out_dir)  # jamais de vieilles images melangees aux nouvelles
    return write_yolo_dataset(  # ecrit les images, les etiquettes et data.yaml ; renvoie le chemin de data.yaml
        flat,  # toutes les annotations
        splits,  # chemin d'image -> train / val / test
        class_names,  # noms des classes
        out_dir,  # dossier de sortie
        task=task,  # format des etiquettes selon la tache
        # pretraitement seulement s'il y en a un (ou du gris) ; sinon copie directe des fichiers, plus rapide
        transform=transform if len(pipeline) or color == "gray" else None,
    )  # type: ignore[arg-type]


def train(config: DetectionConfig, data_yaml: Path, output: Path) -> Path:  # entraine YOLO11 ; renvoie le chemin de best.pt
    from ultralytics import YOLO  # import tardif : la bibliotheque n'est chargee que si on utilise ce moteur

    uc = config.ultralytics  # raccourci vers la section ultralytics du YAML
    model = YOLO(uc.weights)  # charge le modele et ses poids pre-entraines (telecharges s'ils manquent)
    quiet_libraries()  # Ultralytics remet ses messages a INFO a l'import : on les refait taire
    results = model.train(  # lance l'entrainement complet (Ultralytics gere lots, augmentations, validation)
        data=str(data_yaml),  # le fichier data.yaml qui decrit le jeu
        epochs=uc.epochs,  # nombre maximal d'epoques
        imgsz=uc.imgsz,  # taille d'entree des images
        batch=uc.batch,  # images par lot
        patience=uc.patience,  # arret anticipe
        close_mosaic=uc.close_mosaic,  # epoques finales sans augmentation mosaique
        warmup_epochs=uc.warmup_epochs,  # montee progressive du taux d'apprentissage
        device=0 if torch.cuda.is_available() else "cpu",  # premier GPU s'il existe, sinon CPU
        seed=config.runtime.seed,  # graine : resultats reproductibles
        workers=config.runtime.num_workers,  # processus paralleles de lecture des images
        project=str((output / "runs").resolve()),  # chemin absolu, sinon Ultralytics prefixe runs/detect/
        name="train",  # sous-dossier du run
        exist_ok=True,  # reutilise le dossier s'il existe au lieu de creer train2, train3...
        plots=False,  # pas de graphiques Ultralytics (la toolbox fait les siens)
        verbose=False,  # moins de messages
        **uc.extra,  # tout argument supplementaire ecrit dans le YAML (lr0, mosaic...)
    )
    best = Path(results.save_dir) / "weights" / "best.pt"  # meilleur modele choisi par Ultralytics sur la validation
    shutil.copy2(best, output / "best.pt")  # le copie dans le dossier de sortie de l'experience
    return output / "best.pt"  # renvoie son chemin


def load(output: Path):  # recharge le modele entraine ; renvoie l'objet YOLO
    from ultralytics import YOLO  # import tardif : seulement si on utilise ce moteur

    model = YOLO(str(output / "best.pt"))  # charge les poids entraines
    quiet_libraries()  # refait taire les messages d'Ultralytics
    return model  # renvoie le modele pret a predire


# predit sur une liste d'images ; renvoie une ImageAnnotation par image
def predict(model, images: list[np.ndarray], class_names: list[str], imgsz: int, conf: float, iou: float) -> list[ImageAnnotation]:
    """Images RGB pretraitees -> annotations predites (boites, plus boites orientees ou polygones selon la tache)."""
    results = model.predict(  # inference Ultralytics (letterbox, reseau, NMS) sur toutes les images
        # chaque image en 3 canaux puis RGB -> BGR (Ultralytics suppose du BGR pour un tableau numpy)
        [cv2.cvtColor(im if im.ndim == 3 else np.stack([im] * 3, -1), cv2.COLOR_RGB2BGR) for im in images],
        imgsz=imgsz,  # taille d'entree du reseau
        conf=conf,  # confiance minimale pour garder une boite
        iou=iou,  # seuil de recouvrement de la NMS (suppression des doublons)
        verbose=False,  # pas de message par image
        device="cpu" if not torch.cuda.is_available() else 0,  # CPU sans GPU, sinon premier GPU
    )
    out = []  # annotations predites, une par image
    for image, result in zip(images, results):  # pour chaque image et son resultat
        ann = ImageAnnotation("", image.shape[1], image.shape[0])  # annotation vide (chemin inconnu ici), largeur, hauteur
        if result.obb is not None:  # si la tache produit des boites orientees (obb)
            for (cx, cy, w, h, angle), c, s in zip(  # pour chaque boite : centre, taille, angle, classe, score
                result.obb.xywhr.cpu().numpy(),  # centre, taille et angle (radians) de chaque boite, en numpy sur CPU
                result.obb.cls.cpu().numpy(),  # numero de classe de chaque boite
                result.obb.conf.cpu().numpy(),  # score de confiance de chaque boite
            ):
                ann.rotated.append(  # ajoute la boite tournee au format pivot
                    # nom de classe, centre, taille, angle converti en degres, score
                    RotatedBox(class_names[int(c)], float(cx), float(cy), float(w), float(h), float(np.degrees(angle)), float(s))
                )
            ann.boxes = [rotated_to_box(r) for r in ann.rotated]  # boites droites englobantes (pour l'evaluateur COCO)
        elif result.boxes is not None:  # sinon, s'il y a des boites droites (detect ou segment)
            # tableau (n, 6) : coins x1 y1 x2 y2, score, classe
            rows = np.column_stack([result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()])
            ann.boxes = boxes_from_array(rows, class_names)  # converties au format pivot
            if result.masks is not None:  # si la tache est la segmentation (un masque par objet)
                for points, box in zip(result.masks.xy, ann.boxes):  # contour de chaque masque et sa boite
                    if len(points) >= 3:  # un polygone a besoin d'au moins 3 points
                        # contour au format pivot, avec la classe et le score de la boite
                        ann.polygons.append(Polygon(box.label, [(float(x), float(y)) for x, y in points], box.score))
        out.append(ann)  # garde l'annotation de cette image
    return out  # renvoie toutes les annotations predites


def val_metrics(model, data_yaml: Path, config: DetectionConfig, output: Path) -> dict:  # metriques Ultralytics ; renvoie un dict
    """Metriques d'Ultralytics sur le test (pour boites orientees et masques, que l'evaluateur COCO commun ne couvre pas)."""
    metrics = model.val(  # evaluation Ultralytics
        data=str(data_yaml),  # description du jeu
        split="test",  # sur le sous-ensemble test
        imgsz=config.ultralytics.imgsz,  # meme taille d'entree qu'a l'entrainement
        batch=config.ultralytics.batch,  # images par lot
        conf=config.evaluation.conf_eval,  # seuil tres bas : toute la courbe precision-rappel
        iou=config.evaluation.iou_nms,  # seuil de la NMS
        device="cpu" if not torch.cuda.is_available() else 0,  # CPU ou premier GPU
        project=str((output / "runs").resolve()),  # dossier des runs, en chemin absolu
        name="test",  # sous-dossier de ce run
        exist_ok=True,  # reutilise le dossier s'il existe
        plots=False,  # pas de graphiques
        verbose=False,  # moins de messages
    )
    # mAP = precision moyenne (aire sous la courbe precision-rappel) ; 50 = boite juste si IoU >= 0.5 ; 50_95 = moyenne de 0.5 a 0.95
    result = {"box_map50": float(metrics.box.map50), "box_map50_95": float(metrics.box.map)}
    if getattr(metrics, "seg", None) is not None and config.ultralytics.task == "segment":  # si on a aussi des masques
        result.update({"mask_map50": float(metrics.seg.map50), "mask_map50_95": float(metrics.seg.map)})  # mAP des masques
    return result  # renvoie les metriques


# --- export et post-traitement de production -------------------------------------------------


# prepare une image pour le modele ONNX ; renvoie (tenseur, ratio de redimension, marges)
def preprocess_for_onnx(image_rgb: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """RGB uint8 -> letterbox (gris 114, centre) -> [0, 1] -> NCHW float32 ; renvoie aussi (ratio, (pad_x, pad_y))."""
    ratio, _, _, pad_x, pad_y = letterbox_params(image_rgb.shape[0], image_rgb.shape[1], size)  # ratio et marges du letterbox
    boxed = letterbox(image_rgb, size=size, pad_value=114)  # image mise dans un carre size x size, bords gris 114
    # (H, W, C) -> (1, C, H, W) = NCHW (lot, canaux, hauteur, largeur), valeurs de 0-255 ramenees dans [0, 1]
    return np.ascontiguousarray(boxed.astype(np.float32).transpose(2, 0, 1)[None] / 255.0), ratio, (pad_x, pad_y)


def postprocess_detect(  # transforme la sortie brute du reseau en boites finales ; renvoie un tableau (n, 6)
    output: np.ndarray,  # sortie brute du reseau ONNX
    ratio: float,  # ratio de redimensionnement du letterbox
    pad: tuple[int, int],  # marges (gauche, haut) ajoutees par le letterbox
    conf: float,  # confiance minimale pour garder une boite
    iou: float,  # seuil de recouvrement de la NMS
    shape: tuple[int, int] | None = None,  # (hauteur, largeur) de l'image d'origine pour borner les boites ; None = pas de bornage
) -> np.ndarray:  # renvoie (n, 6) [x1, y1, x2, y2, score, classe]
    """Sortie brute YOLO11 (1, 4 + C, A) -> (n, 6) [x1, y1, x2, y2, score, classe] dans l'image d'origine.

    Pour chaque ancre : cx, cy, w, h en pixels de l'image letterboxee, deja decodes, puis un score
    par classe (pas de score d'objet). Score max et classe, filtrage, coins, NMS par classe, puis
    on annule le letterbox (retirer la bordure, diviser par le ratio).
    """
    pred = output[0].T  # retire la dimension du lot et transpose : une ligne par ancre (position candidate), (A, 4 + C)
    scores, classes = pred[:, 4:].max(axis=1), pred[:, 4:].argmax(axis=1)  # meilleur score par ancre et sa classe
    keep = scores > conf  # garde seulement les ancres assez sures
    pred, scores, classes = pred[keep], scores[keep], classes[keep]  # applique ce filtre aux trois tableaux
    if len(pred) == 0:  # si rien ne passe le seuil
        return np.zeros((0, 6), np.float32)  # aucune boite
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]  # centre et taille de chaque boite
    # centre-taille -> coins (x1, y1, x2, y2), plus score et classe, en float32
    rows = np.column_stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, scores, classes]).astype(np.float32)
    rows = nms_per_class(rows, iou)  # supprime les boites en double, classe par classe
    rows[:, [0, 2]] = (rows[:, [0, 2]] - pad[0]) / ratio  # x : retire la marge gauche puis annule le redimensionnement
    rows[:, [1, 3]] = (rows[:, [1, 3]] - pad[1]) / ratio  # y : retire la marge du haut puis annule le redimensionnement
    if shape is not None:  # boites ramenees dans l'image (hauteur, largeur), comme Ultralytics
        rows[:, [0, 2]] = rows[:, [0, 2]].clip(0, shape[1])  # x bornes entre 0 et la largeur
        rows[:, [1, 3]] = rows[:, [1, 3]].clip(0, shape[0])  # y bornes entre 0 et la hauteur
    return rows  # boites finales dans l'image d'origine


# exporte en ONNX et verifie l'export ; renvoie un dict (chemin, rapport, sorties, reseau torch, exemple)
def export(model, config: DetectionConfig, output: Path, samples: list[np.ndarray], class_names: list[str]) -> dict:
    """ONNX sans NMS (le programme de production la fait) ; verifie (1) les sorties brutes contre le
    reseau torch, (2) pour detect, la chaine ONNX + postprocess_detect contre model.predict d'Ultralytics."""
    import onnxruntime as ort  # onnxruntime : execute un modele ONNX sans PyTorch (comme en production)

    size = config.ultralytics.imgsz  # taille d'entree du reseau
    exported = Path(  # chemin du fichier ONNX produit par Ultralytics
        # export ONNX : lot dynamique, sans NMS integree, sans simplification, sur CPU
        model.export(format="onnx", imgsz=size, dynamic=True, nms=False, simplify=False, opset=config.export.opset, device="cpu")
    )
    onnx_path = output / "model.onnx"  # emplacement voulu dans le dossier de l'experience
    shutil.move(str(exported), onnx_path)  # deplace le fichier a cet emplacement
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])  # ouvre le modele ONNX sur CPU
    name = session.get_inputs()[0].name  # nom de l'entree du modele ONNX
    net = model.model.eval().float()  # le reseau PyTorch interne, en mode evaluation et en float32
    raw_gap, scale, mine, reference = 0.0, 1.0, [], []  # plus grand ecart brut, echelle des sorties, boites des deux chaines
    for image in samples:  # pour chaque image d'exemple
        tensor, ratio, pad = preprocess_for_onnx(image, size)  # pretraitement de production
        onnx_out = session.run(None, {name: tensor})  # sortie brute du modele ONNX
        with torch.no_grad():  # pas de gradient : simple inference
            torch_out = net(torch.from_numpy(tensor))  # sortie brute du reseau PyTorch sur le meme tenseur
        first = _first_tensor(torch_out).numpy()  # premier tenseur de la sortie PyTorch, en numpy
        # retient le plus grand ecart PyTorch / ONNX et la plus grande valeur de sortie (echelle)
        raw_gap, scale = max(raw_gap, float(np.abs(first - onnx_out[0]).max())), max(scale, float(np.abs(first).max()))
        if config.ultralytics.task == "detect":  # pour la detection simple, on verifie aussi le post-traitement
            ev = config.evaluation  # raccourci vers les seuils
            # nos boites : sortie ONNX + notre post-traitement ecrit en clair
            mine.append(postprocess_detect(onnx_out[0], ratio, pad, VERIFY_CONF, ev.iou_nms, image.shape[:2]))
            ref = predict(model, [image], class_names, size, VERIFY_CONF, ev.iou_nms)[0]  # boites de reference : Ultralytics
            reference.append(boxes_to_array(ref.boxes, class_names))  # converties en tableau (n, 6)
    # rapport : ecart brut, echelle, et comparaison des boites (si detect)
    report = {"raw_max_abs_diff": raw_gap, "raw_scale": scale, **(mean_best_iou(mine, reference) if mine else {})}
    logger.info("verification ONNX Ultralytics : %s", report)  # affiche le rapport
    if raw_gap > config.export.atol + config.export.rtol * scale * 10:  # si l'ecart brut depasse la tolerance (elargie x10)
        raise RuntimeError(f"export ONNX incorrect : ecart brut {raw_gap:.3e} pour une echelle {scale:.1f}")  # export refuse
    if report.get("mean_iou") is not None and report["mean_iou"] < 0.98:  # si nos boites different de celles d'Ultralytics
        raise RuntimeError(f"le post-traitement ONNX ne reproduit pas Ultralytics (IoU moyen {report['mean_iou']:.3f})")  # refuse
    return {  # tout ce dont la suite (carte du modele, mesure de latence) a besoin
        "onnx": onnx_path,  # chemin du modele ONNX
        "report": report,  # rapport de verification
        "outputs": [(o.name, o.shape) for o in session.get_outputs()],  # nom et forme de chaque sortie ONNX
        "net": net,  # reseau PyTorch (pour comparer les latences)
        "sample": torch.from_numpy(preprocess_for_onnx(samples[0], size)[0]),  # une entree d'exemple en tenseur
    }


def _first_tensor(output):  # renvoie le premier tenseur trouve dans une sortie (tenseur seul ou listes imbriquees)
    """Premier tenseur d'une sortie Ultralytics : detect rend un tenseur, segment un tuple (sortie, (..., prototypes))."""
    while isinstance(output, (list, tuple)):  # tant que c'est une liste ou un tuple
        output = output[0]  # on descend dans le premier element
    return output  # c'est un tenseur
