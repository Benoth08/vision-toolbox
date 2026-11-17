"""Moteur RF-DETR (Roboflow, Apache-2.0 pour nano / small / medium / base) : un DETR temps reel a dos DINOv2.

Pourquoi l'avoir a cote des YOLO : c'est un transformer de detection (DETR) sans ancres NI
suppression des non-maxima : chaque "requete" du decodeur predit au plus un objet, l'appariement
hongrois a l'entrainement interdit les doublons. Le dos DINOv2 (auto-supervise sur 142 millions
d'images) transfere tres bien vers des domaines eloignes de COCO, et la famille est sous licence
Apache-2.0, comme YOLOX (pas la contrainte AGPL d'Ultralytics).

Ce que la toolbox ajoute autour de la librairie `rfdetr` :
  - le jeu au format COCO de Roboflow (train / valid / test, `_annotations.coco.json`), ecrit
    depuis le pivot avec le meme pretraitement qu'en production ;
  - un post-traitement ecrit en numpy (sigmoide, top-k sur requetes x classes, boites cxcywh
    normalisees) que le programme de production reproduit sans PyTorch, verifie contre
    `model.predict` a l'export.

Entree du reseau : image RGB redimensionnee SANS conserver les proportions a resolution x resolution
(INTER_LINEAR), valeurs / 255, normalisation ImageNet. Sorties ONNX : `dets` (1, Q, 4) boites
cxcywh normalisees dans [0, 1], `labels` (1, Q, C + 1) logits (le dernier indice ne sert pas).

Installation : `pip install "rfdetr[train]"` (extra rfdetr). Poids COCO : telecharges par rfdetr
(stockage Roboflow) au premier usage, ou poses dans weights/ (tools/fetch_weights.py --rfdetr).
"""

from __future__ import annotations  # les annotations de type restent du texte : syntaxe moderne "X | None" partout

import json  # ecriture des fichiers d'annotations au format COCO (JSON)
import logging  # journal des messages (info, avertissements) au lieu de print
import shutil  # copie, deplacement et suppression de fichiers et dossiers
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import cv2  # OpenCV : redimensionnement d'images
import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)
import torch  # PyTorch : ici seulement pour savoir si l'on est sur CPU ou GPU

from vtb.annotate.schema import ImageAnnotation  # format pivot : toutes les etiquettes d'une image
from vtb.io.images import write_image  # ecrit une image sur disque
from vtb.preprocess import Pipeline  # suite d'operations OpenCV appliquee a chaque image
from vtb.tasks.detection.config import RFDETRConfig  # section rfdetr du YAML

logger = logging.getLogger(__name__)  # journal propre a ce module

SUBSETS = {"train": "train", "val": "valid", "test": "test"}  # noms de dossiers attendus par rfdetr
# resolution d'entree par defaut de chaque taille de modele (cote du carre en pixels)
DEFAULT_RESOLUTION = {"nano": 384, "small": 512, "medium": 576, "base": 560, "large": 704}


def model_class(size: str):  # renvoie la classe Python du modele RF-DETR de la taille demandee
    import rfdetr  # import tardif : la bibliotheque n'est chargee que si on utilise ce moteur

    return {  # table taille -> classe, puis on prend celle demandee
        "nano": rfdetr.RFDETRNano,  # le plus petit
        "small": rfdetr.RFDETRSmall,  # petit
        "medium": rfdetr.RFDETRMedium,  # moyen
        "base": rfdetr.RFDETRBase,  # base
        "large": rfdetr.RFDETRLarge,  # le plus gros
    }[size]  # on prend la classe correspondant a la taille demandee


def resolution_of(cfg: RFDETRConfig) -> int:  # renvoie la resolution d'entree effective
    return cfg.resolution or DEFAULT_RESOLUTION[cfg.size]  # celle du YAML si donnee, sinon celle de la taille


def _three_channels(image: np.ndarray) -> np.ndarray:  # renvoie l'image avec 3 canaux
    return image if image.ndim == 3 else np.stack([image] * 3, axis=-1)  # gris (H, W) -> (H, W, 3) en repetant le canal


def write_dataset(  # ecrit le jeu au format COCO de Roboflow ; renvoie le dossier du jeu
    annotations: dict[str, list[ImageAnnotation]],  # annotations par sous-ensemble (train / val / test)
    class_names: list[str],  # noms des classes, dans l'ordre des numeros
    pipeline: Pipeline,  # pretraitement a appliquer aux images
    color: str,  # "rgb" ou "gray"
    out_dir: Path,  # dossier ou ecrire le jeu
) -> Path:  # renvoie out_dir
    """train/ valid/ test/ : images APRES pretraitement + `_annotations.coco.json` (categories 0..C-1, boites a l'echelle)."""
    from vtb.io.images import read_image  # lit une image sur disque en tableau numpy

    if out_dir.exists():  # si le dossier existe deja (run precedent)
        shutil.rmtree(out_dir)  # jamais de vieilles images melangees aux nouvelles
    index = {n: i for i, n in enumerate(class_names)}  # nom de classe -> numero (categories 0..C-1)
    for subset, folder in SUBSETS.items():  # pour chaque sous-ensemble et son nom de dossier rfdetr
        target = out_dir / folder  # dossier de ce sous-ensemble
        target.mkdir(parents=True)  # le cree (avec les dossiers parents)
        images, objects = [], []  # listes COCO : fiches d'images et fiches d'objets
        for k, ann in enumerate(annotations.get(subset, [])):  # pour chaque image du sous-ensemble (k = numero)
            image = _three_channels(pipeline(read_image(ann.image_path, color=color)))  # lit, pretraite, 3 canaux
            sx, sy = image.shape[1] / ann.width, image.shape[0] / ann.height  # facteurs image pretraitee / image d'origine
            name = f"{k:06d}_{Path(ann.image_path).stem}.png"  # nom de fichier unique : numero sur 6 chiffres + nom d'origine
            write_image(target / name, image)  # ecrit l'image pretraitee
            images.append({"id": k, "file_name": name, "width": image.shape[1], "height": image.shape[0]})  # fiche COCO de l'image
            for b in ann.boxes:  # pour chaque boite de l'image
                w, h = (b.x2 - b.x1) * sx, (b.y2 - b.y1) * sy  # largeur et hauteur dans l'image pretraitee
                objects.append(  # ajoute la fiche COCO de l'objet
                    {  # fiche d'un objet au format COCO
                        "id": len(objects) + 1,  # identifiant unique de l'objet (a partir de 1)
                        "image_id": k,  # image a laquelle il appartient
                        "category_id": index[b.label],  # numero de sa classe
                        "bbox": [b.x1 * sx, b.y1 * sy, w, h],  # boite COCO : coin haut gauche, largeur, hauteur
                        "area": w * h,  # aire de la boite
                        "iscrowd": 0,  # 0 = objet individuel (pas une foule)
                    }
                )
        # liste des classes au format COCO ("defect" : categorie parente, purement descriptive)
        categories = [{"id": i, "name": n, "supercategory": "defect"} for i, n in enumerate(class_names)]
        (target / "_annotations.coco.json").write_text(  # ecrit le fichier d'annotations attendu par rfdetr
            json.dumps({"images": images, "annotations": objects, "categories": categories}),  # images, objets et classes dans un seul JSON
            encoding="utf-8",  # fichier texte en UTF-8
        )
    logger.info("jeu RF-DETR (COCO) ecrit sous %s", out_dir)  # indique ou le jeu a ete ecrit
    return out_dir  # renvoie le dossier du jeu


# entraine RF-DETR ; renvoie le chemin de best.pt
def train(cfg: RFDETRConfig, dataset_dir: Path, output: Path, device: torch.device, num_workers: int, seed: int) -> Path:
    """Entrainement par la librairie (EMA, augmentations multi-echelles, AdamW avec taux par couche) ; rend le meilleur point."""
    model = model_class(cfg.size)(pretrain_weights=cfg.weights, resolution=resolution_of(cfg))  # modele + poids COCO
    run_dir = output / "rfdetr_run"  # dossier de travail de la librairie
    model.train(  # lance l'entrainement complet (la librairie gere tout)
        dataset_dir=str(dataset_dir),  # le jeu au format COCO ecrit par write_dataset
        output_dir=str(run_dir),  # ou la librairie ecrit ses points de controle
        epochs=cfg.epochs,  # nombre maximal d'epoques
        batch_size=cfg.batch_size,  # images par lot
        grad_accum_steps=cfg.grad_accum_steps,  # lots cumules avant chaque pas d'optimisation
        lr=cfg.lr,  # taux d'apprentissage du decodeur
        lr_encoder=cfg.lr_encoder,  # taux d'apprentissage de l'encodeur DINOv2
        early_stopping=cfg.patience > 0,  # arret anticipe active si patience > 0
        early_stopping_patience=max(cfg.patience, 1),  # epoques sans progres avant l'arret (au moins 1)
        num_workers=num_workers,  # processus paralleles de lecture
        device="cuda" if device.type == "cuda" else "cpu",  # GPU ou CPU
        augmentation_backend="torchvision",  # "cpu" peut se resoudre en kornia (GPU seulement) selon la version
        tensorboard=False,  # pas de journal TensorBoard
        progress_bar=None,  # pas de barre de progression
        seed=seed,  # graine : resultats reproductibles
        **cfg.extra,  # tout argument supplementaire ecrit dans le YAML
    )
    best = next(  # premier fichier de meilleur modele qui existe, sinon None
        (  # generateur : parcourt les noms candidats
            run_dir / n  # chemin candidat
            # par ordre de preference : meilleur global, meilleur EMA, meilleur sans EMA
            for n in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint_best_regular.pth")
            if (run_dir / n).exists()  # seulement s'il existe sur le disque
        ),
        None,  # valeur si aucun n'existe
    )
    if best is None:  # si la librairie n'a ecrit aucun point de controle
        raise FileNotFoundError(f"aucun point de controle RF-DETR dans {run_dir}")  # erreur claire
    shutil.copy2(best, output / "best.pt")  # copie le meilleur modele dans le dossier de l'experience
    return output / "best.pt"  # renvoie son chemin


def load(output: Path):  # recharge le modele entraine ; renvoie l'objet RF-DETR
    from rfdetr import RFDETR  # classe generique capable de relire un point de controle

    return RFDETR.from_checkpoint(str(output / "best.pt"), trust_checkpoint=True)  # notre propre point de controle


# predit sur des images en memoire ; renvoie un tableau (n, 6) par image
def predict_images(model, images: list[np.ndarray], num_classes: int, conf: float) -> list[np.ndarray]:
    """Images pipeline (RGB ou gris) -> (n, 6) [x1 y1 x2 y2 score classe] dans leur repere (pas de NMS : DETR)."""
    out = []  # resultats, un par image
    for image in images:  # pour chaque image
        detections = model.predict(_three_channels(image), threshold=conf)  # inference de la librairie
        # coins, confiances et classes en un tableau (n, 6) float32
        rows = np.column_stack([detections.xyxy, detections.confidence, detections.class_id]).astype(np.float32).reshape(-1, 6)
        out.append(rows[rows[:, 5] < num_classes])  # retire les classes hors de nos classes (indice "fond" inutilise)
    return out  # renvoie toutes les detections


# --- export de production ----------------------------------------------------------------------


# pretraitement de production RF-DETR ; renvoie le tenseur d'entree (1, 3, R, R)
def preprocess_for_onnx(image_rgb: np.ndarray, resolution: int, mean: list[float], std: list[float]) -> np.ndarray:
    """RGB -> [0, 1] -> redimensionnement sans proportions (INTER_LINEAR) -> normalisation -> (1, 3, R, R)."""
    # 3 canaux, float, valeurs ramenees dans [0, 1] (divise par 65535 pour une image 16 bits, 255 sinon)
    x = _three_channels(image_rgb).astype(np.float32) / (65535.0 if image_rgb.dtype == np.uint16 else 255.0)
    x = cv2.resize(x, (resolution, resolution), interpolation=cv2.INTER_LINEAR)  # carre R x R, proportions non gardees
    x = (x - np.array(mean, np.float32)) / np.array(std, np.float32)  # normalisation ImageNet canal par canal
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)  # (H, W, C) -> (1, C, H, W), contigu


def postprocess_onnx(  # transforme les sorties ONNX d'une image en boites finales ; renvoie un tableau (n, 6)
    dets: np.ndarray,  # boites (Q, 4) cx cy w h normalisees d'une image
    logits: np.ndarray,  # logits (Q, C + 1) : scores bruts par requete et par classe
    width: int,  # largeur de l'image donnee au pretraitement
    height: int,  # hauteur de cette image
    num_classes: int,  # nombre de vraies classes (le dernier indice ne sert pas)
    conf: float,  # seuil de probabilite
    top_k: int = 300,  # nombre maximal de couples (requete, classe) gardes
) -> np.ndarray:  # renvoie (n, 6) [x1, y1, x2, y2, score, classe]
    """Sorties ONNX d'UNE image -> (n, 6) dans l'image donnee au pretraitement.

    sigmoide de chaque logit (requete x classe), les top_k meilleurs couples, classe = indice modulo
    (C + 1), requete = indice divise par (C + 1), boite cxcywh normalisee -> coins x (largeur, hauteur).
    Pas de NMS : l'appariement hongrois de l'entrainement a deja appris a ne pas dupliquer.
    """
    prob = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))  # sigmoide : logit (score brut) -> probabilite entre 0 et 1
    flat = prob.ravel()  # tableau (requetes, classes) aplati en une seule ligne de couples
    order = np.argsort(-flat, kind="stable")[:top_k]  # indices des top_k plus grandes probabilites (tri stable)
    order = order[flat[order] >= conf]  # garde ceux au-dessus du seuil
    labels, queries = order % prob.shape[1], order // prob.shape[1]  # indice aplati -> (classe, requete)
    keep = labels < num_classes  # retire le dernier indice de classe (ne sert pas)
    order, labels, queries = order[keep], labels[keep], queries[keep]  # applique ce filtre
    b = dets[queries]  # boites (cx, cy, w, h) normalisees des requetes retenues
    # centre-taille -> coins (x1, y1, x2, y2), toujours normalises dans [0, 1]
    corners = np.column_stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2])
    # en pixels de l'image (x par la largeur, y par la hauteur), bornes dans l'image
    corners = (corners * np.array([width, height, width, height])).clip(0, [width, height, width, height])
    return np.column_stack([corners, flat[order], labels]).astype(np.float32).reshape(-1, 6)  # coins, score, classe


# compare nos detections a celles de reference ; renvoie un dict de mesures
def compare_detections(mine: list[np.ndarray], reference: list[np.ndarray], tie: float = 1e-4) -> dict:
    """Comparaison robuste aux ex aequo de deux listes de detections (ONNX + numpy contre model.predict).

    1. les scores tries doivent coincider (insensible a l'ordre des ex aequo) ;
    2. l'IoU n'est mesure que sur les boites dont le score se distingue de tous les autres de plus
       de `tie` : entre deux requetes de meme score, l'ordre (donc la boite retenue au rang 300)
       depend d'un arrondi et ne signale pas un defaut d'export. Un modele peu entraine a
       presque tous ses scores egaux : la verification repose alors sur les scores.
    """
    from vtb.tasks.detection.common import box_iou  # IoU de deux boites (intersection / union)

    gaps, ious, n_mine, n_ref = [0.0], [], 0, 0  # ecarts de scores, IoU, nombres de boites de chaque cote
    for rows, ref in zip(mine, reference):  # pour chaque image : nos boites et celles de reference
        n_mine, n_ref = n_mine + len(rows), n_ref + len(ref)  # compte les boites
        a, b = np.sort(rows[:, 4])[::-1], np.sort(ref[:, 4])[::-1]  # scores tries du plus grand au plus petit
        n = min(len(a), len(b))  # nombre de scores comparables
        if n:  # s'il y en a au moins un
            gaps.append(float(np.abs(a[:n] - b[:n]).max()))  # plus grand ecart de score rang par rang
        for k, row in enumerate(rows):  # pour chacune de nos boites
            others = np.delete(rows[:, 4], k)  # scores de toutes les autres boites de l'image
            # seulement si une reference existe et que ce score est unique (pas d'ex aequo a tie pres)
            if len(ref) and (len(others) == 0 or np.abs(others - row[4]).min() > tie):
                # boite identique (a 0,5 px) = accord parfait, meme degeneree : un modele peu entraine sort des boites
                # ecrasees sur le bord de l'image (aire nulle, IoU indefini)
                ious.append(max(1.0 if np.allclose(row[:4], r[:4], atol=0.5) else box_iou(row[:4], r[:4]) for r in ref))
    return {  # resume chiffre
        "boxes": n_mine,  # nombre de nos boites
        "reference_boxes": n_ref,  # nombre de boites de reference
        "max_score_gap": max(gaps),  # pire ecart de score
        "distinct_boxes": len(ious),  # boites a score unique comparees par IoU
        "mean_iou_distinct": float(np.mean(ious)) if ious else None,  # IoU moyen de ces boites (None si aucune)
    }


# exporte en ONNX et verifie ; renvoie un dict (rapport, resolution, normalisation, sorties)
def export(model, samples: list[np.ndarray], output: Path, num_classes: int, conf: float, opset: int) -> dict:
    """ONNX par la librairie (lot fixe de 1), puis verification : ONNX + postprocess_onnx == model.predict (300 boites par image)."""
    import onnxruntime as ort  # onnxruntime : execute un modele ONNX sans PyTorch (comme en production)

    resolution = model.model_config.resolution  # resolution d'entree du modele entraine
    exported = Path(  # chemin du fichier ONNX ecrit par la librairie
        model.export(  # export ONNX par rfdetr
            output_dir=str(output / "rfdetr_export"),  # dossier de travail de l'export
            shape=(resolution, resolution),  # entree carree resolution x resolution
            opset_version=max(17, min(opset, 18)),  # version des operations ONNX, bornee entre 17 et 18
            verbose=False,  # sans messages
        )
    )
    onnx_path = output / "model.onnx"  # emplacement voulu dans le dossier de l'experience
    shutil.move(str(exported), onnx_path)  # deplace le fichier a cet emplacement
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])  # ouvre le modele ONNX sur CPU
    mean, std = [float(v) for v in model.means], [float(v) for v in model.stds]  # normalisation utilisee par le modele
    conf = 0.0  # DETR rend toujours ses 300 meilleurs couples : on les compare tous, meme pour un modele peu entraine
    mine, reference = [], []  # detections ONNX + numpy, detections de la librairie
    for image in samples:  # pour chaque image d'exemple
        # inference ONNX : boites et logits
        dets, logits = session.run(None, {session.get_inputs()[0].name: preprocess_for_onnx(image, resolution, mean, std)})
        mine.append(postprocess_onnx(dets[0], logits[0], image.shape[1], image.shape[0], num_classes, conf))  # nos boites
        reference.append(predict_images(model, [image], num_classes, conf)[0])  # boites de reference (model.predict)
    report = compare_detections(mine, reference)  # compare les deux chaines
    logger.info("verification ONNX RF-DETR : %s", report)  # affiche le rapport
    # refuse si les scores different ou si les boites a score unique ne se recouvrent pas assez
    if report["max_score_gap"] > 1e-3 or (report["mean_iou_distinct"] is not None and report["mean_iou_distinct"] < 0.98):
        raise RuntimeError(f"le post-traitement ONNX ne reproduit pas RF-DETR : {report}")  # message avec le rapport
    return {  # informations pour la carte du modele
        "report": report,  # rapport de verification
        "resolution": resolution,  # cote de l'entree
        "mean": mean,  # moyenne de normalisation
        "std": std,  # ecart-type de normalisation
        "outputs": [(o.name, o.shape) for o in session.get_outputs()],  # nom et forme de chaque sortie ONNX
    }


# mesure la latence de chaque moteur ; renvoie un dict nom -> millisecondes
def measure_latency(model, onnx_path: Path, image: np.ndarray, runs: int, openvino_xml: Path | None = None) -> dict[str, float]:
    """Millisecondes par image : model.predict (PyTorch, pretraitement compris), onnxruntime et OpenVINO (reseau seul)."""
    import time  # chronometrage

    import onnxruntime as ort  # moteur ONNX

    def timed(fn) -> float:  # execute fn plusieurs fois ; renvoie la duree moyenne en millisecondes
        for _ in range(2):  # deux passages de chauffe (non mesures : caches, allocations)
            fn()  # appel de chauffe
        start = time.perf_counter()  # heure de depart (horloge precise)
        for _ in range(runs):  # runs passages mesures
            fn()  # appel mesure
        return round(1000 * (time.perf_counter() - start) / runs, 2)  # duree moyenne en ms, 2 decimales

    resolution = model.model_config.resolution  # resolution d'entree
    # entree ONNX preparee une fois (on ne mesure que le reseau)
    x = preprocess_for_onnx(image, resolution, [float(v) for v in model.means], [float(v) for v in model.stds])
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])  # ouvre le modele ONNX
    name = session.get_inputs()[0].name  # nom de l'entree
    # PyTorch via la librairie (pretraitement compris), lot d'une image
    result = {"torch_cpu_ms": timed(lambda: model.predict(_three_channels(image), threshold=0.5)), "batch": 1}
    result["onnxruntime_cpu_ms"] = timed(lambda: session.run(None, {name: x}))  # onnxruntime, reseau seul
    if openvino_xml is not None:  # si une version OpenVINO existe
        import openvino as ov  # OpenVINO : moteur d'Intel, rapide sur CPU Intel

        compiled = ov.Core().compile_model(str(openvino_xml), "CPU")  # charge et compile le modele pour le CPU
        result["openvino_cpu_ms"] = timed(lambda: compiled(x))  # latence OpenVINO
    logger.info("latence : %s", result)  # affiche les latences
    return result  # renvoie les latences
