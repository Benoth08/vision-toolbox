"""Prediction a partir de model.onnx + model_card.json SEULEMENT : numpy, OpenCV et onnxruntime, sans PyTorch.

C'est la reference Python de ce que le programme de production (C#, C++) doit faire, etape par
etape : si ce module donne les memes reponses que le workflow PyTorch, la carte du modele est
complete et le programme de production peut s'ecrire en la lisant.

    predictor = OnnxPredictor("outputs/classification/resnet18_partial")
    result = predictor(read_image("piece.png", color=predictor.color))   # dict pret pour du JSON

Etapes communes : Pipeline OpenCV de la carte -> mise a l'echelle de l'entree (input.resize) ->
couleur (RGB, GRAY, BGR) -> valeurs (division par 255 / 65535, ou rien) -> mean / std -> NCHW ->
onnxruntime -> post-traitement propre a la tache (et au decodeur, pour la detection).
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[str], X | None) partout

from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import cv2  # OpenCV : redimensionnement d'image
import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)

from vtb.export.card import ModelCard, read_card  # carte du modele : description de l'entree, des sorties, des seuils
from vtb.io.images import max_value  # valeur maximale d'un pixel selon son type (255, 65535 ou 1)
from vtb.preprocess import Pipeline  # pretraitement OpenCV decrit dans la carte (le meme qu'a l'entrainement)

# letterbox = reduire l'image sans la deformer et completer par des bandes grises jusqu'au carre attendu
from vtb.preprocess.ops.geometry import letterbox, letterbox_params


# NMS (suppression des non-maxima) : parmi des boites qui se recouvrent, garde la plus sure ; renvoie les indices gardes
def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    """Suppression des non-maxima en numpy (indices gardes, par score decroissant) : 15 lignes a porter en C#."""
    order = np.argsort(-scores)  # indices des boites, du score le plus haut au plus bas
    keep = []  # indices des boites gardees
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # aire de chaque boite (largeur x hauteur)
    while len(order):  # tant qu'il reste des boites a examiner
        i = order[0]  # la boite restante la plus sure
        keep.append(i)  # on la garde
        # coin haut gauche de l'intersection avec chacune des autres boites restantes
        xx1, yy1 = np.maximum(boxes[i, 0], boxes[order[1:], 0]), np.maximum(boxes[i, 1], boxes[order[1:], 1])
        # coin bas droit de l'intersection
        xx2, yy2 = np.minimum(boxes[i, 2], boxes[order[1:], 2]), np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)  # aire d'intersection (0 si disjointes)
        overlap = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)  # IoU (intersection sur union) ; 1e-9 evite /0
        order = order[1:][overlap <= iou]  # retire la boite gardee et celles qui la recouvrent trop (doublons)
    return np.array(keep, dtype=np.int64)  # indices gardes, en tableau d'entiers


def nms_per_class(rows: np.ndarray, iou: float) -> np.ndarray:  # NMS separe pour chaque classe ; renvoie les lignes gardees
    """(n, 6) [x1 y1 x2 y2 score classe] : NMS classe par classe (deux classes ne s'eliminent pas)."""
    # pour chaque classe c presente : lignes de cette classe, filtrees par la NMS sur leurs boites (:4) et scores (4)
    kept = [rows[rows[:, 5] == c][nms(rows[rows[:, 5] == c, :4], rows[rows[:, 5] == c, 4], iou)] for c in np.unique(rows[:, 5])]
    out = np.concatenate(kept) if kept else np.zeros((0, 6), np.float32)  # recolle les classes ; aucune ligne = tableau vide
    return out[np.argsort(-out[:, 4])]  # trie toutes les detections par score decroissant


# decode la sortie brute de YOLOX (positions relatives a une grille) ; renvoie cx, cy, w, h en pixels + scores
def yolox_decode(raw: np.ndarray, size: int, strides: tuple[int, ...] = (8, 16, 32)) -> np.ndarray:
    """Sortie YOLOX non decodee (A, 5 + C) -> cx, cy, w, h en pixels de l'entree (comme demo_postprocess officiel)."""
    grids, expanded = [], []  # coordonnees des cases de grille ; pas (stride) de chaque case
    for stride in strides:  # pour chaque echelle (case de 8, 16 puis 32 pixels)
        n = size // stride  # nombre de cases par cote a cette echelle
        yv, xv = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")  # numeros de ligne et de colonne de chaque case
        grids.append(np.stack((xv, yv), 2).reshape(-1, 2))  # liste des cases (x, y), une par ligne
        expanded.append(np.full((n * n, 1), stride))  # pas de chaque case (meme valeur pour toute l'echelle)
    grid, stride = np.concatenate(grids).astype(np.float32), np.concatenate(expanded).astype(np.float32)  # toutes echelles
    out = raw.copy()  # copie : la sortie brute reste intacte
    out[:, :2] = (out[:, :2] + grid) * stride  # centre = (decalage predit + position de la case) x taille de case
    out[:, 2:4] = np.exp(out[:, 2:4]) * stride  # largeur, hauteur = exp(valeur predite) x taille de case
    return out  # sortie decodee, en pixels de l'entree du reseau


class OnnxPredictor:  # predicteur complet : pretraitement, execution ONNX, post-traitement selon la tache
    """Un modele exporte par la toolbox (dossier contenant model.onnx et model_card.json)."""

    def __init__(self, model_dir: str | Path, providers: list[str] | None = None) -> None:  # charge la carte et le modele
        import onnxruntime as ort  # onnxruntime : moteur qui execute le modele ONNX (import a la demande)

        self.model_dir = Path(model_dir)  # dossier du modele exporte
        self.card: ModelCard = read_card(self.model_dir / "model_card.json")  # lit et valide la carte du modele
        self.session = ort.InferenceSession(  # charge le modele ONNX
            str(self.model_dir / self.card.files.get("onnx", "model.onnx")),  # fichier indique par la carte (model.onnx par defaut)
            providers=providers or ["CPUExecutionProvider"],  # materiel d'execution ; CPU par defaut
        )
        self.input_name = self.session.get_inputs()[0].name  # nom de l'entree du reseau
        self.output_names = [o.name for o in self.session.get_outputs()]  # noms des sorties, dans l'ordre
        self.pipeline = Pipeline(self.card.preprocess)  # pretraitement OpenCV decrit dans la carte
        # couleur de LECTURE du fichier (avant le pretraitement) ; cartes anciennes : deduite de l'entree du reseau
        self.color = self.card.input.read_color or ("gray" if self.card.input.color == "GRAY" else "rgb")
        # detection : moteur d'origine (ultralytics, yolox...) = debut du nom avant ":" ; autres taches : ""
        self.decoder = self.card.model_name.split(":")[0] if self.card.task == "detection" else ""

    # --- entree ---------------------------------------------------------------------------------

    # prepare une image pour le reseau ; renvoie (tenseur 1 x C x H x W, infos pour revenir a l'image d'origine)
    def prepare(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        """Image lue (RGB ou gris) -> tenseur NCHW float32 + ce qu'il faut pour revenir aux coordonnees d'origine."""
        spec = self.card.input  # description de l'entree attendue
        processed = self.pipeline(image)  # applique le pretraitement de l'entrainement
        # tailles avant / apres pretraitement, facteur d'echelle et decalage du letterbox (neutres par defaut)
        meta = {"original_hw": image.shape[:2], "processed_hw": processed.shape[:2], "ratio": 1.0, "pad": (0, 0)}
        x = processed  # image de travail
        if spec.channels == 3 and x.ndim == 2:  # le reseau attend 3 canaux mais l'image est grise
            x = np.stack([x] * 3, axis=-1)  # recopie le gris sur 3 canaux
        if spec.resize == "letterbox":  # letterbox centre (YOLO Ultralytics)
            ratio, _, _, pad_x, pad_y = letterbox_params(x.shape[0], x.shape[1], spec.height)  # echelle et decalages
            x = letterbox(x, size=spec.height, pad_value=spec.letterbox_pad_value or 114)  # image carree, bandes grises
            meta.update(ratio=ratio, pad=(pad_x, pad_y))  # memorise pour corriger les boites predites
        elif spec.resize == "letterbox_top_left":  # YOLOX : colle en haut a gauche, pas de decalage
            ratio = min(spec.height / x.shape[0], spec.width / x.shape[1])  # facteur pour tenir dans le cadre sans deformer
            # redimensionne en gardant les proportions (INTER_LINEAR : interpolation lissee)
            resized = cv2.resize(x, (int(x.shape[1] * ratio), int(x.shape[0] * ratio)), interpolation=cv2.INTER_LINEAR)
            canvas = np.full((spec.height, spec.width, 3), spec.letterbox_pad_value or 114, dtype=x.dtype)  # cadre gris (114)
            canvas[: resized.shape[0], : resized.shape[1]] = resized  # colle l'image dans le coin haut gauche
            x = canvas  # l'image de travail devient le cadre complet
            meta.update(ratio=ratio)  # memorise l'echelle (pas de decalage)
        if spec.color == "BGR":  # le reseau attend l'ordre bleu, vert, rouge
            x = x[..., ::-1]  # inverse l'ordre des canaux (RGB -> BGR)
        values = x.astype(np.float32)  # passe en flottants pour les calculs
        if not spec.scale.startswith("none"):  # sauf si la carte dit "pas de mise a l'echelle"
            values /= max_value(x)  # ramene les pixels entre 0 et 1 (divise par 255 ou 65535)
        if spec.resize == "stretch" and values.shape[:2] != (spec.height, spec.width):  # RF-DETR : taille fixe sans proportions
            values = cv2.resize(values, (spec.width, spec.height), interpolation=cv2.INTER_LINEAR)  # etire a la taille attendue
        if values.ndim == 2:  # image grise (H, W)
            values = values[None]  # ajoute l'axe des canaux : (1, H, W)
        else:  # image couleur (H, W, C)
            values = values.transpose(2, 0, 1)  # reordonne en (C, H, W) : canaux d'abord, comme PyTorch
        if spec.mean:  # normalisation demandee
            # (valeur - moyenne) / ecart-type, canal par canal ; [:, None, None] aligne un nombre par canal sur l'image
            values = (values - np.array(spec.mean, np.float32)[:, None, None]) / np.array(spec.std, np.float32)[:, None, None]
        # values[None] ajoute l'axe du lot (1, C, H, W) ; tableau contigu en memoire, comme l'exige onnxruntime
        return np.ascontiguousarray(values[None], dtype=np.float32), meta

    # --- sortie ---------------------------------------------------------------------------------

    def __call__(self, image: np.ndarray) -> dict:  # permet d'ecrire predictor(image) ; renvoie un dict pret pour du JSON
        tensor, meta = self.prepare(image)  # image -> tenseur d'entree
        outputs = dict(zip(self.output_names, self.session.run(None, {self.input_name: tensor})))  # execute ; nom -> sortie
        return getattr(self, f"_{self.card.task}")(outputs, meta)  # appelle _classification, _detection... selon la tache

    def _classification(self, outputs: dict, meta: dict) -> dict:  # renvoie la classe predite et les probabilites
        logits = outputs[self.output_names[0]][0].astype(np.float64)  # logits = scores bruts du reseau, un par classe (image 0)
        probs = np.exp(logits - logits.max())  # softmax (scores -> probabilites) ; retirer le max evite un exp trop grand
        probs /= probs.sum()  # divise par la somme : les probabilites somment a 1
        k = int(probs.argmax())  # indice de la classe la plus probable
        return {  # resultat
            "label": self.card.class_names[k],  # nom de la classe predite
            "probability": float(probs[k]),  # sa probabilite
            "probabilities": dict(zip(self.card.class_names, probs.round(6).tolist())),  # toutes les classes, 6 decimales
        }

    def _anomaly(self, outputs: dict, meta: dict) -> dict:  # renvoie le score d'anomalie et la decision
        score = float(outputs["score"][0])  # score d'anomalie de l'image (grand = anormal)
        threshold = self.card.thresholds["image"]  # seuil de decision choisi sur la validation
        result = {"score": score, "anomaly": score > threshold, "threshold": threshold}  # anomalie si score > seuil
        if "pixel" in self.card.thresholds:  # si un seuil par pixel existe
            # nombre de pixels de la carte d'anomalie au-dessus du seuil pixel ([0, 0] = image 0, canal 0)
            result["anomalous_pixels"] = int((outputs["anomaly_map"][0, 0] > self.card.thresholds["pixel"]).sum())
        return result  # renvoie le resultat

    def _segmentation(self, outputs: dict, meta: dict) -> dict:  # renvoie la surface de defaut et le masque des classes
        probs = outputs["probabilities"][0]  # probabilites par classe et par pixel (C, H, W), image 0
        # 2 classes (fond / defaut) : seuil sur la proba de defaut ; plus de classes : classe la plus probable par pixel
        classes = (probs[1] > self.card.thresholds["defect_probability"]).astype(np.int64) if len(probs) == 2 else probs.argmax(0)
        area = int((classes > 0).sum())  # nombre de pixels de defaut (toute classe autre que le fond 0)
        return {  # resultat
            "defect_pixels": area,  # surface totale de defaut, en pixels
            "flagged": area >= self.card.thresholds.get("min_defect_area", 0),  # piece signalee si surface >= minimum
            # surface par classe de defaut (k > 0 : on saute le fond)
            "pixels_per_class": {n: int((classes == k).sum()) for k, n in enumerate(self.card.class_names) if k > 0},
            "mask": classes,  # retire par le service JSON ; utile en Python
        }

    def _detection(self, outputs: dict, meta: dict) -> dict:  # renvoie la liste des objets detectes
        conf, iou = self.card.thresholds["conf"], self.card.thresholds.get("iou_nms", 0.6)  # seuil de confiance ; seuil NMS
        first = outputs[self.output_names[0]]  # premiere sortie du reseau
        if self.decoder == "ultralytics":  # YOLO d'Ultralytics
            pred = first[0].T  # (A, 4 + C)
            scores, classes = pred[:, 4:].max(1), pred[:, 4:].argmax(1)  # meilleur score de classe et sa classe, par ancre
            keep = scores > conf  # ancres assez sures
            c = pred[keep]  # predictions gardees
            rows = np.column_stack(  # assemble les colonnes x1, y1, x2, y2, score, classe
                # centre +/- demi-taille -> coins ; puis score et classe
                [c[:, 0] - c[:, 2] / 2, c[:, 1] - c[:, 3] / 2, c[:, 0] + c[:, 2] / 2, c[:, 1] + c[:, 3] / 2, scores[keep], classes[keep]]
            )
            rows = nms_per_class(rows.astype(np.float32), iou) if len(rows) else np.zeros((0, 6), np.float32)  # retire doublons
            rows[:, [0, 2]] = (rows[:, [0, 2]] - meta["pad"][0]) / meta["ratio"]  # annule le letterbox en x
            rows[:, [1, 3]] = (rows[:, [1, 3]] - meta["pad"][1]) / meta["ratio"]  # annule le letterbox en y
        elif self.decoder == "yolox":  # YOLOX
            decoded = yolox_decode(first[0], self.card.input.height)  # positions en pixels de l'entree
            probs = decoded[:, 5:]  # probabilites de classe
            classes = probs.argmax(1)  # classe la plus probable par ancre
            scores = decoded[:, 4] * probs[np.arange(len(classes)), classes]  # score = "objet present" x proba de la classe
            keep = scores >= conf  # ancres assez sures
            c = decoded[keep]  # predictions gardees
            rows = np.column_stack(  # assemble les colonnes x1, y1, x2, y2, score, classe
                # centre +/- demi-taille -> coins ; puis score et classe
                [c[:, 0] - c[:, 2] / 2, c[:, 1] - c[:, 3] / 2, c[:, 0] + c[:, 2] / 2, c[:, 1] + c[:, 3] / 2, scores[keep], classes[keep]]
            )
            rows = nms_per_class(rows.astype(np.float32), iou) if len(rows) else np.zeros((0, 6), np.float32)  # retire doublons
            rows[:, :4] /= meta["ratio"]  # annule la mise a l'echelle (letterbox en haut a gauche : pas de decalage)
        elif self.decoder == "rfdetr":  # DETR : top-k sur (requete, classe), pas de NMS
            ph, pw = meta["processed_hw"]  # taille de l'image pretraitee (les boites RF-DETR sont relatives, entre 0 et 1)
            logits = outputs["labels"][0].astype(np.float64)  # scores bruts (requetes x classes)
            flat = (1.0 / (1.0 + np.exp(-logits))).ravel()  # sigmoide (score -> probabilite 0-1), mise a plat
            order = np.argsort(-flat, kind="stable")[:300]  # les 300 couples (requete, classe) les plus probables
            order = order[flat[order] >= conf]  # garde ceux au-dessus du seuil
            labels, queries = order % logits.shape[1], order // logits.shape[1]  # position a plat -> (requete, classe)
            keep = labels < len(self.card.class_names)  # ignore les classes en trop (au-dela de nos classes)
            order, labels, queries = order[keep], labels[keep], queries[keep]  # applique ce filtre aux trois tableaux
            b = outputs["dets"][0][queries]  # boites des requetes retenues (cx, cy, w, h relatifs)
            rows = np.column_stack(  # assemble les colonnes x1, y1, x2, y2, score, classe
                [  # liste des colonnes
                    (b[:, 0] - b[:, 2] / 2) * pw,  # x1 en pixels
                    (b[:, 1] - b[:, 3] / 2) * ph,  # y1 en pixels
                    (b[:, 0] + b[:, 2] / 2) * pw,  # x2 en pixels
                    (b[:, 1] + b[:, 3] / 2) * ph,  # y2 en pixels
                    flat[order],  # score
                    labels,  # classe
                ]
            )
            rows = rows.astype(np.float32).reshape(-1, 6)  # flottants 32 bits, forme (n, 6) meme si n = 0
        elif self.decoder == "fasterrcnn":  # Faster R-CNN de torchvision (sorties deja decodees et filtrees par NMS)
            # boites (n, 4), scores (n,) et classes (n,) cote a cote : tableau (n, 6)
            rows = np.column_stack([outputs["boxes"], outputs["scores"], outputs["labels"]]).astype(np.float32).reshape(-1, 6)
            rows = rows[rows[:, 4] >= conf]  # garde les detections assez sures
        else:  # moteur non prevu
            raise NotImplementedError(f"decodeur de detection inconnu : {self.card.model_name}")  # arret avec message clair
        ph, pw = meta["processed_hw"]  # taille apres pretraitement
        oh, ow = meta["original_hw"]  # taille d'origine
        rows[:, [0, 2]] = (rows[:, [0, 2]] * ow / pw).clip(0, ow)  # repere de l'image pretraitee -> image d'origine
        rows[:, [1, 3]] = (rows[:, [1, 3]] * oh / ph).clip(0, oh)  # meme chose en y, borne a l'image
        objects = [  # une entree par objet detecte
            # classe (nom), score a 4 decimales, boite [x1, y1, x2, y2] a 1 decimale
            {"label": self.card.class_names[int(r[5])], "score": round(float(r[4]), 4), "box": [round(float(v), 1) for v in r[:4]]}
            for r in rows  # pour chaque ligne gardee
        ]
        return {"objects": objects, "count": len(objects)}  # liste des objets et leur nombre
