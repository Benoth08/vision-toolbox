"""Les cinq etapes de la detection, communes aux quatre moteurs (ultralytics, yolox, fasterrcnn, rfdetr).

prepare(config)          annotations (VOC, YOLO, COCO, LabelMe, masques...) -> pivot split.vtb.json + split.csv
train(config)            moteur choisi -> best.pt + history.csv + detector.json
evaluate(config)         test, une seule fois, par le MEME chemin que la production -> mAP COCO + point de fonctionnement
predict(config, source)  dossier d'images -> CSV, une ligne par objet (+ images annotees)
export(config)           ONNX verifie (sorties brutes ET post-traitement) + model_card.json (+ OpenVINO)

Le moteur change ; le decoupage, les annotations, l'evaluateur et la carte du modele ne changent
pas : les moteurs se comparent honnetement sur les memes images.
"""

from __future__ import annotations  # les annotations de type restent du texte : syntaxe moderne "X | None" partout

import json  # lecture / ecriture de fichiers JSON (detector.json, metriques)
import logging  # journal des messages (info, avertissements) au lieu de print
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)
import pandas as pd  # pandas : tableaux de donnees (catalogue, historique, predictions)
import torch  # PyTorch : tenseurs (tableaux de nombres, sur CPU ou GPU) ; ici surtout torch.device

# format pivot des annotations : fiche d'une image, liste des classes d'un jeu, ecriture du fichier pivot
from vtb.annotate.schema import ImageAnnotation, class_names_of, save_annotations
from vtb.core.runtime import timer  # chronometre un bloc de code

# mAP COCO (precision moyenne sur toute la courbe) ; point de fonctionnement (precision / rappel a un seuil donne)
from vtb.evaluate.detection import coco_metrics, operating_point

# carte du modele (model_card.json) : fiche qui decrit entree, sorties, classes, pour l'utiliser hors Python
from vtb.export.card import InputSpec, ModelCard, OutputSpec, framework_versions, write_card

# lecture des annotations, ajout des colonnes groupe et date, sous-echantillon pour un essai rapide
from vtb.io.catalogs import add_groups, add_timestamps, limit_catalog, read_data_annotations
from vtb.io.images import list_images, read_image, write_image  # lister, lire et ecrire des images
from vtb.preprocess import Pipeline  # suite d'operations OpenCV appliquee a chaque image
from vtb.split import check_no_leakage, make_split, save_split  # decoupage train / val / test et controle de fuite
from vtb.tasks.detection.common import (  # outils communs aux quatre moteurs
    annotations_for,  # annotations d'un sous-ensemble (train, val ou test)
    boxes_from_array,  # tableau (n, 6) -> boites du format pivot
    boxes_to_array,  # boites du format pivot -> tableau (n, 6)
    class_names_for,  # liste des classes lue dans le fichier pivot
    ensure_boxes,  # ajoute des boites droites aux images qui n'ont que des polygones / boites orientees
    predict_tiled,  # prediction tuile par tuile sur une grande image
    rotated_to_box,  # boite orientee -> boite droite englobante
    scale_annotation,  # ramene des predictions dans l'image d'origine
)
from vtb.tasks.detection.config import DetectionConfig  # le YAML de l'experience de detection, valide
from vtb.tasks.detection.ultralytics_backend import VERIFY_CONF  # seuil de confiance bas utilise pour verifier les exports

logger = logging.getLogger(__name__)  # journal propre a ce module


def pivot_path(split_file: str | Path) -> Path:  # renvoie le chemin du fichier d'annotations pivot
    """Les annotations pivot vivent a cote du split : split.csv -> split.vtb.json."""
    return Path(split_file).with_suffix(".vtb.json")  # meme nom, autre extension


# --- etape 1 : annotations + decoupage -------------------------------------------------------


def prepare(config: DetectionConfig) -> pd.DataFrame:  # lit, decoupe, ecrit ; renvoie le catalogue decoupe
    """Lit les annotations (n'importe quel format de vtb.annotate), decoupe PAR IMAGE, ecrit split.csv et le pivot.

    Stratification sur la classe principale de l'image (la plus frequente) ; group_regex garde les
    images d'une meme bobine / piece du meme cote. Les images sans objet sont gardees (classe
    "background") : elles apprennent au detecteur a se taire.
    """
    annotations = read_data_annotations(config.data)  # lit toutes les annotations, quel que soit leur format
    if not annotations:  # si rien n'a ete lu
        # erreur claire : dossier et format a verifier
        raise FileNotFoundError(f"aucune annotation lue sous {config.data.root} (format {config.data.annotation_format})")
    class_names = class_names_of(annotations)  # liste triee de toutes les classes presentes
    rows = [  # une ligne de catalogue par image
        # chemin, classe principale (pour stratifier), nombre d'objets, taille de l'image
        {"path": a.image_path, "label": a.main_label(), "n_objects": len(a.object_labels()), "width": a.width, "height": a.height}
        for a in annotations  # pour chaque image annotee
    ]
    # tableau du catalogue, eventuellement reduit (essai rapide), avec la colonne groupe (piece, bobine...)
    catalog = add_groups(limit_catalog(pd.DataFrame(rows), config.data.max_images), config.data.group_regex, config.data.groups_csv)
    if config.split.strategy == "temporal" or config.data.timestamp_regex:  # si on a besoin des dates
        catalog = add_timestamps(catalog, config.data.timestamp_regex)  # ajoute la colonne date
    split = make_split(catalog, config.split, config.runtime.seed)  # decoupe train / val / test (graine fixe)
    check_no_leakage(split)  # verifie qu'aucune image (ou groupe) n'est dans deux sous-ensembles
    save_split(split, config.split.file)  # ecrit split.csv
    kept = set(split["path"])  # images gardees dans le decoupage (ensemble : recherche rapide)
    # ecrit le fichier pivot avec seulement ces images et la liste des classes
    save_annotations([a for a in annotations if a.image_path in kept], pivot_path(config.split.file), class_names)
    by_subset = {s: annotations_for(config.split.file, s)[0] for s in ("train", "val", "test")}  # annotations par sous-ensemble
    for subset, items in by_subset.items():  # pour chaque sous-ensemble
        # nombre d'objets par classe dans ce sous-ensemble
        objects = pd.Series([label for a in items for label in a.object_labels()]).value_counts().to_dict()
        logger.info("%-5s : %4d images | objets par classe %s", subset, len(items), objects)  # resume dans le journal
    return split  # renvoie le catalogue decoupe


# --- etape 2 : entrainement --------------------------------------------------------------------


def _yolo_dir(output: Path) -> Path:  # renvoie le dossier ou ecrire le jeu au format YOLO
    """Jeu au format YOLO regenere a chaque entrainement (jamais versionne)."""
    return Path("data/processed/yolo") / "_".join(output.parts[-2:])  # nom tire des deux derniers dossiers de sortie


def _rfdetr_dir(output: Path) -> Path:  # renvoie le dossier ou ecrire le jeu au format COCO pour RF-DETR
    """Jeu au format COCO de Roboflow regenere a chaque entrainement RF-DETR."""
    return Path("data/processed/rfdetr") / "_".join(output.parts[-2:])  # nom tire des deux derniers dossiers de sortie


def train(config: DetectionConfig, output: Path, device: torch.device) -> dict:  # entraine le moteur choisi ; renvoie un resume
    pipeline = Pipeline(config.preprocess)  # pretraitement OpenCV decrit dans le YAML
    class_names = class_names_for(config.split.file)  # noms des classes du fichier pivot
    subsets = {s: annotations_for(config.split.file, s)[0] for s in ("train", "val", "test")}  # annotations par sous-ensemble
    color, backend = config.data.color, config.backend  # couleur de lecture et moteur choisi
    # resume sauve dans detector.json : tout ce qu'il faut pour recharger et utiliser le modele
    info = {"backend": backend, "class_names": class_names, "color": color, "preprocess": pipeline.describe()}
    with timer(f"entrainement {backend}") as elapsed:  # chronometre tout l'entrainement
        if backend == "ultralytics":  # moteur YOLO11 d'Ultralytics
            from vtb.tasks.detection import ultralytics_backend as engine  # import tardif du moteur

            # ecrit le jeu au format YOLO (images pretraitees + etiquettes)
            data_yaml = engine.write_dataset(subsets, class_names, pipeline, color, _yolo_dir(output), config.ultralytics.task)
            engine.train(config, data_yaml, output)  # entrainement par la librairie
            results = output / "runs" / "train" / "results.csv"  # historique ecrit par Ultralytics
            history = pd.read_csv(results) if results.exists() else pd.DataFrame()  # le relit s'il existe
            history.columns = [c.strip() for c in history.columns]  # retire les espaces autour des noms de colonnes
            # retient le data.yaml, la taille d'entree et la tache
            info.update({"data_yaml": str(data_yaml), "input_size": config.ultralytics.imgsz, "task": config.ultralytics.task})
        elif backend == "yolox":  # moteur YOLOX
            from yolox.data.data_augment import TrainTransform  # augmentation officielle de YOLOX

            from vtb.tasks.detection import yolox_backend as engine  # import tardif du moteur

            yc = config.yolox  # raccourci vers la section yolox du YAML
            # augmentation du train : retournement et couleurs, cibles completees jusqu'a max_labels
            transform = TrainTransform(max_labels=yc.max_labels, flip_prob=yc.flip_prob, hsv_prob=yc.hsv_prob)
            # petite fonction qui fabrique un YoloxDataset (t = augmentation ou None) a partir d'une liste d'annotations
            make = lambda items, t: engine.YoloxDataset(ensure_boxes(items), class_names, pipeline, color, yc.input_size, yc.max_labels, t)  # noqa: E731
            history = engine.train(  # lance la boucle YOLOX ; renvoie l'historique
                make(subsets["train"], transform),  # jeu du train, avec augmentation
                make(subsets["val"], None),  # jeu de validation, sans augmentation
                yc,  # reglages yolox
                class_names,  # noms des classes
                config.evaluation.conf_eval,  # seuil tres bas pour le mAP
                config.evaluation.iou_nms,  # seuil de la NMS (suppression des doublons)
                device,  # CPU ou GPU
                output,  # dossier de sortie
                config.runtime.seed,  # graine
                config.runtime.num_workers,  # processus de lecture
            )
            info.update({"input_size": yc.input_size, "variant": yc.variant, "task": "detect"})  # retient taille et variante
        elif backend == "rfdetr":  # moteur RF-DETR
            from vtb.tasks.detection import rfdetr_backend as engine  # import tardif du moteur

            rc = config.rfdetr  # raccourci vers la section rfdetr du YAML
            dataset = engine.write_dataset(  # ecrit le jeu au format COCO de Roboflow
                {k: ensure_boxes(v) for k, v in subsets.items()},  # chaque sous-ensemble avec des boites droites garanties
                class_names,  # noms des classes
                pipeline,  # pretraitement a appliquer aux images
                color,  # "rgb" ou "gray"
                _rfdetr_dir(output),  # dossier ou ecrire le jeu COCO
            )
            engine.train(rc, dataset, output, device, config.runtime.num_workers, config.runtime.seed)  # entrainement
            metrics_csv = output / "rfdetr_run" / "metrics.csv"  # historique ecrit par la librairie
            history = pd.read_csv(metrics_csv) if metrics_csv.exists() else pd.DataFrame()  # le relit s'il existe
            info.update({"input_size": engine.resolution_of(rc), "size": rc.size, "task": "detect"})  # retient resolution et taille
        else:  # sinon : Faster R-CNN
            from vtb.augment.transforms import build_augmentation  # fabrique les augmentations decrites dans le YAML
            from vtb.tasks.detection import fasterrcnn_backend as engine  # import tardif du moteur

            augment = build_augmentation(config.augment, task="detection") if config.augment else None  # augmentations ou rien
            # jeu du train (avec augmentation) ; boites droites garanties
            train_set = engine.DetectionDataset(ensure_boxes(subsets["train"]), class_names, pipeline, color, augment)
            val_set = engine.DetectionDataset(ensure_boxes(subsets["val"]), class_names, pipeline, color)  # val sans augmentation
            history = engine.train(  # lance la boucle Faster R-CNN ; renvoie l'historique
                train_set,  # jeu du train
                val_set,  # jeu de validation
                config.fasterrcnn,  # reglages fasterrcnn
                class_names,  # noms des classes
                config.evaluation.conf_eval,  # seuil tres bas pour le mAP
                device,  # CPU ou GPU
                output,  # dossier de sortie
                config.runtime.seed,  # graine
                config.runtime.num_workers,  # processus de lecture
            )
            info.update({"input_size": config.fasterrcnn.min_size, "task": "detect"})  # retient la taille d'entree
    info["train_seconds"] = round(elapsed["seconds"], 1)  # duree de l'entrainement, arrondie a 0.1 s
    history.to_csv(output / "history.csv", index=False)  # ecrit l'historique en CSV
    _plot_history(history, output / "history.png")  # trace les courbes
    (output / "detector.json").write_text(json.dumps(info, indent=2), encoding="utf-8")  # ecrit le resume en JSON lisible
    logger.info("entrainement %s termine en %.0f s -> %s", backend, elapsed["seconds"], output / "best.pt")  # message final
    return info  # renvoie le resume


def _plot_history(history: pd.DataFrame, path: Path) -> None:  # trace et enregistre les courbes (ne renvoie rien)
    """Pertes d'entrainement a gauche, mAP de validation a droite (noms de colonnes de chaque moteur)."""
    import matplotlib.pyplot as plt  # matplotlib : trace de graphiques

    if history.empty:  # si aucun historique (fichier absent)
        return  # rien a tracer
    x = history["epoch"] if "epoch" in history else np.arange(1, len(history) + 1)  # axe des epoques (1, 2, 3...)
    losses = [c for c in history.columns if "loss" in c and not c.startswith("val")]  # colonnes de pertes du train
    maps = [c for c in history.columns if "map" in c.lower()]  # colonnes de mAP (quel que soit le moteur)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))  # une figure, deux graphiques cote a cote
    for c in losses:  # pour chaque perte
        axes[0].plot(x, history[c], label=c)  # une courbe dans le graphique de gauche
    for c in maps:  # pour chaque mAP
        axes[1].plot(x, history[c], label=c)  # une courbe dans le graphique de droite
    axes[0].set_title("pertes (entrainement)")  # titre de gauche
    axes[1].set_title("mAP (validation)")  # titre de droite
    for ax in axes:  # pour chacun des deux graphiques
        ax.set_xlabel("epoque")  # nom de l'axe horizontal
        ax.grid(alpha=0.3)  # quadrillage leger
        if ax.lines:  # s'il y a au moins une courbe
            ax.legend(fontsize=7)  # legende en petit
    fig.tight_layout()  # evite que les titres se chevauchent
    fig.savefig(path, dpi=110)  # enregistre l'image
    plt.close(fig)  # libere la memoire de la figure


# --- chargement et prediction (le chemin de production) ----------------------------------------


class Detector:  # enveloppe commune : le meme code de prediction quel que soit le moteur
    """Modele entraine d'un moteur ; predict() prend des images PRETRAITEES et rend des annotations dans leur repere."""

    def __init__(self, config: DetectionConfig, output: Path, device: torch.device) -> None:  # recharge le modele entraine
        self.config, self.output, self.device = config, output, device  # garde les reglages
        self.info = json.loads((output / "detector.json").read_text(encoding="utf-8"))  # relit le resume ecrit par train
        self.class_names: list[str] = self.info["class_names"]  # noms des classes
        backend = self.info["backend"]  # moteur utilise a l'entrainement
        if backend == "ultralytics":  # YOLO11
            from vtb.tasks.detection import ultralytics_backend as engine  # module du moteur

            self.model = engine.load(output)  # recharge best.pt
        elif backend == "yolox":  # YOLOX
            from vtb.tasks.detection import yolox_backend as engine  # module du moteur

            self.model = engine.load_trained(output, self.info["variant"], len(self.class_names), device)  # recharge best.pt
        elif backend == "rfdetr":  # RF-DETR
            from vtb.tasks.detection import rfdetr_backend as engine  # module du moteur

            self.model = engine.load(output)  # recharge best.pt
        else:  # Faster R-CNN
            from vtb.tasks.detection import fasterrcnn_backend as engine  # module du moteur

            self.model = engine.load_trained(output, config.fasterrcnn, len(self.class_names), device)  # recharge best.pt
        self.engine, self.backend = engine, backend  # garde le module du moteur et son nom

    def arrays(self, images: list[np.ndarray], conf: float) -> list[np.ndarray]:  # predit ; renvoie un tableau (n, 6) par image
        """(n, 6) [x1, y1, x2, y2, score, classe] par image (boites droites seulement)."""
        if self.backend == "ultralytics":  # Ultralytics rend des annotations : on les convertit en tableaux
            return [boxes_to_array(a.boxes, self.class_names) for a in self.annotations(images, conf)]  # une par image
        iou = self.config.evaluation.iou_nms  # seuil de la NMS
        if self.backend == "yolox":  # YOLOX
            # taille d'entree, nombre de classes, seuils et device
            return self.engine.predict_images(self.model, images, self.info["input_size"], len(self.class_names), conf, iou, self.device)
        if self.backend == "rfdetr":  # RF-DETR
            return self.engine.predict_images(self.model, images, len(self.class_names), conf)  # DETR : pas de NMS
        return self.engine.predict_images(self.model, images, conf, self.device)  # Faster R-CNN (NMS interne au modele)

    def annotations(self, images: list[np.ndarray], conf: float) -> list[ImageAnnotation]:  # renvoie une annotation par image
        """Annotations predites ; avec Ultralytics, aussi les boites orientees (obb) ou les polygones (segment)."""
        if self.backend == "ultralytics":  # Ultralytics sait rendre boites orientees et polygones
            iou = self.config.evaluation.iou_nms  # seuil de la NMS
            return self.engine.predict(self.model, images, self.class_names, self.info["input_size"], conf, iou)  # annotations
        out = []  # autres moteurs : on convertit les tableaux en annotations
        for image, rows in zip(images, self.arrays(images, conf)):  # pour chaque image et ses detections
            ann = ImageAnnotation("", image.shape[1], image.shape[0])  # annotation vide (chemin inconnu ici), largeur, hauteur
            ann.boxes = boxes_from_array(rows, self.class_names)  # boites au format pivot
            out.append(ann)  # garde l'annotation
        return out  # renvoie toutes les annotations

    def predict(self, images: list[np.ndarray], conf: float) -> list[ImageAnnotation]:  # prediction, par tuiles si demande
        """Avec tiling.enabled, chaque image plus grande qu'une tuile passe par predict_tiled (boites droites)."""
        tiling = self.config.tiling  # reglages du decoupage en tuiles
        if not tiling.enabled:  # si les tuiles sont desactivees
            return self.annotations(images, conf)  # prediction directe sur les images entieres
        out = []  # une annotation par image
        for image in images:  # pour chaque image
            if max(image.shape[:2]) <= tiling.tile:  # si elle tient dans une seule tuile
                out.extend(self.annotations([image], conf))  # prediction directe
                continue  # image suivante
            # grande image : prediction tuile par tuile, boites ramenees dans l'image et doublons fusionnes
            rows = predict_tiled(lambda t: self.arrays([t], conf)[0], image, tiling.tile, tiling.overlap, tiling.iou_merge)
            ann = ImageAnnotation("", image.shape[1], image.shape[0])  # annotation vide de la taille de l'image
            ann.boxes = boxes_from_array(rows, self.class_names)  # boites au format pivot
            out.append(ann)  # garde l'annotation
        return out  # renvoie toutes les annotations


# fichiers images -> annotations predites dans l'image d'origine ; renvoie une annotation par fichier
def predict_paths(detector: Detector, paths: list[str | Path], conf: float, batch_size: int = 16) -> list[ImageAnnotation]:
    """Fichiers -> annotations predites dans le repere de l'image d'ORIGINE (lecture, pretraitement, moteur, remise a l'echelle)."""
    pipeline = Pipeline(detector.info["preprocess"])  # le pretraitement SAUVE avec le modele (pas celui du YAML actuel)
    results = []  # annotations predites, une par fichier
    for start in range(0, len(paths), batch_size):  # parcourt les fichiers par paquets
        chunk = [str(p) for p in paths[start : start + batch_size]]  # chemins du paquet, en texte
        originals = [read_image(p, color=detector.info["color"]) for p in chunk]  # images d'origine
        images = [pipeline(o) for o in originals]  # images pretraitees (ce que voit le modele)
        # pour chaque fichier : chemin, image d'origine, image pretraitee, prediction
        for path, original, image, pred in zip(chunk, originals, images, detector.predict(images, conf)):
            h, w = original.shape[:2]  # hauteur et largeur d'origine
            # remet les predictions a l'echelle de l'image d'origine (facteurs largeur et hauteur)
            results.append(scale_annotation(pred, w / image.shape[1], h / image.shape[0], path, w, h))
    return results  # renvoie toutes les annotations


def load_trained(config: DetectionConfig, output: Path, device: torch.device) -> Detector:  # renvoie le detecteur pret
    return Detector(config, output, device)  # meme nom de fonction que les autres taches


# --- etape 3 : evaluation ----------------------------------------------------------------------


def evaluate(config: DetectionConfig, output: Path, device: torch.device) -> dict:  # mesure sur le test ; renvoie les metriques
    """mAP COCO (toute la courbe, conf_eval tres bas) + point de fonctionnement (conf_predict) + galerie."""
    detector = Detector(config, output, device)  # recharge le modele entraine
    test, class_names = annotations_for(config.split.file, "test")  # annotations du test (jamais vues) et classes
    with timer("prediction du test") as elapsed:  # chronometre la prediction
        # predit par le meme chemin que la production, seuil tres bas (toute la courbe precision-rappel)
        predictions = predict_paths(detector, [a.image_path for a in test], config.evaluation.conf_eval)
    boxes = [p.boxes for p in predictions]  # boites predites de chaque image
    metrics = coco_metrics(test, boxes, class_names)  # mAP50, mAP50-95 et detail par classe
    # precision et rappel au seuil de production (ce que verra l'operateur)
    metrics["operating_point"] = operating_point(test, boxes, class_names, config.evaluation.conf_predict)
    metrics.update(  # ajoute quelques informations generales
        # nombre d'images, moteur, temps moyen par image en millisecondes
        {"n_images": len(test), "backend": detector.backend, "ms_per_image": round(1000 * elapsed["seconds"] / max(1, len(test)), 1)}
    )
    # boites orientees ou masques : l'evaluateur COCO commun ne les couvre pas, on ajoute les metriques Ultralytics
    if detector.backend == "ultralytics" and detector.info.get("task") in ("obb", "segment"):
        from vtb.tasks.detection.ultralytics_backend import val_metrics  # metriques d'Ultralytics

        metrics["ultralytics"] = val_metrics(detector.model, Path(detector.info["data_yaml"]), config, output)  # sur le test
    (output / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")  # sauve les metriques en JSON
    # tableau par classe : AP (precision moyenne) a cote de precision / rappel au seuil de production
    table = pd.DataFrame(metrics["per_class"]).T.join(pd.DataFrame(metrics["operating_point"]["per_class"]).T)
    table.to_csv(output / "per_class.csv")  # ecrit ce tableau en CSV
    # planche verite terrain / predictions sur quelques images du test
    _gallery(test, predictions, class_names, config.evaluation.conf_predict, detector.info["color"], output / "test_predictions.png")
    op = metrics["operating_point"]["overall"]  # chiffres globaux au seuil de production
    logger.info(  # resume du test dans le journal
        "test (%d images) : mAP50 %.3f | mAP50-95 %.3f | au seuil %.2f : precision %.3f rappel %.3f",  # gabarit du message
        len(test),  # nombre d'images
        metrics["map50"],  # mAP a IoU 0.5
        metrics["map50_95"],  # mAP moyen de IoU 0.5 a 0.95
        config.evaluation.conf_predict,  # seuil de production
        op["precision"],  # part des boites predites qui sont justes
        op["recall"],  # part des vrais objets retrouves
    )
    return metrics  # renvoie les metriques


def _gallery(  # dessine une planche verite / predictions pour quelques images (ne renvoie rien)
    truth: list[ImageAnnotation],  # annotations de verite terrain du test
    preds: list[ImageAnnotation],  # annotations predites, dans le meme ordre
    class_names: list[str],  # noms des classes
    conf: float,  # seuil de production : on n'affiche que les predictions au-dessus
    color: str,  # couleur de lecture des images
    path: Path,  # fichier image de sortie
    n: int = 8,  # nombre d'images a montrer
) -> None:  # ne renvoie rien
    """Verite terrain (gauche) / predictions au seuil de production (droite) pour quelques images de test."""
    import matplotlib.pyplot as plt  # matplotlib : trace de graphiques

    from vtb.annotate.draw import draw_annotation  # dessine boites et contours sur une image

    rng = np.random.default_rng(0)  # generateur aleatoire de graine 0 : toujours les memes images choisies
    # tire au plus n images differentes du test, triees ; liste vide si le test est vide
    picks = sorted(rng.choice(len(truth), size=min(n, len(truth)), replace=False).tolist()) if truth else []
    if not picks:  # si aucune image a montrer
        return  # rien a dessiner
    fig, axes = plt.subplots(len(picks), 2, figsize=(7, 3.2 * len(picks)), squeeze=False)  # une ligne par image, 2 colonnes
    for row, i in enumerate(picks):  # pour chaque image choisie (row = ligne de la planche, i = indice dans le test)
        image = read_image(truth[i].image_path, color=color)  # image d'origine
        shown = ImageAnnotation(preds[i].image_path, preds[i].width, preds[i].height)  # predictions a afficher
        shown.boxes = [b for b in preds[i].boxes if (b.score or 0) >= conf]  # boites au-dessus du seuil de production
        shown.rotated = [r for r in preds[i].rotated if (r.score or 0) >= conf]  # boites orientees au-dessus du seuil
        shown.polygons = [p for p in preds[i].polygons if (p.score or 0) >= conf]  # contours au-dessus du seuil
        if shown.rotated or shown.polygons:  # s'il y a des formes plus precises que des boites droites
            shown.boxes = []  # la boite englobante cacherait la forme
        # colonne 0 : verite terrain ; colonne 1 : predictions filtrees
        for col, (ann, title) in enumerate(((truth[i], "verite terrain"), (shown, f"predictions (conf >= {conf})"))):
            axes[row, col].imshow(draw_annotation(image, ann, class_names, thickness=1))  # image avec ses annotations
            axes[row, col].set_title(f"{Path(truth[i].image_path).name} - {title}", fontsize=7)  # nom du fichier et titre
            axes[row, col].axis("off")  # pas d'axes gradues
    fig.tight_layout()  # evite les chevauchements
    fig.savefig(path, dpi=100)  # enregistre l'image
    plt.close(fig)  # libere la memoire


# --- etape 4 : prediction de production --------------------------------------------------------


# predit sur une image ou un dossier ; renvoie un tableau avec une ligne par objet detecte
def predict(config: DetectionConfig, output: Path, device: torch.device, source: Path, save_images: Path | None = None) -> pd.DataFrame:
    """Une ligne par objet detecte (et une ligne vide pour une image sans detection : chaque image est tracee)."""
    from vtb.annotate.draw import draw_annotation  # dessine les detections sur une image

    detector = Detector(config, output, device)  # recharge le modele entraine
    paths = [source] if Path(source).is_file() else list_images(source, config.data.extensions)  # un fichier ou un dossier
    predictions = predict_paths(detector, paths, config.evaluation.conf_predict)  # predictions au seuil de production
    rows = []  # lignes du tableau de resultats
    for ann in predictions:  # pour chaque image predite
        # boites droites, sauf si l'image a des formes plus precises (boites orientees ou contours)
        objects = [("box", b, {}) for b in ann.boxes] if not (ann.rotated or ann.polygons) else []
        # boites orientees : boite droite englobante + centre, taille et angle en colonnes supplementaires
        objects += [("rotated", rotated_to_box(r), {"cx": r.cx, "cy": r.cy, "w": r.w, "h": r.h, "angle": r.angle}) for r in ann.rotated]
        objects += [  # contours : boite englobante + liste des points en JSON
            # points arrondis a 0.1 pixel
            ("polygon", p.bounding_box(), {"points": json.dumps([[round(x, 1), round(y, 1)] for x, y in p.points])})
            for p in ann.polygons  # pour chaque contour de l'image
        ]
        if not objects:  # si rien n'a ete detecte
            rows.append({"path": ann.image_path, "label": "", "score": np.nan})  # ligne vide : l'image reste tracee
        for kind, box, extra in objects:  # pour chaque objet : type, boite englobante, colonnes supplementaires
            # chemin, type d'objet, classe, score arrondi
            row = {"path": ann.image_path, "kind": kind, "label": box.label, "score": round(float(box.score or 0), 4)}
            row.update({"x1": round(box.x1, 1), "y1": round(box.y1, 1), "x2": round(box.x2, 1), "y2": round(box.y2, 1)})  # coins
            row.update({k: (round(v, 2) if isinstance(v, float) else v) for k, v in extra.items()})  # colonnes en plus, arrondies
            rows.append(row)  # ajoute la ligne
        if save_images is not None:  # si on veut aussi les images annotees
            image = read_image(ann.image_path, color=detector.info["color"])  # relit l'image d'origine
            # dessine les detections et ecrit <nom>_pred.png dans le dossier demande
            write_image(Path(save_images) / f"{Path(ann.image_path).stem}_pred.png", draw_annotation(image, ann, detector.class_names))
    table = pd.DataFrame(rows)  # tableau final
    n_found = int(table["label"].astype(bool).sum()) if len(table) else 0  # nombre d'objets (etiquette non vide)
    # resume dans le journal
    logger.info("%d images | %d objets detectes (conf >= %.2f)", len(predictions), n_found, config.evaluation.conf_predict)
    return table  # renvoie le tableau des detections


# --- etape 5 : export --------------------------------------------------------------------------


def export(config: DetectionConfig, output: Path) -> ModelCard:  # exporte en ONNX, verifie, ecrit la carte ; renvoie la carte
    from vtb.export.onnx import batch_size_of, measure_latency, to_openvino  # lot fige ou non, latences, conversion OpenVINO

    cpu = torch.device("cpu")  # export toujours sur CPU
    detector = Detector(config, output, cpu)  # recharge le modele entraine sur CPU
    pipeline = Pipeline(detector.info["preprocess"])  # pretraitement sauve avec le modele
    val, class_names = annotations_for(config.split.file, "val")  # images de validation et classes
    samples = [pipeline(read_image(a.image_path, color=detector.info["color"])) for a in val[:4]]  # 4 vraies images pretraitees
    samples = [s if s.ndim == 3 else np.stack([s] * 3, -1) for s in samples]  # images grises -> 3 canaux
    ev, size = config.evaluation, detector.info["input_size"]  # seuils et taille d'entree du modele
    onnx_path = output / "model.onnx"  # fichier ONNX a produire
    if detector.backend == "ultralytics":  # export YOLO11
        result = detector.engine.export(detector.model, config, output, samples, class_names)  # export + verification
        net, sample = result["net"], result["sample"]  # reseau PyTorch et entree d'exemple (pour la latence)
        n_out = 4 + len(class_names)  # 4 coordonnees + un score par classe
        spec = InputSpec(  # description de l'entree attendue
            name="images",  # nom de l'entree ONNX
            height=size,  # hauteur du carre d'entree
            width=size,  # largeur du carre d'entree
            channels=3,  # 3 canaux
            color="RGB",  # ordre des couleurs
            scale="divide by 255",  # valeurs ramenees dans [0, 1]
            resize="letterbox",  # redimension sans deformer, centree, bordure grise
            letterbox_pad_value=114,  # valeur de gris de la bordure
        )
        # une description par sortie ONNX (nom et forme lus dans le fichier)
        outputs = [OutputSpec(name=n, shape=list(s), description="sortie brute YOLO11 (avant NMS)") for n, s in result["outputs"]]
        postprocess = {  # recette du post-traitement a reproduire en production
            "steps": (  # les etapes dans l'ordre
                "letterbox centre gris 114 -> /255 -> reseau -> transposer (A, 4 + C) -> score max par ancre -> filtre conf"
                " -> cx,cy,w,h -> coins -> NMS par classe -> retirer la bordure et diviser par le ratio"
            ),
            "reference": "vtb.tasks.detection.ultralytics_backend.postprocess_detect",  # fonction Python de reference
            # forme de la sortie selon la tache
            "output_layout": f"(batch, {n_out}, anchors) pour detect ; obb ajoute l'angle ; segment ajoute coefficients et prototypes",
        }
    elif detector.backend == "yolox":  # export YOLOX
        result = detector.engine.export(  # export + deux verifications
            detector.model,  # le reseau
            samples,  # images de verification
            size,  # cote de l'entree
            len(class_names),  # nombre de classes
            onnx_path,  # fichier de sortie
            config.export.opset,  # version des operations ONNX
            VERIFY_CONF,  # seuil de confiance bas pour la verification
            ev.iou_nms,  # seuil de NMS
            config.export.atol,  # tolerance absolue
            config.export.rtol,  # tolerance relative
        )
        net, sample = result["model"], result["sample"]  # reseau exporte et entree d'exemple
        spec = InputSpec(  # description de l'entree attendue
            name="images",  # nom de l'entree ONNX
            height=size,  # hauteur
            width=size,  # largeur
            channels=3,  # 3 canaux
            color="BGR",  # YOLOX attend du BGR
            scale="none (values 0-255)",  # pas de division : valeurs 0-255
            resize="letterbox_top_left",  # sans deformer, collee en haut a gauche
            letterbox_pad_value=114,  # gris de la bordure
        )
        outputs = [  # description de la sortie
            OutputSpec(  # une seule sortie
                name="output",  # son nom dans le fichier ONNX
                shape=["batch", "anchors", 5 + len(class_names)],  # (lot, positions, 4 coordonnees + objet + classes)
                description="sortie brute NON decodee (x, y, w, h, objet, classes)",  # comment l'interpreter
            )
        ]
        postprocess = {  # recette du post-traitement a reproduire en production
            "steps": (  # les etapes dans l'ordre
                "decoder les grilles de pas 8/16/32 (cx = (x + col) * pas, w = exp(w) * pas) -> score = objet * proba"
                " -> filtre conf -> NMS par classe -> diviser par le ratio"
            ),
            "reference": "vtb.tasks.detection.yolox_backend.postprocess_onnx",  # fonction Python de reference
        }
    elif detector.backend == "rfdetr":  # export RF-DETR
        # export + verification (le seuil passe est remplace par 0 dans rfdetr_backend.export)
        result = detector.engine.export(detector.model, samples, output, len(class_names), VERIFY_CONF, config.export.opset)
        net, sample = None, None  # latence mesuree a part (le modele torch de rfdetr ne prend pas un simple tenseur)
        res = result["resolution"]  # cote de l'entree
        spec = InputSpec(  # description de l'entree attendue
            height=res,  # hauteur
            width=res,  # largeur
            channels=3,  # 3 canaux
            color="RGB",  # ordre des couleurs
            scale="divide by 255",  # valeurs ramenees dans [0, 1]
            mean=result["mean"],  # moyenne de normalisation
            std=result["std"],  # ecart-type de normalisation
            resize="stretch",  # redimension SANS garder les proportions
            dynamic_batch=False,  # lot fixe de 1
        )
        outputs = [  # description des deux sorties
            OutputSpec(  # sortie des boites
                name="dets",  # nom de la sortie dans le fichier ONNX
                shape=[1, "queries", 4],  # forme : (1, requetes, 4)
                description="boites cxcywh normalisees dans [0, 1] (par rapport a l'image recue)",  # comment l'interpreter
            ),
            OutputSpec(  # sortie des scores
                name="labels",  # nom de la sortie dans le fichier ONNX
                shape=[1, "queries", len(class_names) + 1],  # forme : (1, requetes, classes + 1)
                description="logits par classe (le dernier indice ne sert pas)",  # logits = scores bruts avant la sigmoide
            ),
        ]
        postprocess = {  # recette du post-traitement a reproduire en production
            "steps": (  # les etapes dans l'ordre
                "redimensionner l'image pretraitee a resolution x resolution SANS garder les proportions (INTER_LINEAR) -> /255 -> mean/std"
                " -> reseau -> sigmoide des logits -> 300 meilleurs couples (requete, classe) -> classe = indice % (C + 1)"
                " -> garder classe < C et score >= conf -> cxcywh x (largeur, hauteur) de l'image recue ; PAS de NMS"
            ),
            "reference": "vtb.tasks.detection.rfdetr_backend.postprocess_onnx",  # fonction Python de reference
        }
    else:  # export Faster R-CNN
        result = detector.engine.export(detector.model, samples, onnx_path, config.export.opset, VERIFY_CONF)  # export + verif
        net, sample = result["model"], result["sample"]  # modele enveloppe et entree d'exemple
        h, w = samples[0].shape[:2]  # taille de la premiere image d'exemple (l'entree reelle est de taille libre)
        # description de l'entree : le modele redimensionne lui-meme (resize "none"), une image a la fois
        spec = InputSpec(height=h, width=w, channels=3, color="RGB", scale="divide by 255", resize="none", dynamic_batch=False)
        outputs = [  # description des trois sorties
            OutputSpec(name="boxes", shape=["n", 4], description="x1, y1, x2, y2 dans l'image recue"),  # les boites
            OutputSpec(name="scores", shape=["n"], description="score de la classe"),  # les scores
            OutputSpec(name="labels", shape=["n"], description="indice dans class_names"),  # les classes
        ]
        postprocess = {  # recette du post-traitement (presque tout est deja dans le graphe)
            # ce qui reste a faire en production
            "steps": "redimensionnement et NMS dans le graphe (scores >= 0.05) ; garder score >= thresholds.conf ; une image a la fois"
        }
    spec.read_color = detector.info["color"]  # lecture en gris puis copie sur 3 canaux, comme a l'entrainement
    xml = None  # chemin du modele OpenVINO (aucun par defaut)
    if config.export.openvino:  # si la conversion OpenVINO est demandee
        try:  # la conversion peut echouer selon le modele
            xml = to_openvino(onnx_path)  # convertit l'ONNX en OpenVINO (CPU Intel)
        except Exception as error:  # noqa: BLE001 - OpenVINO ne couvre pas tous les graphes (Faster R-CNN)
            logger.warning("conversion OpenVINO impossible : %s", error)  # on previent mais on continue
    if detector.backend == "rfdetr":  # RF-DETR a sa propre mesure de latence
        # latence PyTorch (via la librairie), onnxruntime et OpenVINO
        latency = detector.engine.measure_latency(detector.model, onnx_path, samples[0], config.export.latency_runs, xml)
    else:  # autres moteurs : mesure generique
        latency = measure_latency(net, onnx_path, sample, config.export.latency_runs, xml)  # PyTorch, onnxruntime, OpenVINO
    # relit les metriques du test si l'etape evaluate a deja tourne, sinon dict vide
    metrics = json.loads((output / "test_metrics.json").read_text()) if (output / "test_metrics.json").exists() else {}
    name = {  # nom lisible du modele selon le moteur
        "ultralytics": Path(config.ultralytics.weights).stem,  # ex. yolo11n
        "yolox": f"yolox_{config.yolox.variant}",  # ex. yolox_tiny
        "fasterrcnn": f"fasterrcnn_{config.fasterrcnn.backbone}",  # ex. fasterrcnn_resnet18_fpn
        "rfdetr": f"rfdetr_{config.rfdetr.size}",  # ex. rfdetr_nano
    }[detector.backend]  # on prend le nom du moteur utilise
    card = ModelCard(  # carte du modele : tout ce que le programme de production doit savoir
        task="detection",  # type de tache
        model_name=f"{detector.backend}:{name}",  # moteur et nom du modele
        framework=framework_versions(),  # versions des bibliotheques (tracabilite)
        input=spec,  # description de l'entree
        preprocess=detector.info["preprocess"],  # pretraitement OpenCV a refaire a l'identique
        outputs=outputs,  # description des sorties
        class_names=class_names,  # noms des classes
        thresholds={"conf": ev.conf_predict, "iou_nms": ev.iou_nms},  # seuils de production
        postprocess=postprocess,  # recette du post-traitement
        metrics={k: metrics[k] for k in ("map50", "map50_95", "n_images") if k in metrics}  # metriques du test disponibles
        # | = fusion de dicts : ajoute le point de fonctionnement global s'il existe
        | ({"operating_point": metrics["operating_point"]["overall"]} if "operating_point" in metrics else {}),
        latency_ms={k: v for k, v in latency.items() if k.endswith("_ms")},  # latences en millisecondes
        files={"onnx": "model.onnx", **({"openvino": "model.xml"} if xml else {})},  # fichiers livres
        notes=f"verification ONNX : {result['report']}",  # rapport de verification de l'export
    )
    card.input.dynamic_batch = batch_size_of(onnx_path) is None  # certains modeles figent le lot (anomalib PatchCore, DFM)
    write_card(card, output / "model_card.json")  # ecrit la carte en JSON a cote du modele
    return card  # renvoie la carte
