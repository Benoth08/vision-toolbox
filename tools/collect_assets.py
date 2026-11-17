"""Copie les figures et journaux des experiences dans docs/assets/<sujet>/ pour les articles du site.

    python tools/collect_assets.py

Les images sont converties en JPEG (largeur maximale 1400 px) : le site integre chaque image dans
sa page, il doit rester leger. Une source absente est signalee et ignoree (experience non lancee).
"""

from __future__ import annotations  # permet d'ecrire les types modernes sans les evaluer a l'execution

import logging  # journal de messages (INFO, WARNING) plutot que des print
import shutil  # copie de fichiers
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import cv2  # OpenCV : lecture, redimensionnement et ecriture d'images

logger = logging.getLogger("collect_assets")  # journal nomme d'apres le script

# Liste des fichiers a recopier, rangee par sujet : sujet -> [(fichier produit par l'experience, nom dans docs/assets/<sujet>/)].
# Une cible en .jpg veut dire "convertir l'image en JPEG" ; les autres extensions sont copiees telles quelles.
ASSETS = {
    "classification": [  # articles sur la classification
        ("outputs/classification/resnet18_partial/confusion_matrix.png", "resnet18_partial_confusion.jpg"),  # matrice de confusion
        ("outputs/classification/resnet18_partial/curves.png", "resnet18_partial_curves.jpg"),  # courbes d'entrainement
        ("outputs/classification/simple_cnn/test_errors.png", "simple_cnn_errors.jpg"),  # erreurs du petit CNN sur le test
        ("outputs/classification/resnet18_head/test_errors.png", "resnet18_head_errors.jpg"),  # erreurs du ResNet18 (tete seule)
        ("outputs/classification/resnet18_partial/run.log", "resnet18_partial_run.log"),  # journal de l'entrainement
    ],
    "detection": [  # articles sur la detection d'objets
        ("outputs/detection/quick_yolo/test_predictions.png", "quick_yolo_predictions.jpg"),  # predictions de YOLO
        ("outputs/detection/quick_fasterrcnn/test_predictions.png", "quick_fasterrcnn_predictions.jpg"),  # predictions de Faster R-CNN
        ("outputs/detection/quick_obb_tile/test_predictions.png", "quick_obb_tile_predictions.jpg"),  # boites orientees sur tuiles
        ("outputs/detection/quick_yolo/run.log", "quick_yolo_run.log"),  # journal YOLO
        ("outputs/detection/quick_yolox/run.log", "quick_yolox_run.log"),  # journal YOLOX
        ("outputs/detection/quick_fasterrcnn/run.log", "quick_fasterrcnn_run.log"),  # journal Faster R-CNN
    ],
    "segmentation": [  # articles sur la segmentation (un masque par pixel)
        ("outputs/segmentation/unet_resnet18_tile_cpu/test_predictions.png", "unet_cpu_predictions.jpg"),  # masques predits par U-Net
        ("outputs/segmentation/unet_resnet18_tile_cpu/history.png", "unet_cpu_history.jpg"),  # courbes d'entrainement
        ("outputs/segmentation/unet_resnet18_tile_cpu/run.log", "unet_cpu_run.log"),  # journal U-Net
        ("outputs/segmentation/quick/run.log", "quick_run.log"),  # journal de l'essai rapide
    ],
    "explore": [  # articles sur l'exploration des donnees
        ("outputs/explore/neu_cls/quality_report.md", "neu_cls_quality_report.md"),  # rapport qualite NEU-CLS
        ("outputs/explore/neu_cls/stats_by_class.png", "neu_cls_stats_by_class.jpg"),  # statistiques par classe
        ("outputs/explore/neu_cls/duplicates.png", "neu_cls_duplicates.jpg"),  # images en double
        ("outputs/explore/magnetic_tile/quality_report.md", "magnetic_tile_quality_report.md"),  # rapport qualite Magnetic Tile
        ("outputs/explore/magnetic_tile/stats_by_class.png", "magnetic_tile_stats_by_class.jpg"),  # statistiques par classe
        ("outputs/explore/magnetic_tile/duplicates.png", "magnetic_tile_duplicates.jpg"),  # images en double
        ("outputs/explore/neu_cls/projection_by_label.png", "neu_cls_projection.jpg"),  # projection 2D coloree par classe
        ("outputs/explore/neu_cls/clusters_gallery.png", "neu_cls_clusters.jpg"),  # galerie par groupe d'images proches
        ("outputs/explore/drift_none/drift_report.json", "drift_none_report.json"),  # derive : cas sans derive
        ("outputs/explore/drift_simulated/drift_report.json", "drift_simulated_report.json"),  # derive simulee : rapport
        ("outputs/explore/drift_simulated/drift_projection.png", "drift_simulated_projection.jpg"),  # derive simulee : projection
        ("outputs/explore/drift_simulated/most_novel.png", "drift_simulated_most_novel.jpg"),  # images les plus nouvelles
        ("outputs/explore/drift_simulated/statistics_drift.csv", "drift_simulated_statistics.csv"),  # derive des statistiques
    ],
    "anomaly": [  # articles sur la detection d'anomalies
        ("outputs/anomaly/patchcore_lite_bottle/maps.png", "patchcore_lite_bottle_maps.jpg"),  # cartes d'anomalie PatchCore (bouteilles)
        ("outputs/anomaly/patchcore_lite_bottle/scores.png", "patchcore_lite_bottle_scores.jpg"),  # distribution des scores
        ("outputs/anomaly/patchcore_lite_tile/maps.png", "patchcore_lite_tile_maps.jpg"),  # cartes PatchCore (carreaux)
        ("outputs/anomaly/autoencoder_bottle/maps.png", "autoencoder_bottle_maps.jpg"),  # cartes de l'auto-encodeur
    ],
}


def copy(source: Path, target: Path, max_width: int = 1400) -> None:  # copie un fichier (en JPEG reduit si cible .jpg) ; ne renvoie rien
    target.parent.mkdir(parents=True, exist_ok=True)  # cree le dossier de destination s'il n'existe pas
    if target.suffix == ".jpg":  # si la cible est une image JPEG
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)  # lit l'image source en couleur
        if image.shape[1] > max_width:  # si elle est plus large que la largeur maximale
            # reduit a max_width de large, hauteur calculee pour garder les proportions (INTER_AREA : bon pour reduire)
            image = cv2.resize(image, (max_width, round(image.shape[0] * max_width / image.shape[1])), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, 85])  # ecrit en JPEG qualite 85 : leger et encore net
    else:  # autre fichier (journal, rapport, CSV, JSON)
        shutil.copy2(source, target)  # simple copie, dates comprises


def main() -> None:  # copie tous les fichiers de ASSETS ; ne renvoie rien
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")  # affiche les messages INFO et plus graves
    for topic, items in ASSETS.items():  # pour chaque sujet
        for source, name in items:  # pour chaque fichier de ce sujet
            source_path, target = Path(source), Path("docs/assets") / topic / name  # chemin source et chemin cible
            if not source_path.exists():  # si l'experience n'a pas ete lancee, le fichier n'existe pas
                logger.warning("absent (experience non lancee ?) : %s", source)  # on previent
                continue  # et on passe au fichier suivant
            copy(source_path, target)  # copie (ou convertit) le fichier
            logger.info("%s -> %s (%.0f Ko)", source, target, target.stat().st_size / 1024)  # dit ce qui a ete fait et la taille obtenue


if __name__ == "__main__":  # vrai seulement quand le script est lance directement (pas importe)
    main()  # lance la copie
