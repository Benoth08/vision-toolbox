# %% [markdown]
# # Detection : ou sont les defauts, et de quel type ?
#
# Quatre moteurs derriere le meme contrat : YOLO11 (Ultralytics), YOLOX (Megvii), Faster R-CNN
# (torchvision) et RF-DETR (Roboflow, transformer sans NMS). Le decoupage, le format pivot des annotations, l'evaluateur COCO et la carte du
# modele sont communs, donc les quatre se comparent sur les memes images. Ce notebook deroule la
# chaine avec YOLO11n sur un sous-ensemble de NEU-DET (essai court sur CPU), puis montre ce que
# l'export garantit : la chaine ONNX + post-traitement ecrit en clair rend les memes boites.

# %%
import json  # json : affiche joliment une partie de la carte du modele
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import matplotlib.pyplot as plt  # matplotlib : trace les figures
import pandas as pd  # pandas : tableaux de donnees (comptages, mesures par classe)
from IPython.display import Image, display  # affiche dans le notebook une image PNG deja enregistree sur disque

from vtb.annotate.draw import draw_annotation  # dessine les boites d'une annotation sur une copie de l'image
from vtb.core.config import load_config  # lit le fichier YAML de l'experience et verifie chaque reglage
from vtb.core.experiment import start  # prepare l'experience : dossier de sortie, journal, graines, choix CPU / GPU
from vtb.io.images import read_image  # lit une image en tableau numpy (couleur ou niveaux de gris)
from vtb.tasks.detection import workflow  # les etapes de la recette detection (prepare, train, evaluate, predict, export)
from vtb.tasks.detection.common import annotations_for  # relit les annotations (format pivot) d'une partie du decoupage
from vtb.tasks.detection.config import DetectionConfig  # le modele de configuration : la liste des cles autorisees dans le YAML

config = load_config("recipes/detection/configs/quick_yolo.yaml", DetectionConfig)  # config d'essai rapide avec YOLO11n
# meme config, mais les sorties vont dans un dossier a part pour ne pas melanger avec les experiences des scripts
config = config.model_copy(update={"runtime": config.runtime.model_copy(update={"output_dir": "outputs/detection/notebook"})})
output, device = start(config, "notebook")  # cree le dossier de sortie, le journal, fixe les graines ; device = CPU ou GPU

# %% [markdown]
# ## 1. Les annotations : un format pivot pour tous les formats d'origine
#
# `prepare` lit les annotations (ici Pascal VOC ; aussi YOLO, COCO, LabelMe, Label Studio, CVAT
# ou des masques), les convertit au format pivot de `vtb.annotate`, decoupe PAR IMAGE
# (stratifie sur la classe principale) et ecrit `split.csv` + `split.vtb.json`.

# %%
if not Path(config.split.file).exists():  # si le decoupage n'a pas encore ete fait
    workflow.prepare(config)  # annotations au format pivot + decoupage train / val / test
train, class_names = annotations_for(config.split.file, "train")  # annotations des images de train et noms des classes
print(len(train), "images d'entrainement ;", class_names)  # combien d'images, quelles classes
counts = pd.Series([label for a in train for label in a.object_labels()]).value_counts()  # nombre d'objets par classe
print(counts)  # affiche le comptage (verifie qu'aucune classe n'est quasi absente)

# %%
fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))  # 4 cases cote a cote
for ax, ann in zip(axes, train[:4]):  # pour chaque case et chacune des 4 premieres images de train
    ax.imshow(draw_annotation(read_image(ann.image_path, color="gray"), ann, class_names, thickness=1))  # image + ses boites
    ax.set_title(Path(ann.image_path).name, fontsize=8)  # titre : nom du fichier
    ax.axis("off")  # masque les axes gradues
plt.tight_layout()  # resserre les cases
plt.show()  # affiche la figure

# %% [markdown]
# ## 2. Entrainer (2 epoques : verification de la chaine, pas precision)
#
# Ultralytics fait lui-meme Dataset, augmentations (mosaic...) et boucle ; la toolbox ecrit le
# jeu au format YOLO depuis le pivot, avec le meme pretraitement qu'en production.

# %%
info = workflow.train(config, output, device)  # entraine le detecteur ; renvoie ses infos (moteur, taille d'entree, duree...)
print({k: info[k] for k in ("backend", "input_size", "train_seconds")})  # affiche trois de ces infos

# %% [markdown]
# ## 3. Evaluer : mAP COCO (toute la courbe) ET point de fonctionnement (le seuil de production)
#
# Le mAP resume la courbe precision-rappel ; l'operateur, lui, voit le seuil `conf_predict`.
# On regarde donc aussi precision et rappel a ce seuil, par classe, et la decision par image.

# %%
metrics = workflow.evaluate(config, output, device)  # mesures sur le jeu de test
# mAP50 : precision moyenne avec une boite jugee juste si IoU (recouvrement) >= 0,5 ; mAP50-95 : moyenne sur IoU 0,5 a 0,95
print("mAP50", round(metrics["map50"], 3), "| mAP50-95", round(metrics["map50_95"], 3))
print(pd.read_csv(output / "per_class.csv", index_col=0))  # detail par classe (index_col=0 : 1re colonne = nom de ligne)
display(Image(str(output / "test_predictions.png"), width=520))  # galerie d'images de test avec les boites predites

# %% [markdown]
# ## 4. Exporter et verifier
#
# L'ONNX sort les predictions BRUTES (sans NMS). La verification compare (1) ces sorties au
# reseau PyTorch, (2) les boites obtenues par `postprocess_detect` (letterbox defait, NMS par
# classe, ecrit en numpy) a celles de `model.predict` d'Ultralytics : IoU moyen attendu ~1.

# %%
card = workflow.export(config, output)  # export ONNX verifie ; renvoie la carte du modele (model_card.json)
print(card.notes)  # remarques ecrites pendant l'export (resultat des verifications...)
print(json.dumps(card.input.model_dump(), indent=1))  # description de l'entree attendue par le modele (taille, couleur...)

# %% [markdown]
# ## 5. Comparer les moteurs
#
# Lancer les essais rapides (`quick_yolo`, `quick_yolox`, `quick_fasterrcnn`, `quick_rfdetr`) puis
# `compare.py` : memes images, meme evaluateur. Les vrais chiffres viennent des configurations
# completes, sur GPU.

# %%
comparison = Path("docs/assets/detection/quick_comparison.md")  # tableau comparatif ecrit par compare.py
print(  # affiche le tableau s'il existe
    # sinon : la commande a lancer pour le produire
    comparison.read_text() if comparison.exists() else "lancer : python recipes/detection/scripts/compare.py --markdown " + str(comparison)
)
