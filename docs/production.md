# Du modèle entraîné au programme de production

**Dans le fil : étapes 12 (Sauvegarder et exporter) et 13 (Prédire et mettre en production).** Ce sont les mêmes numéros dans le cours, dans les notebooks et dans l'index `ETAPES.md`.

Un modèle n'est utile qu'une fois branché sur la ligne, dans un programme qui n'a souvent ni Python ni PyTorch : un service C#, une application de supervision, un PC industriel. La toolbox fait de cette étape une partie du contrat de chaque recette : `export` produit un modèle ONNX **vérifié** et une **carte du modèle** qui dit tout ce que le programme de production doit savoir, puis deux implémentations de référence relisent ces deux fichiers sans PyTorch (Python, puis C#).

## Les trois fichiers

| Fichier | Contenu |
|---|---|
| `model.onnx` | le réseau, avec un lot dynamique ; pour les anomalies et la segmentation, normalisation et softmax sont dans le graphe |
| `model.xml` / `model.bin` | la même chose au format OpenVINO (souvent 2 à 4 fois plus rapide sur un CPU Intel) |
| `model_card.json` | tâche, entrée (taille, couleur de lecture, mise à l'échelle, normalisation), prétraitement OpenCV, sorties, classes, seuils, post-traitement, métriques du test, latences |

L'export **échoue** si ONNX ne reproduit pas PyTorch sur de vraies images (tolérance relative à l'amplitude des sorties), avec un lot d'une taille différente de celle de l'export. En détection, il vérifie aussi les boîtes finales : chaîne ONNX + post-traitement écrit en numpy contre les boîtes du moteur.

Deux pièges vus en écrivant le notebook `17_export_onnx` :

- **Un lot figé peut passer inaperçu.** Un `reshape` écrit avec la taille du lot d'exemple tourne parfois avec un autre lot, mais rend une sortie de la mauvaise taille. L'export vérifie donc que chaque sortie a autant de lignes que d'images en entrée ; sinon il réexporte avec un lot fixe de 1 et l'écrit dans la carte (`dynamic_batch: false`).
- **OpenVINO calcule en bfloat16 par défaut sur les CPU qui le permettent.** Sur la machine de test, l'écart avec onnxruntime atteint 0,14 sur des logits et 0,03 sur des probabilités (même classe prédite partout). Les latences OpenVINO des recettes sont mesurées dans ce mode, le plus rapide. En précision f32 (`core.compile_model(xml, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})`), l'écart retombe à 0,016 sur les logits et 0,004 sur les probabilités. Dans tous les cas, valider le seuil de décision dans le mode retenu pour la production.

## La carte du modèle

Exemple d'une carte de détection (YOLOX) : la couleur de lecture du fichier (`read_color`), le prétraitement à appliquer, puis la façon de construire le tenseur d'entrée.

```json
{
  "task": "detection",
  "model_name": "yolox:yolox_tiny",
  "input": {"name": "images", "height": 320, "width": 320, "channels": 3, "color": "BGR",
            "scale": "none (values 0-255)", "resize": "letterbox_top_left", "letterbox_pad_value": 114,
            "read_color": "gray"},
  "preprocess": [],
  "outputs": [{"name": "output", "shape": ["batch", "anchors", 11], "description": "sortie brute NON decodee"}],
  "class_names": ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"],
  "thresholds": {"conf": 0.25, "iou_nms": 0.6},
  "postprocess": {"steps": "decoder les grilles de pas 8/16/32 ... -> NMS par classe -> diviser par le ratio"}
}
```

:::memo
La règle : tout ce qui transforme l'image entre le fichier et le réseau est **dans la carte**, sous forme d'opérations nommées et de paramètres. Rien ne se réécrit « de mémoire » dans le programme de production.
:::

## La référence Python sans PyTorch

`vtb.serve.predictor.OnnxPredictor` relit un dossier exporté et prédit avec numpy, OpenCV et onnxruntime seulement. Il implémente les quatre tâches (classification, anomalie, segmentation, détection avec les quatre décodeurs : YOLO11, YOLOX, Faster R-CNN, RF-DETR) et la suppression des non-maxima en numpy. Un test vérifie, pour chaque tâche, qu'il rend **les mêmes réponses** que le workflow PyTorch : c'est la preuve que la carte suffit.

{{file:src/vtb/serve/predictor.py|95-132}}

## Le service HTTP

`python -m vtb.serve --model outputs/<tâche>/<expérience>` démarre un service FastAPI générique : `GET /health`, `GET /card`, `POST /predict` (un fichier image, réponse JSON). Le même code sert toutes les tâches. Le `docker/Dockerfile.serve` construit une image sans PyTorch (environ 400 Mo) :

```
docker build -f docker/Dockerfile.serve -t vtb-serve .
docker run -p 8000:8000 -v %CD%/outputs/classification/resnet18_partial:/model vtb-serve
curl -F "file=@piece.png" http://localhost:8000/predict
```

## Le programme C#

`csharp/VtbInference` est le portage C# du prédicteur Python (ONNX Runtime et OpenCvSharp) : lecture de la carte, opérations de prétraitement courantes portées à l'identique (resize, crop, flous, CLAHE, gamma, étirement par percentiles, division par le fond), construction du tenseur, post-traitement des quatre tâches. La vérification se fait par **fichiers de référence** :

```
python tools/make_golden.py --model outputs/detection/yolox_tiny_neudet --images data/raw/neu_det/Validation_Images --n 5
dotnet run -c Release --project csharp/VtbInference -- --model outputs/detection/yolox_tiny_neudet --golden outputs/detection/yolox_tiny_neudet/golden.json
```

`golden.json` contient, pour chaque image, la somme du tenseur d'entrée (un écart vient du prétraitement) et la réponse attendue (un écart seul vient du post-traitement). Le programme affiche `entree OK / sortie OK` image par image.

:::warning
Sources de divergence connues entre Python et C# : la couleur de lecture (OpenCV lit en BGR), les arrondis (`Mat.ConvertTo` arrondit déjà, Python ajoute 0,5 puis tronque), et surtout le letterbox, centré pour YOLO11 mais collé en haut à gauche pour YOLOX. Chaque nouveau modèle passe par la vérification golden avant d'aller sur la ligne.
:::

## Entraîner sur GPU à la demande

Les configurations complètes (détection 60 époques, segmentation 256 pixels, EfficientAd) demandent un GPU. `tools/runpod/run_remote.py` en loue un sur RunPod le temps d'une expérience, avec trois garde-fous : il affiche le type de carte, le prix horaire, la durée et le coût maximaux et demande de taper « oui » avant de créer le pod ; chaque étape distante a un délai ; le pod est supprimé dans un bloc `finally`, quoi qu'il arrive, puis la liste des pods restants est affichée. La clé est lue dans `RUNPOD_API_KEY` ou un fichier `.env` ignoré par git, jamais écrite ailleurs.

```
python tools/runpod/run_remote.py run --gpu "NVIDIA RTX A5000" --max-minutes 60 \
    --setup "pip install -e .[detection] && python tools/get_data/get_neu_det.py" \
    --command "python recipes/detection/scripts/run.py --config recipes/detection/configs/yolo11n_neudet.yaml" \
    --fetch outputs/detection/yolo11n_neudet
python tools/runpod/run_remote.py --list
```

## Poste sans Internet

`tools/fetch_weights.py` télécharge à l'avance les poids pré-entraînés (torchvision, Ultralytics, YOLOX, Hugging Face) dans `weights/`, avec un manifeste (taille et SHA-256 de chaque fichier). Sur le poste isolé, `TORCH_HOME=weights/torch` et `HF_HOME=weights/hf` suffisent à la toolbox pour les retrouver ; le zoo de modèles affiche un message explicite si un poids manque.
