# csharp/ : inference des modeles de la toolbox en C# (ONNX Runtime + OpenCvSharp)

Le programme de production ne lit que deux fichiers, ecrits par l'etape `export` de chaque
recette : `model.onnx` et `model_card.json`. La carte dit tout ce qu'il faut savoir : taille et
couleur d'entree, pretraitement OpenCV a appliquer, mise a l'echelle, normalisation, sorties,
classes, seuils et post-traitement. Ce projet est le portage C# de `src/vtb/serve/predictor.py`
(le predicteur Python sans PyTorch), qui sert de reference.

**Etat : ecrit mais non compile dans l'environnement ou il a ete produit (pas de .NET).** La
compilation et la verification se font sur un poste Windows avec le SDK .NET 8 (voir
`prompts/` pour la procedure guidee).

## Contenu

| Fichier | Role |
|---|---|
| `VtbInference.csproj` | projet .NET 8 ; paquets Microsoft.ML.OnnxRuntime et OpenCvSharp4 (+ runtime Windows) |
| `ModelCard.cs` | lecture de model_card.json |
| `Preprocessing.cs` | operations de `vtb.preprocess` portees a l'identique : resize, crop, to_gray, gaussian_blur, median_blur, clahe, gamma, percentile_stretch, shading_division |
| `Predictor.cs` | lecture d'image, tenseur d'entree, inference, post-traitement des 4 taches (classification, anomalie, segmentation, detection Ultralytics / YOLOX / Faster R-CNN / RF-DETR) et NMS |
| `Golden.cs` | comparaison aux reponses de reference Python (`--golden`) |
| `Program.cs` | ligne de commande |

## Construire et lancer (Windows, SDK .NET 8)

```
cd csharp\VtbInference
dotnet build -c Release
dotnet run -c Release -- --model ..\..\outputs\classification\resnet18_partial --image ..\..\data\raw\neu_cls_flat\crazing
```

Derriere un proxy d'entreprise, les paquets NuGet viennent du depot interne : adapter les
versions du `.csproj` a celles qui y sont disponibles (ONNX Runtime >= 1.16, OpenCvSharp4 >= 4.8).

## Verifier contre Python (a faire pour chaque nouveau modele)

1. Cote Python, ecrire les reponses de reference :
   `python tools/make_golden.py --model outputs/<recette>/<experience> --images <dossier> --n 5`
2. Cote C# :
   `dotnet run -c Release -- --model <meme dossier> --golden <meme dossier>\golden.json`
3. Attendu : `entree OK` et `sortie OK` pour chaque image. Un ecart d'entree vient du
   pretraitement (operation mal portee, interpolation, arrondi) ; un ecart de sortie seul vient
   du post-traitement (seuil, NMS, decodage).

## Points d'attention (les sources de divergence connues)

- **Couleur de lecture** : `input.read_color` dit si le fichier est lu en gris ou en RGB AVANT
  le pretraitement (un modele de detection entraine en gris lit en gris puis copie sur 3 canaux).
  OpenCV lit en BGR : la conversion en RGB est faite dans `Predictor.Read`.
- **Arrondis** : Python ecrit `(v * peak + 0,5)` tronque ; `Mat.ConvertTo` arrondit deja, sans
  le +0,5. `round()` de Python et `Math.Round` arrondissent tous deux au pair le plus proche.
- **Letterbox** : YOLO11 centre l'image (bordure 114 de chaque cote) ; YOLOX la colle en haut a
  gauche, en BGR, sans division par 255. Ne jamais melanger les deux.
- **Faster R-CNN** : redimensionnement et NMS sont dans le graphe ONNX ; une image a la fois ;
  les etiquettes sortent en int64.
- **RF-DETR** : redimensionnement SANS garder les proportions (`resize: stretch`), normalisation
  ImageNet, pas de NMS (top-k des couples requete / classe) ; le dernier indice de classe ne sert pas.
- **Operation absente** : `NotSupportedException` avec son nom. La porter depuis
  `src/vtb/preprocess/ops/` (chaque docstring donne l'equivalent OpenCvSharp) puis refaire la
  verification golden.
- **Performances** : le remplissage du tenseur par indexeur est simple a lire ; pour des images
  de plusieurs megapixels, passer par `Span<float>` sur `tensor.Buffer`.
