# Tache : compiler le projet C# et prouver qu'il donne les memes reponses que Python

Contexte : `csharp/VtbInference/` est le portage C# (ONNX Runtime + OpenCvSharp) de
`src/vtb/serve/predictor.py`. Il a ete ecrit sans compilateur : attends-toi a des erreurs de
compilation mineures. Lis `csharp/README.md`.

## Etapes

1. `dotnet --info` (SDK 8 requis). `cd csharp/VtbInference && dotnet build -c Release`.
   Corriger les erreurs de compilation sans changer le comportement ; aligner les versions des
   paquets NuGet sur celles disponibles (depot interne eventuel).
2. Produire un modele exporte par tache (essais rapides de `01_setup_and_checks.md`) :
   classification, anomalie (patchcore_lite et un modele anomalib), segmentation, detection
   (YOLO11, YOLOX, Faster R-CNN).
3. Pour chacun : `python tools/make_golden.py --model <dossier> --images <dossier d'images> --n 5`
   puis `dotnet run -c Release -- --model <dossier> --golden <dossier>/golden.json`.
4. Tout ecart : identifier l'etape (entree = pretraitement ; sortie = post-traitement), corriger
   le C# pour reproduire Python (Python est la reference ; si Python est faux, le dire et
   corriger les deux, avec un test Python).
5. Porter les operations de pretraitement manquantes utilisees par les configurations du depot
   (`NotSupportedException`), chacune verifiee par un golden.
6. Mesurer la latence C# (CPU) par tache et la comparer a `latency_ms` de la carte.

## Livrable

Le projet compile, `--golden` affiche "tout est identique a Python" pour chaque modele, le
README C# est mis a jour (versions, latences, operations portees).
