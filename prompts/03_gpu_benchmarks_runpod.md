# Tache : lancer les configurations completes sur GPU (RunPod) et remplir les tableaux de resultats

Contexte : sur CPU seuls les essais `quick` ont tourne. Il faut les vrais chiffres des
configurations nominales. Le GPU est loue sur RunPod via `tools/runpod/run_remote.py`.

## Regles (imperatives)

- La cle est dans `RUNPOD_API_KEY` ou dans `.env`. Ne jamais l'afficher, l'ecrire dans un
  fichier, un message ou une commande ; si elle manque, s'arreter et le dire.
- Avant CHAQUE creation de pod : afficher type de GPU, prix horaire, duree maximale, cout
  maximal, et attendre mon accord explicite dans la conversation. `--yes` seulement apres.
- Toujours `--max-minutes` ; le script supprime le pod dans un `finally`. Apres chaque lancement
  et a la fin : `python tools/runpod/run_remote.py --list` doit montrer zero pod.
- Commencer par un essai de 15 minutes maximum (`pytest -q` + une config quick) pour valider la
  chaine distante avant les longues.
- Budget total a ne pas depasser : celui que je donne au debut (le demander s'il manque).

## A lancer (regrouper plusieurs experiences par pod pour limiter les demarrages)

1. Detection NEU-DET : `yolo11n_neudet`, `yolox_tiny_neudet`, `fasterrcnn_resnet18_neudet`,
   `fasterrcnn_resnet50_coco_neudet`, `rfdetr_nano_neudet`, `rfdetr_base_neudet` (extra `rfdetr`,
   poids telecharges sur le pod), puis `compare.py`.
2. Detection Magnetic Tile : `yolo11n_obb_tile`, `yolo11n_seg_tile`.
3. Segmentation : `unet_resnet18_tile`, `deeplabv3plus_resnet18_tile`,
   `unetplusplus_resnet34_tile`, `unet_resnet18_tile_multiclass`.
4. Classification : `efficientnet_b0_partial`, `mobilenet_v3_large_partial`,
   `convnext_tiny_partial`, `timm_example` (poids telechargeables sur le pod).
5. Rapatrier chaque dossier `outputs/...` (`--fetch`), puis en local : `evaluate` deja fait sur
   le pod, verifier `model_card.json` et relancer `export` en local si la verification ONNX a
   besoin du CPU local.

## Livrable

Les tableaux "Resultats obtenus" des README `recipes/detection`, `recipes/segmentation` et
`recipes/classification` remplis avec les chiffres reels (GPU, temps, mAP / IoU / F1,
latences), un fichier `docs/assets/<recette>/comparison.md` par recette, le cout total reel, et
la preuve que plus aucun pod ne tourne.
