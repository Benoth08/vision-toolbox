# Tache : banc complet des modeles d'anomalies, y compris EfficientAd et DRAEM

Contexte : `recipes/anomaly/` compare l'auto-encodeur, le PatchCore de la toolbox et les modeles
anomalib sous un protocole unique (normales seules en train / val, seuil = quantile 0,99 des
normales de validation, test une seule fois). Sur CPU, sans acces a Hugging Face, une partie
seulement a tourne (voir le README de la recette). EfficientAd (jeu auxiliaire Imagenette) et
DRAEM (textures DTD) telechargent des donnees supplementaires : a lancer sur GPU.

## Etapes

1. Regles RunPod de `03_gpu_benchmarks_runpod.md` (confirmation, duree maximale, suppression).
2. Sur le pod : `pip install -e .[anomaly,export]`, donnees MVTec bottle et Magnetic Tile.
3. Lancer toutes les configurations `recipes/anomaly/configs/*.yaml` (bouteille puis tuiles),
   avec les vrais poids pre-entraines (timm / Hugging Face, accessibles sur le pod).
4. `python recipes/anomaly/scripts/benchmark.py` -> `docs/assets/anomaly/benchmark.md`.
5. Si un modele echoue (version d'anomalib, parametre renomme), corriger l'adaptateur
   `src/vtb/tasks/anomaly/anomalib_adapter.py` en gardant le contrat (carte brute ->
   MapToScore), avec un test.
6. Comparer aux chiffres CPU deja dans le README : les ecarts dus aux poids reels (au lieu des
   poids de secours de l'environnement de developpement) sont attendus ; les expliquer.

## Livrable

Le tableau complet (AUROC image et pixel, rappel, fausses alarmes, temps, latence ONNX) dans le
README de la recette, et la liste des modeles recommandes par cas (texture, objet, peu de
normales, temps reel).
