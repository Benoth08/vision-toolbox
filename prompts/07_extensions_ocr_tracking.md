# Tache : etendre la toolbox (OCR de marquage, suivi d'objets, jeux Kaggle)

Chaque extension suit le contrat de `CLAUDE.md` : `src/vtb/tasks/<tache>/`, une recette avec
cinq verbes, une configuration quick, un test synthetique, un export verifie et une carte de
modele relisible par `vtb.serve.predictor`.

## Pistes (une a la fois, me demander laquelle)

1. **OCR de marquage** (numero de bobine, frappe a froid, jet d'encre) : detection de la zone
   (recette detection) puis lecture. Comparer une solution pre-entrainee (docTR ou PaddleOCR,
   verifier licence et export ONNX) et un CRNN + CTC entraine sur des marquages synthetiques
   generes avec des polices industrielles. Metriques : exactitude par caractere et par chaine.
2. **Suivi** (pieces sur convoyeur, defaut vu sur plusieurs images) : ByteTrack par
   Ultralytics (`model.track`) branche sur la recette detection ; compter une fois chaque objet.
3. **Severstal Steel Defect Detection** (Kaggle, segmentation de bande) : la cle Kaggle est
   dans les variables `KAGGLE_USERNAME` / `KAGGLE_KEY` (jamais dans un fichier) ; ecrire
   `tools/get_data/get_severstal.py` (decodage RLE -> masques), puis une configuration de la
   recette segmentation. Respecter les conditions d'utilisation du jeu (pas de redistribution).
4. **Kolektor SDD2** ou **DAGM 2007** : configurations de la recette anomalie / segmentation.

## Livrable

Pour la piste choisie : code, recette, test, README avec chiffres reels.
