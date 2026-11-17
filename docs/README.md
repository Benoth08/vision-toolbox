# docs/ : les articles du site

Chaque fichier `docs/<sujet>.md` devient une page du site (voir le generateur du site) ; sa
premiere ligne `# Titre` devient le titre de la page (le titre court du menu est dans
`site/site.yaml`, cle `titles`). Les
articles sont ecrits en francais AVEC accents (c'est du texte a lire, comme le cours) ; le code
reste en ASCII.

Directives reconnues, seules sur leur ligne, chemins relatifs a la racine du depot :

    {{file:src/vtb/preprocess/ops/illumination.py}}        fichier entier, avec son nom
    {{file:src/vtb/preprocess/ops/illumination.py|43-90}}  lignes 40 a 80
    {{img:docs/assets/preprocessing/shading.png|legende}}  image (integree en base64)
    {{text:docs/assets/classification/train.log|0}}        sortie texte (0 = tout, N = N dernieres lignes)
    :::note / :::warning / :::memo / :::tip ... :::        encadres
    :::details Titre ... :::                               bloc repliable

Les figures et journaux utilises par les articles sont copies dans `docs/assets/<sujet>/`
(petits fichiers, versionnes) par les scripts des recettes, pour que le site se reconstruise
sans relancer les entrainements.
