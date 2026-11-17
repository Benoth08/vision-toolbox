# Recette annotation : outils, conversions, pre-annotation, accord

Tout passe par le format pivot de la toolbox (`vtb.annotate.schema.ImageAnnotation`) : chaque
outil exporte dans son format, `convert.py` le ramene au pivot (ou vers un autre format), et les
recettes d'entrainement ne lisent que le pivot ou le catalogue. Rien ne sort du poste ou du
reseau du site : aucun outil en ligne.

| Besoin | Outil conseille |
|---|---|
| Trier seul quelques centaines d'images en classes | `sort_images.py` (trieuse au clavier, sans serveur) |
| Encadrer seul une centaine d'images | `annotate_boxes.py` (boites a la souris, sans serveur) |
| Relire les images signalees par un detecteur d'anomalies | `review_anomalies.py` |
| Annoter seul, polygones et boites orientees, avec aide automatique (SAM, YOLO) | X-AnyLabeling (application de bureau) |
| Annoter a plusieurs, toutes les taches, pre-annotations | Label Studio (serveur web local) |
| Gros volumes de boites / polygones, video, SAM integre | CVAT (Docker) |
| Polygones simples, rapide a installer | Labelme |

## Commandes

```
python recipes/annotation/scripts/sort_images.py --images data/a_trier --classes ok,rayure,inclusion --csv outputs/annotation/tri.csv --copy-to data/raw/mon_jeu
python recipes/annotation/scripts/annotate_boxes.py --images data/a_annoter --classes rayure,inclusion --out outputs/annotation/boites.vtb.json --yolo-dir outputs/annotation/labels
python recipes/annotation/scripts/review_anomalies.py --csv outputs/<recette anomalies>/predictions.csv --out outputs/annotation/revue.csv

python recipes/annotation/scripts/convert.py --from voc --images data/raw/neu_det/IMAGES --annotations data/raw/neu_det/ANNOTATIONS --to coco --out outputs/annotation/neu_det.coco.json --check
python recipes/annotation/scripts/convert.py --from masks --images data/raw/magnetic_tile/MT_Crack/Imgs --annotations data/raw/magnetic_tile/MT_Crack/Imgs --extensions .jpg --kind rotated --classes crack --to yolo --task obb --out outputs/annotation/mt_crack_obb --check
python recipes/annotation/scripts/preannotate.py --predictions outputs/<recette>/predictions.vtb.json --tool labelstudio --out outputs/annotation/pre.json --document-root data/raw --order outputs/annotation/ordre.csv
python recipes/annotation/scripts/agreement.py boxes --images data/raw/neu_det/IMAGES --a voc:annotateur_a --b coco:annotateur_b.json
```

Touches des outils maison (fenetres OpenCV) :
- trieuse : 1-9 classe, retour arriere annuler, s passer, q quitter ;
- boites : glisser = boite, 1-9 classe courante, x / Suppr effacer la derniere, c tout effacer, n / p image suivante / precedente, q quitter ;
- revue : g normal, d defaut, s passer, retour arriere annuler, q quitter.
Les decisions sont enregistrees a chaque touche : relancer la commande reprend la session.
Les fenetres n'ont pas pu etre testees dans l'environnement de redaction (pas d'ecran) ; la
logique (etat, touches, souris) est testee dans `tests/test_annotate_tools.py`.

## Les etapes du fil dans cette recette

Le fil en 13 etapes (le meme que dans le cours et les notebooks : `ETAPES.md`, a la racine) : cette
recette sert l'etape 2 (Importer et explorer : produire et controler les etiquettes) et l'etape 3
(Cataloguer : tout ramener au format pivot). Elle n'a pas de YAML : chaque script prend des arguments.

| script | etapes du fil couvertes | cles du YAML |
|---|---|---|
| `sort_images.py`, `annotate_boxes.py` | 2 Importer et explorer : produire les etiquettes (classes, boites) | aucune (arguments) |
| `convert.py` | 2 et 3 : lire un jeu annote, le convertir au format pivot ou vers un autre format, verifier l'aller-retour | aucune ; le resultat se lit ensuite par `data.annotations_dir` et `data.annotation_format` des recettes |
| `agreement.py` | 2 Importer et explorer : mesurer l'accord entre deux annotateurs (le plafond de tout modele) | aucune (arguments) |
| `preannotate.py` | retour de l'etape 13 vers l'etape 2 : les predictions d'un modele deviennent des pre-annotations a corriger | aucune (arguments) |
| `review_anomalies.py` | retour de l'etape 13 vers l'etape 2 : relire les images signalees par un detecteur d'anomalies | aucune (arguments) |

Le meme contenu, cellule par cellule : `notebooks/04_annotation.py`.

## Label Studio sous Windows (serveur local, multi-utilisateurs)

```
python -m venv C:\outils\labelstudio-venv          # environnement separe : Label Studio a beaucoup de dependances
C:\outils\labelstudio-venv\Scripts\Activate.ps1
pip install label-studio
$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = "true"
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = "C:\donnees"   # dossier racine des images
label-studio start                                   # puis http://localhost:8080
```

1. Creer un projet ; Settings > Labeling Interface > Code : coller un modele de
   `src/vtb/annotate/templates/labelstudio_*.xml` (classification, boxes, rotated, polygons) et
   remplacer les classes.
2. Settings > Cloud Storage > Add Source Storage > Local files : chemin absolu sous la racine
   (par exemple `C:\donnees\neu_det\IMAGES`), cocher "Treat every bucket object as a source file".
   Ou importer directement un fichier de taches produit par `preannotate.py` (les images y sont
   referencees en `/data/local-files/?d=<chemin relatif a la racine>`).
3. Export : format "JSON" (pas "JSON_MIN"), puis
   `convert.py --from labelstudio --images C:\donnees --annotations export.json --document-root C:\donnees --to vtb --out ...`.
   Le pinceau (brush) n'est pas lu : exporter ces projets en "COCO" ou en masques PNG.

## CVAT (Docker Desktop)

```
git clone https://github.com/cvat-ai/cvat && cd cvat
docker compose up -d                                  # puis http://localhost:8080
docker exec -it cvat_server bash -ic "python3 ~/manage.py createsuperuser"
```

Creer une tache, coller les labels de `templates/cvat_labels.json` (Raw), charger les images.
Pre-annotations : `preannotate.py --tool cvat` puis Actions > Upload annotations > "CVAT 1.1".
Export : Actions > Export task dataset > "CVAT for images 1.1", puis `convert.py --from cvat`.

## X-AnyLabeling et Labelme (applications de bureau)

- X-AnyLabeling : telecharger l'executable Windows depuis les "releases" GitHub du projet ;
  Fichier > Ouvrir le dossier d'images ; les modeles d'auto-annotation (SAM, YOLO) se chargent
  depuis l'application (telechargement de poids : a faire chez soi si le reseau le bloque).
  Le format d'enregistrement est celui de labelme (un JSON par image), boites orientees
  comprises (type "rotation") : `convert.py --from labelme`.
- Labelme : `pip install labelme` puis `labelme <dossier>` ; polygones et rectangles, meme lecture.
- Pre-annotations : `preannotate.py --tool xanylabeling --out <dossier des images>` (les JSON
  doivent etre a cote des images, de meme nom).

## La boucle de pre-annotation

1. Annoter a la main 100 a 200 images (et tout le jeu de test, toujours a la main).
2. Entrainer un premier modele (configuration `quick` d'une recette).
3. Predire le reste ; `preannotate.py` exporte vers l'outil et ecrit l'ordre de relecture
   (images ou le modele hesite d'abord).
4. Corriger dans l'outil, exporter, `convert.py` vers le pivot, reentrainer. Recommencer.

## Resultats obtenus (notebook, environnement de redaction)

- NEU-DET (1770 images, 4126 boites) : VOC -> COCO -> Labelme -> Label Studio -> CVAT -> vtb,
  0 difference a chaque etape (tolerance 0,01 px), de 0,1 a 4 s par format sur CPU.
- Magnetic Tile : 392 images de defauts -> 442 polygones (YOLO-seg) et 442 boites orientees
  (YOLO-OBB) ; un contour de fissure passe de 138 a 41 points avec epsilon = 1 px.
- Accord simule (decalages de 3 px, 5 % d'oublis, 3 % de fausses boites, 4 % de classes
  changees) : precision 0,938, rappel 0,910, IoU moyen 0,844 ; kappa de classification 0,942.

Voir aussi `docs/annotation.md` (article du site) et `src/vtb/annotate/templates/annotation_guide_template.md`.
