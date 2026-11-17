# Le mode notebook : toute la toolbox, cellule par cellule

La toolbox existe en deux modes qui font la meme chose :

| | mode production | mode notebook |
|---|---|---|
| ou | `src/vtb/` + `recipes/<tache>/` | `notebooks/` |
| comment | un YAML, cinq scripts (`prepare`, `train`, `evaluate`, `predict`, `export`) | un notebook qu'on lit et execute de haut en bas |
| le code | range en briques reutilisables, teste | deroule dans les cellules : chargement, decoupage, modele, boucle d'entrainement, metriques, export |
| pour | livrer, refaire a l'identique, comparer des experiences | comprendre, explorer, montrer, prototyper sur de nouvelles donnees |

Les deux modes utilisent les memes donnees, le meme decoupage (meme algorithme, meme graine) et les
memes reglages ; un notebook exporte `model.onnx` + `model_card.json` au format de la production :
le modele est servi tel quel par `vtb.serve` (service HTTP) et par le programme C#.

## Le fil en 13 etapes

Le cours, les notebooks, les recettes et la bibliotheque suivent le meme fil, toujours dans le meme
ordre et avec les memes noms. Le detail (ou trouver chaque etape dans la toolbox) est dans
[`../ETAPES.md`](../ETAPES.md).

| phase | etapes |
|---|---|
| Donnees | 1 Imports et reglages, 2 Importer et explorer, 3 Cataloguer, 4 Decouper, 5 Pretraiter, 6 Augmenter, 7 Charger par lots |
| Modele | 8 Modele, 9 Perte et optimiseur |
| Apprentissage | 10 Entrainer, 11 Evaluer |
| Livraison | 12 Sauvegarder et exporter, 13 Predire |

- Notebooks de tache (00, 07 a 15) : chaque section porte son etape dans le titre
  (`## Etape 4 - Decouper : ...`, ou `## Etapes 3 et 4 - Cataloguer et decouper : ...` quand une
  section en couvre plusieurs) et l'en-tete donne "Le fil" du notebook, dans l'ordre des cellules.
  Les sections sans numero (`Resultats`, `Adapter a tes donnees`, `Passer en production`,
  `Pour aller plus loin`) sont hors fil.
- Notebooks thematiques (01 a 06, 16 a 19) : chacun approfondit une ou plusieurs etapes ; l'en-tete
  le dit ("Dans le fil : ...") et les titres de section concernes portent l'etape entre parentheses.

## Les notebooks

Etape(s) : les etapes du fil que le notebook couvre. Temps : mode `QUICK` sur un CPU 2 coeurs
(indicatif). Resultats : ceux de l'execution livree (sous-ensembles et peu d'epoques en `QUICK` ;
les essais de detection verifient la chaine, pas la precision).

| notebook | etape(s) | contenu | QUICK | resultat de l'execution livree | recette |
|---|---|---|---|---|---|
| `00_vue_ensemble` | 1 a 13 | la chaine entiere, courte : dossier -> decoupage -> pretraitement -> modele pre-entraine -> entrainement -> evaluation -> ONNX verifie -> prediction sans PyTorch | 41 s | exactitude test 0,989 ; ONNX = PyTorch a 5e-6 | classification |
| `01_donnees_decoupage` | 2, 3, 4 | lecture 8/16 bits, catalogues, decoupages stratifie / groupe / temporel, fuites, moyenne et ecart-type | 6 s | decoupages identiques a 100 % a ceux des recettes | toutes |
| `02_opencv_pretraitements` | 5 | la boite a recettes OpenCV : ~55 operations en 17 familles, aide-memoire, chaine complete jusqu'aux mesures en mm (100 % OpenCV, copiable partout) | 11 s | 7 ms par tuile pour la chaine complete | preprocessing |
| `03_augmentation` | 6 | le module `vtb.augment` : presets, operations industrielles, boites et masques, mixup / cutmix, copy-paste, controle d'alignement, vitesse | 154 s | 25 images / classe : preset sans gain a acquisition egale (0,888 contre 0,932), +14 points quand l'eclairage change (0,567 contre 0,431) | toutes |
| `04_annotation` | 2, 3 | formats VOC, YOLO, COCO, LabelMe, Label Studio, CVAT, masques ; controle ; accord entre annotateurs ; erreurs d'etiquette | 28 s | aller-retour sur 6 formats sans aucune difference ; 40 erreurs injectees sur 40 retrouvees | annotation |
| `05_exploration_qualite` | 2 | illisibles, doublons, raccourcis, plongements, derive | 42 s | 161 groupes de quasi-doublons (Magnetic Tile) ; AUC de derive 0,39 sans derive, 0,996 simulee | explore |
| `06_modeles_transfert` | 8, 9 | choisir un modele pre-entraine, strategies de transfert comparees | 74 s | partial 0,967, progressive 0,963, features 0,948 (sous-ensemble 96 px) | classification |
| `07_classification` | 1 a 13 | NEU-CLS de bout en bout : pretraitement choisi a la metrique, entrainement, reseau de zero, Grad-CAM, seuil de rejet, export | 194 s | resnet18 1,000 ; reseau de zero 0,948 ; 0,4 % de rejet | classification |
| `08_detection_yolo` | 1 a 13 | YOLO11 : boites, boites orientees, instances, tuiles | 83 s | mAP50 0,130 = recette ; export IoU 1,000 | detection |
| `09_detection_yolox` | 1 a 13 | YOLOX (Apache-2.0) : letterbox haut-gauche, decode + NMS en numpy | 51 s | mAP50 0,137 = recette ; export IoU 1,000 | detection |
| `10_detection_faster_rcnn` | 1 a 13 | Faster R-CNN torchvision, boucle d'entrainement ecrite | 206 s | mAP50 0,068 = recette ; export IoU 1,000 | detection |
| `11_detection_rfdetr` | 1 a 13 | RF-DETR, top-k sans NMS (poids aleatoires ici : les poids COCO n'etaient pas accessibles) | 202 s | mAP50 0,002 = recette ; scores ONNX a 3e-8 | detection |
| `12_segmentation` | 1 a 13 | U-Net sur Magnetic Tile, decoupage groupe recalcule, perte dice_ce, IoU / Dice, decision par image | 397 s | Dice defauts 0,67 en 8 epoques (recette 15 epoques : 0,83) | segmentation |
| `13_anomalies_autoencodeur` | 1 a 13 | auto-encodeur convolutif, seuil sur les saines de validation, echec explique sur texture | 133 s | bouteille AUROC 0,822 ; tuiles 0,547 | anomaly |
| `14_anomalies_patchcore` | 1 a 13 | PatchCore ecrit a la main (coreset glouton, kNN), exporte d'un bloc | 72 s | AUROC image 1,000 / pixel 0,982 = recette, meme seuil | anomaly |
| `15_anomalies_anomalib` | 1 a 13 | PaDiM, STFPM, FastFlow avec le meme protocole | 367 s | PaDiM 0,995 / 0,982 ; STFPM 1,000 / 0,977 = recette | anomaly |
| `16_vision_classique` | 5, 11 | seuillage, blobs, mesure calibree en mm, comparaison a une reference | 9 s | metriques identiques a la recette ; 60,255 mm mesures pour 60,25 | classical |
| `17_export_onnx` | 12 | export ONNX (dynamo / ancien, lot dynamique ou fixe), verification, OpenVINO, latence, carte du modele | 33 s | ONNX = PyTorch a 7e-6 ; lot fige detecte | toutes |
| `18_prediction_production` | 13 | predicteur ONNX sans PyTorch ecrit a la main, service FastAPI, fichiers de reference C# | 8 s | memes reponses que `vtb.serve` sur 6 modeles | toutes |
| `19_boite_a_outils` | toutes | `vtb.viz` (affichage) et `vtb.toolkit` (outils de travail, ligne de commande `vtb-kit`) | 61 s | exactitude 0,950 [0,908 ; 0,983] par bootstrap ; McNemar p = 0,039 | - |

Les notebooks 12 a 18 utilisent les donnees de `tools/get_data/` ; 18 relit aussi les modeles
exportes par 17 et par les recettes quand ils existent (les absents sont ignores).

## Lancer un notebook

Depuis la racine du depot (les chemins des donnees sont relatifs a la racine) :

```
python tools/get_data/get_neu_cls.py                              # les donnees publiques (une fois)
python tools/notebooks.py notebooks/07_classification.py --execute  # .py -> .ipynb execute
```

La source est le `.py` (cellules `# %%`) : dans VS Code, chaque cellule s'execute avec
Shift+Entree, sans Jupyter. Le `.ipynb` est genere a partir du `.py` et garde ses sorties.

## La cellule REGLAGES

La premiere cellule de code de chaque notebook regroupe tout ce qu'on change :

- `DATA_DIR`, le format des donnees et les classes : pour passer a ses propres images ;
- `QUICK = True` : sous-ensemble et peu d'epoques, quelques minutes sur CPU (verifie la chaine) ;
  `QUICK = False` : la configuration complete de la recette (GPU conseille) ;
- `SEED`, `DEVICE` (cuda si disponible), `OUT` (dossier des sorties, `outputs/notebooks/<nom>`).

Chaque notebook se termine par "Passer en production" : le YAML equivalent de la recette, la
commande, et une cellule qui verifie que le modele exporte par le notebook est servi a l'identique
par `vtb.serve.predictor.OnnxPredictor`.

## Regles

- Chaque ligne de code est commentee (a droite, ou juste au-dessus quand la ligne est longue) :
  ce qu'elle fait et pourquoi, avec les termes techniques expliques a leur premiere apparition.
- Le code metier est dans les cellules ; seuls l'affichage (`vtb.viz`) et, dans 03 et 19,
  `vtb.augment` / `vtb.toolkit` viennent de la toolbox.
- Les seuils viennent de la validation, jamais du test ; le test est lu une seule fois.
- Un notebook n'ecrit que dans `outputs/notebooks/<nom>/`.
