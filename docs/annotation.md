# Annotation : outils, formats, pré-annotation, qualité

**Dans le fil : étape 2 (Importer et explorer les données)**, quand les images sont à étiqueter ou arrivent déjà annotées ; la lecture des formats sert ensuite à l'étape 3 (Cataloguer). Ce sont les mêmes numéros dans le cours, dans les notebooks et dans l'index `ETAPES.md`.

L'étiquetage est presque toujours la partie la plus longue d'un projet de vision industrielle, et celle qui limite le plus le résultat : un modèle n'apprend jamais mieux que ses étiquettes. Cette page couvre le choix de l'outil, le format pivot de la toolbox et les conversions, la pré-annotation par un modèle, le guide d'annotation, la mesure de l'accord entre annotateurs et la recherche d'étiquettes suspectes.

## Quel outil pour quel besoin

| Outil | Type | Pour | Points forts | Limites |
|---|---|---|---|---|
| **Outils de la toolbox** (`sort_images.py`, `annotate_boxes.py`, `review_anomalies.py`) | fenêtres OpenCV, aucun serveur | trier en classes, encadrer, relire des anomalies, seul | s'installent avec la toolbox, fonctionnent hors ligne sous Windows, reprise automatique | une personne à la fois, boîtes alignées seulement |
| **Label Studio** | serveur web local (pip) | équipe, toutes les tâches | multi-utilisateurs, pré-annotations importables, configurations prêtes dans `templates/` | serveur à faire tourner, pinceau non lu par la toolbox |
| **CVAT** | serveur web (Docker) | gros volumes de boîtes et polygones, vidéo | le plus complet, SAM intégré, interpolation vidéo | Docker obligatoire |
| **X-AnyLabeling** | application de bureau | annoter seul avec aide automatique | SAM et YOLO intégrés, boîtes orientées, format labelme | poids des modèles à télécharger |
| **Labelme** | application de bureau | polygones simples | installation en une commande | pas d'aide automatique |
| **FiftyOne** | revue de jeux (pas d'annotation) | regarder, filtrer, trouver les doublons et les erreurs | navigation rapide dans des milliers d'images | à installer à part |

:::note
Règle de site industriel : les images ne quittent pas le réseau autorisé. Pas de service d'annotation en ligne, pas de copie hors du site ; tous les outils ci-dessus tournent en local.
:::

## Le format pivot et les conversions

Tous les formats sont lus vers une même structure, `ImageAnnotation` : étiquettes d'image, boîtes, boîtes orientées, polygones, masque. Convertir A vers B, c'est lire A puis écrire B : huit formats demandent seize fonctions, pas cinquante-six convertisseurs.

{{file:src/vtb/annotate/schema.py|77-89}}

| Format | Fichier | Boîtes | Orientées | Polygones | Étiquettes d'image |
|---|---|---|---|---|---|
| VOC | un `.xml` par image | oui | non | non | non |
| YOLO | un `.txt` par image | oui | oui | oui | non |
| COCO | un `.json` | oui | oui (extension) | oui | oui (extension) |
| Labelme / X-AnyLabeling | un `.json` par image | oui | oui | oui | oui (flags) |
| Label Studio | un `.json` | oui | oui | oui | oui (choices) |
| CVAT 1.1 | un `.xml` | oui | oui | oui | oui (tags) |
| Masques PNG | un `.png` par image | déduites | déduites | déduits | non |

Les boîtes orientées sont le piège classique : chaque outil a sa convention (centre et angle pour OpenCV, coin et rotation horaire pour Label Studio, rotation autour du centre pour CVAT, quatre coins pour labelme). Elles passent toutes par le même repère dans `vtb/annotate/geometry.py`, sans approximation.

Sur NEU-DET (1 770 images, 4 126 boîtes), la chaîne VOC → COCO → Labelme → Label Studio → CVAT → pivot ne perd rien :

{{text:docs/assets/annotation/conversion_chain.log|0}}

:::warning
Un format mal lu ne lève pas d'erreur : il donne des boîtes décalées, et l'entraînement « marche » quand même. Après toute conversion d'un nouveau jeu, deux contrôles : `convert.py --check` (relecture et comparaison automatique) et dessiner quelques images.
:::

{{img:docs/assets/annotation/neu_det_boxes.png|NEU-DET : boîtes relues après la chaîne complète de conversions, une image par classe}}

## Des masques aux objets

Beaucoup de jeux industriels fournissent des masques (Magnetic Tile, MVTec, Severstal). `vtb.annotate.masks` en tire des polygones pour YOLO-seg, des boîtes orientées pour YOLO-OBB (la forme naturelle d'une rayure ou d'une fissure) ou des boîtes simples, une par composante connexe. Les bords anticrénelés des PNG sont ignorés, et les composantes trop petites aussi (`min_area`).

{{img:docs/assets/annotation/magnetic_tile_masks_to_objects.png|Magnetic Tile : masque, polygones (YOLO-seg) et boîtes orientées (YOLO-OBB)}}

`epsilon` simplifie les contours : sur une fissure, 138 points avec `epsilon = 0`, 41 avec 1 pixel, 10 avec 2 pixels. Au-delà d'un pixel, le contour s'éloigne visiblement des fissures fines.

## Pré-annotation et apprentissage actif

La boucle qui fait gagner le plus de temps :

1. annoter à la main 100 à 200 images ;
2. entraîner un premier modèle (configuration `quick` d'une recette) ;
3. prédire le reste et exporter les prédictions vers l'outil (`preannotate.py`) ;
4. faire corriger **en commençant par les images où le modèle hésite** (`order_by_uncertainty`) : ce sont celles qui apprennent le plus au modèle suivant ;
5. réimporter, réentraîner, recommencer.

L'incertitude d'une image détectée est le score le plus proche de 0,5 parmi ses objets ; pour la classification, l'écart entre les deux classes les plus probables (marge).

:::warning
Deux règles sans exception. Une proposition du modèle est relue par un humain avant de servir à l'entraînement. Le jeu de **test** est annoté entièrement à la main, sans pré-annotation : sinon on mesure l'accord du modèle avec lui-même.
:::

## Le guide d'annotation

Il s'écrit **avant** la première image, et il évolue : chaque question rencontrée devient une règle, et une nouvelle version. Le modèle `src/vtb/annotate/templates/annotation_guide_template.md` donne la structure :

- des classes **exhaustives** et **sans recouvrement**, chacune avec une définition, des exemples et des contre-exemples ;
- une classe `a_verifier` pour les cas douteux, relus par un expert (jamais une classe « autre » fourre-tout) ;
- les règles de dessin (boîte au plus près, deux défauts distincts = deux boîtes, taille minimale) ;
- les images sans défaut gardées et marquées explicitement ;
- le tableau des cas limites et des décisions prises.

## Mesurer l'accord entre annotateurs

Faire annoter deux fois 5 à 10 % du jeu, par deux personnes qui ne se voient pas. L'accord entre humains est le plafond qu'aucun modèle ne dépassera durablement sur ces étiquettes : à 0,80, c'est souvent le guide qu'il faut corriger, pas le réseau.

| Tâche | Mesure | Lecture |
|---|---|---|
| Classification | kappa de Cohen (accord corrigé du hasard) | < 0,4 faible, 0,6 à 0,8 bon, > 0,8 très bon |
| Détection | appariement optimal des boîtes par IoU ≥ 0,5 : précision, rappel, IoU moyen, accord de classe | le rappel dit ce qu'un annotateur oublie ; l'IoU moyen, la précision du placement |
| Segmentation | IoU et Dice par image, IoU cumulé sur le jeu | l'IoU cumulé est moins sensible aux très petits masques |

Exemple simulé sur NEU-DET (décalages de 3 pixels, 5 % d'oublis, 3 % de fausses boîtes, 4 % de classes changées) : précision 0,938, rappel 0,910, IoU moyen 0,844, accord de classe 0,964 ; kappa 0,942 en classification.

## Étiquettes suspectes

`vtb.annotate.label_issues.find_label_issues` classe les images à relire à partir de probabilités **hors pli** : chaque image est prédite par un modèle qui ne l'a pas vue à l'entraînement (validation croisée). Une image dont la classe donnée reçoit une probabilité très faible, alors qu'une autre classe dépasse son seuil de confiance, est probablement mal étiquetée. En pratique, une bonne partie des images ainsi désignées sont de vraies erreurs. La bibliothèque `cleanlab` fait la version complète si elle est installée.

:::tip
Sans probabilités hors pli, la méthode ne vaut rien : un modèle évalué sur les images qu'il a apprises a aussi appris leurs erreurs, et ne les signale plus.
:::

## Installer les outils sous Windows

Les commandes détaillées (Label Studio dans un environnement séparé avec les fichiers locaux, CVAT avec Docker Desktop, X-AnyLabeling, Labelme) sont dans `recipes/annotation/README.md`. Les fenêtres des outils maison n'ont pas pu être testées dans l'environnement de rédaction (pas d'écran) ; leur logique (touches, souris, reprise de session) est couverte par les tests.
