# Tache : verifier les outils d'annotation interactifs et les allers-retours de formats

Contexte : `src/vtb/annotate/tools/` contient des outils OpenCV a fenetre (tri d'images au
clavier, annotation de boites, revue) et `src/vtb/annotate/` lit / ecrit VOC, YOLO, COCO,
LabelMe, Label Studio, CVAT et masques. Les tests automatiques couvrent les formats mais pas
les fenetres. Lis `recipes/annotation/README.md` et `docs/annotation.md`.

## Etapes

1. Lancer chaque outil sur 20 images de NEU-DET ; verifier raccourcis clavier, annulation,
   sauvegarde, reprise apres fermeture. Corriger ce qui ne marche pas sous Windows (codes de
   touches, DPI, chemins avec accents).
2. Allers-retours : NEU-DET VOC -> COCO -> YOLO -> Label Studio -> pivot, et verifier que les
   boites sont identiques a 0,5 pixel pres (ajouter le test s'il manque).
3. Si Label Studio ou CVAT est disponible sur le poste : importer un export de la toolbox avec
   les modeles de `src/vtb/annotate/templates/`, annoter 5 images, reexporter, relire.
4. Pre-annotation : `preannotate` avec un modele de detection exporte ; verifier l'import des
   propositions dans l'outil.

## Livrable

Liste des verifications (OK / corrige), corrections faites, README d'annotation mis a jour avec
les captures utiles (dans `docs/assets/annotation/`).
