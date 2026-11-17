# Détection d'anomalies : apprendre le normal, signaler l'écart

En inspection, les défauts graves sont rares : quelques exemples par an, jamais deux identiques. Impossible d'entraîner un classifieur sur une classe qu'on n'a presque jamais vue. Les méthodes d'anomalie renversent le problème : elles apprennent l'aspect **normal** à partir d'images saines, qu'on a en abondance, et signalent tout ce qui s'en écarte. Les défauts ne servent qu'à **évaluer**.

## Un protocole unique pour toutes les méthodes

La comparaison n'est honnête que si toutes les méthodes suivent les mêmes règles. Dans la toolbox, elles sont fixées par la recette, quelle que soit la méthode :

1. **Découpage** : l'entraînement et la validation ne contiennent que des images saines ; tous les défauts sont dans le test. Un défaut vu à l'entraînement fausserait la mesure.
2. **Seuil** : le quantile 0,99 des scores des images saines de **validation**. On accepte une fausse alarme sur cent pièces saines, sans avoir vu un seul défaut.
3. **Carte et score** : chaque méthode produit une carte brute ; un même module la lisse (filtre gaussien), la ramène à la taille de l'image et en tire le score de l'image (maximum, ou moyenne des pixels les plus anormaux).
4. **Évaluation** : AUROC image et pixel (indépendants du seuil), puis rappel et taux de fausses alarmes **au seuil**, et rappel par type de défaut.

{{file:src/vtb/tasks/anomaly/workflow.py|44-52}}

Le modèle exporté contient tout : réseau, carte brute, lissage et score. Il prend une image dans [0, 1] et rend la carte et le score ; le seuil est dans la carte du modèle.

## Les méthodes

| Méthode | Principe | Entraînement | Forces | Limites |
|---|---|---|---|---|
| PatchCore (toolbox et anomalib) | mémoire de vecteurs de zones normales ; score = distance au plus proche | aucun (une passe) | la référence ; excellent sur textures et objets | mémoire et temps d'inférence qui croissent avec le nombre d'images |
| PaDiM | une gaussienne par position de l'image | aucun | très rapide | demande des pièces alignées |
| DFM | PCA des caractéristiques profondes | aucun | base de comparaison très rapide | moins précis |
| STFPM | un étudiant imite un professeur pré-entraîné ; l'écart signale l'anomalie | oui | bonne localisation | à entraîner |
| Reverse Distillation | un décodeur reconstruit les caractéristiques du professeur | oui | très bon sur MVTec | plus lourd |
| FastFlow | flot normalisant sur les caractéristiques | oui | rapide en inférence | réglage délicat |
| SuperSimpleNet | discriminateur entraîné sur des anomalies synthétiques | oui | récent, bon compromis | à entraîner |
| EfficientAd | étudiant-professeur léger + auto-encodeur | oui (jeu auxiliaire) | très rapide en inférence | télécharge Imagenette |
| DRAEM | reconstruction + segmentation d'anomalies synthétiques (textures) | oui (jeu auxiliaire) | localisation fine | télécharge DTD, long |
| Auto-encodeur convolutif | reconstruire le normal ; l'erreur signale l'anomalie | oui | simple, explicable | faible sur textures fines |

Le PatchCore de la toolbox (`patchcore_lite`) est écrit en clair, en une centaine de lignes : un ResNet18 gelé, les caractéristiques de deux niveaux agrégées par zone, une mémoire réduite par **coreset k-center glouton** (à chaque pas, on garde le vecteur le plus éloigné de ceux déjà choisis), et la distance au plus proche voisin calculée par un produit matriciel. Il s'exporte en ONNX d'un bloc.

{{file:src/vtb/tasks/anomaly/patchcore_lite.py|68-98}}

## Résultats

Banc sur CPU (deux cœurs partagés, temps indicatifs), backbone resnet18 pour toutes les méthodes, images de 256 pixels. Les configurations de référence des articles (wide_resnet50, plus d'époques, EfficientAd et DRAEM) se lancent sur GPU avec `prompts/04_anomalib_full_benchmark.md`.

**MVTec AD bottle** (objet sur fond propre ; test de 20 bouteilles saines et 63 défectueuses) :

| Méthode | AUROC image | AUROC pixel | Rappel au seuil | Fausses alarmes | Construction (s) |
|---|---|---|---|---|---|
| PatchCore de la toolbox (mémoire 2 %) | 1,000 | 0,982 | 1,000 | 0,000 | 140 |
| PatchCore d'anomalib (coreset 1 %) | 1,000 | 0,982 | 1,000 | 0,000 | 461 |
| PaDiM | 0,995 | 0,982 | 0,984 | 0,050 | 90 |
| DFM | 1,000 | 0,935 | 1,000 | 0,050 | 30 |
| STFPM (15 époques) | 1,000 | 0,977 | 1,000 | 0,050 | 273 |
| Reverse Distillation | 0,998 | 0,984 | 1,000 | 0,050 | 232 |
| FastFlow | 1,000 | 0,972 | 1,000 | 0,000 | 227 |
| SuperSimpleNet (10 époques) | 0,948 | 0,658 | 0,746 | 0,000 | 774 |
| Auto-encodeur | 0,788 | 0,720 | 0,460 | 0,000 | 1 136 |

{{img:docs/assets/anomaly/patchcore_lite_bottle_maps.jpg|PatchCore de la toolbox sur la bouteille : image, vérité terrain, carte d'anomalie lissée.}}

**Magnetic Tile** (surface texturée, la plus proche de l'acier ; test de 191 tuiles saines et 392 défectueuses, cinq types) :

| Méthode | AUROC image | AUROC pixel | Rappel au seuil | Fausses alarmes |
|---|---|---|---|---|
| PatchCore de la toolbox (mémoire 0,5 %) | 0,965 | 0,749 | 0,651 | 0,010 |
| PaDiM | 0,928 | 0,727 | 0,520 | 0,016 |
| STFPM (10 époques) | 0,663 | 0,582 | 0,097 | 0,016 |
| Auto-encodeur | 0,541 | 0,595 | 0,028 | 0,005 |

Sur la bouteille, presque toutes les méthodes séparent parfaitement les images saines des défectueuses : ce sont la localisation et les fausses alarmes qui les départagent, et PatchCore est en tête sans aucun entraînement. Sur la texture, l'écart se creuse. PatchCore reste le meilleur, les méthodes entraînées peu d'époques sur CPU décrochent, et l'auto-encodeur ne voit presque rien : il reconstruit aussi bien la texture que les défauts. Même pour PatchCore, le rappel au seuil 0,99 plafonne à 65 % (fissures et zones irrégulières surtout) : sur ce type de surface, il faut soit accepter plus de fausses alarmes, soit passer à une méthode supervisée dès que quelques défauts sont annotés.

:::note
Deux modèles d'anomalib (PatchCore et DFM) figent la taille du lot dans leur graphe. L'export le détecte en exécutant le modèle ONNX avec un autre lot que celui de l'exemple, réexporte avec un lot fixe de 1 et l'écrit dans la carte (`dynamic_batch: false`) : le programme de production envoie alors les images une par une.
:::

## Choisir

- **Premier essai** : PatchCore, sans entraînement, avec 1 à 2 % de mémoire. Si l'AUROC image est bonne mais le seuil donne trop de fausses alarmes, regarder la carte des images saines les plus mal notées : bord de pièce, reflet, étiquette.
- **Pièces bien alignées et contrainte de temps** : PaDiM.
- **Peu d'images saines** (quelques dizaines) : PatchCore reste utilisable ; les méthodes à entraîner souffrent davantage.
- **Inférence très rapide** : EfficientAd ou FastFlow, à entraîner sur GPU.
- **Textures régulières** (tôle, tissu, bois) : PatchCore et l'auto-encodeur se comportent bien ; les objets complexes demandent PatchCore ou Reverse Distillation.
