# Guide d'annotation : <nom du projet>

Version : 1.0 - date : AAAA-MM-JJ - responsable : <role, par exemple "responsable qualite">

Ce guide est ecrit AVANT la premiere annotation et relu par toutes les personnes qui annotent.
Toute question qui se pose pendant l'annotation donne lieu a une regle ajoutee ici (et a une
nouvelle version), puis les images deja annotees concernees sont relues.

## 1. But et tache

- Ce que le modele devra faire : <classer l'image / encadrer chaque defaut / detourer chaque defaut>.
- Ce qui sera decide avec : <rejet de la piece, alerte operateur, statistique de production>.
- Unite annotee : <l'image entiere / chaque defaut visible>.

## 2. Les classes

Les classes sont EXHAUSTIVES (toute image ou tout objet entre dans une classe) et SANS
RECOUVREMENT (un objet ne peut pas relever de deux classes). Une classe par ligne :

| classe | definition | exemples (fichiers) | contre-exemples (fichiers) |
|---|---|---|---|
| <rayure> | <trait fin, longueur > 5 x largeur, oriente dans le sens de laminage> | <ex_01.png, ex_07.png> | <calamine allongee : ex_12.png> |
| <inclusion> | <...> | <...> | <...> |
| a_verifier | cas douteux : l'annotateur ne tranche pas, un expert relit | | |

Regles :
- Jamais de classe "autre" fourre-tout : si un type de defaut revient, il devient une classe.
- Les cas douteux vont dans "a_verifier" (jamais au hasard dans une classe) ; ils sont relus
  par un expert et la decision devient une regle de ce guide.

## 3. Regles de dessin

- Boites : la boite englobe tout le defaut visible, au plus pres (pas de marge), y compris les
  parties peu contrastees ; deux defauts qui se touchent mais sont distincts = deux boites ;
  un defaut coupe par le bord de l'image est annote jusqu'au bord.
- Boites orientees : le grand cote suit l'axe du defaut (rayure, fissure).
- Polygones / masques : le contour suit la limite visible du defaut ; les trous ne sont pas
  representes ; taille minimale annotee : <N> pixels (en dessous : ignorer).
- Images sans defaut : elles sont gardees et marquees explicitement (aucun objet, ou classe
  "ok") ; ce n'est pas un oubli.

## 4. Cas limites rencontres (a completer au fil de l'eau)

| situation | decision | depuis la version |
|---|---|---|
| <reflet sur la surface ressemblant a une rayure> | <pas un defaut> | 1.0 |

## 5. Qualite

- 5 a 10 % des images sont annotees deux fois par deux personnes differentes ; l'accord est
  mesure (kappa pour les classes, IoU pour les boites et masques) et note ici.
- Le jeu de TEST est annote entierement a la main, sans pre-annotation par un modele, et relu.
- Les images designees par la recherche d'etiquettes suspectes sont relues en priorite.

## 6. Outils et formats

- Outil : <Label Studio / CVAT / X-AnyLabeling / outils vtb>, configuration : <fichier>.
- Export : <format>, converti vers le format pivot de la toolbox par recipes/annotation/scripts/convert.py.
- Les images ne quittent pas le site : aucun outil en ligne, aucune copie hors du reseau autorise.
