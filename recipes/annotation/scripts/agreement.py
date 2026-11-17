"""Mesure l'accord entre deux annotateurs (classification, boites ou masques).

    python recipes/annotation/scripts/agreement.py classification --a annotateur_a.csv --b annotateur_b.csv
    python recipes/annotation/scripts/agreement.py boxes --images data/raw/neu_det/IMAGES \
        --a voc:annotateur_a/ --b coco:annotateur_b.json --iou 0.5
    python recipes/annotation/scripts/agreement.py masks --a masques_a/ --b masques_b/

classification : deux CSV (path, label), apparies par nom de fichier. boxes : FORMAT:CHEMIN pour
chaque annotateur, images appariees par nom. masks : deux dossiers de PNG de meme nom.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[str], X | None) partout

import argparse  # argparse : lit les options passees en ligne de commande (--a, --b, ...)
import json  # json : met les resultats sous forme de texte lisible pour le journal
import logging  # logging : affiche des messages de suivi (mieux que print dans un script)
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import pandas as pd  # pandas : tableaux de donnees (lecture des CSV, fusion de deux tableaux)

# fonctions de la toolbox qui calculent l'accord pour chaque type d'annotation
from vtb.annotate.agreement import box_agreement, classification_agreement, mask_agreement
from vtb.annotate.formats import read_annotations  # lit un jeu annote dans n'importe quel format connu (voc, coco, yolo...)
from vtb.core.runtime import setup_logging  # configure l'affichage des messages de suivi sur la console
from vtb.io.images import read_mask  # lit un masque PNG et le renvoie en tableau de 0 et de 1

logger = logging.getLogger(__name__)  # "journal" propre a ce fichier : logger.info(...) ecrit un message de suivi


def _read_spec(spec: str, images: Path):  # lit "FORMAT:CHEMIN" et renvoie un dict {nom d'image: annotations de l'image}
    fmt, _, path = spec.partition(":")  # coupe au premier ":" : "voc:dossier/" -> fmt="voc", path="dossier/"
    # indexe par nom de fichier pour pouvoir apparier les images des deux annotateurs
    return {Path(a.image_path).name: a for a in read_annotations(fmt, images, Path(path))}


def main() -> None:  # point d'entree du script : lit les options, calcule l'accord, l'affiche ; ne renvoie rien
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # l'aide reprend la 1re ligne de la docstring
    parser.add_argument("mode", choices=["classification", "boxes", "masks"])  # 1er argument obligatoire : le type d'annotation
    parser.add_argument("--a", required=True)  # annotations du premier annotateur (CSV, FORMAT:CHEMIN ou dossier)
    parser.add_argument("--b", required=True)  # annotations du second annotateur, meme forme que --a
    parser.add_argument("--images", type=Path, default=None)  # dossier des images (utile seulement pour le mode boxes)
    # seuil d'IoU (Intersection over Union = surface commune / surface totale de deux boites) pour dire "meme objet"
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()  # lit la ligne de commande ; args.mode, args.a... contiennent les valeurs
    setup_logging()  # active l'affichage des messages logger.info sur la console

    if args.mode == "classification":  # cas 1 : chaque annotateur a donne une classe par image
        a, b = pd.read_csv(args.a), pd.read_csv(args.b)  # lit les deux CSV (colonnes path et label)
        # ajoute une colonne "name" (nom du fichier seul) aux deux tableaux, puis les fusionne sur ce nom :
        # seules les images vues par les deux restent ; les colonnes label deviennent label_x (a) et label_y (b)
        merged = a.assign(name=a["path"].map(lambda p: Path(p).name)).merge(b.assign(name=b["path"].map(lambda p: Path(p).name)), on="name")
        # kappa de Cohen (accord corrige du hasard), accord brut et matrice de confusion entre les deux
        result = classification_agreement(merged["label_x"], merged["label_y"])
        logger.info(  # affiche le resume ; chaque %... est remplace par la valeur correspondante ci-dessous
            "kappa %.3f | accord brut %.3f | %d images\n%s",  # modele du message (%.3f = nombre a 3 decimales)
            result["kappa"],  # kappa : 1 = accord parfait, 0 = pas mieux que le hasard
            result["raw_agreement"],  # part des images ou les deux annotateurs ont donne la meme classe
            result["n_images"],  # nombre d'images communes aux deux annotateurs
            result["confusion"],  # matrice de confusion : qui a dit quoi quand l'autre a dit autre chose
        )
    elif args.mode == "boxes":  # cas 2 : chaque annotateur a dessine des boites autour des objets
        a, b = _read_spec(args.a, args.images), _read_spec(args.b, args.images)  # dict {nom d'image: annotations} par annotateur
        common = sorted(set(a) & set(b))  # noms d'images presents chez les deux (& = intersection), tries
        # compare les boites image par image (a sert de reference) ; deux boites "coincident" si IoU >= seuil
        result = box_agreement([a[n] for n in common], [b[n] for n in common], args.iou)
        # affiche le nombre d'images et chaque mesure arrondie a 3 decimales
        logger.info("%d images communes | %s", len(common), json.dumps({k: round(v, 3) for k, v in result.items()}))
    else:  # cas 3 (masks) : chaque annotateur a colorie les pixels du defaut (masques PNG)
        # noms des PNG du dossier a qui existent aussi dans le dossier b, tries
        names = sorted(p.name for p in Path(args.a).glob("*.png") if (Path(args.b) / p.name).exists())
        # lit les masques des deux annotateurs dans le meme ordre et calcule IoU et Dice (recouvrement des pixels)
        result = mask_agreement([read_mask(Path(args.a) / n) for n in names], [read_mask(Path(args.b) / n) for n in names])
        logger.info("%s", json.dumps({k: round(v, 3) for k, v in result.items()}))  # affiche les mesures arrondies


if __name__ == "__main__":  # vrai seulement quand on lance le fichier directement (pas quand on l'importe)
    main()  # lance le script
