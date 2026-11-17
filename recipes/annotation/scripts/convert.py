"""Convertit un jeu annote d'un format a un autre, en passant par le format pivot, et verifie le resultat.

    python recipes/annotation/scripts/convert.py --from voc --images data/raw/neu_det/IMAGES \
        --annotations data/raw/neu_det/ANNOTATIONS --to coco --out outputs/annotation/neu_det.coco.json
    python recipes/annotation/scripts/convert.py --from masks --images data/raw/magnetic_tile/MT_Crack/Imgs \
        --annotations data/raw/magnetic_tile/MT_Crack/Imgs --extensions .jpg --kind rotated --classes crack \
        --to yolo --task obb --out outputs/annotation/mt_obb

Formats : voc, yolo, coco, labelme, labelstudio, cvat, masks, vtb (voir vtb.annotate.formats).
--check relit le fichier ecrit et le compare a ce qui a ete lu : a faire sur tout nouveau jeu.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[str], X | None) partout

import argparse  # argparse : lit les options passees en ligne de commande (--from, --to, ...)
import logging  # logging : affiche des messages de suivi (mieux que print dans un script)
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

from vtb.annotate.check import same_annotations  # compare deux jeux d'annotations et liste les differences

# READERS : dict {nom de format: fonction de lecture} ; convert : lecture + ecriture ; read_annotations : lecture seule
from vtb.annotate.formats import READERS, convert, read_annotations
from vtb.core.runtime import setup_logging  # configure l'affichage des messages de suivi sur la console

logger = logging.getLogger(__name__)  # "journal" propre a ce fichier : logger.info(...) ecrit un message de suivi


def parse_classes(value: str | None) -> list[str] | None:  # renvoie la liste des classes, ou None si l'option est absente
    """--classes a,b,c ou --classes fichier.txt (une classe par ligne)."""
    if not value:  # option non donnee (None ou texte vide)
        return None  # None = laisser le lecteur deviner les classes
    path = Path(value)  # on essaie d'abord de lire la valeur comme un chemin de fichier
    if path.is_file():  # si c'est un fichier texte existant : une classe par ligne
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]  # lignes vides ignorees
    return [c.strip() for c in value.split(",") if c.strip()]  # sinon : "a,b,c" -> ["a", "b", "c"]


def main() -> None:  # point d'entree du script : convertit le jeu, et le verifie si --check ; ne renvoie rien
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # l'aide reprend la 1re ligne de la docstring
    # format source ; dest="src_fmt" car "from" est un mot reserve de Python (args.from serait invalide)
    parser.add_argument("--from", dest="src_fmt", required=True, choices=sorted(READERS))
    parser.add_argument("--images", required=True, type=Path, help="dossier des images")  # dossier des images du jeu
    # selon le format : un dossier (un fichier par image, ex. voc) ou un seul fichier (ex. coco)
    parser.add_argument("--annotations", required=True, type=Path, help="dossier ou fichier d'annotations source")
    parser.add_argument("--to", dest="dst_fmt", required=True, choices=sorted(READERS))  # format de destination
    parser.add_argument("--out", required=True, type=Path, help="dossier ou fichier de sortie")  # ou ecrire le resultat
    # liste des classes : l'ordre donne le numero de classe (indices YOLO, valeurs des pixels des masques)
    parser.add_argument("--classes", default=None, help="a,b,c ou fichier texte ; ordre = indices YOLO / masques")
    # variante YOLO : detect = boites droites, obb = boites tournees, segment = contours (polygones)
    parser.add_argument("--task", default=None, choices=["detect", "obb", "segment"], help="variante YOLO")
    # Label Studio : dossier racine a partir duquel les chemins d'images sont ecrits
    parser.add_argument("--document-root", type=Path, default=None, help="Label Studio : LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT")
    # pour une source "masks" : quel objet extraire de chaque tache du masque (contour, boite tournee, boite droite, masque)
    parser.add_argument("--kind", default=None, choices=["polygons", "rotated", "boxes", "mask"], help="masks : objets a extraire")
    parser.add_argument("--mask-suffix", default=None)  # fin de nom des fichiers masques (ex. "_mask") pour les apparier aux images
    # extensions des images a prendre (utile quand masques et images sont dans le meme dossier)
    parser.add_argument("--extensions", nargs="*", default=None, help="extensions des images (ex. .jpg si les masques sont a cote)")
    # store_true : l'option ne prend pas de valeur ; sa presence suffit a la mettre a True
    parser.add_argument("--check", action="store_true", help="relit la sortie et la compare a l'entree")
    args = parser.parse_args()  # lit la ligne de commande ; args.src_fmt, args.dst_fmt... contiennent les valeurs
    setup_logging()  # active l'affichage des messages de suivi

    options = {  # options facultatives passees aux fonctions de lecture / ecriture
        "class_names": parse_classes(args.classes),  # liste des classes (ou None)
        "task": args.task,  # variante YOLO (ou None)
        "document_root": args.document_root,  # racine Label Studio (ou None)
        "kind": args.kind,  # objet a extraire des masques (ou None)
        "mask_suffix": args.mask_suffix,  # suffixe des masques (ou None)
        "extensions": args.extensions,  # extensions d'images (ou None)
    }
    options = {k: v for k, v in options.items() if v is not None}  # garde seulement les options donnees (None = valeur par defaut)
    # lit la source dans le format pivot, puis l'ecrit au format cible ; renvoie les annotations lues (pour le controle)
    annotations = convert(args.src_fmt, args.images, args.annotations, args.dst_fmt, args.out, **options)
    n_objects = sum(len(a.object_labels()) for a in annotations)  # nombre total d'objets annotes, toutes images confondues
    # resume : nombre d'images, d'objets, formats source -> cible et chemin de sortie
    logger.info("%d images, %d objets : %s -> %s (%s)", len(annotations), n_objects, args.src_fmt, args.dst_fmt, args.out)
    if args.check:  # si --check : on relit ce qui vient d'etre ecrit pour verifier que rien n'a ete perdu
        # le format YOLO range les images et les labels dans des sous-dossiers images/train et labels/train
        images_back = args.out / "images" / "train" if args.dst_fmt == "yolo" else args.images
        target = args.out / "labels" / "train" if args.dst_fmt == "yolo" else args.out  # ou relire les annotations ecrites
        # relit la sortie avec les memes options, sauf "kind" qui ne concerne que la lecture des masques source
        back = read_annotations(args.dst_fmt, images_back, target, **{k: v for k, v in options.items() if k != "kind"})
        issues = same_annotations(annotations, back, tol=1e-2)  # differences image par image (ecart de coordonnees tolere : 0,01 px)
        if issues:  # s'il y a au moins une difference
            raise SystemExit("conversion avec pertes :\n" + "\n".join(issues[:20]))  # arrete le script et montre les 20 premieres
        logger.info("controle : relecture identique (tolerance 0,01 px)")  # tout est identique : la conversion est fiable


if __name__ == "__main__":  # vrai seulement quand on lance le fichier directement (pas quand on l'importe)
    main()  # lance le script
