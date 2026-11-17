"""Boite a outils en ligne de commande : python -m vtb.toolkit <commande> --help

summary        inventaire d'un dossier d'images (tailles, types, illisibles) par sous-dossier
mean-std       moyenne et ecart-type par canal (normalisation propre au jeu)
class-weights  poids de classes pour une perte ponderee (dossiers de classes ou CSV)
resize         redimensionne un dossier entier (arborescence conservee, en parallele)
to-8bit        convertit un dossier 16 bits en 8 bits
frames         extrait des images d'une video
patches        decoupe un dossier en patchs (et leurs masques)
imagefolder    range un split.csv en dossiers <split>/<classe>/
threshold      seuil pour un rappel / taux de fausses alarmes / precision vise (CSV de scores)
ci             intervalle de confiance par bootstrap d'une metrique (CSV de predictions)
mcnemar        compare deux modeles sur les memes images (CSV de predictions)
"""

# Ce fichier __main__.py est execute par "python -m vtb.toolkit" : il lit la commande tapee et appelle
# la bonne fonction de dataset.py ou stats.py. Les imports lourds sont faits dans chaque commande,
# pour que "--help" reste instantane.
from __future__ import annotations  # permet d'ecrire les types "list[str] | None" sans les evaluer a l'execution

import argparse  # lecture des arguments de la ligne de commande (sous-commandes, options --xxx)
import functools  # functools.partial : fige certains arguments d'une fonction
import sys  # sys.exit : renvoie un code de sortie au terminal
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import pandas as pd  # pandas : tableaux avec des colonnes nommees, lecture de CSV
from tabulate import tabulate  # tabulate : met un tableau en forme pour l'afficher joliment dans le terminal


def _print_table(frame: pd.DataFrame, max_rows: int = 40) -> None:  # affiche un tableau (au plus max_rows lignes) ; ne renvoie rien
    shown = frame.head(max_rows)  # les premieres lignes seulement, pour ne pas inonder le terminal
    print(tabulate(shown, headers="keys", tablefmt="github", showindex=False, floatfmt=".4g"))  # tableau style GitHub, 4 chiffres
    if len(frame) > max_rows:  # s'il reste des lignes non affichees
        print(f"... {len(frame) - max_rows} lignes de plus")  # on dit combien


def _print_dict(values: dict) -> None:  # affiche un dictionnaire en tableau a deux colonnes ; ne renvoie rien
    print(tabulate([(k, v) for k, v in values.items()], tablefmt="github", headers=["", "valeur"], floatfmt=".4g"))  # une ligne par cle


def cmd_summary(args: argparse.Namespace) -> None:  # commande "summary" : inventaire d'un dossier ; ne renvoie rien
    from vtb.toolkit.dataset import folder_summary, summarize  # fonctions d'inventaire (importees seulement si on en a besoin)

    kwargs = {"extensions": args.ext} if args.ext else {}  # extensions choisies par --ext, sinon celles par defaut
    files = folder_summary(args.root, recursive=not args.flat, workers=args.workers, **kwargs)  # une ligne par image
    _print_table(summarize(files))  # affiche le resume par sous-dossier
    bad = files[~files["readable"]]  # lignes des images illisibles (~ = "non")
    if len(bad):  # s'il y en a
        print(f"\n{len(bad)} image(s) illisible(s) :")  # titre
        _print_table(bad[["path", "error"]], max_rows=20)  # chemin et cause de chacune (20 au plus)
    if args.csv:  # si --csv a ete donne
        files.to_csv(args.csv, index=False)  # ecrit l'inventaire complet dans ce fichier CSV
        print(f"\ninventaire complet : {args.csv}")  # dit ou il est


def cmd_mean_std(args: argparse.Namespace) -> None:  # commande "mean-std" : moyenne et ecart-type ; ne renvoie rien
    from vtb.io.images import list_images  # liste des images d'un dossier
    from vtb.toolkit.dataset import mean_std  # calcul de la moyenne et de l'ecart-type

    paths = list_images(args.root)  # toutes les images du dossier
    # calcule moyenne et ecart-type par canal (1 canal si --gray, sinon 3)
    mean, std = mean_std(paths, size=args.size, max_images=args.max_images, color="gray" if args.gray else "rgb", seed=args.seed)
    print(f"images : {min(len(paths), args.max_images or len(paths))} sur {len(paths)}")  # combien d'images ont servi
    print(f"mean: {mean}")  # moyennes a recopier dans la normalisation
    print(f"std:  {std}")  # ecarts-types a recopier dans la normalisation


def cmd_class_weights(args: argparse.Namespace) -> None:  # commande "class-weights" : poids par classe ; ne renvoie rien
    from vtb.toolkit.dataset import class_weights, folder_summary  # calcul des poids et inventaire

    if args.csv:  # si les etiquettes viennent d'un CSV
        labels = pd.read_csv(args.csv)[args.label_column]  # lit la colonne d'etiquettes
    else:  # sinon : dossier range en <classe>/<image>
        files = folder_summary(args.root)  # inventaire du dossier
        labels = files["folder"].map(lambda f: Path(f).parts[0] if f else ".")  # classe = premier sous-dossier du chemin
    counts = pd.Series(list(labels)).astype(str).value_counts().sort_index()  # nombre d'images par classe, classes triees
    weights = class_weights(labels, mode=args.mode)  # poids de chaque classe
    # tableau a afficher : classe, nombre d'images, poids
    table = pd.DataFrame({"classe": counts.index, "images": counts.to_numpy(), "poids": [weights[c] for c in counts.index]})
    _print_table(table)  # affiche le tableau
    print(f"\nliste pour la perte : {list(weights.values())}")  # liste a copier dans nn.CrossEntropyLoss(weight=...)


def cmd_resize(args: argparse.Namespace) -> None:  # commande "resize" : redimensionne un dossier ; ne renvoie rien
    from vtb.toolkit.dataset import resize_folder  # redimensionnement d'un dossier entier

    size = (args.width, args.height) if args.width and args.height else args.size  # (largeur, hauteur) si donnees, sinon --size
    if size is None:  # si aucune taille n'a ete donnee
        raise SystemExit("donner --size, ou --width et --height")  # quitte le programme avec ce message
    # redimensionne tout le dossier ; rapport : une ligne par image
    report = resize_folder(args.src, args.dst, size, keep_aspect=not args.stretch, pad=args.pad, extension=args.ext, workers=args.workers)
    print(f"{int(report['ok'].sum())} image(s) ecrites dans {args.dst}, {int((~report['ok']).sum())} en echec")  # bilan
    print(report["shape"].value_counts().head(5).to_string())  # les 5 tailles obtenues les plus frequentes


def cmd_to_8bit(args: argparse.Namespace) -> None:  # commande "to-8bit" : conversion 16 -> 8 bits ; ne renvoie rien
    from vtb.toolkit.dataset import convert_folder_8bit  # conversion d'un dossier entier

    report = convert_folder_8bit(  # convertit chaque image ; rapport : une ligne par image
        args.src,  # dossier source
        args.dst,  # dossier destination
        method=args.method,  # methode de conversion
        low=args.low,  # percentile bas
        high=args.high,  # percentile haut
        extension=args.ext,  # format de sortie
        workers=args.workers,  # nombre de threads
    )
    print(f"{int(report['ok'].sum())} image(s) converties dans {args.dst}, {int((~report['ok']).sum())} en echec")  # bilan


def cmd_frames(args: argparse.Namespace) -> None:  # commande "frames" : images d'une video ; ne renvoie rien
    from vtb.toolkit.dataset import extract_frames  # extraction d'images d'une video

    count = extract_frames(  # extrait les images avec les reglages donnes ; renvoie leur nombre
        args.video,  # fichier video
        args.dst,  # dossier de sortie
        every=args.every,  # une image sur N
        fps=args.fps,  # ou cadence voulue
        start=args.start,  # debut en secondes
        end=args.end,  # fin en secondes
        max_frames=args.max_frames,  # nombre maximum d'images
        extension=args.ext,  # format des images
    )
    print(f"{count} image(s) ecrites dans {args.dst}")  # bilan


def cmd_patches(args: argparse.Namespace) -> None:  # commande "patches" : decoupe en patchs ; ne renvoie rien
    from vtb.toolkit.dataset import patch_folder  # decoupe d'un dossier en patchs

    # masques : ignores avec --no-masks ; obligatoires si on a dit ou les trouver ; sinon utilises s'ils existent
    masks = False if args.no_masks else (True if (args.masks_dir or args.mask_suffix) else None)
    table = patch_folder(  # decoupe et ecrit les patchs ; renvoie le catalogue
        args.src,  # dossier source
        args.dst,  # dossier de sortie
        size=args.size,  # taille d'un patch
        stride=args.stride,  # decalage entre patchs
        masks=masks,  # politique des masques choisie plus haut
        masks_dir=args.masks_dir,  # dossier des masques
        mask_suffix=args.mask_suffix,  # suffixe des noms de masques
        keep_empty=args.keep_empty,  # part des patchs sains a garder
        seed=args.seed,  # germe du hasard
    )
    csv = Path(args.dst) / "patches.csv"  # le catalogue est ecrit a cote des patchs
    csv.parent.mkdir(parents=True, exist_ok=True)  # cree le dossier s'il n'existe pas (cas d'aucun patch ecrit)
    table.to_csv(csv, index=False)  # ecrit le catalogue
    with_defect = int(table["has_defect"].sum()) if len(table) else 0  # nombre de patchs avec defaut
    sources = table["source"].nunique() if len(table) else 0  # nombre d'images d'origine differentes
    print(f"{len(table)} patch(s) de {args.size} px depuis {sources} image(s) ; {with_defect} avec defaut")  # bilan
    print(f"liste : {csv} (colonne source = groupe pour le decoupage)")  # rappel : decouper train/val/test par image d'origine


def cmd_imagefolder(args: argparse.Namespace) -> None:  # commande "imagefolder" : range un split.csv en dossiers ; ne renvoie rien
    from vtb.toolkit.dataset import export_imagefolder  # rangement en dossiers split/classe

    frame = pd.read_csv(args.csv)  # lit le catalogue (une ligne par image)
    result = export_imagefolder(frame, args.dst, mode=args.mode, split=args.split_column, label=args.label_column)  # copie ou lie
    # tableau croise : nombre d'images par split (lignes) et par classe (colonnes)
    _print_table(result.groupby([args.split_column, args.label_column]).size().unstack(fill_value=0).reset_index())
    print(f"\n{len(result)} image(s) rangees dans {args.dst}")  # bilan


def _defect_column(frame: pd.DataFrame, args: argparse.Namespace) -> pd.Series:  # renvoie une colonne True/False "a un defaut"
    if args.defect_labels:  # si on a donne les noms des etiquettes defectueuses
        wanted = {s.strip() for s in args.defect_labels.split(",")}  # ensemble de ces noms (espaces retires)
        return frame[args.label].astype(str).isin(wanted)  # True si l'etiquette fait partie de ces noms
    return frame[args.defect].astype(str).str.lower().isin({"1", "true", "yes", "defect"})  # sinon colonne 0/1 ou true/false


def cmd_threshold(args: argparse.Namespace) -> None:  # commande "threshold" : choix du seuil de decision ; ne renvoie rien
    from vtb.toolkit.stats import threshold_for  # choix du seuil sous contraintes

    frame = pd.read_csv(args.csv)  # lit le CSV de validation (scores et verite)
    result = threshold_for(  # cherche le seuil qui respecte les contraintes
        frame[args.score].to_numpy(),  # scores du modele
        _defect_column(frame, args).to_numpy(),  # verite : defaut ou non
        min_recall=args.min_recall,  # rappel minimum
        max_false_alarm=args.max_false_alarm,  # taux de fausses alarmes maximum
        min_precision=args.min_precision,  # precision minimum
    )
    _print_dict(result)  # affiche le seuil et ses chiffres


# Metriques disponibles pour "ci" : nom -> (module ou elle se trouve, nom de la fonction, arguments fixes)
METRICS = {
    "accuracy": ("sklearn.metrics", "accuracy_score", {}),  # exactitude : part de bonnes reponses
    "balanced_accuracy": ("sklearn.metrics", "balanced_accuracy_score", {}),  # moyenne des rappels par classe
    "f1_macro": ("sklearn.metrics", "f1_score", {"average": "macro"}),  # F1 (melange rappel / precision) moyen sur les classes
    "recall_macro": ("sklearn.metrics", "recall_score", {"average": "macro"}),  # rappel moyen sur les classes
    "precision_macro": ("sklearn.metrics", "precision_score", {"average": "macro", "zero_division": 0}),  # precision moyenne
}


def _metric(name: str):  # renvoie la fonction de metrique demandee, prete a appeler avec (verite, prediction)
    import importlib  # importlib : importe un module dont on connait le nom en texte

    module, function, kwargs = METRICS[name]  # lit la fiche de la metrique
    return functools.partial(getattr(importlib.import_module(module), function), **kwargs)  # fonction avec ses arguments fixes


def cmd_ci(args: argparse.Namespace) -> None:  # commande "ci" : intervalles de confiance par bootstrap ; ne renvoie rien
    from vtb.toolkit.stats import bootstrap_ci  # intervalle de confiance par bootstrap

    frame = pd.read_csv(args.csv)  # lit le CSV (verite et prediction)
    groups = frame[args.group].to_numpy() if args.group else None  # colonne de groupe (piece) si --group est donne
    rows = []  # une ligne de resultat par metrique
    for name in args.metric:  # pour chaque metrique demandee
        interval = bootstrap_ci(  # intervalle de confiance de cette metrique
            _metric(name),  # la fonction de metrique
            frame[args.true].astype(str).to_numpy(),  # verite, en texte (les classes peuvent etre des noms ou des numeros)
            frame[args.pred].astype(str).to_numpy(),  # predictions, en texte aussi pour pouvoir comparer
            n=args.n,  # nombre de tirages
            groups=groups,  # groupes (ou None)
        )
        # ligne : nom, valeur, bornes basse et haute
        rows.append({"metrique": name, "valeur": interval.value, "bas (95 %)": interval.low, "haut (95 %)": interval.high})
    print(f"{len(frame)} exemples, {args.n} tirages bootstrap" + (f", par groupe ({args.group})" if args.group else ""))  # rappel
    _print_table(pd.DataFrame(rows))  # affiche le tableau des intervalles


def cmd_mcnemar(args: argparse.Namespace) -> None:  # commande "mcnemar" : comparaison de deux modeles ; ne renvoie rien
    from vtb.toolkit.stats import mcnemar  # test de McNemar

    frame = pd.read_csv(args.csv)  # lit le CSV (verite et predictions des deux modeles)
    columns = [frame[c].astype(str).to_numpy() for c in (args.true, args.pred_a, args.pred_b)]  # les trois colonnes en texte
    _print_dict(mcnemar(*columns))  # fait le test et affiche le resultat


def build_parser() -> argparse.ArgumentParser:  # decrit toutes les commandes et leurs options ; renvoie l'analyseur
    parser = argparse.ArgumentParser(  # analyseur principal
        prog="python -m vtb.toolkit",  # nom affiche dans l'aide
        description=__doc__,  # texte d'aide = la docstring du haut du fichier
        formatter_class=argparse.RawDescriptionHelpFormatter,  # garde les retours a la ligne de ce texte
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="commande")  # une sous-commande est obligatoire

    p = sub.add_parser("summary", help="inventaire d'un dossier d'images")  # sous-commande summary
    p.add_argument("root", type=Path)  # argument obligatoire : le dossier
    p.add_argument("--flat", action="store_true", help="ne pas descendre dans les sous-dossiers")  # interrupteur (True si present)
    p.add_argument("--csv", type=Path, help="ecrire l'inventaire complet (une ligne par image)")  # fichier de sortie optionnel
    # extensions a compter (plusieurs valeurs possibles)
    p.add_argument("--ext", nargs="+", default=None, help="extensions a compter (ex. .jpg quand les masques .png sont a cote)")
    p.add_argument("--workers", type=int, default=None)  # nombre de threads
    p.set_defaults(func=cmd_summary)  # fonction a appeler pour cette sous-commande

    p = sub.add_parser("mean-std", help="moyenne et ecart-type par canal, valeurs dans [0, 1]")  # sous-commande mean-std
    p.add_argument("root", type=Path, help="dossier d'images (du TRAIN)")  # dossier des images d'entrainement
    p.add_argument("--size", type=int, default=None, help="redimensionner en carre avant calcul")  # taille du carre
    p.add_argument("--max-images", type=int, default=None)  # nombre maximum d'images
    p.add_argument("--gray", action="store_true", help="un seul canal")  # niveaux de gris
    p.add_argument("--seed", type=int, default=0)  # germe du hasard
    p.set_defaults(func=cmd_mean_std)  # fonction a appeler

    p = sub.add_parser("class-weights", help="poids de classes pour une perte ponderee")  # sous-commande class-weights
    p.add_argument("root", type=Path, nargs="?", help="dossier <classe>/<image>")  # dossier, facultatif (nargs="?")
    p.add_argument("--csv", type=Path, help="ou un CSV (ex. split.csv filtre sur le train)")  # ou un CSV
    p.add_argument("--label-column", default="label")  # colonne des etiquettes du CSV
    p.add_argument("--mode", choices=["inverse", "sqrt", "effective"], default="inverse")  # formule des poids
    p.set_defaults(func=cmd_class_weights)  # fonction a appeler

    p = sub.add_parser("resize", help="redimensionner un dossier entier")  # sous-commande resize
    p.add_argument("src", type=Path)  # dossier source
    p.add_argument("dst", type=Path)  # dossier destination
    p.add_argument("--size", type=int, help="plus grand cote (ou cote du carre avec --stretch / --pad)")  # taille voulue
    p.add_argument("--width", type=int)  # largeur exacte
    p.add_argument("--height", type=int)  # hauteur exacte
    p.add_argument("--stretch", action="store_true", help="ne pas garder les proportions")  # etirer
    p.add_argument("--pad", action="store_true", help="completer en carre (letterbox centre)")  # letterbox : bandes autour de l'image
    p.add_argument("--ext", default=None, help="changer le format, ex. .png")  # nouveau format
    p.add_argument("--workers", type=int, default=None)  # nombre de threads
    p.set_defaults(func=cmd_resize)  # fonction a appeler

    p = sub.add_parser("to-8bit", help="convertir un dossier 16 bits en 8 bits")  # sous-commande to-8bit
    p.add_argument("src", type=Path)  # dossier source
    p.add_argument("dst", type=Path)  # dossier destination
    p.add_argument("--method", choices=["percentile", "minmax", "scale"], default="percentile")  # methode de conversion
    p.add_argument("--low", type=float, default=0.5)  # percentile bas
    p.add_argument("--high", type=float, default=99.5)  # percentile haut
    p.add_argument("--ext", default=".png")  # format de sortie
    p.add_argument("--workers", type=int, default=None)  # nombre de threads
    p.set_defaults(func=cmd_to_8bit)  # fonction a appeler

    p = sub.add_parser("frames", help="extraire des images d'une video")  # sous-commande frames
    p.add_argument("video", type=Path)  # fichier video
    p.add_argument("dst", type=Path)  # dossier de sortie
    p.add_argument("--every", type=int, default=1, help="une image sur N")  # une image sur N
    p.add_argument("--fps", type=float, default=None, help="cadence voulue (remplace --every)")  # images par seconde voulues
    p.add_argument("--start", type=float, default=None, help="debut en secondes")  # debut
    p.add_argument("--end", type=float, default=None, help="fin en secondes")  # fin
    p.add_argument("--max-frames", type=int, default=None)  # nombre maximum d'images
    p.add_argument("--ext", default=".png")  # format des images
    p.set_defaults(func=cmd_frames)  # fonction a appeler

    p = sub.add_parser("patches", help="decouper un dossier en patchs (et masques)")  # sous-commande patches
    p.add_argument("src", type=Path)  # dossier source
    p.add_argument("dst", type=Path)  # dossier de sortie
    p.add_argument("--size", type=int, required=True)  # taille des patchs (obligatoire)
    p.add_argument("--stride", type=int, default=None, help="pas entre patchs (defaut : size, sans recouvrement)")  # decalage
    p.add_argument("--masks-dir", type=Path, default=None)  # dossier des masques
    p.add_argument("--mask-suffix", default="")  # suffixe des masques
    p.add_argument("--no-masks", action="store_true")  # ignorer les masques
    p.add_argument("--keep-empty", type=float, default=1.0, help="fraction des patchs sans defaut a garder")  # reequilibrage
    p.add_argument("--seed", type=int, default=0)  # germe du hasard
    p.set_defaults(func=cmd_patches)  # fonction a appeler

    p = sub.add_parser("imagefolder", help="ranger un split.csv en dossiers <split>/<classe>/")  # sous-commande imagefolder
    p.add_argument("csv", type=Path)  # catalogue CSV
    p.add_argument("dst", type=Path)  # dossier de sortie
    p.add_argument("--mode", choices=["copy", "link"], default="copy")  # copier ou lier
    p.add_argument("--split-column", default="split")  # colonne train/val/test
    p.add_argument("--label-column", default="label")  # colonne de classe
    p.set_defaults(func=cmd_imagefolder)  # fonction a appeler

    # sous-commande threshold
    p = sub.add_parser("threshold", help="seuil de decision pour une contrainte (rappel, fausses alarmes, precision)")
    p.add_argument("csv", type=Path, help="CSV de la VALIDATION avec une colonne de score")  # CSV de validation
    p.add_argument("--score", default="score")  # colonne des scores
    p.add_argument("--defect", default="is_defect", help="colonne 0/1 (ou true/false)")  # colonne verite 0/1
    p.add_argument("--label", default="label", help="avec --defect-labels : colonne d'etiquette")  # colonne d'etiquette
    p.add_argument("--defect-labels", default=None, help="etiquettes defectueuses separees par des virgules")  # noms des defauts
    p.add_argument("--min-recall", type=float, default=None)  # rappel minimum
    p.add_argument("--max-false-alarm", type=float, default=None)  # fausses alarmes maximum
    p.add_argument("--min-precision", type=float, default=None)  # precision minimum
    p.set_defaults(func=cmd_threshold)  # fonction a appeler

    p = sub.add_parser("ci", help="intervalle de confiance bootstrap d'une metrique")  # sous-commande ci
    p.add_argument("csv", type=Path, help="CSV avec verite et prediction")  # CSV de predictions
    p.add_argument("--true", default="label")  # colonne de verite
    p.add_argument("--pred", default="pred")  # colonne de prediction
    p.add_argument("--metric", nargs="+", choices=sorted(METRICS), default=["accuracy", "f1_macro"])  # metriques a calculer
    p.add_argument("--n", type=int, default=1000)  # nombre de tirages bootstrap
    p.add_argument("--group", default=None, help="colonne de groupe (piece, bobine) : bootstrap par groupe")  # colonne de groupe
    p.set_defaults(func=cmd_ci)  # fonction a appeler

    p = sub.add_parser("mcnemar", help="deux modeles sur les memes images : difference significative ?")  # sous-commande mcnemar
    p.add_argument("csv", type=Path)  # CSV de predictions
    p.add_argument("--true", default="label")  # colonne de verite
    p.add_argument("--pred-a", required=True)  # colonne des predictions du modele A
    p.add_argument("--pred-b", required=True)  # colonne des predictions du modele B
    p.set_defaults(func=cmd_mcnemar)  # fonction a appeler
    return parser  # renvoie l'analyseur complet


def main(argv: list[str] | None = None) -> int:  # point d'entree ; renvoie 0 si tout s'est bien passe
    args = build_parser().parse_args(argv)  # lit les arguments tapes (argv=None : ceux du terminal)
    if args.command == "class-weights" and args.root is None and args.csv is None:  # class-weights sans dossier ni CSV
        raise SystemExit("class-weights : donner un dossier ou --csv")  # quitte avec un message clair
    args.func(args)  # appelle la fonction de la sous-commande choisie
    return 0  # code 0 = succes


if __name__ == "__main__":  # vrai seulement quand le fichier est lance (python -m vtb.toolkit), pas quand il est importe
    sys.exit(main())  # lance main et transmet son code de sortie au terminal
