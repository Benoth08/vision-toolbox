"""Outils autour des augmentations : decrire, regarder, controler les cibles, mesurer la vitesse.

    describe(aug)                       # tableau des etapes (ce qui tourne vraiment)
    preview(image, aug, n=8)            # 8 tirages a regarder AVANT d'entrainer
    check_alignment(aug, "detection")   # boites et masques suivent-ils l'image ?
    benchmark_ops(size=(256, 256))      # images / s par operation : trouver le goulot
    augmentation_catalog()              # toutes les operations, familles, taches, parametres
"""

from __future__ import annotations  # permet d'ecrire les types (ex. "str | None") sans qu'ils soient evalues au chargement

import inspect  # inspect : lit la signature d'une fonction (ses parametres)
import random  # random : generateur aleatoire standard de Python (a semer pour la reproductibilite)
import time  # time : mesure du temps (chronometre pour le benchmark)
from collections.abc import Callable, Sequence  # types : fonction appelable, liste ou tuple
from typing import Any  # Any : "n'importe quel type"

import numpy as np  # numpy : calcul sur des tableaux de nombres
import pandas as pd  # pandas : tableaux avec colonnes nommees (DataFrame), pratiques a afficher
import torch  # PyTorch : tenseurs (tableaux de nombres multidimensionnels) et generateur aleatoire
from torchvision import tv_tensors  # tenseurs etiquetes de torchvision (BoundingBoxes = boites, Mask = masque)
from torchvision.transforms import v2  # transformations torchvision v2 (Compose, RandomApply...)

from .presets import PRESETS  # les presets : listes d'augmentations pretes

# registre des operations, leurs parametres par defaut, et les fonctions qui construisent une chaine ou une etape
from .transforms import OPS, available_augmentations, build_augmentation, build_step

# --- catalogue et description ---------------------------------------------------------------------


def augmentation_catalog() -> pd.DataFrame:  # renvoie un tableau de toutes les operations disponibles
    """Une ligne par operation : nom, famille, taches, parametres par defaut, resume."""
    defaults = available_augmentations()  # nom -> {parametre: valeur par defaut}
    rows = [  # une ligne (dictionnaire) par operation du registre
        # colonnes : nom, famille, taches separees par des virgules, parametres par defaut, resume
        {"name": name, "family": info.family, "tasks": ", ".join(info.tasks), "defaults": defaults[name], "summary": info.summary}
        for name, info in OPS.items()  # pour chaque operation du registre
    ]
    return pd.DataFrame(rows)  # transforme la liste de lignes en tableau


def presets_table() -> pd.DataFrame:  # renvoie un tableau de tous les presets
    """Une ligne par preset : nom, operations, raisonnement."""
    # une ligne par preset : son nom, la liste de ses operations, son explication
    rows = [{"preset": name, "steps": ", ".join(s["name"] for s in p.steps), "summary": p.summary} for name, p in PRESETS.items()]
    return pd.DataFrame(rows)  # transforme la liste de lignes en tableau


_HIDDEN = {"training", "p", "transforms"}  # attributs internes a ne pas afficher comme parametres


def _params_of(transform: Any) -> dict[str, Any]:  # renvoie les reglages d'une transformation, lisibles a l'affichage
    out = {}  # reglages trouves
    for key, value in vars(transform).items():  # pour chaque attribut de l'objet (vars donne son dictionnaire interne)
        # si l'attribut est prive (commence par _), masque, ou un objet lourd (sous-module, tenseur)
        if key.startswith("_") or key in _HIDDEN or isinstance(value, (torch.nn.Module, torch.Tensor)):
            continue  # on ne l'affiche pas
        # valeur simple gardee telle quelle ; autre objet remplace par son texte (repr)
        out[key] = value if isinstance(value, (int, float, str, bool, tuple, list, type(None))) else repr(value)
    return out  # renvoie les reglages


# parcourt une chaine de transformations et ajoute une ligne par operation dans `rows` (rien a renvoyer)
def _walk(transform: Any, p: float, rows: list[dict[str, Any]]) -> None:
    if isinstance(transform, v2.Compose):  # si c'est une chaine (Compose)
        for t in transform.transforms:  # pour chaque transformation de la chaine
            _walk(t, p, rows)  # on descend dedans (fonction recursive : elle s'appelle elle-meme)
    elif isinstance(transform, v2.RandomApply):  # si c'est un "applique avec la probabilite p"
        for t in transform.transforms:  # pour chaque transformation enveloppee
            _walk(t, p * transform.p, rows)  # on descend en multipliant les probabilites
    else:  # sinon c'est une operation simple
        own = getattr(transform, "p", 1.0)  # sa propre probabilite (1 si elle n'en a pas)
        # ajoute la ligne : numero d'etape, nom de la classe, probabilite finale arrondie, reglages
        rows.append({"step": len(rows) + 1, "transform": type(transform).__name__, "p": round(p * own, 3), "params": _params_of(transform)})


def describe(aug: Any, task: str = "classification") -> pd.DataFrame:  # renvoie le tableau des etapes qui tournent vraiment
    """Tableau des etapes d'un Compose (ou d'une liste du YAML, construite pour `task`) : classe, probabilite, parametres."""
    if aug is None:  # si pas d'augmentation
        return pd.DataFrame(columns=["step", "transform", "p", "params"])  # tableau vide avec les bonnes colonnes
    if isinstance(aug, (list, tuple)):  # si on a donne la liste du YAML plutot qu'une chaine construite
        aug = build_augmentation(aug, task)  # type: ignore[arg-type]  # on construit la chaine (le commentaire calme mypy)
    rows: list[dict[str, Any]] = []  # lignes du tableau
    _walk(aug, 1.0, rows)  # parcourt la chaine en partant d'une probabilite de 1
    return pd.DataFrame(rows)  # transforme les lignes en tableau


# --- regarder -------------------------------------------------------------------------------------


def _to_tensor(image: Any) -> torch.Tensor:  # renvoie l'image en tenseur (C, H, W) de decimaux dans [0, 1]
    if isinstance(image, torch.Tensor):  # si c'est deja un tenseur torch
        return image.float()  # simple passage en decimaux (suppose deja dans [0, 1])
    array = np.asarray(image)  # sinon : tableau numpy
    peak = {np.dtype(np.uint8): 255.0, np.dtype(np.uint16): 65535.0}.get(array.dtype, 1.0)  # valeur max selon le type
    array = array.astype(np.float32) / peak  # ramene dans [0, 1]
    # image grise : ajoute l'axe des canaux devant ; couleur : (H, W, C) -> (C, H, W) ; copy() pour une memoire contigue
    return torch.from_numpy(array[None] if array.ndim == 2 else array.transpose(2, 0, 1).copy())


# montre n tirages de l'augmentation sur une meme image ; renvoie la figure (et l'enregistre si save est donne)
def preview(image: Any, aug: Callable, n: int = 8, target: Any = None, save: str | None = None):
    """n tirages de l'augmentation sur une image (tenseur (C, H, W) ou tableau uint8 / uint16 / float).

    Utilise vtb.viz.show_augmentations quand le module d'affichage est present (meme rendu que le
    reste de la toolbox), sinon une grille matplotlib simple. Renvoie la Figure.
    """
    x = _to_tensor(image)  # image en tenseur (C, H, W) dans [0, 1]
    try:  # on essaie d'utiliser l'affichage de la toolbox
        from vtb.viz import show_augmentations  # import paresseux : le module reste copiable seul

        return show_augmentations(x, aug, n=n, target=target, save=save)  # affichage complet, renvoie la figure
    except ImportError:  # si le module d'affichage n'est pas la (dossier augment copie seul dans un autre projet)
        pass  # on continue avec la grille simple ci-dessous
    import matplotlib.pyplot as plt  # matplotlib : dessin de graphiques et d'images

    columns = min(n, 4)  # 4 colonnes au plus
    rows = int(np.ceil(n / columns))  # nombre de lignes necessaire (arrondi au-dessus)
    # grille de cases ; squeeze=False : axes reste un tableau 2D meme avec une seule ligne
    fig, axes = plt.subplots(rows, columns, figsize=(2.6 * columns, 2.6 * rows), squeeze=False)
    for k, ax in enumerate(axes.ravel()):  # pour chaque case de la grille, avec son numero k
        ax.axis("off")  # pas d'axes gradues autour des images
        if k >= n:  # si la case est en trop (grille plus grande que n)
            continue  # on la laisse vide
        out = aug(x) if target is None else aug(x, target)[0]  # nouveau tirage ; avec une cible, on ne garde que l'image
        array = out.clamp(0, 1).numpy()  # bornee a [0, 1] puis convertie en tableau numpy
        # affiche l'image : grise (1 canal) telle quelle, couleur remise en (H, W, C) ; vmin/vmax : 0 = noir, 1 = blanc
        ax.imshow(array[0] if array.shape[0] == 1 else array.transpose(1, 2, 0), cmap="gray", vmin=0, vmax=1)
    fig.tight_layout()  # resserre les marges
    if save:  # si un chemin d'enregistrement est donne
        fig.savefig(save, dpi=100)  # enregistre la figure en image
    return fig  # renvoie la figure


# --- controler que les cibles suivent l'image -----------------------------------------------------


# renvoie une image test, ses 3 boites et son masque : sert a verifier que les augmentations deplacent bien les cibles
def alignment_pattern(size: Sequence[int] = (96, 128), channels: int = 1) -> tuple[torch.Tensor, tv_tensors.BoundingBoxes, tv_tensors.Mask]:
    """Image test asymetrique (rectangle, L, disque clairs sur fond sombre), ses boites et son masque (1, 2, 3)."""
    height, width = int(size[0]), int(size[1])  # taille de l'image test
    mask = torch.zeros(height, width, dtype=torch.int64)  # masque vide (0 = fond)
    mask[int(0.15 * height) : int(0.45 * height), int(0.10 * width) : int(0.35 * width)] = 1  # rectangle
    mask[int(0.55 * height) : int(0.90 * height), int(0.15 * width) : int(0.22 * width)] = 2  # L : barre verticale
    mask[int(0.80 * height) : int(0.90 * height), int(0.15 * width) : int(0.45 * width)] = 2  # L : barre horizontale
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")  # numero de ligne et de colonne de chaque pixel
    # disque : pixels a une distance du centre (0.35 H, 0.72 W) inferieure au rayon (15 % du petit cote)
    disc = ((yy - 0.35 * height) ** 2 + (xx - 0.72 * width) ** 2) <= (0.15 * min(height, width)) ** 2
    mask[disc] = 3  # le disque est l'objet 3
    # image : objets clairs (0.8) sur fond sombre (0.2), copiee sur le nombre de canaux voulu ; clone() : memoire propre
    image = torch.where(mask > 0, 0.8, 0.2).float()[None].expand(channels, -1, -1).clone()
    boxes = []  # une boite par objet
    for k in (1, 2, 3):  # pour chaque objet
        ys, xs = torch.nonzero(mask == k, as_tuple=True)  # coordonnees de ses pixels
        boxes.append([xs.min().item(), ys.min().item(), xs.max().item() + 1, ys.max().item() + 1])  # rectangle englobant XYXY
    return (  # renvoie les trois elements
        image,  # l'image test
        # les boites au format torchvision : XYXY, avec la taille de l'image (canvas_size)
        tv_tensors.BoundingBoxes(torch.tensor(boxes, dtype=torch.float32), format="XYXY", canvas_size=(height, width)),
        tv_tensors.Mask(mask),  # le masque au format torchvision
    )


def _iou(a: torch.Tensor, b: torch.Tensor) -> float:  # IoU de deux masques vrai/faux
    # IoU (intersection sur union) : surface commune / surface totale ; 1 = superposition parfaite, 0 = aucune
    union = (a | b).sum().item()  # nombre de pixels dans a OU b
    return 1.0 if union == 0 else (a & b).sum().item() / union  # pixels dans a ET b, divises par l'union (1 si tout est vide)


def _box_iou(a: Sequence[float], b: Sequence[float]) -> float:  # IoU de deux boites (x1, y1, x2, y2)
    # largeur et hauteur de la zone commune (0 si les boites ne se touchent pas)
    iw, ih = max(0.0, min(a[2], b[2]) - max(a[0], b[0])), max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih  # surface commune
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter  # surface a + surface b - partie comptee deux fois
    return inter / union if union > 0 else 0.0  # IoU (0 si les deux boites sont vides)


def check_alignment(  # verifie que l'augmentation transforme boites et masque comme l'image ; renvoie un bilan
    aug: Any,  # chaine d'augmentation construite, ou liste du YAML
    task: str = "detection",  # tache utilisee pour construire la chaine si on donne une liste
    *,  # les parametres suivants doivent etre nommes a l'appel
    size: Sequence[int] = (96, 128),  # taille de l'image test
    channels: int = 1,  # nombre de canaux de l'image test
    trials: int = 20,  # nombre de tirages a controler
    seed: int = 0,  # graine de depart (tirages reproductibles)
) -> dict[str, Any]:  # renvoie un dictionnaire : ok, taux de reussite, IoU medianes, tableau des tirages
    """Verifie sur une image test que masque et boites suivent l'image a travers l'augmentation.

    Pour chaque tirage : IoU entre la zone claire de l'image et le masque (doit rester haute, sauf
    si un trou cache un objet), et IoU entre chaque boite et le rectangle englobant son objet dans
    le masque (>= 0,5 et la boite contient l'objet : une rotation agrandit la boite englobante).
    ok = au moins 90 % des tirages passent les deux controles. Une operation ecrite a la main qui
    oublie de transformer les cibles echoue ici.
    """
    if isinstance(aug, (list, tuple)):  # si on a donne la liste du YAML
        aug = build_augmentation(aug, task)  # type: ignore[arg-type]  # on construit la chaine
    image, boxes, mask = alignment_pattern(size, channels)  # image test, ses boites et son masque
    rows = []  # une ligne de resultat par tirage
    for t in range(trials):  # pour chaque tirage
        torch.manual_seed(seed + t)  # graine differente mais reproductible a chaque tirage
        # applique l'augmentation aux trois en meme temps (copies : les originaux restent intacts)
        out_image, out_boxes, out_mask = aug(image.clone(), boxes.clone(), tv_tensors.Mask(mask.clone()))
        m = out_mask.as_subclass(torch.Tensor).reshape(out_mask.shape[-2:])  # masque augmente en tenseur simple (H, W)
        gray = out_image.mean(0)  # image augmentee en niveaux de gris (moyenne des canaux)
        fg = m > 0  # pixels d'objet selon le masque (fg = foreground = premier plan)
        if fg.all() or not fg.any():  # si le masque est tout objet ou tout fond, la comparaison n'a pas de sens
            mask_iou = float("nan")  # NaN = "non mesure"
        else:  # sinon on compare la zone claire de l'image au masque
            inside, outside = gray[fg].mean(), gray[~fg].mean()  # luminosite moyenne dans les objets et dans le fond
            threshold = (inside + outside) / 2  # seuil a mi-chemin
            pred = gray > threshold if inside >= outside else gray < threshold  # pixels "objet" d'apres l'image seule
            mask_iou = _iou(pred, fg)  # accord image / masque : proche de 1 si le masque a suivi l'image
        box_ious, contained = [], True  # IoU de chaque boite et "toutes les boites contiennent leur objet"
        b = out_boxes.as_subclass(torch.Tensor).reshape(-1, 4).tolist()  # boites augmentees en liste Python
        for k, box in zip((1, 2, 3), b, strict=False):  # pour chaque objet et sa boite
            ys, xs = torch.nonzero(m == k, as_tuple=True)  # pixels de l'objet dans le masque augmente
            if len(ys) == 0 or box[2] - box[0] < 1 or box[3] - box[1] < 1:  # si l'objet ou sa boite a disparu
                continue  # objet sorti de l'image
            ref = [xs.min().item(), ys.min().item(), xs.max().item() + 1, ys.max().item() + 1]  # rectangle de l'objet d'apres le masque
            box_ious.append(_box_iou(box, ref))  # accord boite / masque
            tol = 2.5  # tolerance en pixels (arrondis des transformations)
            # la boite doit contenir le rectangle de l'objet (a la tolerance pres) ; &= : reste faux des qu'un objet echoue
            contained &= box[0] <= ref[0] + tol and box[1] <= ref[1] + tol and box[2] >= ref[2] - tol and box[3] >= ref[3] - tol
        rows.append(  # ajoute le resultat du tirage
            # numero, IoU du masque, plus petite IoU de boite (la pire), et "boites contiennent leurs objets"
            {"trial": t, "mask_iou": mask_iou, "min_box_iou": min(box_ious) if box_ious else float("nan"), "box_contains": contained}
        )
    frame = pd.DataFrame(rows)  # tableau des tirages
    mask_ok = frame["mask_iou"].isna() | (frame["mask_iou"] >= 0.8)  # masque correct : non mesure ou IoU >= 0.8
    box_ok = frame["min_box_iou"].isna() | ((frame["min_box_iou"] >= 0.5) & frame["box_contains"])  # boites correctes
    passed = mask_ok & box_ok  # tirage reussi si les deux controles passent
    return {  # bilan
        "ok": bool(passed.mean() >= 0.9),  # vrai si au moins 90 % des tirages reussissent
        "pass_rate": float(passed.mean()),  # part des tirages reussis
        "median_mask_iou": float(frame["mask_iou"].median()),  # IoU mediane du masque
        "median_box_iou": float(frame["min_box_iou"].median()),  # IoU mediane de la pire boite
        "trials": frame,  # le detail tirage par tirage
    }


# --- vitesse --------------------------------------------------------------------------------------


def benchmark_ops(  # mesure le temps de chaque operation ; renvoie un tableau trie du plus lent au plus rapide
    names: Sequence[str] | None = None,  # operations a mesurer ; None = toutes celles permises pour la tache
    *,  # les parametres suivants doivent etre nommes a l'appel
    size: Sequence[int] = (256, 256),  # taille de l'image de test
    channels: int = 1,  # nombre de canaux de l'image de test
    repeats: int = 30,  # nombre de repetitions par operation (moyenne plus fiable)
    task: str = "classification",  # tache : seules les operations permises pour elle sont mesurees
    seed: int = 0,  # graine aleatoire (mesures reproductibles)
) -> pd.DataFrame:  # renvoie le tableau des temps
    """Temps moyen par image de chaque operation appliquee a coup sur (p = 1), du plus lent au plus rapide.

    Le chargement des donnees est souvent le goulot de l'entrainement sur CPU : une operation a
    plus de quelques millisecondes par image merite d'etre allegee ou retiree.
    """
    torch.manual_seed(seed)  # fixe le generateur torch
    image = torch.rand(channels, int(size[0]), int(size[1]))  # image de test aleatoire (C, H, W) dans [0, 1]
    rows = []  # une ligne par operation
    for name in names or [n for n, info in OPS.items() if task in info.tasks]:  # pour chaque operation a mesurer
        info = OPS[name]  # fiche de l'operation
        if task not in info.tasks:  # si elle n'est pas permise pour cette tache
            continue  # on la saute
        params: dict[str, Any] = {"name": name}  # etape minimale : seulement le nom (parametres par defaut)
        if "p" in inspect.signature(info.builder).parameters:  # si l'operation a une probabilite p
            params["p"] = 1.0  # on la force a 1 : on mesure le cout quand elle s'applique vraiment
        transform = build_step(params, task)  # type: ignore[arg-type]  # construit l'operation
        transform(image)  # echauffement
        start = time.perf_counter()  # demarre le chronometre (horloge precise)
        for _ in range(repeats):  # repete l'operation
            transform(image)  # applique l'operation (resultat ignore : seul le temps compte)
        ms = (time.perf_counter() - start) / repeats * 1000  # temps moyen par image, en millisecondes
        # ajoute la ligne : nom, famille, millisecondes par image, images par seconde
        rows.append({"op": name, "family": info.family, "ms_per_image": round(ms, 3), "images_per_s": round(1000 / max(ms, 1e-6), 1)})
    return pd.DataFrame(rows).sort_values("ms_per_image", ascending=False, ignore_index=True)  # tableau trie, plus lente en tete


# --- reproductibilite -----------------------------------------------------------------------------


def worker_init_fn(worker_id: int) -> None:  # appelee par le DataLoader au demarrage de chaque worker (processus de chargement)
    """A passer au DataLoader si le Dataset utilise numpy ou `random` : seme les deux depuis la graine torch du worker.

    Les operations de vtb.augment n'en ont pas besoin (elles tirent avec torch, re-seme par
    DataLoader dans chaque worker). Les versions actuelles de PyTorch sement deja numpy et `random`
    dans chaque worker ; avant la version 1.9, numpy donnait les MEMES tirages a tous les workers :
    cette fonction garantit le bon comportement quelle que soit la version.
    """
    seed = torch.initial_seed() % 2**32  # graine torch de ce worker, ramenee a 32 bits (limite de numpy)
    np.random.seed(seed)  # seme le generateur numpy avec elle
    random.seed(seed)  # seme aussi le generateur standard de Python
