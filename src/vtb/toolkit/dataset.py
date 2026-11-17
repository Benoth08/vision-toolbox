"""Outils de travail sur un dossier d'images : inventaire, statistiques, conversions en lot.

    from vtb.toolkit import dataset as ds
    files = ds.folder_summary("data/raw/neu_cls_flat")     # une ligne par fichier
    print(ds.summarize(files))                              # une ligne par dossier
    mean, std = ds.mean_std(files["path"], size=224)        # normalisation propre au jeu
    weights = ds.class_weights(files["folder"])             # perte ponderee si classes desequilibrees

Les lectures passent par `vtb.io.images` (chemins Windows avec accents, 16 bits conserves) ; les
traitements en lot tournent en parallele sur des threads (OpenCV relache le GIL pendant le
decodage, le redimensionnement et l'encodage).
"""

from __future__ import annotations  # permet d'ecrire les types "int | None" partout sans les evaluer a l'execution

import logging  # journal de messages (avertissements) plutot que des print
import os  # acces au systeme : nombre de coeurs, liens de fichiers
import shutil  # copie de fichiers
from collections.abc import Callable, Iterable, Sequence  # types pour annoter : fonction, objet parcourable, liste
from concurrent.futures import ThreadPoolExecutor  # lance des taches en parallele sur plusieurs threads
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux
from typing import Literal  # Literal : type qui n'accepte que quelques valeurs fixes ("rgb" ou "gray")

import cv2  # OpenCV : lecture, ecriture et traitement d'images (redimensionnement, video)
import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)
import pandas as pd  # pandas : tableaux avec des colonnes nommees (DataFrame), pour les inventaires

from vtb.core.config import IMAGE_EXTENSIONS  # liste des extensions d'images reconnues (.png, .jpg, .tif...)
from vtb.io.images import list_images, max_value, read_image, read_mask, to_uint8, write_image  # lecture / ecriture d'images de la boite
from vtb.preprocess.tiles import Tile, tile_grid  # decoupe d'une image en tuiles (carres) qui couvrent toute l'image

logger = logging.getLogger(__name__)  # journal propre a ce fichier (son nom apparait dans les messages)

PathLike = str | Path  # raccourci de type : un chemin donne en texte ou en Path


def _workers(workers: int | None) -> int:  # renvoie le nombre de threads a utiliser
    """Nombre de threads : un par coeur par defaut, au plus 16."""
    return max(1, min(16, workers or os.cpu_count() or 1))  # valeur demandee, sinon nombre de coeurs ; borne entre 1 et 16


def _map(func: Callable, items: Sequence, workers: int | None) -> list:  # applique func a chaque element ; renvoie la liste des resultats
    """map parallele qui garde l'ordre ; sequentiel s'il n'y a qu'un thread (traces d'erreur plus lisibles)."""
    n = _workers(workers)  # nombre de threads
    if n == 1 or len(items) < 2:  # si un seul thread ou un seul element : inutile de paralleliser
        return [func(item) for item in items]  # boucle simple, dans l'ordre
    with ThreadPoolExecutor(max_workers=n) as pool:  # cree un groupe de n threads, ferme proprement a la fin du bloc
        return list(pool.map(func, items))  # repartit le travail ; les resultats reviennent dans l'ordre des elements


# --------------------------------------------------------------------------- inventaire


def _file_row(path: Path, root: Path) -> dict:  # decrit une image ; renvoie un dict (une ligne du tableau d'inventaire)
    """Une ligne d'inventaire ; readable=False et la cause si l'image ne se decode pas."""
    relative = path.relative_to(root) if path != root else Path(path.name)  # chemin de l'image vu depuis le dossier racine
    row = {  # informations connues sans ouvrir l'image
        "path": str(path),  # chemin complet
        "folder": str(relative.parent) if str(relative.parent) != "." else "",  # sous-dossier (souvent = la classe), "" a la racine
        "name": path.name,  # nom du fichier
        "extension": path.suffix.lower(),  # extension en minuscules (.PNG et .png comptent pareil)
        "size_kb": round(path.stat().st_size / 1024, 1),  # taille du fichier sur le disque en kilo-octets
    }
    try:  # on essaie de decoder l'image
        image = read_image(path, color="unchanged")  # lecture telle quelle (16 bits et nombre de canaux d'origine conserves)
    except Exception as error:  # noqa: BLE001 - on veut la cause, quelle qu'elle soit
        return {**row, "readable": False, "error": f"{type(error).__name__}: {error}"[:200]}  # image illisible : la cause (200 car. max)
    return {  # image lisible : on ajoute ses dimensions et son type
        **row,  # reprend toutes les cases de row
        "readable": True,  # l'image se decode
        "error": "",  # pas d'erreur
        "height": image.shape[0],  # hauteur en pixels
        "width": image.shape[1],  # largeur en pixels
        "channels": 1 if image.ndim == 2 else image.shape[2],  # 1 canal (gris) si tableau 2D, sinon 3 (couleur) ou 4 (avec transparence)
        "dtype": str(image.dtype),  # type des pixels : uint8 (0-255) ou uint16 (0-65535)
    }


def folder_summary(  # inventaire d'un dossier ; renvoie un DataFrame (une ligne par image)
    root: PathLike,  # dossier a inventorier (ou un seul fichier)
    *,  # les arguments suivants doivent etre nommes
    recursive: bool = True,  # True : descend aussi dans les sous-dossiers
    extensions: Sequence[str] = IMAGE_EXTENSIONS,  # extensions acceptees comme images
    workers: int | None = None,  # nombre de threads (None = un par coeur)
) -> pd.DataFrame:  # renvoie le tableau d'inventaire
    """Inventaire d'un dossier : une ligne par image (path, folder, height, width, channels, dtype, size_kb, readable).

    `folder` est le sous-dossier relatif a root : avec un rangement root/<classe>/image, c'est la
    classe. Plus leger que `vtb.quality.scan` (pas de statistiques de pixels ni de doublons) : le
    premier coup d'oeil sur des donnees qu'on vient de recevoir.
    """
    root = Path(root)  # en Path pour manipuler le chemin
    paths = list_images(root, extensions, recursive=recursive)  # liste triee des fichiers images
    if not paths:  # si aucune image n'a ete trouvee
        raise FileNotFoundError(f"aucune image ({', '.join(extensions)}) dans {root}")  # erreur claire avec les extensions cherchees
    base = root if root.is_dir() else root.parent  # dossier de reference pour les chemins relatifs
    return pd.DataFrame(_map(lambda p: _file_row(p, base), paths, workers))  # decrit chaque image en parallele, puis en fait un tableau


def summarize(files: pd.DataFrame, by: str = "folder") -> pd.DataFrame:  # resume par dossier ; renvoie un DataFrame
    """Une ligne par dossier : nombre d'images, illisibles, tailles et types rencontres, poids moyen."""

    def sizes(frame: pd.DataFrame) -> str:  # texte des tailles les plus frequentes d'un groupe (ex. "200x200 (300)")
        ok = frame[frame["readable"]]  # garde les images lisibles seulement
        if ok.empty:  # si aucune n'est lisible
            return ""  # pas de taille a afficher
        # texte "hauteur x largeur" pour chaque image, puis nombre d'images par taille (de la plus frequente a la moins)
        counts = (ok["height"].astype(int).astype(str) + "x" + ok["width"].astype(int).astype(str)).value_counts()
        text = ", ".join(f"{size} ({n})" for size, n in counts.head(3).items())  # les 3 tailles les plus frequentes
        return text + (f", +{len(counts) - 3} autres" if len(counts) > 3 else "")  # signale combien d'autres tailles existent

    rows = []  # une ligne de resume par dossier
    for name, frame in files.groupby(by, sort=True):  # pour chaque dossier (groupe d'images de meme valeur "by")
        ok = frame[frame["readable"]]  # images lisibles du dossier
        rows.append(  # ajoute la ligne de resume du dossier
            {  # dictionnaire nom de colonne -> valeur pour ce dossier
                by: name if name != "" else ".",  # nom du dossier ("." pour la racine)
                "images": len(frame),  # nombre d'images
                "unreadable": int((~frame["readable"]).sum()),  # nombre d'images illisibles (~ = "non")
                "sizes (HxW)": sizes(frame),  # tailles les plus frequentes
                # nombres de canaux rencontres (ex. "1,3")
                "channels": ",".join(str(int(c)) for c in sorted(ok["channels"].dropna().unique())) if not ok.empty else "",
                "dtypes": ",".join(sorted(ok["dtype"].dropna().unique())) if not ok.empty else "",  # types de pixels rencontres
                "extensions": ",".join(sorted(frame["extension"].unique())),  # extensions rencontrees
                "mean_kb": round(float(frame["size_kb"].mean()), 1),  # taille moyenne d'un fichier
            }
        )
    table = pd.DataFrame(rows)  # tableau des resumes
    total = {  # ligne TOTAL ajoutee en bas du tableau
        by: "TOTAL",  # etiquette de la ligne
        "images": int(table["images"].sum()),  # total des images
        "unreadable": int(table["unreadable"].sum()),  # total des illisibles
        "sizes (HxW)": "",  # pas de sens au total
        "channels": "",  # pas de sens au total
        "dtypes": ",".join(sorted(files.loc[files["readable"], "dtype"].dropna().unique())),  # tous les types de pixels du jeu
        "extensions": ",".join(sorted(files["extension"].unique())),  # toutes les extensions du jeu
        "mean_kb": round(float(files["size_kb"].mean()), 1),  # taille moyenne sur tout le jeu
    }
    return pd.concat([table, pd.DataFrame([total])], ignore_index=True)  # colle la ligne TOTAL sous le tableau et le renvoie


# --------------------------------------------------------------------------- statistiques


def mean_std(  # moyenne et ecart-type des pixels par canal ; renvoie (liste des moyennes, liste des ecarts-types)
    paths: Iterable[PathLike],  # chemins des images (du TRAIN seulement)
    *,  # les arguments suivants doivent etre nommes
    size: int | None = None,  # taille du carre auquel on ramene chaque image avant calcul (None = taille d'origine)
    max_images: int | None = None,  # nombre maximum d'images (tirees au hasard) pour aller plus vite
    color: Literal["rgb", "gray"] = "rgb",  # lecture en couleur (3 canaux) ou en niveaux de gris (1 canal)
    seed: int = 0,  # germe du hasard pour le tirage : meme germe = memes images
    workers: int | None = None,  # nombre de threads
) -> tuple[list[float], list[float]]:  # (moyennes, ecarts-types), un nombre par canal
    """Moyenne et ecart-type par canal, sur des valeurs ramenees dans [0, 1] (8 ou 16 bits).

    A calculer sur le TRAIN seulement. Utile pour un reseau appris de zero ou des images tres loin
    d'ImageNet ; avec un reseau pre-entraine, on garde en general la normalisation ImageNet.
    size : redimensionnement carre avant calcul (plus rapide, et c'est ce que voit le reseau).
    """
    # Normalisation : on soustrait la moyenne et on divise par l'ecart-type de chaque canal, pour que
    # les valeurs d'entree du reseau soient centrees autour de 0 avec une dispersion proche de 1.
    paths = [str(p) for p in paths]  # liste de chemins en texte
    if max_images is not None and len(paths) > max_images:  # si on a demande une limite et qu'il y a plus d'images
        paths = list(np.random.default_rng(seed).choice(paths, size=max_images, replace=False))  # tirage au hasard sans remise
    if not paths:  # si la liste est vide
        raise ValueError("aucune image pour calculer la moyenne")  # on arrete avec un message clair

    def sums(path: str) -> tuple[np.ndarray, np.ndarray, int]:  # pour une image : (somme des pixels, somme des carres, nombre)
        image = read_image(path, color=color)  # lit l'image (8 ou 16 bits)
        values = image.astype(np.float64) / max_value(image)  # en decimaux entre 0 et 1 (divise par 255 ou 65535)
        if size is not None:  # si une taille est demandee
            values = cv2.resize(values, (size, size), interpolation=cv2.INTER_AREA)  # ramene a size x size (INTER_AREA : bon pour reduire)
        values = values.reshape(-1, 1 if values.ndim == 2 else values.shape[2])  # une ligne par pixel, une colonne par canal
        return values.sum(axis=0), (values**2).sum(axis=0), values.shape[0]  # sommes par canal et nombre de pixels

    parts = _map(sums, paths, workers)  # calcule les sommes de chaque image en parallele
    total = sum(p[0] for p in parts)  # somme de tous les pixels, par canal
    total_sq = sum(p[1] for p in parts)  # somme de tous les carres, par canal
    count = sum(p[2] for p in parts)  # nombre total de pixels
    mean = total / count  # moyenne par canal
    std = np.sqrt(np.maximum(total_sq / count - mean**2, 0.0))  # ecart-type = racine(moyenne des carres - carre de la moyenne)
    return [round(float(v), 4) for v in mean], [round(float(v), 4) for v in std]  # deux listes de nombres arrondis a 4 chiffres


# poids par classe pour une perte ponderee (les classes rares comptent plus) ; renvoie un dict classe -> poids
def class_weights(labels: Iterable, *, mode: Literal["inverse", "sqrt", "effective"] = "inverse", beta: float = 0.999) -> dict[str, float]:
    """Poids par classe pour une perte ponderee, dans l'ordre alphabetique des classes.

    inverse   : n / (K * n_c), le "balanced" de scikit-learn (compense tout le desequilibre) ;
    sqrt      : 1 / sqrt(n_c), compensation partielle, souvent plus stable ;
    effective : (1 - beta) / (1 - beta ** n_c), "nombre effectif d'exemples" (Cui et al. 2019).
    Les poids sont normalises pour que la moyenne ponderee par les effectifs vaille 1 : l'echelle
    de la perte (donc le bon taux d'apprentissage) ne change pas.

        weights = class_weights(train["label"])
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(list(weights.values()), dtype=torch.float32))
    """
    counts = pd.Series(list(labels)).astype(str).value_counts().sort_index()  # nombre d'exemples par classe, classes triees par nom
    n_c = counts.to_numpy(dtype=np.float64)  # ces effectifs en tableau de decimaux
    if mode == "inverse":  # mode "inverse" : poids inversement proportionnel a l'effectif
        raw = n_c.sum() / (len(n_c) * n_c)  # n / (nombre de classes x effectif de la classe)
    elif mode == "sqrt":  # mode "sqrt" : compensation plus douce
        raw = 1.0 / np.sqrt(n_c)  # 1 / racine de l'effectif
    elif mode == "effective":  # mode "effective" : nombre effectif d'exemples
        raw = (1.0 - beta) / (1.0 - np.power(beta, n_c))  # formule de Cui et al. (beta proche de 1)
    else:  # mode inconnu
        raise ValueError(f"mode inconnu : {mode} (inverse, sqrt, effective)")  # on arrete avec la liste des modes valides
    raw = raw * n_c.sum() / (raw * n_c).sum()  # remise a l'echelle : moyenne des poids sur tous les exemples = 1
    return {name: round(float(w), 4) for name, w in zip(counts.index, raw)}  # dict nom de classe -> poids arrondi


# un poids par exemple (pour tirer plus souvent les classes rares) ; renvoie un tableau numpy
def sample_weights(labels: Iterable, *, mode: Literal["inverse", "sqrt", "effective"] = "inverse") -> np.ndarray:
    """Un poids par exemple (pour torch.utils.data.WeightedRandomSampler) : l'autre facon de compenser."""
    labels = pd.Series(list(labels)).astype(str)  # etiquettes en texte dans une Series pandas
    weights = class_weights(labels, mode=mode)  # poids de chaque classe
    return labels.map(weights).to_numpy(dtype=np.float64)  # chaque exemple recoit le poids de sa classe


# sous-ensemble equilibre : au plus n lignes par classe ; renvoie un DataFrame
def sample_per_class(frame: pd.DataFrame, n: int, *, label: str = "label", seed: int = 0) -> pd.DataFrame:
    """Au plus n lignes par classe, tirees au hasard (reproductible) : sous-ensemble equilibre pour un essai rapide."""
    # pour chaque classe : tire n lignes au hasard (ou toutes s'il y en a moins), avec un germe fixe
    parts = [group.sample(n=min(n, len(group)), random_state=seed) for _, group in frame.groupby(label, sort=True)]
    return pd.concat(parts).reset_index(drop=True)  # recolle les morceaux et renumerote les lignes 0, 1, 2...


# --------------------------------------------------------------------------- rangements


def _place(source: Path, target: Path, mode: str) -> None:  # place un fichier a sa nouvelle adresse ; ne renvoie rien
    """Copie, lien physique ou lien symbolique (repli sur la copie si le systeme refuse, ex. Windows sans droits)."""
    target.parent.mkdir(parents=True, exist_ok=True)  # cree le dossier de destination (et ses parents) s'il n'existe pas
    if target.exists():  # si un fichier occupe deja la place
        target.unlink()  # on le supprime pour le remplacer
    if mode == "link":  # mode lien : pas de copie, on economise la place disque
        for make_link in (os.link, os.symlink):  # essaie d'abord un lien physique, puis un lien symbolique
            try:  # le systeme peut refuser (droits, disques differents)
                make_link(source.resolve(), target)  # cree le lien vers le fichier d'origine
                return  # reussi : rien d'autre a faire
            except OSError:  # si le systeme refuse ce type de lien
                continue  # on essaie le suivant
    shutil.copy2(source, target)  # mode copie (ou aucun lien possible) : copie le fichier avec ses dates


def export_imagefolder(  # range les images en dossiers split/classe ; renvoie le catalogue avec la colonne new_path
    frame: pd.DataFrame,  # catalogue : une ligne par image avec au moins path, split et label
    out_dir: PathLike,  # dossier de sortie
    *,  # les arguments suivants doivent etre nommes
    mode: Literal["copy", "link"] = "copy",  # copier les fichiers, ou creer des liens (pas de place disque en plus)
    split: str = "split",  # nom de la colonne qui dit train / val / test
    label: str = "label",  # nom de la colonne qui donne la classe
) -> pd.DataFrame:  # le catalogue complete
    """Range un catalogue decoupe en out_dir/<split>/<classe>/<image> (format ImageFolder de torchvision, Keras, Ultralytics-cls).

    Deux images de meme nom dans une classe (venues de dossiers differents) sont prefixees par leur
    numero de ligne au lieu de s'ecraser. Renvoie le catalogue avec la colonne new_path.
    """
    out_dir = Path(out_dir)  # en Path pour construire les chemins
    frame = frame.reset_index(drop=True).copy()  # copie numerotee 0, 1, 2... (on ne modifie pas le tableau de l'appelant)
    names = frame["path"].map(lambda p: Path(p).name)  # nom de fichier de chaque image
    keys = frame[split].astype(str) + "/" + frame[label].astype(str) + "/" + names  # adresse cible "split/classe/nom"
    duplicated = keys.duplicated(keep=False)  # True pour les images qui auraient la meme adresse qu'une autre
    new_paths = []  # nouveaux chemins, dans l'ordre des lignes
    for i, row in frame.iterrows():  # pour chaque image du catalogue (i = numero de ligne)
        name = names[i] if not duplicated[i] else f"{i:06d}_{names[i]}"  # nom garde, ou prefixe par le numero de ligne si doublon
        new_paths.append(str(out_dir / str(row[split]) / str(row[label]) / name))  # out_dir/split/classe/nom
    # copie (ou lie) chaque fichier vers sa nouvelle adresse, en parallele
    _map(lambda pair: _place(Path(pair[0]), Path(pair[1]), mode), list(zip(frame["path"], new_paths)), None)
    frame["new_path"] = new_paths  # garde la nouvelle adresse dans le catalogue
    return frame  # renvoie le catalogue complete


# --------------------------------------------------------------------------- conversions en lot


def resize_image(  # redimensionne une image ; renvoie la nouvelle image
    image: np.ndarray,  # image d'entree (8 ou 16 bits, gris ou couleur)
    size: int | tuple[int, int],  # entier : taille du plus grand cote ; couple : (largeur, hauteur) exactes
    *,  # les arguments suivants doivent etre nommes
    keep_aspect: bool = True,  # True : garde les proportions (pas de deformation)
    pad: bool = False,  # True : complete avec des bandes pour obtenir un carre size x size
    pad_value: int = 0,  # valeur des pixels de bande (0 = noir)
) -> np.ndarray:  # l'image redimensionnee
    """Redimensionne une image (8 ou 16 bits conserves).

    size entier + keep_aspect : le plus grand cote vaut size ; + pad : complete en carre size x size
    (centre, valeur pad_value). size (largeur, hauteur) sans keep_aspect : etirement exact.
    INTER_AREA pour reduire (pas de crenelage), INTER_LINEAR pour agrandir.
    """
    height, width = image.shape[:2]  # dimensions d'origine (les 2 premieres cases de shape)
    if isinstance(size, int):  # si size est un seul entier
        if keep_aspect:  # en gardant les proportions
            scale = size / max(height, width)  # facteur d'echelle pour que le plus grand cote vaille size
            new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))  # nouvelles dimensions (au moins 1 pixel)
        else:  # sans garder les proportions
            new_w = new_h = size  # carre size x size (l'image est deformee)
    else:  # size est un couple (largeur, hauteur)
        new_w, new_h = int(size[0]), int(size[1])  # dimensions exactes demandees
    shrink = new_w * new_h < width * height  # True si on reduit l'image
    # INTER_AREA pour reduire (moyenne les pixels, pas de crenelage), INTER_LINEAR pour agrandir
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)
    if pad and isinstance(size, int) and keep_aspect:  # si on veut un carre complete par des bandes
        top, left = (size - new_h) // 2, (size - new_w) // 2  # marges haut et gauche pour centrer l'image
        value = [pad_value] * (1 if resized.ndim == 2 else resized.shape[2])  # valeur de bande, une par canal
        # ajoute les bandes en haut, en bas, a gauche et a droite (valeur constante)
        resized = cv2.copyMakeBorder(resized, top, size - new_h - top, left, size - new_w - left, cv2.BORDER_CONSTANT, value=value)
    return resized  # renvoie l'image redimensionnee


def _convert_folder(  # applique une conversion a toutes les images d'un dossier ; renvoie un rapport (une ligne par image)
    src: PathLike,  # dossier source
    dst: PathLike,  # dossier destination (meme arborescence)
    convert: Callable[[np.ndarray], np.ndarray],  # fonction image -> image a appliquer
    extension: str | None,  # nouvelle extension (ex. ".png"), ou None pour garder celle d'origine
    workers: int | None,  # nombre de threads
) -> pd.DataFrame:  # le rapport
    """Applique convert a chaque image de src et l'ecrit dans dst en gardant l'arborescence."""
    src, dst = Path(src), Path(dst)  # en Path pour construire les chemins
    paths = list_images(src)  # toutes les images du dossier source
    if not paths:  # si le dossier ne contient aucune image
        raise FileNotFoundError(f"aucune image dans {src}")  # on arrete avec un message clair

    def one(path: Path) -> dict:  # convertit une image ; renvoie sa ligne de rapport
        target = dst / path.relative_to(src)  # meme chemin relatif, mais sous dst
        if extension is not None:  # si on change de format
            target = target.with_suffix(extension)  # remplace l'extension du fichier cible
        try:  # une image peut echouer (fichier abime...)
            result = convert(read_image(path, color="unchanged"))  # lit l'image telle quelle puis la convertit
            write_image(target, result)  # ecrit le resultat (cree les dossiers au besoin)
            # ligne de rapport : reussite et nouvelle forme (ex. "224x224x3")
            return {"source": str(path), "target": str(target), "ok": True, "error": "", "shape": "x".join(map(str, result.shape))}
        except Exception as error:  # noqa: BLE001 - une image ratee ne doit pas arreter le lot
            # ligne de rapport : echec et sa cause (200 caracteres max)
            return {"source": str(path), "target": str(target), "ok": False, "error": f"{type(error).__name__}: {error}"[:200], "shape": ""}

    report = pd.DataFrame(_map(one, paths, workers))  # convertit toutes les images en parallele ; rapport en tableau
    failed = int((~report["ok"]).sum())  # nombre d'images ratees
    if failed:  # s'il y a eu des echecs
        logger.warning("%d image(s) non converties sur %d (colonne error)", failed, len(report))  # previent dans le journal
    return report  # renvoie le rapport


def resize_folder(  # redimensionne tout un dossier ; renvoie un rapport par image
    src: PathLike,  # dossier source
    dst: PathLike,  # dossier destination
    size: int | tuple[int, int],  # taille voulue (voir resize_image)
    *,  # les arguments suivants doivent etre nommes
    keep_aspect: bool = True,  # garder les proportions
    pad: bool = False,  # completer en carre
    extension: str | None = None,  # nouvelle extension, ou None pour garder l'actuelle
    workers: int | None = None,  # nombre de threads
) -> pd.DataFrame:  # le rapport
    """Redimensionne tout un dossier (arborescence conservee) en parallele ; renvoie un rapport par image."""
    # lambda : petite fonction sans nom qui redimensionne une image avec les reglages choisis
    return _convert_folder(src, dst, lambda im: resize_image(im, size, keep_aspect=keep_aspect, pad=pad), extension, workers)


def to_8bit(  # convertit une image 16 bits (ou decimale) en 8 bits ; renvoie l'image uint8
    image: np.ndarray,  # image d'entree
    *,  # les arguments suivants doivent etre nommes
    method: Literal["percentile", "minmax", "scale"] = "percentile",  # facon de ramener les valeurs dans 0-255
    low: float = 0.5,  # percentile bas : ce qui est en dessous devient noir (0)
    high: float = 99.5,  # percentile haut : ce qui est au-dessus devient blanc (255)
) -> np.ndarray:  # l'image 8 bits
    """Ramene une image 16 bits (ou flottante) en 8 bits.

    scale      : division par la valeur maximale du type (aucun contraste invente ; souvent tres sombre) ;
    minmax     : etirement entre le minimum et le maximum de l'image (sensible a un seul pixel chaud) ;
    percentile : etirement entre les percentiles low et high (robuste, le choix par defaut).
    Attention : percentile et minmax normalisent chaque image a sa propre dynamique ; si l'intensite
    absolue a un sens (thermique, radiographie), preferer scale ou des bornes fixes communes.
    """
    if method == "scale":  # simple division par la valeur maximale du type
        return to_uint8(image)  # sans percentiles : divise par 65535 (ou 1) puis multiplie par 255
    if method == "minmax":  # etirement entre le plus petit et le plus grand pixel
        return to_uint8(image, 0.0, 100.0)  # percentile 0 = minimum, percentile 100 = maximum
    if method == "percentile":  # etirement robuste entre deux percentiles
        return to_uint8(image, low, high)  # ignore les quelques pixels extremes
    raise ValueError(f"methode inconnue : {method} (percentile, minmax, scale)")  # aucune methode connue : erreur claire


def convert_folder_8bit(  # convertit tout un dossier en 8 bits ; renvoie un rapport par image
    src: PathLike,  # dossier source (images 16 bits)
    dst: PathLike,  # dossier destination
    *,  # les arguments suivants doivent etre nommes
    method: Literal["percentile", "minmax", "scale"] = "percentile",  # methode de conversion (voir to_8bit)
    low: float = 0.5,  # percentile bas
    high: float = 99.5,  # percentile haut
    extension: str | None = ".png",  # PNG par defaut : compression sans perte
    workers: int | None = None,  # nombre de threads
) -> pd.DataFrame:  # le rapport
    """Convertit un dossier d'images 16 bits en 8 bits (PNG par defaut : sans perte)."""
    # applique to_8bit, avec les reglages choisis, a chaque image du dossier
    return _convert_folder(src, dst, lambda im: to_8bit(im, method=method, low=low, high=high), extension, workers)


# --------------------------------------------------------------------------- video


def extract_frames(  # extrait des images d'une video ; renvoie le nombre d'images ecrites
    video: PathLike,  # chemin du fichier video
    out_dir: PathLike,  # dossier ou ecrire les images
    *,  # les arguments suivants doivent etre nommes
    every: int = 1,  # garder une image sur "every"
    fps: float | None = None,  # ou bien : nombre d'images a garder par seconde de video (remplace every)
    start: float | None = None,  # instant de debut en secondes (None = debut de la video)
    end: float | None = None,  # instant de fin en secondes (None = jusqu'au bout)
    max_frames: int | None = None,  # nombre maximum d'images a ecrire
    extension: str = ".png",  # format des images ecrites
    prefix: str | None = None,  # debut du nom des fichiers (None = nom de la video)
) -> int:  # nombre d'images ecrites
    """Extrait des images d'une video ; renvoie le nombre d'images ecrites.

    every : une image sur `every` ; fps : cadence voulue (remplace every, ex. fps=2 -> deux images
    par seconde de video) ; start / end en secondes ; noms <prefix>_<numero d'image>.png, le numero
    etant celui de la video (on retrouve l'instant : numero / fps de la video).
    Les images sautees sont seulement "grab" (pas decodees) : beaucoup plus rapide.
    """
    capture = cv2.VideoCapture(str(video))  # ouvre la video avec OpenCV
    if not capture.isOpened():  # si OpenCV n'a pas pu l'ouvrir
        raise ValueError(f"video illisible : {video}")  # on arrete avec un message clair
    video_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0  # cadence de la video (images par seconde) ; 25 si inconnue
    step = max(1, round(video_fps / fps)) if fps else max(1, int(every))  # pas entre deux images gardees
    first = int(round(start * video_fps)) if start else 0  # numero de la premiere image (secondes x cadence)
    last = int(round(end * video_fps)) if end is not None else None  # numero de la derniere image, ou None
    if first:  # si on ne commence pas au debut
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)  # saute directement a la premiere image voulue
    out_dir = Path(out_dir)  # en Path pour construire les chemins
    out_dir.mkdir(parents=True, exist_ok=True)  # cree le dossier de sortie s'il n'existe pas
    prefix = prefix or Path(video).stem  # prefixe des noms : celui donne, sinon le nom de la video sans extension
    index, written = first, 0  # numero de l'image courante dans la video, et nombre d'images ecrites
    try:  # le bloc finally plus bas liberera la video meme en cas d'erreur
        while last is None or index <= last:  # tant qu'on n'a pas depasse la derniere image voulue
            if not capture.grab():  # avance d'une image sans la decoder (rapide) ; False = fin de la video
                break  # plus d'image : on sort de la boucle
            if (index - first) % step == 0:  # si cette image tombe sur le pas choisi
                ok, frame = capture.retrieve()  # decode l'image qu'on vient d'attraper
                if not ok:  # si le decodage a echoue
                    break  # on arrete la lecture
                ok, buffer = cv2.imencode(extension, frame)  # frame est en BGR, imencode attend du BGR
                if not ok:  # si l'encodage a echoue (extension non geree)
                    raise ValueError(f"encodage impossible en {extension}")  # on arrete avec un message clair
                buffer.tofile(str(out_dir / f"{prefix}_{index:06d}{extension}"))  # ecrit les octets (accepte les chemins avec accents)
                written += 1  # une image de plus ecrite
                if max_frames is not None and written >= max_frames:  # si on a atteint le maximum demande
                    break  # on s'arrete
            index += 1  # passe a l'image suivante
    finally:  # dans tous les cas (fin normale ou erreur)
        capture.release()  # ferme la video et libere le fichier
    return written  # renvoie le nombre d'images ecrites


# --------------------------------------------------------------------------- patchs


# decoupe une image (et son masque) en patchs = petits carres ; renvoie une liste de dicts (un par patch)
def make_patches(image: np.ndarray, mask: np.ndarray | None = None, *, size: int, stride: int | None = None) -> list[dict]:
    """Decoupe une image (et son masque) en patchs size x size espaces de stride (= size : sans recouvrement).

    La grille est celle de `vtb.preprocess.tiles` (dernier patch colle au bord : toute l'image est
    couverte, aucun patch partiel). Chaque patch : {x, y, image, mask, defect_fraction}.
    """
    # masque : image de meme taille ou les pixels du defaut sont non nuls ; stride : decalage entre deux patchs
    stride = stride or size  # par defaut, decalage = taille : patchs cote a cote, sans recouvrement
    if stride > size:  # si le decalage est plus grand que le patch
        raise ValueError("stride > size laisserait des bandes non couvertes")  # des pixels ne seraient dans aucun patch
    # positions des patchs sur toute l'image ; recouvrement entre deux patchs = size - stride pixels
    tiles: list[Tile] = tile_grid(image.shape[0], image.shape[1], size, size - stride)
    patches = []  # liste des patchs produits
    for tile in tiles:  # pour chaque position de patch
        rows, cols = tile.slice()  # tranches de lignes et de colonnes qui correspondent au patch
        patch_mask = mask[rows, cols] if mask is not None else None  # morceau du masque au meme endroit (s'il y a un masque)
        patches.append(  # ajoute le patch a la liste
            {  # description du patch
                "x": tile.x,  # colonne du coin haut-gauche dans l'image d'origine
                "y": tile.y,  # ligne du coin haut-gauche
                "image": image[rows, cols].copy(),  # pixels du patch (copie : independante de l'image d'origine)
                "mask": None if patch_mask is None else patch_mask.copy(),  # morceau de masque (copie), ou None
                # part des pixels du patch qui sont du defaut (NaN = "pas un nombre" quand il n'y a pas de masque)
                "defect_fraction": float((patch_mask > 0).mean()) if patch_mask is not None else float("nan"),
            }
        )
    return patches  # renvoie tous les patchs


# cherche le fichier masque d'une image ; renvoie son chemin, ou None s'il n'existe pas
def _find_mask(path: Path, relative: Path, masks_dir: PathLike | None, mask_suffix: str) -> Path | None:
    """Masque <nom><suffixe>.png dans masks_dir (meme arborescence) ou a cote de l'image ; None s'il n'existe pas."""
    base = (Path(masks_dir) / relative).parent if masks_dir is not None else path.parent  # dossier ou chercher le masque
    candidate = base / f"{path.stem}{mask_suffix}.png"  # nom attendu : nom de l'image + suffixe + .png
    if candidate.exists() and candidate.resolve() != path.resolve():  # s'il existe et n'est pas l'image elle-meme
        return candidate  # on a trouve le masque
    return None  # pas de masque pour cette image


def patch_folder(  # decoupe toutes les images d'un dossier en patchs ; renvoie un DataFrame (une ligne par patch)
    src: PathLike,  # dossier source
    dst: PathLike,  # dossier de sortie (dst/images et dst/masks)
    *,  # les arguments suivants doivent etre nommes
    size: int,  # taille d'un patch en pixels (carre)
    stride: int | None = None,  # decalage entre deux patchs (None = size, sans recouvrement)
    masks: bool | None = None,  # None : masques utilises s'ils existent ; True : obligatoires ; False : ignores
    masks_dir: PathLike | None = None,  # dossier des masques (None = a cote des images)
    mask_suffix: str = "",  # suffixe du nom des masques (ex. "_mask")
    keep_empty: float = 1.0,  # part des patchs sans defaut a garder (1.0 = tous)
    seed: int = 0,  # germe du hasard pour le tri des patchs sans defaut
    extension: str = ".png",  # format des patchs d'image ecrits
) -> pd.DataFrame:  # le catalogue des patchs
    """Decoupe toutes les images d'un dossier en patchs : dst/images/... et, avec masques, dst/masks/...

    Masques : meme convention que le catalogue image + masque (`<nom><suffixe>.png` dans masks_dir
    avec la meme arborescence, ou a cote de l'image, comme Magnetic Tile). masks=None : utilises
    s'ils existent ; True : obligatoires ; False : ignores. Les fichiers reconnus comme masques ne
    sont pas decoupes comme des images.
    keep_empty : fraction des patchs SANS defaut a garder (ex. 0.2 quand les defauts sont rares :
    on reequilibre sans rien jeter de defectueux). Renvoie une ligne par patch (source, path,
    mask_path, x, y, has_defect, defect_fraction).
    A faire AVANT le decoupage train/val/test, en gardant la colonne source comme groupe : deux
    patchs d'une meme image ne doivent jamais etre de part et d'autre.
    """
    src, dst = Path(src), Path(dst)  # en Path pour construire les chemins
    rng = np.random.default_rng(seed)  # generateur aleatoire reproductible
    paths = list_images(src)  # toutes les images du dossier (masques compris a ce stade)
    # pour chaque image : le chemin de son masque (ou None) ; on ne cherche pas si masks=False
    found = {p: (_find_mask(p, p.relative_to(src), masks_dir, mask_suffix) if masks is not False else None) for p in paths}
    mask_files = {m.resolve() for m in found.values() if m is not None}  # ensemble des fichiers qui sont des masques
    rows = []  # une ligne de catalogue par patch ecrit
    for path in paths:  # pour chaque fichier image du dossier
        if path.resolve() in mask_files:  # si ce fichier est en fait un masque
            continue  # c'est le masque d'une autre image
        mask_path = found[path]  # chemin du masque de cette image, ou None
        if masks is True and mask_path is None:  # si les masques sont obligatoires et qu'il manque celui-ci
            raise FileNotFoundError(f"masque absent pour {path}")  # on arrete avec un message clair
        mask = read_mask(mask_path) if mask_path is not None else None  # lit le masque (0 = sain, autre = defaut), ou None
        image = read_image(path, color="unchanged")  # lit l'image telle quelle
        stem_base = path.relative_to(src).with_suffix("").as_posix().replace("/", "__")  # nom unique : sous-dossiers relies par "__"
        for patch in make_patches(image, mask, size=size, stride=stride):  # pour chaque patch de l'image
            has_defect = bool(patch["mask"] is not None and patch["mask"].any())  # True si au moins un pixel de defaut
            if patch["mask"] is not None and not has_defect and rng.random() >= keep_empty:  # patch sain tire pour etre jete
                continue  # on ne l'ecrit pas (reequilibrage sains / defauts)
            stem = f"{stem_base}_y{patch['y']}_x{patch['x']}"  # nom du patch : image d'origine + position
            image_path = dst / "images" / f"{stem}{extension}"  # chemin du patch d'image
            write_image(image_path, patch["image"])  # ecrit le patch d'image
            mask_out = ""  # chemin du patch de masque ("" s'il n'y en a pas)
            if patch["mask"] is not None:  # si l'image a un masque
                mask_out = str(dst / "masks" / f"{stem}.png")  # chemin du patch de masque (toujours en PNG)
                write_image(mask_out, (patch["mask"] > 0).astype(np.uint8) * 255)  # ecrit le masque en noir (0) et blanc (255)
            rows.append(  # ajoute la ligne du patch au catalogue
                {  # dictionnaire nom de colonne -> valeur pour ce patch
                    "source": str(path),  # image d'origine (a garder comme groupe pour le decoupage train/val/test)
                    "path": str(image_path),  # chemin du patch d'image
                    "mask_path": mask_out,  # chemin du patch de masque
                    "x": patch["x"],  # position horizontale dans l'image d'origine
                    "y": patch["y"],  # position verticale
                    "has_defect": has_defect,  # True si le patch contient du defaut
                    "defect_fraction": round(patch["defect_fraction"], 5),  # part de pixels de defaut
                }
            )
    return pd.DataFrame(rows)  # renvoie le catalogue des patchs
