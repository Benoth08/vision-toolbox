"""Le catalogue : quel que soit le rangement sur le disque, un seul tableau, une ligne par image.

Colonnes standard (toutes ne sont pas toujours presentes) :
    path             chemin de l'image (seule colonne obligatoire)
    label            classe de l'image ("" si inconnue) ; en anomalie : good / defect
    split            train / val / test quand le rangement l'impose (sinon ajoute par vtb.split)
    group            identifiant de source (piece, bobine, brame) : deux images du meme groupe
                     restent du meme cote du decoupage
    timestamp        date d'acquisition (decoupage temporel)
    mask_path        masque de verite terrain (segmentation, anomalies)
    defect_type      type de defaut (anomalies MVTec)
    annotation_path  fichier d'annotation de l'image (detection)
    n_objects        nombre d'objets annotes

La suite du code ne lit jamais le disque autrement que par ce tableau.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (ex. Path | None)

import logging  # journal : messages horodates sur la console et dans un fichier
import re  # expressions regulieres : chercher un motif dans un texte (ici dans les noms de fichiers)
from datetime import datetime  # type date + heure
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import pandas as pd  # pandas : tableaux de donnees (DataFrame = tableau avec des colonnes nommees)

from vtb.core.config import DataConfig  # section "data" de la configuration (ou et comment lire les images)
from vtb.io.images import list_images, read_mask  # liste des images d'un dossier ; lecture d'un masque binaire

logger = logging.getLogger(__name__)  # journal propre a ce module


def from_class_folders(root: Path, extensions: list[str]) -> pd.DataFrame:  # catalogue d'un rangement "un dossier par classe"
    """root/<classe>/image : le rangement le plus courant."""
    rows = [  # liste de lignes du futur tableau, construite en une seule expression (comprehension de liste)
        {"path": str(p), "label": class_dir.name}  # une ligne : chemin de l'image et classe = nom du dossier
        for class_dir in sorted(d for d in root.iterdir() if d.is_dir())  # pour chaque sous-dossier de root (une classe), dans l'ordre
        for p in list_images(class_dir, extensions)  # pour chaque image de ce dossier de classe
    ]
    return pd.DataFrame(rows)  # transforme la liste de dicts en tableau (une ligne par image)


def from_split_folders(root: Path, extensions: list[str]) -> pd.DataFrame:  # catalogue d'un rangement train/val/test puis classes
    """root/{train,val,test}/<classe>/image : le decoupage fourni est conserve."""
    frames = []  # un tableau par sous-ensemble, assembles a la fin
    for subset in ("train", "val", "test"):  # pour chaque sous-ensemble possible
        if (root / subset).is_dir():  # si le dossier existe
            frames.append(from_class_folders(root / subset, extensions).assign(split=subset))  # lit ses classes et note le sous-ensemble
    catalog = pd.concat(frames, ignore_index=True)  # colle les tableaux bout a bout avec une numerotation continue
    per_split = catalog.groupby("split")["label"].apply(lambda s: tuple(sorted(set(s))))  # pour chaque sous-ensemble : ses classes
    if per_split.nunique() > 1:  # si les listes de classes ne sont pas toutes identiques (decalerait les indices sans erreur)
        raise ValueError(f"les classes different entre sous-ensembles : {per_split.to_dict()}")  # on arrete avec le detail
    return catalog  # renvoie le catalogue avec la colonne split


def from_csv(root: Path, csv_path: Path, path_column: str, label_column: str) -> pd.DataFrame:  # catalogue lu depuis un CSV d'etiquettes
    """Un dossier d'images et un CSV (chemin relatif a root, etiquette)."""
    table = pd.read_csv(csv_path)  # lit le CSV en tableau
    # chemin complet = root / chemin relatif ; etiquette convertie en texte (une classe "1" reste "1")
    return pd.DataFrame({"path": [str(root / f) for f in table[path_column]], "label": table[label_column].astype(str)})


def from_flat_folder(root: Path, extensions: list[str]) -> pd.DataFrame:  # catalogue d'images sans classe
    """Images sans etiquette (exploration, pre-annotation) : label vide."""
    return pd.DataFrame({"path": [str(p) for p in list_images(root, extensions)], "label": ""})  # toutes les images, label vide


def from_mvtec(root: Path, extensions: list[str]) -> pd.DataFrame:  # catalogue du format MVTec AD (detection d'anomalies)
    """Arborescence MVTec AD : train/good, test/<type>, ground_truth/<type>/<nom>_mask.png."""
    rows = [  # d'abord les images d'entrainement : toutes normales ("good"), sans masque
        {"path": str(p), "label": "good", "defect_type": "good", "split": "train", "mask_path": ""}  # une ligne par image normale
        for p in list_images(root / "train" / "good", extensions)  # pour chaque image de train/good
    ]
    for type_dir in sorted(d for d in (root / "test").iterdir() if d.is_dir()):  # pour chaque dossier de test (un type de defaut ou good)
        for p in list_images(type_dir, extensions):  # pour chaque image de ce dossier
            mask = root / "ground_truth" / type_dir.name / f"{p.stem}_mask.png"  # chemin attendu du masque (zone du defaut)
            rows.append(  # ajoute une ligne au catalogue
                {  # contenu de la ligne
                    "path": str(p),  # chemin de l'image
                    "label": "good" if type_dir.name == "good" else "defect",  # classe binaire : normale ou defectueuse
                    "defect_type": type_dir.name,  # type precis de defaut (nom du dossier)
                    "split": "test",  # ces images servent seulement au test
                    "mask_path": str(mask) if mask.exists() else "",  # chemin du masque s'il existe (pas de masque pour good)
                }
            )
    return pd.DataFrame(rows)  # tableau final : une ligne par image


# catalogue d'images accompagnees chacune d'un masque (segmentation)
def from_image_mask(root: Path, extensions: list[str], masks_dir: Path | None, mask_suffix: str) -> pd.DataFrame:
    """Une image et son masque. Masque dans masks_dir (meme arborescence) ou a cote de l'image en .png.

    label : le premier sous-dossier sous root s'il y en a un (ex. MT_Crack), sinon defect / free
    d'apres le contenu du masque. has_defect dit toujours si le masque contient au moins un pixel.
    """
    rows = []  # lignes du futur tableau
    for p in list_images(root, extensions):  # pour chaque image sous root
        relative = p.relative_to(root)  # chemin de l'image relatif a root (ex. MT_Crack/img1.jpg)
        base = (masks_dir / relative).parent if masks_dir is not None else p.parent  # dossier du masque : masks_dir ou a cote de l'image
        mask = base / f"{p.stem}{mask_suffix}.png"  # nom du masque : nom de l'image + suffixe, en .png
        if not mask.exists():  # si le masque attendu n'existe pas
            raise FileNotFoundError(f"masque absent pour {p} : {mask}")  # erreur claire avec les deux chemins
        has_defect = bool(read_mask(mask).any())  # True si au moins un pixel du masque vaut 1 (defaut present)
        # classe = 1er sous-dossier s'il existe, sinon "defect" ou "free" d'apres le contenu du masque
        label = relative.parts[0] if len(relative.parts) > 1 else ("defect" if has_defect else "free")
        rows.append({"path": str(p), "label": label, "mask_path": str(mask), "has_defect": has_defect})  # ajoute la ligne
    return pd.DataFrame(rows)  # tableau final


def read_data_annotations(config: DataConfig) -> list:  # lit les annotations d'un jeu et renvoie une liste (une entree par image)
    """Les annotations d'un jeu (format pivot), avec les options du lecteur et le reetiquetage par dossier si demande."""
    # lecteur multi-formats (VOC, YOLO, COCO...) ; importe ici pour ne le charger que si besoin
    from vtb.annotate.formats import read_annotations

    root = Path(config.root)  # dossier racine des images
    options = dict(config.annotation_options)  # copie des options du lecteur (on ne modifie pas la config)
    if config.annotation_format == "masks" and "extensions" not in options:  # format masques sans extensions precisees
        options["extensions"] = config.extensions  # on lui passe les extensions d'images de la config
    # lit les annotations ; dossier d'annotations = annotations_dir, ou root s'il n'est pas donne ; **options = options nommees
    annotations = read_annotations(config.annotation_format, root, Path(config.annotations_dir or config.root), **options)
    if config.object_label_from == "folder":  # si la classe des objets doit venir du dossier de l'image
        for ann in annotations:  # pour chaque image annotee
            label = Path(ann.image_path).relative_to(root).parts[0]  # classe = premier sous-dossier sous root
            for obj in (*ann.boxes, *ann.rotated, *ann.polygons):  # pour chaque objet : boites, boites tournees, polygones
                obj.label = label  # remplace sa classe par celle du dossier
    return annotations  # renvoie la liste des annotations


def from_annotations(config: DataConfig) -> pd.DataFrame:  # catalogue d'un jeu de detection : une ligne par image annotee
    """Jeu de detection / segmentation d'instances : une ligne par image, annotations relues par vtb.annotate."""
    rows = []  # lignes du futur tableau
    for ann in read_data_annotations(config):  # pour chaque image annotee
        rows.append(  # ajoute une ligne
            {  # contenu de la ligne
                "path": ann.image_path,  # chemin de l'image
                "label": ann.main_label(),  # classe principale de l'image (sert au decoupage stratifie)
                "n_objects": len(ann.object_labels()),  # nombre d'objets annotes dans l'image
                "width": ann.width,  # largeur de l'image en pixels
                "height": ann.height,  # hauteur de l'image en pixels
            }
        )
    return pd.DataFrame(rows)  # tableau final


# ajoute la colonne group (images d'une meme source) et renvoie le catalogue complete
def add_groups(catalog: pd.DataFrame, group_regex: str | None, groups_csv: str | None = None) -> pd.DataFrame:
    """Colonne group : premier groupe capture par la regex dans le nom de fichier, sinon le nom lui-meme.

    Exemple : `^(C\\d+)_` donne le numero de bobine de `C1234_cam2_0007.png`. Sans regex, chaque
    image est son propre groupe (aucune contrainte) : le groupe est alors le chemin complet, car
    deux images de dossiers differents peuvent porter le meme nom (MVTec : train/good/000.png et
    test/good/000.png).
    """
    catalog = catalog.copy()  # copie : on ne modifie pas le tableau recu
    if groups_csv is not None:  # groupes calcules ailleurs (quasi-doublons, numero de piece d'une base)
        # lit le CSV (keep_default_na=False : une case vide reste "" et non NaN), puis table chemin -> groupe
        table = pd.read_csv(groups_csv, keep_default_na=False).set_index("path")["group"].astype(str)
        catalog["group"] = [table.get(str(p), str(p)) for p in catalog["path"]]  # groupe du CSV, ou le chemin lui-meme s'il est absent
        return catalog  # renvoie le catalogue avec les groupes
    if group_regex is None:  # pas de regle de groupe : chaque image est son propre groupe
        catalog["group"] = catalog["path"].astype(str)  # groupe = chemin complet (unique)
        return catalog  # renvoie le catalogue
    pattern = re.compile(group_regex)  # prepare l'expression reguliere une seule fois
    groups = []  # groupe de chaque image, dans l'ordre du catalogue
    for p in catalog["path"]:  # pour chaque chemin d'image
        match = pattern.search(Path(p).name)  # cherche le motif dans le nom du fichier
        if match is None:  # si le motif n'est pas trouve
            raise ValueError(f"group_regex {group_regex!r} ne trouve rien dans {Path(p).name}")  # erreur claire avec le nom fautif
        groups.append(match.group(1) if match.groups() else match.group(0))  # 1er groupe entre parentheses, sinon tout le texte trouve
    catalog["group"] = groups  # nouvelle colonne group
    return catalog  # renvoie le catalogue complete


def add_timestamps(catalog: pd.DataFrame, timestamp_regex: str | None) -> pd.DataFrame:  # ajoute la colonne timestamp (date de prise)
    """Colonne timestamp : date lue dans le nom (regex avec groupes nommes Y, m, d, et H, M, S facultatifs), sinon date du fichier."""
    catalog = catalog.copy()  # copie : on ne modifie pas le tableau recu
    if timestamp_regex is None:  # pas de regle pour lire la date dans le nom
        catalog["timestamp"] = [datetime.fromtimestamp(Path(p).stat().st_mtime) for p in catalog["path"]]  # date de modification du fichier
        return catalog  # renvoie le catalogue
    pattern = re.compile(timestamp_regex)  # prepare l'expression reguliere une seule fois
    stamps = []  # date de chaque image
    for p in catalog["path"]:  # pour chaque chemin d'image
        match = pattern.search(Path(p).name)  # cherche le motif dans le nom du fichier
        if match is None:  # si le motif n'est pas trouve
            raise ValueError(f"timestamp_regex {timestamp_regex!r} ne trouve rien dans {Path(p).name}")  # erreur claire
        # groupes nommes trouves (Y, m, d...) convertis en entiers ; les groupes absents sont ignores
        parts = {k: int(v) for k, v in match.groupdict().items() if v is not None}
        # construit la date ; heure, minute et seconde valent 0 si absentes du nom
        stamps.append(datetime(parts["Y"], parts["m"], parts["d"], parts.get("H", 0), parts.get("M", 0), parts.get("S", 0)))
    catalog["timestamp"] = stamps  # nouvelle colonne timestamp
    return catalog  # renvoie le catalogue complete


def limit_catalog(catalog: pd.DataFrame, max_images: int | None) -> pd.DataFrame:  # garde au plus max_images lignes (essai rapide)
    """Sous-echantillon aleatoire fixe (graine 0) pour un essai rapide ; l'ordre par chemin est conserve."""
    if max_images is None or len(catalog) <= max_images:  # pas de limite, ou deja assez petit
        return catalog  # rien a faire
    logger.warning("max_images=%d : %d images sur %d gardees (essai rapide)", max_images, max_images, len(catalog))  # previent
    # tirage aleatoire reproductible (random_state=0), retrie par chemin, numerotation des lignes refaite
    return catalog.sample(n=max_images, random_state=0).sort_values("path").reset_index(drop=True)


def build_catalog(config: DataConfig, with_timestamps: bool = False) -> pd.DataFrame:  # construit le catalogue complet d'apres la config
    """Choisit le lecteur d'apres data.layout, puis ajoute group (et timestamp si demande)."""
    root = Path(config.root)  # dossier racine des donnees
    if not root.exists():  # si le dossier n'existe pas
        raise FileNotFoundError(f"data.root introuvable : {root}")  # erreur claire
    readers = {  # un lecteur par rangement ; lambda = petite fonction sans nom, appelee seulement pour le rangement choisi
        "class_folders": lambda: from_class_folders(root, config.extensions),  # un dossier par classe
        "split_folders": lambda: from_split_folders(root, config.extensions),  # train/val/test puis classes
        "csv": lambda: from_csv(root, Path(config.labels_csv or ""), config.path_column, config.label_column),  # CSV d'etiquettes
        "flat": lambda: from_flat_folder(root, config.extensions),  # images sans classe
        "mvtec": lambda: from_mvtec(root, config.extensions),  # format MVTec AD
        "image_mask": lambda: from_image_mask(  # image + masque
            root,  # dossier racine des images
            config.extensions,  # extensions d'images acceptees
            Path(config.masks_dir) if config.masks_dir else None,  # dossier des masques, ou None (masque a cote de l'image)
            config.mask_suffix,  # suffixe du nom de masque (ex. "_mask")
        ),
        "annotations": lambda: from_annotations(config),  # fichiers d'annotations (detection)
    }
    catalog = readers[config.layout]()  # choisit le lecteur du rangement demande et l'appelle
    if catalog.empty:  # si aucune image n'a ete trouvee
        # erreur claire qui rappelle ou on a cherche et avec quels reglages
        raise FileNotFoundError(f"aucune image trouvee sous {root} (layout {config.layout}, extensions {config.extensions})")
    catalog = limit_catalog(catalog, config.max_images)  # sous-echantillon eventuel (essai rapide)
    catalog = add_groups(catalog, config.group_regex, config.groups_csv)  # ajoute la colonne group
    if with_timestamps or config.timestamp_regex is not None:  # si les dates sont demandees ou une regle de date est donnee
        catalog = add_timestamps(catalog, config.timestamp_regex)  # ajoute la colonne timestamp
    # resume dans le journal : nombre d'images, rangement, nombre de groupes distincts
    logger.info("catalogue : %d images | layout %s | %d groupes", len(catalog), config.layout, catalog["group"].nunique())
    return catalog  # renvoie le catalogue final
