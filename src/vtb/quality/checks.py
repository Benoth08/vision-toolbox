"""Controle d'un jeu d'images AVANT d'entrainer : la moitie des "bons resultats" faux vient des donnees.

Ce que l'on cherche, et pourquoi :
  - fichiers illisibles ou tronques      : plantent l'entrainement au milieu de la nuit ;
  - tailles, types, canaux heterogenes   : un uint16 parmi des uint8, une image RGB parmi des grises ;
  - images vides, saturees, floues       : une camera mal reglee, un capot ferme ;
  - doublons exacts et quasi-doublons    : s'ils tombent dans deux sous-ensembles, le test est truque
                                          (le modele a deja vu l'image) ; avec deux etiquettes
                                          differentes, c'est une erreur d'annotation ;
  - raccourcis                           : si la luminosite ou la taille seule predit la classe, le
                                          modele apprendra la luminosite, pas le defaut (classes
                                          prises sur des lignes ou a des heures differentes).
"""

from __future__ import annotations  # permet d'ecrire les types (ex. "str | Path") sans qu'ils soient evalues au chargement

import hashlib  # empreintes cryptographiques (md5) : deux fichiers identiques octet par octet ont la meme empreinte
import logging  # journal des messages (infos, avertissements) plutot que des print
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import cv2  # OpenCV : lecture, ecriture et traitement d'images
import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)
import pandas as pd  # pandas : tableaux avec colonnes nommees (DataFrame), une ligne par image

# max_value : valeur maximale d'un pixel selon le type ; read_image : lit une image ; to_uint8 : convertit en 8 bits (0-255)
from vtb.io.images import max_value, read_image, to_uint8

logger = logging.getLogger(__name__)  # journal propre a ce module (son nom apparait dans les messages)

# statistiques simples calculees pour chaque image : luminosite moyenne, contraste, taille, nettete, part saturee, part noire
STAT_COLUMNS = ["mean", "std", "height", "width", "sharpness", "saturated_fraction", "dark_fraction"]


def dhash(gray: np.ndarray, size: int = 8) -> int:  # renvoie l'empreinte visuelle d'une image grise, un entier de 64 bits
    """Empreinte perceptive 64 bits : signe du gradient horizontal sur une vignette 9 x 8.

    Deux images presque identiques (recompression JPEG, leger changement de luminosite) ont des
    empreintes a quelques bits pres ; deux images differentes, a ~32 bits.
    """
    # reduit l'image a 9 colonnes x 8 lignes ; int16 : entiers signes pour comparer sans debordement
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()  # 1 si un pixel est plus clair que son voisin de gauche : 8 x 8 = 64 bits
    return int(np.packbits(bits).view(">u8")[0])  # range les 64 bits dans 8 octets, puis les lit comme un seul entier 64 bits


def image_stats(path: str | Path) -> dict:  # renvoie un dictionnaire de statistiques pour une image (une ligne du tableau)
    """Une ligne de statistiques par fichier ; readable=False et la cause si la lecture echoue."""
    row: dict = {"path": str(path)}  # la ligne commence par le chemin du fichier
    try:  # on essaie de lire le fichier ; s'il est abime, on note l'erreur au lieu de tout arreter
        data = Path(path).read_bytes()  # octets bruts du fichier (pour l'empreinte md5)
        image = read_image(path, color="unchanged")  # image telle que stockee (8 ou 16 bits, gris ou couleur)
    except Exception as error:  # noqa: BLE001 - on veut la cause, quelle qu'elle soit
        # renvoie la ligne marquee illisible avec le type et le message de l'erreur (coupes a 200 caracteres)
        return {**row, "readable": False, "error": f"{type(error).__name__}: {error}"[:200]}
    peak = max_value(image)  # valeur maximale possible d'un pixel : 255 en 8 bits, 65535 en 16 bits
    # version grise de l'image : telle quelle si deja grise, sinon conversion des 3 premiers canaux couleur
    gray = image if image.ndim == 2 else cv2.cvtColor(image[..., :3], cv2.COLOR_RGB2GRAY)
    gray8 = gray if gray.dtype == np.uint8 else to_uint8(gray)  # copie en 8 bits : certaines fonctions OpenCV l'exigent
    values = gray.astype(np.float32) / peak  # pixels ramenes entre 0 et 1, pour comparer des images 8 et 16 bits
    row.update(  # ajoute les statistiques a la ligne
        {  # dictionnaire des statistiques de l'image
            "readable": True,  # le fichier a pu etre lu
            "error": "",  # pas d'erreur
            "height": image.shape[0],  # hauteur en pixels
            "width": image.shape[1],  # largeur en pixels
            "channels": 1 if image.ndim == 2 else image.shape[2],  # nombre de canaux : 1 = gris, 3 = couleur
            "dtype": str(image.dtype),  # type des pixels (uint8, uint16...) : un melange signale un probleme d'export
            "mean": float(values.mean()),  # luminosite moyenne (0 = noir, 1 = blanc)
            "std": float(values.std()),  # ecart-type = contraste global (0 = image uniforme)
            "saturated_fraction": float((gray >= peak * 0.995).mean()),  # part des pixels presque blancs (surexposition)
            "dark_fraction": float((gray <= peak * 0.005).mean()),  # part des pixels presque noirs (sous-exposition)
            "sharpness": float(cv2.Laplacian(gray8, cv2.CV_64F).var()),  # variance du laplacien : faible = flou
            "md5": hashlib.md5(data).hexdigest(),  # empreinte md5 du fichier : identique = copie exacte
            "dhash": dhash(gray8),  # empreinte visuelle : proche = images qui se ressemblent
        }
    )
    return row  # renvoie la ligne complete


def scan(catalog: pd.DataFrame) -> pd.DataFrame:  # renvoie le catalogue complete par les statistiques de chaque image
    """Statistiques de toutes les images du catalogue (colonnes du catalogue conservees : label, split, group...)."""
    rows = [image_stats(p) for p in catalog["path"]]  # une ligne de statistiques par image du catalogue
    stats = pd.DataFrame(rows)  # transforme ces lignes en tableau
    extra = [c for c in catalog.columns if c != "path"]  # autres colonnes du catalogue a garder (etiquette, sous-ensemble...)
    # colle catalogue et statistiques en associant les lignes par chemin ; how="left" garde toutes les images du catalogue
    return catalog[["path", *extra]].merge(stats, on="path", how="left")


def detail_signature(path: str | Path, size: int = 64) -> np.ndarray:  # renvoie un vecteur de 64 x 64 = 4096 nombres
    """Vignette 64 x 64 sans ses basses frequences, centree et normee : le "detail" de l'image.

    Deux textures d'acier differentes ont des empreintes dHash proches (meme eclairage, meme
    degrade) mais des details independants : leur correlation est proche de 0. Une vraie copie
    (reenregistrement JPEG, luminosite un peu changee) garde une correlation proche de 1.
    """
    gray = read_image(path, color="gray")  # lit l'image en niveaux de gris
    # vignette 64 x 64 (convertie en 8 bits si besoin), en nombres a virgule pour les calculs suivants
    small = cv2.resize(to_uint8(gray) if gray.dtype != np.uint8 else gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    # retire le flou gaussien (basses frequences = eclairage, degrades) : il ne reste que les details fins
    detail = small - cv2.GaussianBlur(small, (0, 0), 3)
    detail -= detail.mean()  # centre : moyenne a 0
    # divise par la longueur du vecteur (au moins 1e-6 pour ne pas diviser par 0) : le produit scalaire devient une correlation
    return (detail / max(float(np.linalg.norm(detail)), 1e-6)).ravel()


# renvoie le tableau des paires de doublons : chemins, distance dHash, correlation, copie exacte ou non
def find_duplicates(stats: pd.DataFrame, max_hamming: int = 6, min_correlation: float = 0.9, max_candidates: int = 200_000) -> pd.DataFrame:
    """Paires de doublons exacts (meme md5) et de quasi-doublons, en deux temps.

    1. candidats : empreintes dHash a <= max_hamming bits (XOR + comptage de bits, toutes les paires) ;
    2. confirmation : correlation des details (detail_signature) >= min_correlation, sauf copie exacte.
    L'etape 2 evite les milliers de faux doublons des surfaces texturees, ou dHash ne voit que
    l'eclairage. Colonnes cross_split et label_conflict : les deux cas graves.
    """
    ok = stats[stats["readable"].astype(bool)].reset_index(drop=True)  # images lisibles seulement, renumerotees 0, 1, 2...
    columns = ["path_a", "path_b", "hamming", "correlation", "exact"]  # colonnes du tableau renvoye s'il est vide
    if len(ok) < 2:  # si moins de 2 images, il ne peut pas y avoir de paire
        return pd.DataFrame(columns=columns)  # renvoie un tableau vide mais avec les bonnes colonnes
    hashes = ok["dhash"].astype("uint64").to_numpy()  # empreintes dHash en entiers 64 bits non signes
    md5 = ok["md5"].to_numpy()  # empreintes md5 en tableau numpy
    candidates = []  # paires candidates : (indice a, indice b, distance de Hamming)
    for i in range(len(ok) - 1):  # pour chaque image (sauf la derniere), on la compare a toutes les suivantes
        # distance de Hamming = nombre de bits differents : XOR (^) met 1 la ou les bits different, puis on compte les 1
        distances = np.bitwise_count(hashes[i] ^ hashes[i + 1 :])
        # garde les images suivantes assez proches ; o est la position dans la tranche, donc l'indice reel est i + 1 + o
        candidates.extend((i, i + 1 + int(o), int(distances[o])) for o in np.flatnonzero(distances <= max_hamming))
        if len(candidates) >= max_candidates:  # si trop de candidats : seuil trop large, on arrete pour ne pas saturer la memoire
            logger.warning("plus de %d candidats : recherche arretee (max_hamming trop large ?)", max_candidates)  # previent
            break  # sort de la boucle
    signatures: dict[int, np.ndarray] = {}  # memoire des signatures deja calculees (une image peut etre dans plusieurs paires)

    def signature(k: int) -> np.ndarray:  # renvoie la signature de detail de l'image numero k, calculee une seule fois
        if k not in signatures:  # si elle n'a pas encore ete calculee
            signatures[k] = detail_signature(ok["path"].iloc[k])  # on la calcule et on la garde
        return signatures[k]  # renvoie la signature memorisee

    rows = []  # paires confirmees
    for i, j, distance in candidates:  # pour chaque paire candidate
        exact = md5[i] == md5[j]  # meme md5 = fichiers identiques octet par octet
        correlation = 1.0 if exact else float(signature(i) @ signature(j))  # correlation des details (1 si copie exacte)
        if not exact and correlation < min_correlation:  # si les details ne se ressemblent pas assez : faux doublon
            continue  # on ignore cette paire
        a, b = ok.iloc[i], ok.iloc[j]  # les deux lignes du tableau (une par image)
        # ligne de resultat : chemins, distance dHash, correlation arrondie, copie exacte ou non
        row = {"path_a": a["path"], "path_b": b["path"], "hamming": distance, "correlation": round(correlation, 4), "exact": bool(exact)}
        if "split" in ok:  # si on connait le sous-ensemble (train/val/test) de chaque image
            # cross_split vrai : les deux copies sont dans deux sous-ensembles differents = fuite
            row.update({"split_a": a["split"], "split_b": b["split"], "cross_split": a["split"] != b["split"]})
        if "label" in ok:  # si on connait l'etiquette de chaque image
            # label_conflict vrai : la meme image porte deux etiquettes = erreur d'annotation
            row.update({"label_a": a["label"], "label_b": b["label"], "label_conflict": a["label"] != b["label"]})
        rows.append(row)  # garde la paire confirmee
    logger.info("doublons : %d candidats dHash, %d confirmes", len(candidates), len(rows))  # bilan dans le journal
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)  # tableau des paires (vide avec colonnes si aucune)


# renvoie un tableau chemin -> nom de groupe : toutes les copies d'une meme image ont le meme groupe
def duplicate_groups(paths: list[str], duplicates: pd.DataFrame) -> pd.DataFrame:
    """Groupes de quasi-doublons (composantes connexes du graphe des paires) -> table path, group.

    A donner a data.groups_csv avec split.strategy: grouped : toutes les copies d'une meme piece
    (memes images sous plusieurs expositions, rafales) tombent du meme cote du decoupage.
    """
    # union-find : chaque image pointe vers un "parent" ; au depart chacune est son propre parent (un groupe par image)
    parent = {p: p for p in paths}

    def root(p: str) -> str:  # renvoie la racine du groupe de p (l'image qui represente tout le groupe)
        while parent[p] != p:  # tant que p n'est pas sa propre racine, on remonte
            parent[p] = parent[parent[p]]  # raccourci : p pointe directement vers son grand-parent (chemins plus courts)
            p = parent[p]  # on monte d'un cran
        return p  # p est la racine

    # pour chaque paire de doublons (get renvoie une liste vide si la colonne n'existe pas)
    for a, b in zip(duplicates.get("path_a", []), duplicates.get("path_b", [])):
        if a in parent and b in parent:  # si les deux images font partie de la liste
            parent[root(a)] = root(b)  # fusionne leurs deux groupes : la racine de a pointe vers celle de b
    roots = {p: root(p) for p in paths}  # racine finale de chaque image
    sizes = pd.Series(list(roots.values())).value_counts()  # nombre d'images par racine = taille de chaque groupe
    names = {r: f"dup_{k:04d}" for k, r in enumerate(sizes[sizes > 1].index)}  # nom dup_0000, dup_0001... pour les groupes >= 2
    # tableau final : une image seule garde son chemin comme nom de groupe, les copies prennent le nom dup_ de leur groupe
    return pd.DataFrame({"path": paths, "group": [names.get(roots[p], p) for p in paths]})


# renvoie un tableau : pour chaque statistique, a quel point elle suffit a deviner la classe
def shortcut_scores(stats: pd.DataFrame, label_column: str = "label", seed: int = 0) -> pd.DataFrame:
    """Pour chaque statistique simple, exactitude equilibree d'un petit arbre qui predit la classe avec elle SEULE.

    Proche du hasard (1 / nombre de classes) : normal. Tres au-dessus : la classe est lisible dans la
    luminosite, la taille ou la nettete -> verifier l'acquisition avant d'accuser le modele.
    """
    from sklearn.model_selection import cross_val_score  # validation croisee : score moyen sur des images non vues
    from sklearn.tree import DecisionTreeClassifier  # arbre de decision : suite de questions "valeur > seuil ?"

    ok = stats[stats["readable"].astype(bool)]  # images lisibles seulement
    labels = ok[label_column].astype(str)  # etiquettes en texte
    if labels.nunique() < 2 or labels.value_counts().min() < 3:  # si moins de 2 classes, ou une classe avec moins de 3 images
        return pd.DataFrame()  # pas assez de donnees pour un test fiable : tableau vide
    folds = min(5, int(labels.value_counts().min()))  # nombre de plis : 5, ou moins si la plus petite classe est petite
    rows = []  # un score par statistique
    for column in STAT_COLUMNS:  # pour chaque statistique simple
        x = ok[[column]].to_numpy(dtype=float)  # une seule colonne en entree (tableau N x 1), en nombres decimaux
        if np.nanstd(x) == 0:  # si la statistique est la meme pour toutes les images, elle ne peut rien predire
            continue  # on passe a la suivante
        tree = DecisionTreeClassifier(max_depth=3, random_state=seed)  # petit arbre (3 questions au plus) : pas de par coeur
        # exactitude equilibree : moyenne des taux de reussite par classe (juste meme si une classe est rare)
        score = cross_val_score(tree, x, labels, cv=folds, scoring="balanced_accuracy").mean()
        rows.append({"statistic": column, "balanced_accuracy": round(float(score), 3)})  # garde le score de cette statistique
    # tableau trie du score le plus haut (raccourci le plus dangereux) au plus bas, lignes renumerotees
    table = pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False).reset_index(drop=True)
    table["chance"] = round(1 / labels.nunique(), 3)  # score du hasard pour comparaison (1 / nombre de classes)
    return table  # renvoie le tableau des scores


def flag_outliers(stats: pd.DataFrame, z: float = 4.0) -> pd.DataFrame:  # renvoie le tableau des images atypiques
    """Images atypiques : statistique a plus de z ecarts robustes (MAD) de la mediane de sa classe."""
    ok = stats[stats["readable"].astype(bool)].copy()  # images lisibles ; copy() evite de modifier le tableau d'origine
    # classe de chaque image ; sans etiquettes, toutes les images forment un seul groupe "all"
    group = ok["label"].astype(str) if "label" in ok else pd.Series("all", index=ok.index)
    flags = []  # morceaux de tableau d'images signalees, concatenes a la fin
    for column in ("mean", "std", "sharpness"):  # pour la luminosite, le contraste et la nettete
        median = ok.groupby(group)[column].transform("median")  # mediane de la classe, recopiee sur chaque ligne
        # MAD (ecart absolu median) x 1.4826 : estimation de l'ecart-type peu sensible aux valeurs extremes
        mad = ok.groupby(group)[column].transform(lambda s: (s - s.median()).abs().median()) * 1.4826
        score = (ok[column] - median).abs() / mad.replace(0, np.nan)  # z robuste : distance a la mediane en nombre de MAD
        # images au-dela du seuil z : chemin, statistique en cause et score arrondi a 1 decimale
        flags.append(ok.loc[score > z, ["path"]].assign(statistic=column, robust_z=score[score > z].round(1)))
    # images vides ou mal exposees : contraste quasi nul, plus de 50 % de pixels blancs ou plus de 90 % de pixels noirs
    empty = ok[(ok["std"] < 1e-3) | (ok["saturated_fraction"] > 0.5) | (ok["dark_fraction"] > 0.9)]
    flags.append(empty[["path"]].assign(statistic="vide_ou_sature", robust_z=np.nan))  # ajoutees avec un score vide (NaN)
    return pd.concat(flags, ignore_index=True)  # rassemble tous les morceaux en un seul tableau


# renvoie un dictionnaire : chiffres cles du jeu et liste d'alertes en francais
def summarize(stats: pd.DataFrame, duplicates: pd.DataFrame, shortcuts: pd.DataFrame, outliers: pd.DataFrame) -> dict:
    """Resume chiffre + liste d'alertes en clair (ce que le rapport affiche en premier)."""
    ok = stats[stats["readable"].astype(bool)]  # images lisibles seulement
    alerts = []  # messages d'alerte en clair
    unreadable = int((~stats["readable"].astype(bool)).sum())  # nombre de fichiers illisibles (~ inverse vrai et faux)
    if unreadable:  # s'il y en a au moins un
        alerts.append(f"{unreadable} fichier(s) illisible(s) : a retirer ou a reexporter")  # alerte
    for column, what in (("dtype", "types"), ("channels", "nombres de canaux")):  # pour le type des pixels puis le nombre de canaux
        if ok[column].nunique() > 1:  # si plusieurs valeurs differentes coexistent dans le jeu
            alerts.append(f"{what} melanges : {ok[column].value_counts().to_dict()}")  # alerte avec le compte de chaque valeur
    if ok[["height", "width"]].drop_duplicates().shape[0] > 1:  # si plusieurs tailles d'image differentes
        # alerte avec le nombre de tailles distinctes
        alerts.append(f"{ok[['height', 'width']].drop_duplicates().shape[0]} tailles d'image differentes : prevoir un resize")
    if not duplicates.empty:  # s'il y a des doublons
        if "cross_split" in duplicates and duplicates["cross_split"].any():  # si au moins une paire est a cheval sur 2 sous-ensembles
            alerts.append(  # alerte de fuite : le test contient des images deja vues a l'entrainement
                f"{int(duplicates['cross_split'].sum())} doublon(s) a cheval sur deux sous-ensembles : FUITE, decouper par groupe"
            )
        if "label_conflict" in duplicates and duplicates["label_conflict"].any():  # si au moins une paire a deux etiquettes
            alerts.append(  # alerte d'annotation : une meme image ne peut pas avoir deux classes
                f"{int(duplicates['label_conflict'].sum())} doublon(s) avec deux etiquettes differentes : erreurs d'annotation probables"
            )
    chance = float(shortcuts["chance"].iloc[0]) if not shortcuts.empty else 0.0  # score du hasard (0 si pas de tableau)
    # alerte si une statistique seule fait plus de 60 % du chemin entre le hasard et la perfection
    if not shortcuts.empty and shortcuts["balanced_accuracy"].iloc[0] > chance + 0.6 * (1 - chance):
        top = shortcuts.iloc[0]  # la statistique la plus predictive (le tableau est trie)
        alerts.append(  # alerte de raccourci
            f"raccourci possible : '{top['statistic']}' seule predit la classe (exactitude equilibree {top['balanced_accuracy']})"
        )
    if len(outliers):  # s'il y a des images atypiques
        alerts.append(f"{outliers['path'].nunique()} image(s) atypique(s) a regarder (voir outliers.csv)")  # nombre d'images distinctes
    return {  # le resume renvoye
        "n_images": int(len(stats)),  # nombre total de fichiers
        "n_readable": int(len(ok)),  # nombre de fichiers lisibles
        # effectif de chaque classe (dictionnaire vide si pas d'etiquettes)
        "classes": ok["label"].astype(str).value_counts().to_dict() if "label" in ok else {},
        "sizes": {  # les 5 tailles d'image les plus frequentes, sous la forme "hauteurxlargeur": nombre
            f"{int(h)}x{int(w)}": int(n)  # cle "480x640" -> nombre d'images de cette taille
            # compte les images par (hauteur, largeur), trie du plus frequent au moins frequent, garde les 5 premiers
            for (h, w), n in ok.groupby(["height", "width"]).size().sort_values(ascending=False).head(5).items()
        },
        "n_duplicate_pairs": int(len(duplicates)),  # nombre de paires de doublons
        "n_exact_duplicate_pairs": int(duplicates["exact"].sum()) if len(duplicates) else 0,  # dont copies exactes
        "alerts": alerts or ["aucune alerte"],  # liste des alertes, ou un message rassurant si elle est vide
    }
