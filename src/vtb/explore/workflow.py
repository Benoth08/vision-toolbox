"""Les trois analyses sans etiquette de la recette explore : controle du jeu, plongements, derive.

check_dataset(config)  fichiers, tailles, doublons, fuites, raccourcis -> quality_report.md / .json + CSV
embeddings(config)     plongements -> projection 2D, groupes k-means, galerie par groupe, voisins
drift(config)          reference vs courant -> AUC du classifieur de domaine + statistiques qui ont bouge
"""

from __future__ import annotations  # permet d'ecrire les types (ex. "str | None") sans qu'ils soient evalues au chargement

import json  # lecture / ecriture de fichiers JSON (dictionnaires enregistres en texte)
import logging  # journal des messages (infos, avertissements) plutot que des print
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)
import pandas as pd  # pandas : tableaux avec colonnes nommees (DataFrame), lus et ecrits en CSV
import torch  # PyTorch : ici seulement pour le type torch.device (processeur cpu ou carte graphique cuda)

from vtb.evaluate.plots import plot_gallery  # dessine une planche d'images avec un titre sous chacune
from vtb.explore.config import DriftConfig, ExploreConfig  # reglages des analyses, lus depuis le YAML
from vtb.explore.drift import domain_classifier, statistics_drift, verdict  # les deux mesures de derive et leur verdict

# plongements (un vecteur par image) : calcul, projection 2D, groupes k-means, purete, voisins, dessin
from vtb.explore.embeddings import build_embedder, cluster, cluster_purity, embed_images, nearest, plot_projection, project_2d
from vtb.io.catalogs import build_catalog  # liste les images du jeu (chemin, etiquette...) dans un tableau
from vtb.io.images import read_image, to_uint8, write_image  # lecture, conversion en 8 bits et ecriture d'images
from vtb.preprocess import Pipeline  # suite d'operations de pretraitement appliquee a chaque image

# fonctions du controle qualite : statistiques par image, doublons, images atypiques, raccourcis, resume
from vtb.quality.checks import duplicate_groups, find_duplicates, flag_outliers, scan, shortcut_scores, summarize

logger = logging.getLogger(__name__)  # journal propre a ce module (son nom apparait dans les messages)


def _thumb(path: str, size: int = 128) -> np.ndarray:  # renvoie une vignette carree (128 x 128) d'une image, pour les galeries
    import cv2  # OpenCV : lecture, ecriture et traitement d'images (ici le redimensionnement)

    image = read_image(path, color="rgb")  # lit l'image en couleur (hauteur, largeur, 3 canaux)
    image = image if image.dtype == np.uint8 else to_uint8(image)  # ramene en 8 bits (0-255) si l'image est en 16 bits
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)  # INTER_AREA : la meilleure methode pour reduire


def _catalog(config: ExploreConfig) -> pd.DataFrame:  # renvoie le tableau des images, avec leur sous-ensemble si connu
    catalog = build_catalog(config.data)  # une ligne par image : chemin, etiquette, groupe...
    if config.split_file and Path(config.split_file).exists():  # si un fichier de decoupage train/val/test est donne et existe
        split = pd.read_csv(config.split_file, keep_default_na=False)[["path", "split"]]  # on n'en garde que chemin et sous-ensemble
        # retire l'eventuelle colonne split du catalogue, puis colle celle du fichier en associant les lignes par chemin
        catalog = catalog.drop(columns=[c for c in ("split",) if c in catalog]).merge(split, on="path", how="left")
        catalog["split"] = catalog["split"].fillna("hors_split")  # images absentes du fichier de decoupage : marquees a part
    return catalog  # renvoie le tableau complet


# --- controle du jeu ---------------------------------------------------------------------------


# controle tout le jeu d'images, ecrit les rapports dans output et renvoie le resume (chiffres + alertes)
def check_dataset(config: ExploreConfig, output: Path) -> dict:
    import matplotlib.pyplot as plt  # matplotlib : dessin de graphiques (histogrammes)

    catalog = _catalog(config)  # liste des images a controler
    stats = scan(catalog)  # statistiques de chaque image : lisible ?, taille, luminosite, nettete, empreintes
    duplicates = find_duplicates(stats, config.max_hamming, config.min_correlation)  # paires de doublons exacts ou quasi-doublons
    # raccourcis : seulement s'il y a une colonne d'etiquettes avec au moins 2 classes ; sinon tableau vide
    shortcuts = shortcut_scores(stats) if "label" in stats and stats["label"].astype(str).nunique() > 1 else pd.DataFrame()
    outliers = flag_outliers(stats)  # images atypiques (trop sombres, trop floues...) par rapport a leur classe
    summary = summarize(stats, duplicates, shortcuts, outliers)  # resume chiffre et liste d'alertes en clair
    stats.to_csv(output / "image_stats.csv", index=False)  # enregistre les statistiques (sans le numero de ligne)
    duplicates.to_csv(output / "duplicates.csv", index=False)  # enregistre les paires de doublons
    groups = duplicate_groups(stats["path"].astype(str).tolist(), duplicates)  # regroupe les copies d'une meme image
    groups.to_csv(output / "groups.csv", index=False)  # a donner a data.groups_csv (decoupage par groupe)
    # nombre de groupes de doublons : les groupes de plusieurs images ont un nom qui commence par "dup_"
    summary["n_duplicate_groups"] = int(groups.loc[groups["group"].str.startswith("dup_"), "group"].nunique())
    outliers.to_csv(output / "outliers.csv", index=False)  # enregistre les images atypiques
    shortcuts.to_csv(output / "shortcuts.csv", index=False)  # enregistre les scores de raccourci
    # resume en JSON lisible (indent=2) ; default=str convertit en texte ce que JSON ne sait pas ecrire
    (output / "quality_report.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    ok = stats[stats["readable"].astype(bool)]  # ne garde que les images lisibles pour les graphiques
    if "label" in ok and len(ok):  # si on connait les classes et qu'il reste au moins une image
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))  # une figure avec 2 graphiques cote a cote
        for label, group in ok.groupby(ok["label"].astype(str)):  # pour chaque classe et ses images
            axes[0].hist(group["mean"], bins=30, alpha=0.5, label=label)  # histogramme de la luminosite moyenne (30 barres)
            # histogramme de la nettete ; log10(x + 1) : ecrase les tres grandes valeurs pour tout voir sur un meme axe
            axes[1].hist(np.log10(group["sharpness"] + 1), bins=30, alpha=0.5, label=label)
        axes[0].set_title("luminosite moyenne par classe")  # titre du graphique de gauche
        axes[1].set_title("nettete (log10 variance du laplacien)")  # titre du graphique de droite
        axes[0].legend(fontsize=6)  # legende des classes (une seule suffit, les couleurs sont les memes)
        fig.tight_layout()  # resserre les marges
        fig.savefig(output / "stats_by_class.png", dpi=110)  # enregistre l'image du graphique
        plt.close(fig)  # ferme la figure pour liberer la memoire
    if len(duplicates):  # s'il y a au moins une paire de doublons, on en montre quelques-unes
        pairs = duplicates.sort_values("hamming").head(6)  # les 6 paires les plus proches (moins de bits differents)
        images, titles = [], []  # vignettes et legendes de la galerie
        for _, row in pairs.iterrows():  # pour chaque paire (iterrows donne le numero de ligne, ignore ici, et la ligne)
            images += [_thumb(row["path_a"]), _thumb(row["path_b"])]  # les deux images de la paire, cote a cote
            # legendes : debut du nom de chaque fichier, et la distance d (bits differents) sous la seconde
            titles += [Path(row["path_a"]).name[:22], f"{Path(row['path_b']).name[:16]} d={row['hamming']}"]
        plot_gallery(images, titles, output / "duplicates.png", columns=4)  # planche de 4 colonnes = 2 paires par ligne

    # rapport Markdown : titre, puis la liste des alertes (une puce "- " par alerte)
    lines = [f"# Controle du jeu : {config.data.root}", "", "## Alertes", ""] + [f"- {a}" for a in summary["alerts"]]
    lines += ["", "## Chiffres", "", f"- images : {summary['n_images']} (lisibles : {summary['n_readable']})"]  # nombre d'images
    lines += [f"- classes : {summary['classes']}", f"- tailles les plus frequentes : {summary['sizes']}"]  # classes et tailles
    # nombre de paires de doublons, dont copies exactes
    lines += [f"- paires de doublons : {summary['n_duplicate_pairs']} (exacts : {summary['n_exact_duplicate_pairs']})"]
    if not shortcuts.empty:  # si les raccourcis ont pu etre calcules, on ajoute leur tableau au rapport
        # to_markdown : convertit le tableau pandas en tableau Markdown
        lines += ["", "## Raccourcis possibles (une statistique seule predit-elle la classe ?)", "", shortcuts.to_markdown(index=False)]
    (output / "quality_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")  # ecrit le rapport, une ligne par element
    for alert in summary["alerts"]:  # pour chaque alerte
        logger.info("alerte : %s", alert)  # l'affiche aussi dans le journal
    return summary  # renvoie le resume a l'appelant


# --- plongements -------------------------------------------------------------------------------


# calcule les plongements, les projette en 2D, forme des groupes, dessine galeries et voisins ; renvoie un resume
def embeddings(config: ExploreConfig, output: Path, device: torch.device) -> dict:
    catalog = _catalog(config)  # liste des images
    pipeline = Pipeline(config.preprocess)  # pretraitement (taille fixe) avant le reseau
    model = build_embedder(config.model)  # reseau pre-entraine sans sa tete : sort un vecteur par image
    vectors = embed_images(catalog["path"].tolist(), pipeline, config.data.color, model, device=device)  # (N images, D nombres)
    np.save(output / "embeddings.npy", vectors)  # enregistre les plongements (format numpy) pour les reutiliser sans recalcul
    xy = project_2d(vectors, config.projection, config.runtime.seed)  # un point (x, y) par image pour le dessin
    k = min(config.n_clusters, len(catalog))  # nombre de groupes, jamais plus que le nombre d'images
    groups = cluster(vectors, k, config.runtime.seed)  # numero de groupe k-means de chaque image
    table = catalog.assign(cluster=groups, x=xy[:, 0], y=xy[:, 1])  # ajoute groupe et coordonnees au tableau des images
    table.to_csv(output / "clusters.csv", index=False)  # enregistre ce tableau
    result = {"n_images": len(table), "dimension": int(vectors.shape[1]), "n_clusters": k}  # resume : effectifs et tailles
    if "label" in table and table["label"].astype(str).nunique() > 1:  # si les etiquettes existent et qu'il y a 2 classes ou plus
        # purete : les groupes trouves sans etiquette correspondent-ils aux classes ? (1 = parfaitement)
        result["cluster_purity"] = round(cluster_purity(groups, table["label"].astype(str)), 3)
        # nuage 2D colore par classe
        plot_projection(xy, table["label"].astype(str), output / "projection_by_label.png", f"{config.projection} - couleur = classe")
    plot_projection(  # nuage 2D colore par groupe k-means
        xy,  # coordonnees 2D des points
        [f"groupe {g}" for g in groups],  # couleur de chaque point = son groupe k-means
        output / "projection_by_cluster.png",  # fichier image de sortie
        f"{config.projection} - couleur = groupe k-means",  # titre du graphique
    )
    rng = np.random.default_rng(config.runtime.seed)  # generateur aleatoire avec graine : meme tirage a chaque execution
    images, titles = [], []  # vignettes et legendes de la galerie par groupe
    for g in range(k):  # pour chaque groupe
        members = np.flatnonzero(groups == g)  # indices des images de ce groupe
        # tire au hasard (sans remise) au plus gallery_per_cluster images du groupe
        for i in rng.choice(members, size=min(config.gallery_per_cluster, len(members)), replace=False):
            images.append(_thumb(table["path"].iloc[i]))  # vignette de l'image tiree
            # legende : numero de groupe et classe si elle existe, coupee a 20 caracteres
            titles.append(f"g{g} {table['label'].iloc[i] if 'label' in table else ''}"[:20])
    plot_gallery(images, titles, output / "clusters_gallery.png", columns=config.gallery_per_cluster)  # une ligne par groupe
    order, sims = nearest(vectors, vectors[:5], k=6)  # voisins des 5 premieres images (exemple)
    result["neighbours_example"] = [  # pour chaque image requete : ses voisins et leurs similarites cosinus
        {  # un dictionnaire par image requete
            "query": table["path"].iloc[q],  # chemin de l'image requete
            # chemins des voisins ; [1:] saute le premier, qui est l'image elle-meme (similarite 1)
            "neighbours": [table["path"].iloc[j] for j in order[q][1:]],
            "cosine": [round(float(s), 3) for s in sims[q][1:]],  # similarite cosinus de chaque voisin (1 = identique)
        }
        for q in range(len(order))  # pour chaque image requete
    ]
    (output / "embeddings_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")  # enregistre le resume
    # journal : le resume sans l'exemple de voisins (trop long a afficher)
    logger.info("plongements : %s", {k: v for k, v in result.items() if k != "neighbours_example"})
    return result  # renvoie le resume a l'appelant


# --- derive -------------------------------------------------------------------------------------


def _limited(catalog: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:  # renvoie au plus n images du catalogue
    # si trop d'images : tirage au hasard reproductible de n lignes, retriees par chemin ; sinon le catalogue tel quel
    return catalog.sample(n=n, random_state=seed).sort_values("path") if len(catalog) > n else catalog


def drift(config: DriftConfig, output: Path, device: torch.device) -> dict:  # mesure la derive et renvoie un resume
    """Mesure la derive entre deux dossiers ; avec `simulate`, les images courantes sont alterees (demonstration)."""
    # images de reference (au plus max_images) et images courantes (graine + 1 : un tirage different)
    reference = _limited(build_catalog(config.reference), config.max_images, config.runtime.seed)
    current = _limited(build_catalog(config.current), config.max_images, config.runtime.seed + 1)  # images courantes
    if config.simulate:  # on ecrit les images alterees : toute la suite lit des fichiers, comme en vrai
        # alter : pipeline qui abime les images ; folder : dossier ou les ecrire (un sous-dossier par sortie)
        alter, folder = Pipeline(config.simulate), Path("data/processed/drift_simulation") / output.name
        paths = []  # chemins des images alterees
        for i, p in enumerate(current["path"]):  # pour chaque image courante, avec son numero i
            target = folder / f"{i:05d}_{Path(p).stem}.png"  # nom du fichier : numero sur 5 chiffres + nom d'origine, en PNG
            write_image(target, alter(read_image(p, color=config.current.color)))  # lit, altere et ecrit l'image
            paths.append(str(target))  # retient le nouveau chemin
        current = current.assign(path=paths)  # les images courantes deviennent les images alterees
    pipeline, model = Pipeline(config.preprocess), build_embedder(config.model)  # pretraitement et reseau sans tete
    # plongements des deux periodes, avec le meme reseau et le meme pretraitement
    ref_vectors = embed_images(reference["path"].tolist(), pipeline, config.reference.color, model, device=device)
    cur_vectors = embed_images(current["path"].tolist(), pipeline, config.current.color, model, device=device)  # cote courant
    # AUC : les deux periodes sont-elles separables ? novelty : proba "courant" de chaque image courante
    auc, novelty = domain_classifier(ref_vectors, cur_vectors, config.runtime.seed)
    stats_table = statistics_drift(scan(reference), scan(current))  # test KS sur les statistiques simples (luminosite...)
    stats_table.to_csv(output / "statistics_drift.csv", index=False)  # enregistre ce tableau
    # trie les images courantes de la plus "nouvelle" (la plus differente de la reference) a la moins nouvelle
    newest = current.assign(novelty=novelty).sort_values("novelty", ascending=False)
    newest[["path", "novelty"]].to_csv(output / "current_novelty.csv", index=False)  # enregistre chemin et score de nouveaute
    xy = project_2d(np.concatenate([ref_vectors, cur_vectors]), "pca", config.runtime.seed)  # projette les deux periodes en 2D
    plot_projection(  # nuage 2D colore par periode, avec l'AUC dans le titre
        xy,  # coordonnees 2D des points
        ["reference"] * len(ref_vectors) + ["courant"] * len(cur_vectors),  # periode de chaque point (meme ordre que xy)
        output / "drift_projection.png",  # fichier image de sortie
        f"AUC domaine {auc:.2f}",  # titre : AUC avec 2 decimales
    )
    top = newest.head(8)  # les 8 images courantes les plus nouvelles
    # galerie de ces 8 images, avec leur score de nouveaute en legende
    plot_gallery([_thumb(p) for p in top["path"]], [f"{v:.2f}" for v in top["novelty"]], output / "most_novel.png", columns=8)
    result = {  # resume de la derive
        "domain_auc": round(auc, 3),  # AUC du classifieur de domaine (0.5 = pas de derive, 1 = derive totale)
        "verdict": verdict(auc),  # phrase lisible associee a l'AUC
        "n_reference": len(reference),  # nombre d'images de reference utilisees
        "n_current": len(current),  # nombre d'images courantes utilisees
        "simulated": [s.model_dump() for s in config.simulate],  # alterations simulees (vide en usage reel)
        "top_statistics": stats_table.head(3).to_dict(orient="records"),  # les 3 statistiques qui ont le plus bouge
    }
    # enregistre le resume en JSON ; default=str convertit en texte ce que JSON ne sait pas ecrire
    (output / "drift_report.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    logger.info("derive : AUC %.3f -> %s", auc, result["verdict"])  # affiche le verdict dans le journal
    return result  # renvoie le resume a l'appelant
