"""Plongements (embeddings) d'images par un reseau pre-entraine : un vecteur par image, comparable entre images.

A quoi ca sert sans aucune etiquette :
  - voir la structure du jeu (projection 2D) : les classes se separent-elles ? y a-t-il des
    sous-groupes inattendus (deux cameras, deux eclairages) ?
  - regrouper (k-means) pour annoter plus vite : on etiquette un groupe d'un coup ;
  - chercher les voisins d'une image (une image de defaut -> les images qui lui ressemblent) ;
  - mesurer une derive entre deux periodes (vtb.explore.drift).
Le reseau est celui du zoo (resnet18 ImageNet par defaut), sans sa tete : sortie du pooling global.
"""

from __future__ import annotations  # permet d'ecrire les types (ex. "nn.Module | None") sans qu'ils soient evalues au chargement

import logging  # journal des messages (infos, avertissements) plutot que des print
from collections.abc import Sequence  # Sequence : type "liste ou tuple" (on peut indexer et prendre une tranche)
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import numpy as np  # numpy : calcul sur des tableaux de nombres (ici les plongements)
import torch  # PyTorch : calcul sur des tenseurs (tableaux de nombres multidimensionnels) et reseaux de neurones
from torch import nn  # nn : briques de reseaux de neurones (couches, fonctions utiles)

from vtb.io.images import read_image  # lit une image du disque en tableau numpy (accepte les chemins avec accents)

# get_spec : fiche d'un modele du zoo ; instantiate : construit le reseau ; set_module : remplace une couche par une autre
from vtb.models.zoo import get_spec, instantiate, set_module
from vtb.preprocess import Pipeline  # suite d'operations de pretraitement (redimensionnement...) appliquee a chaque image

# moyenne / ecart-type des images ImageNet ; normalize : (x - moyenne) / ecart-type ; to_float_chw : image -> tenseur [0, 1]
from vtb.preprocess.tensor import IMAGENET_MEAN, IMAGENET_STD, normalize, to_float_chw

logger = logging.getLogger(__name__)  # journal propre a ce module (son nom apparait dans les messages)


def build_embedder(model_name: str = "resnet18", pretrained: bool = True) -> nn.Module:  # renvoie un reseau qui sort un vecteur
    """Reseau du zoo dont la tete est remplacee par l'identite."""
    spec = get_spec(model_name)  # fiche du modele : source, nom de la couche de tete, etc.
    model = instantiate(spec, pretrained=pretrained)  # construit le reseau, avec les poids appris sur ImageNet si pretrained
    # la tete (derniere couche qui donne les classes) devient Identity : le reseau renvoie alors les caracteristiques brutes
    set_module(model, spec.head, nn.Identity())
    return model.eval()  # mode evaluation : desactive dropout et fige la batchnorm, pour des sorties stables


@torch.no_grad()  # pas de calcul de gradient (derivees pour l'entrainement) : plus rapide et moins de memoire
def embed_images(  # calcule le plongement de chaque image et renvoie un tableau (nombre d'images, taille du vecteur)
    paths: Sequence[str | Path],  # chemins des images a traiter
    pipeline: Pipeline,  # pretraitement applique a chaque image avant le reseau (doit donner une taille fixe)
    color: str = "rgb",  # lecture en couleur ("rgb") ou en niveaux de gris ("gray")
    model: nn.Module | None = None,  # reseau a utiliser ; None = resnet18 sans tete construit ici
    batch_size: int = 32,  # 32 images traitees ensemble a chaque passage : plus grand = plus rapide mais plus de memoire
    device: torch.device | None = None,  # ou calculer : cpu ou gpu (cuda) ; None = cpu
) -> np.ndarray:  # renvoie un tableau numpy : une ligne (un vecteur) par image
    """(N, D) float32, normalise L2 (le produit scalaire devient la similarite cosinus).

    Le pipeline doit produire une taille fixe (finir par un resize) ; une image grise est copiee
    sur 3 canaux, puis normalisee comme ImageNet.
    """
    device = device or torch.device("cpu")  # si aucun appareil n'est donne, on calcule sur le processeur
    model = (model or build_embedder()).to(device).eval()  # reseau par defaut si besoin, envoye sur l'appareil, mode evaluation
    out = []  # un tableau de plongements par lot (batch), concatenes a la fin
    for start in range(0, len(paths), batch_size):  # pour chaque lot : indices 0, 32, 64... jusqu'a la fin de la liste
        tensors = []  # tenseurs des images de ce lot
        for p in paths[start : start + batch_size]:  # pour chaque image du lot
            # lit l'image, applique le pretraitement, puis la convertit en tenseur (canaux, hauteur, largeur) de valeurs 0-1
            x = to_float_chw(pipeline(read_image(p, color=color)))
            x = x.repeat(3, 1, 1) if x.shape[0] == 1 else x[:3]  # image grise copiee sur 3 canaux ; au-dela de 3 canaux, on coupe
            # normalisation : chaque canal est centre et reduit comme les images ImageNet vues par le reseau a l'entrainement
            tensors.append(normalize(x, IMAGENET_MEAN, IMAGENET_STD))
        # empile le lot en un tenseur (N, 3, H, W), le passe dans le reseau, puis aplatit chaque sortie en un vecteur
        features = model(torch.stack(tensors).to(device)).flatten(1)
        # normalisation L2 : chaque vecteur ramene a une longueur 1 ; puis retour sur cpu en tableau numpy
        out.append(nn.functional.normalize(features, dim=1).cpu().numpy())
    # message de suivi : nombre total de plongements et taille de chaque vecteur (0 si aucune image)
    logger.info("%d plongements de dimension %d", sum(len(o) for o in out), out[0].shape[1] if out else 0)
    return np.concatenate(out).astype(np.float32) if out else np.zeros((0, 0), np.float32)  # tous les lots en un seul tableau


def project_2d(embeddings: np.ndarray, method: str = "pca", seed: int = 0) -> np.ndarray:  # renvoie un point (x, y) par image
    """(N, 2) pour l'affichage : PCA (rapide, fidele aux distances globales) ou t-SNE (groupes locaux, plus lent)."""
    from sklearn.decomposition import PCA  # PCA : projette sur les directions ou les donnees varient le plus

    if method == "tsne":  # si on veut t-SNE (meilleur pour voir les groupes, mais plus lent)
        from sklearn.manifold import TSNE  # t-SNE : place proches les images proches, sans garder les grandes distances

        # on reduit d'abord a 50 dimensions au plus avec la PCA : t-SNE est plus rapide et moins bruite
        reduced = PCA(n_components=min(50, embeddings.shape[1], len(embeddings)), random_state=seed).fit_transform(embeddings)
        # perplexity ~ nombre de voisins pris en compte (entre 5 et 30 selon la taille du jeu) ; init pca : resultat plus stable
        return TSNE(n_components=2, perplexity=min(30, max(5, len(embeddings) // 10)), random_state=seed, init="pca").fit_transform(reduced)
    return PCA(n_components=2, random_state=seed).fit_transform(embeddings)  # sinon : simple PCA en 2 dimensions


def cluster(embeddings: np.ndarray, k: int, seed: int = 0) -> np.ndarray:  # renvoie le numero de groupe de chaque image
    """Groupes k-means (indices 0..k-1)."""
    from sklearn.cluster import KMeans  # k-means : forme k groupes autour de k centres, chaque image va au centre le plus proche

    # n_init=10 : 10 essais avec des centres de depart differents, on garde le meilleur ; graine fixe = resultat reproductible
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(embeddings)


# renvoie, pour chaque requete, les indices de ses k voisins les plus proches et leurs similarites
def nearest(embeddings: np.ndarray, queries: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Indices et similarites cosinus des k plus proches voisins de chaque requete (plongements normalises)."""
    sims = queries @ embeddings.T  # produit matriciel : similarite cosinus de chaque requete avec chaque image (1 = identique)
    order = np.argsort(-sims, axis=1)[:, :k]  # trie du plus similaire au moins similaire et garde les k premiers indices
    return order, np.take_along_axis(sims, order, axis=1)  # renvoie les indices et les similarites correspondantes


def cluster_purity(clusters: np.ndarray, labels: Sequence[str]) -> float:  # renvoie un score entre 0 et 1
    """Part des images dont l'etiquette est la majoritaire de leur groupe : 1 = groupes "purs"."""
    labels = np.asarray(labels)  # convertit les etiquettes en tableau numpy pour pouvoir les filtrer par masque
    total = 0  # nombre d'images qui portent l'etiquette majoritaire de leur groupe
    for c in np.unique(clusters):  # pour chaque numero de groupe present
        _, counts = np.unique(labels[clusters == c], return_counts=True)  # combien d'images de chaque etiquette dans ce groupe
        total += counts.max()  # on ajoute l'effectif de l'etiquette la plus frequente du groupe
    return float(total / len(labels))  # part des images "bien rangees" : 1 = chaque groupe ne contient qu'une seule classe


def plot_projection(xy: np.ndarray, colors: Sequence[str], path: str | Path, title: str = "") -> None:  # enregistre une image
    """Nuage 2D, une couleur par valeur (classe, groupe, periode)."""
    import matplotlib.pyplot as plt  # matplotlib : dessin de graphiques

    values = np.asarray(colors).astype(str)  # valeur de couleur de chaque point, en texte (une couleur par valeur distincte)
    fig, ax = plt.subplots(figsize=(6.4, 5))  # cree la figure (6.4 x 5 pouces) et sa zone de dessin
    for v in sorted(set(values)):  # pour chaque valeur distincte, dans l'ordre alphabetique
        m = values == v  # masque : vrai pour les points qui ont cette valeur
        # dessine ces points (taille 8, un peu transparents) ; la legende indique la valeur et le nombre de points
        ax.scatter(xy[m, 0], xy[m, 1], s=8, alpha=0.7, label=f"{v} ({m.sum()})")
    ax.set_title(title, fontsize=9)  # titre du graphique en petit
    ax.legend(fontsize=7, markerscale=2, loc="best")  # legende avec des pastilles agrandies, placee la ou elle gene le moins
    ax.set_xticks([])  # pas de graduations : les axes d'une projection n'ont pas d'unite qui ait du sens
    ax.set_yticks([])  # idem pour l'axe vertical
    fig.tight_layout()  # resserre les marges
    fig.savefig(path, dpi=110)  # enregistre l'image sur le disque (110 points par pouce)
    plt.close(fig)  # ferme la figure pour liberer la memoire
