"""Ce qui est commun a toutes les methodes d'anomalie : lissage, score image, seuil, metriques.

Toute methode produit une carte d'anomalie (N, 1, S, S) ; le reste du protocole est identique,
ce qui rend les methodes comparables entre elles et echangeables en production.
"""

from __future__ import annotations  # permet les annotations de type modernes

import math  # fonctions mathematiques de base (ceil = arrondi au-dessus)

import numpy as np  # numpy : calcul sur des tableaux de nombres
import torch  # PyTorch : tenseurs (tableaux de nombres multi-dimensions)
import torch.nn as nn  # nn : modules de reseau (pour rendre le calcul exportable en ONNX)
import torch.nn.functional as F  # F : fonctions de calcul (interpolation, convolution, bourrage)

# f1_score : moyenne equilibree precision / rappel ; roc_auc_score : AUROC, qualite du classement des scores sans seuil
from sklearn.metrics import f1_score, roc_auc_score


def gaussian_kernel(sigma: float) -> torch.Tensor:  # noyau de flou gaussien (poids en cloche) ; renvoie un tenseur (1, 1, k, k)
    """Noyau gaussien 2D normalise (1, 1, k, k), k = 2 ceil(3 sigma) + 1."""
    radius = max(1, math.ceil(3 * sigma))  # rayon du noyau : 3 sigma couvre presque toute la cloche
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)  # positions -radius ... +radius
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))  # courbe en cloche 1D : poids fort au centre, faible sur les bords
    kernel = kernel / kernel.sum()  # normalise : les poids font 1 au total (le flou ne change pas le niveau moyen)
    return torch.outer(kernel, kernel)[None, None]  # produit exterieur -> noyau 2D ; [None, None] ajoute 2 dimensions (1, 1, k, k)


class MapToScore(nn.Module):  # post-traitement commun : lisse la carte et calcule un score par image
    """Carte brute (N, 1, h, w) -> carte lissee a la taille de l'image (N, 1, S, S) et score image (N,). Exportable en ONNX."""

    # size : taille (S, S) de l'image ; sigma : force du flou ; image_score : "max" ou "top_mean"
    def __init__(self, size: tuple[int, int], sigma: float, image_score: str = "max", top_fraction: float = 0.01) -> None:
        super().__init__()  # initialise nn.Module
        self.size, self.image_score, self.top_fraction = size, image_score, top_fraction  # garde les reglages
        # noyau range dans le modele (exporte avec lui) ; sigma = 0 -> noyau 1x1 de valeur 1, donc aucun flou
        self.register_buffer("kernel", gaussian_kernel(sigma) if sigma > 0 else torch.ones(1, 1, 1, 1))

    def forward(self, raw_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # carte brute -> (carte lissee, score)
        if raw_map.ndim == 3:  # si la carte n'a pas de dimension canal (N, h, w)
            raw_map = raw_map[:, None]  # on l'ajoute : (N, 1, h, w)
        # agrandit la carte a la taille de l'image (interpolation bilineaire) si elle n'y est pas deja
        maps = F.interpolate(raw_map, size=self.size, mode="bilinear", align_corners=False) if raw_map.shape[-2:] != self.size else raw_map
        radius = self.kernel.shape[-1] // 2  # rayon du noyau (0 si pas de flou)
        if radius:  # si un flou est demande
            # bourrage "reflect" (bords recopies en miroir, pour ne pas assombrir les bords) puis convolution avec le noyau
            maps = F.conv2d(F.pad(maps, (radius,) * 4, mode="reflect"), self.kernel)
        flat = maps.flatten(1)  # chaque carte mise a plat : (N, S*S)
        if self.image_score == "max":  # score = pixel le plus anormal
            score = flat.amax(dim=1)  # maximum de chaque carte
        else:  # score "top_mean" : moyenne des pixels les plus anormaux
            k = max(1, int(self.top_fraction * flat.shape[1]))  # nombre de pixels a moyenner (au moins 1)
            score = flat.topk(k, dim=1).values.mean(dim=1)  # les k plus grandes valeurs de chaque carte, moyennees
        return maps, score  # carte lissee (N, 1, S, S) et score (N,)


# calcule les seuils de decision a partir des seuls scores des normales de validation ; renvoie un dictionnaire
def calibrate_thresholds(val_scores: np.ndarray, val_maps: np.ndarray, image_quantile: float, pixel_quantile: float) -> dict[str, float]:
    """Seuils = quantiles des scores de normales de validation : un taux de fausses alarmes choisi, sans aucun defaut."""
    return {  # dictionnaire enregistre dans calibration.json
        "image_threshold": float(np.quantile(val_scores, image_quantile)),  # ex. quantile 0.99 : 1 % des normales depassent le seuil
        "pixel_threshold": float(np.quantile(val_maps, pixel_quantile)),  # meme principe sur tous les pixels des cartes
        "image_quantile": image_quantile,  # quantile utilise (garde pour la tracabilite)
        "pixel_quantile": pixel_quantile,  # idem pour les pixels
        "n_val_normals": int(len(val_scores)),  # nombre de normales qui ont servi a calibrer
    }


def anomaly_metrics(  # mesure la qualite du detecteur sur le test ; renvoie un dictionnaire de metriques
    scores: np.ndarray,  # score de chaque image (N,)
    maps: np.ndarray,  # carte d'anomalie de chaque image (N, S, S)
    masks: np.ndarray,  # masque verite de chaque image (N, S, S), 1 = pixel defectueux
    labels: np.ndarray,  # etiquette de chaque image : 1 defaut, 0 normal
    image_threshold: float,  # seuil calibre : score au-dessus = anomalie
    defect_types: np.ndarray | None = None,  # type de defaut de chaque image (pour le rappel par type), facultatif
) -> dict:  # renvoie un dictionnaire nom -> valeur
    """AUROC image et pixel (sans seuil), puis rappel, fausses alarmes et F1 au seuil calibre, et rappel par type de defaut."""
    decisions = (scores > image_threshold).astype(int)  # decision par image : 1 si signalee anormale, 0 sinon
    metrics: dict = {"n_images": int(len(labels)), "n_defects": int(labels.sum())}  # nombre d'images et de defauts du test
    if len(set(labels)) == 2:  # l'AUROC n'a de sens que s'il y a des normales ET des defauts
        metrics["auroc_image"] = float(roc_auc_score(labels, scores))  # 1 = les defauts ont toujours un score plus haut que les normales
    if masks.any():  # si au moins un masque contient un pixel defectueux
        flat_masks = masks.reshape(-1).astype(np.uint8)  # tous les pixels de tous les masques a la suite
        flat_maps = maps.reshape(-1)  # toutes les valeurs de carte, dans le meme ordre
        step = max(1, flat_masks.size // 4_000_000)  # au-dela de quelques millions de pixels, un sous-echantillon regulier suffit
        metrics["auroc_pixel"] = float(roc_auc_score(flat_masks[::step], flat_maps[::step]))  # AUROC au niveau des pixels (1 sur step)
    # rappel : part des defauts detectes (nan s'il n'y a aucun defaut)
    metrics["recall"] = float(decisions[labels == 1].mean()) if (labels == 1).any() else float("nan")
    # taux de fausses alarmes : part des normales signalees a tort
    metrics["false_alarm_rate"] = float(decisions[labels == 0].mean()) if (labels == 0).any() else float("nan")
    metrics["f1"] = float(f1_score(labels, decisions, zero_division=0))  # F1 au seuil (0 au lieu d'une erreur si rien n'est signale)
    if defect_types is not None:  # si on connait le type de chaque defaut
        metrics["recall_per_type"] = {  # rappel calcule separement pour chaque type de defaut
            # pour chaque type present parmi les defauts : part des images de ce type signalees, arrondie a 3 decimales
            str(t): round(float(decisions[defect_types == t].mean()), 3)  # cle = nom du type, valeur = rappel
            for t in sorted(set(defect_types[labels == 1].tolist()))  # types distincts parmi les vrais defauts, tries
        }
    return metrics  # toutes les metriques
