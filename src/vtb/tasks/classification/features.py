"""Strategie `features` : backbone gele + regression logistique scikit-learn, repliee en couche lineaire.

On extrait un vecteur par image (sortie du pooling global), on standardise, on ajuste une
regression logistique. La standardisation et la regression sont ensuite repliees dans une seule
nn.Linear placee a la place de la tete : le resultat est un nn.Module ordinaire, evalue, predit
et exporte en ONNX exactement comme les autres strategies.
"""

from __future__ import annotations  # les annotations de type restent du texte : syntaxe moderne partout

import logging  # journal des messages (info, avertissements) au lieu de print

import numpy as np  # numpy : calcul sur des tableaux de nombres
import torch  # PyTorch : tenseurs (tableaux de nombres, sur CPU ou GPU) et reseaux de neurones
import torch.nn as nn  # nn : briques de reseau (couches, Identity, Linear...)
from sklearn.linear_model import LogisticRegression  # regression logistique : classifieur lineaire simple et rapide
from sklearn.preprocessing import StandardScaler  # standardisation : chaque colonne ramenee a moyenne 0 et ecart-type 1
from torch.utils.data import DataLoader  # DataLoader : distribue les images par lots (batch = groupe d'images)

from vtb.models.zoo import ModelSpec, get_module, set_module  # fiche du modele (nom de la tete...) ; lire / remplacer une couche

logger = logging.getLogger(__name__)  # journal propre a ce module


@torch.no_grad()  # pas de calcul de gradient (derivees pour l'apprentissage) : inutile ici, economise memoire et temps
# passe toutes les images dans le backbone (partie du reseau qui extrait les caracteristiques) ; renvoie (vecteurs, classes)
def extract_features(model: nn.Module, spec: ModelSpec, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Vecteurs juste avant la tete (la tete est remplacee temporairement par l'identite)."""
    head = get_module(model, spec.head)  # garde la tete d'origine (derniere couche, celle qui donne les classes)
    set_module(model, spec.head, nn.Identity())  # la remplace par Identity (sortie = entree) : le modele sort le vecteur
    model.eval()  # mode evaluation : dropout coupe, batchnorm fige (resultats stables)
    feats, targets = [], []  # listes ou l'on accumule les vecteurs et les classes de chaque lot
    try:  # try / finally : la tete sera remise meme si une erreur survient
        for images, y in loader:  # pour chaque lot d'images et leurs classes
            feats.append(model(images.to(device)).flatten(1).cpu())  # passe le lot, aplatit en (lot, D), ramene sur CPU
            targets.append(y)  # garde les classes du lot
    finally:  # execute dans tous les cas, erreur ou pas
        set_module(model, spec.head, head)  # remet la tete d'origine dans le modele
    return torch.cat(feats).numpy(), torch.cat(targets).numpy()  # colle tous les lots et convertit en tableaux numpy


def fit_logistic_head(  # entraine la regression logistique et la place comme tete du modele ; renvoie le classifieur sklearn
    model: nn.Module,  # le reseau dont on remplace la tete
    spec: ModelSpec,  # fiche du modele : donne le nom de la couche de tete
    features: np.ndarray,  # vecteurs extraits par extract_features, un par image (N, D)
    targets: np.ndarray,  # classe de chaque image (N,)
    c: float,  # inverse de la force de regularisation : petit c = modele plus simple, moins de sur-apprentissage
    seed: int,  # graine aleatoire : meme resultat a chaque execution
) -> LogisticRegression:  # renvoie le modele scikit-learn ajuste
    """Ajuste StandardScaler + regression logistique et remplace la tete par la couche lineaire equivalente.

    Repli : z = W (x - mu) / sigma + b = (W / sigma) x + (b - W mu / sigma).
    """
    scaler = StandardScaler().fit(features)  # apprend moyenne (mu) et ecart-type (sigma) de chaque dimension
    # ajuste la regression sur les vecteurs standardises ; max_iter eleve pour bien converger ; graine fixe
    clf = LogisticRegression(C=c, max_iter=3000, random_state=seed).fit(scaler.transform(features), targets)
    weight = clf.coef_ / scaler.scale_  # (C, D) : poids W / sigma, la standardisation est absorbee dans les poids
    bias = clf.intercept_ - weight @ scaler.mean_  # biais b - W mu / sigma (@ = produit matriciel)
    if weight.shape[0] == 1:  # cas binaire : scikit-learn ne donne qu'une ligne ; logits [-z/2, z/2] equivalents
        # deux lignes opposees : logits = scores bruts avant softmax ; softmax(-z/2, z/2) = meme probabilite que sigmoide(z)
        weight, bias = np.vstack([-weight / 2, weight / 2]), np.array([-bias[0] / 2, bias[0] / 2])
    linear = nn.Linear(weight.shape[1], weight.shape[0])  # couche lineaire PyTorch : D entrees, C sorties (une par classe)
    with torch.no_grad():  # copie de poids : ce n'est pas un pas d'apprentissage, pas de gradient
        linear.weight.copy_(torch.tensor(weight, dtype=torch.float32))  # recopie les poids calcules dans la couche
        linear.bias.copy_(torch.tensor(bias, dtype=torch.float32))  # recopie les biais
    set_module(model, spec.head, linear)  # remplace la tete du modele : il devient un nn.Module ordinaire, exportable en ONNX
    # message de suivi : nombre d'images, taille des vecteurs, nombre de classes
    logger.info("regression logistique : %d vecteurs de dimension %d, %d classes", *features.shape, weight.shape[0])
    return clf  # renvoie le classifieur scikit-learn (utile pour inspection)
