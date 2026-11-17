"""Grad-CAM : quelle zone de l'image a fait monter le score de la classe predite.

Deux crochets sur la derniere couche de convolution (le dernier bloc du backbone) : on garde ses
activations et leur gradient, on pondere chaque carte par la moyenne de son gradient, on garde la
partie positive. Une zone chaude hors du defaut (bord, coin, horodatage) signale un modele qui a
appris un artefact et ne generalisera pas.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (ex. int | None)

import cv2  # OpenCV : traitement d'images (ici redimensionnement et palette de couleurs)
import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)
import torch  # PyTorch : tenseurs (tableaux de nombres sur CPU ou GPU) et reseaux de neurones
import torch.nn as nn  # nn : briques de reseaux de neurones (couches, modules)
import torch.nn.functional as F  # F : fonctions sans poids (relu, interpolation...)

from vtb.models.zoo import get_module  # retrouve une couche du modele a partir de son nom (ex. "layer4")


class GradCAM:  # calcule la carte de chaleur Grad-CAM d'un modele de classification
    """`cam = GradCAM(model, "layer4"); heat, cls = cam(tensor)` ; heat de la taille de l'entree, dans [0, 1]."""

    def __init__(self, model: nn.Module, layer: str) -> None:  # prepare Grad-CAM sur la couche "layer" du modele
        self.model = model.eval()  # mode evaluation : dropout et BatchNorm figes
        self.store: dict[str, torch.Tensor] = {}  # memoire pour les activations et gradients captures
        target = get_module(model, layer)  # la couche observee (en general le dernier bloc de convolution)
        # crochet (hook) : fonction appelee automatiquement a chaque passage avant dans cette couche
        self.handle = target.register_forward_hook(self._keep)

    def _keep(self, module: nn.Module, inputs, output: torch.Tensor) -> None:  # crochet : memorise la sortie de la couche
        self.store["activations"] = output  # activations = cartes produites par la couche (N, C, h, w)
        # second crochet sur ce tenseur : lors du retour arriere, memorise son gradient (derivee du score par rapport a lui)
        output.register_hook(lambda grad: self.store.__setitem__("gradients", grad))

    # calcule la carte pour une image ; renvoie (carte de chaleur numpy, classe expliquee)
    def __call__(self, tensor: torch.Tensor, class_index: int | None = None) -> tuple[np.ndarray, int]:
        x = tensor.unsqueeze(0) if tensor.ndim == 3 else tensor  # (C, H, W) -> (1, C, H, W) : le modele attend un lot
        x = x.clone().requires_grad_(True)  # meme backbone gele, le gradient doit traverser jusqu'a la couche visee
        logits = self.model(x)  # logits = scores bruts du modele, un par classe (avant softmax)
        index = int(logits.argmax(1)) if class_index is None else class_index  # classe expliquee : la predite, sauf si imposee
        self.model.zero_grad(set_to_none=True)  # efface les gradients d'un calcul precedent
        logits[0, index].backward()  # score de LA classe, pas la perte ; backward = calcule les gradients (retour arriere)
        weights = self.store["gradients"].mean(dim=(2, 3), keepdim=True)  # importance de chaque carte = moyenne de son gradient
        cam = F.relu((weights * self.store["activations"]).sum(dim=1, keepdim=True))  # somme ponderee des cartes, partie positive (relu)
        # agrandit la carte a la taille de l'entree (interpolation bilineaire), garde la 1re image et le seul canal, sans gradient
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0].detach()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # ramene la carte dans [0, 1] (1e-8 evite une division par 0)
        return cam.cpu().numpy(), index  # carte ramenee sur le CPU (si le modele est sur GPU) puis en tableau numpy, et indice de la classe

    def close(self) -> None:  # retire le crochet (a appeler quand on a fini)
        self.handle.remove()  # la couche redevient normale


def overlay(image: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:  # superpose la carte sur l'image ; renvoie du RGB
    """Superpose une carte [0, 1] (colormap JET) sur une image uint8 grise ou RGB ; renvoie du RGB uint8."""
    base = image if image.ndim == 3 else np.stack([image] * 3, axis=-1)  # image grise -> 3 canaux identiques
    base = base if base.dtype == np.uint8 else (255 * (base.astype(np.float32) / base.max())).astype(np.uint8)  # ramene en 8 bits
    heat_u8 = cv2.resize((heat * 255).astype(np.uint8), (base.shape[1], base.shape[0]))  # carte en 0-255, a la taille de l'image
    colored = cv2.cvtColor(cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)  # palette JET (bleu -> rouge), en RGB
    return cv2.addWeighted(base, 1 - alpha, colored, alpha, 0)  # melange : (1 - alpha) x image + alpha x carte coloree
