"""Augmentations qui melangent plusieurs images : MixUp / CutMix (lot), copy-paste de defauts, mosaique.

Elles ne tiennent pas dans un Compose image par image :
- MixUp / CutMix melangent deux images d'un LOT et leurs etiquettes (classification) ; elles se
  branchent comme `collate_fn` du DataLoader d'entrainement (`batch_augmentation`) ;
- le copy-paste prend des defauts dans une banque (`DefectBank`) et les colle sur d'autres images,
  en mettant a jour boites, etiquettes et masque : le remede direct au manque de defauts ;
- la mosaique assemble 4 images en une (detection : plus d'objets et de contextes par image).

Toutes les classes sont picklables (pas de fonction locale) : elles marchent avec num_workers > 0,
y compris sous Windows (demarrage "spawn" des workers).
"""

from __future__ import annotations  # permet d'ecrire les types (ex. "X | None") sans qu'ils soient evalues au chargement

import math  # fonctions mathematiques de base (floor, ceil) sur des nombres simples
from collections.abc import Sequence  # Sequence : type "liste ou tuple"
from dataclasses import dataclass  # dataclass : cree une petite classe "fiche" a partir de ses seuls champs
from typing import Any  # Any : "n'importe quel type"

import cv2  # OpenCV : traitement d'images (composantes connexes, flou)
import numpy as np  # numpy : calcul sur des tableaux de nombres
import torch  # PyTorch : tenseurs (tableaux de nombres multidimensionnels)
import torch.nn.functional as nnf  # fonctions utiles : one_hot (codage des classes), interpolate (redimensionnement)

# default_collate : la fonction standard qui empile une liste d'exemples en un lot (batch) de tenseurs
from torch.utils.data import default_collate
from torchvision import tv_tensors  # tenseurs etiquetes de torchvision (BoundingBoxes = boites englobantes)
from torchvision.transforms import v2  # transformations torchvision v2 (MixUp, CutMix, RandomChoice)

# aides de tirage au hasard (generateur torch) : element, pile ou face, facteur log, intervalle, entier, decimal
from ._base import choice, coin, log_uniform, pair, randint, uniform

# --- MixUp / CutMix (classification) --------------------------------------------------------------


class MixCollate:  # remplace la fonction qui forme les lots du DataLoader pour y melanger des images
    """collate_fn qui applique MixUp ou CutMix au lot avec la probabilite p.

    Les etiquettes deviennent des probabilites (N, num_classes) : CrossEntropyLoss les accepte
    telles quelles. Pour une precision d'entrainement, comparer argmax(sortie) a argmax(cible).
    Ne pas l'utiliser pour la validation ni le test.
    """

    # num_classes : nombre de classes ; alpha : force du melange (0 = desactive) ; p : probabilite de melanger un lot
    def __init__(self, num_classes: int, mixup_alpha: float = 0.2, cutmix_alpha: float = 1.0, p: float = 0.5) -> None:
        if mixup_alpha <= 0 and cutmix_alpha <= 0:  # si les deux methodes sont desactivees
            raise ValueError("au moins un des deux alpha doit etre > 0")  # rien a faire : erreur claire
        self.num_classes, self.p = int(num_classes), float(p)  # garde les reglages
        mixers: list[v2.Transform] = []  # methodes de melange retenues
        if mixup_alpha > 0:  # MixUp : moyenne ponderee de deux images entieres (et de leurs etiquettes)
            mixers.append(v2.MixUp(alpha=mixup_alpha, num_classes=self.num_classes))  # ajoute MixUp
        if cutmix_alpha > 0:  # CutMix : un rectangle d'une image colle dans une autre (etiquettes au prorata de la surface)
            mixers.append(v2.CutMix(alpha=cutmix_alpha, num_classes=self.num_classes))  # ajoute CutMix
        self.mixer = mixers[0] if len(mixers) == 1 else v2.RandomChoice(mixers)  # une seule methode, ou l'une des deux au hasard

    # appelee par le DataLoader avec la liste des exemples (image, classe) ; renvoie le lot d'images et les cibles
    def __call__(self, batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        images, labels = default_collate(batch)  # empile : images (N, C, H, W) et classes (N,)
        labels = torch.as_tensor(labels, dtype=torch.int64)  # classes en entiers 64 bits (type attendu par MixUp / CutMix)
        if coin(self.p):  # si le tirage dit "melanger ce lot"
            return self.mixer(images, labels)  # renvoie images melangees et cibles en probabilites melangees
        # sinon : images intactes, classes en codage "one-hot" (1 pour la vraie classe, 0 ailleurs) pour garder le meme format
        return images, nnf.one_hot(labels, self.num_classes).float()


# renvoie un MixCollate a passer au DataLoader ; l'etoile oblige a nommer les parametres qui suivent
def batch_augmentation(num_classes: int, *, mixup_alpha: float = 0.2, cutmix_alpha: float = 1.0, p: float = 0.5) -> MixCollate:
    """MixUp / CutMix au niveau du lot : `DataLoader(train_set, collate_fn=batch_augmentation(6))`."""
    return MixCollate(num_classes, mixup_alpha, cutmix_alpha, p)  # simple raccourci vers la classe


# --- banque de defauts et copy-paste --------------------------------------------------------------


@dataclass  # dataclass : __init__ et affichage generes automatiquement a partir des champs
class DefectInstance:  # un defaut decoupe, pret a etre colle ailleurs
    """Un defaut decoupe : pixels (C, h, w), opacite (h, w) dans [0, 1], classe."""

    patch: torch.Tensor  # pixels du defaut et d'une petite marge autour
    alpha: torch.Tensor  # opacite de chaque pixel au collage : 1 = defaut, 0 = fond, entre les deux = bord adouci
    label: int  # numero de classe du defaut


def _as_tensor_image(image: Any) -> torch.Tensor:  # renvoie l'image en tenseur (C, H, W) de decimaux
    """Tenseur (C, H, W) float, ou tableau (H, W[, C]) uint8 / uint16 / float -> tenseur (C, H, W) float [0, 1]."""
    if isinstance(image, torch.Tensor):  # si c'est deja un tenseur torch
        return image.float()  # simple passage en decimaux (suppose deja dans [0, 1])
    array = np.asarray(image)  # sinon : tableau numpy
    peak = {np.dtype(np.uint8): 255.0, np.dtype(np.uint16): 65535.0}.get(array.dtype, 1.0)  # valeur max selon le type
    array = array.astype(np.float32) / peak  # ramene dans [0, 1]
    # image grise : ajoute l'axe des canaux devant ; couleur : (H, W, C) -> (C, H, W) ; copy() pour une memoire contigue
    return torch.from_numpy(array[None] if array.ndim == 2 else array.transpose(2, 0, 1).copy())


def _feathered_box(height: int, width: int, feather: float) -> np.ndarray:  # renvoie une opacite (H, W) a bords adoucis
    """Opacite d'un rectangle : 1 au centre, descend a 0 sur `feather` pixels au bord."""
    ys = np.minimum(np.arange(height), np.arange(height)[::-1])[:, None] + 0.5  # distance de chaque ligne au bord le plus proche
    xs = np.minimum(np.arange(width), np.arange(width)[::-1])[None, :] + 0.5  # distance de chaque colonne au bord le plus proche
    # distance au bord le plus proche, divisee par la largeur d'adoucissement, bornee a [0, 1]
    return np.clip(np.minimum(ys, xs) / max(feather, 1e-6), 0, 1).astype(np.float32)


class DefectBank:  # reserve de defauts decoupes dans les images d'entrainement
    """Banque de defauts a coller : construite depuis des masques (segmentation) ou des boites (detection)."""

    def __init__(self, instances: Sequence[DefectInstance] = ()) -> None:  # instances : defauts deja decoupes
        self.instances = list(instances)  # copie en liste

    def __len__(self) -> int:  # permet d'ecrire len(banque)
        return len(self.instances)  # nombre de defauts dans la banque

    def classes(self) -> dict[int, int]:  # renvoie le nombre de defauts de chaque classe
        """Classe -> nombre de defauts dans la banque."""
        out: dict[int, int] = {}  # compteur par classe
        for inst in self.instances:  # pour chaque defaut
            out[inst.label] = out.get(inst.label, 0) + 1  # ajoute 1 au compteur de sa classe (0 si premiere fois)
        return out  # renvoie le compteur

    @classmethod  # methode appelee sur la classe : DefectBank.from_masks(...) fabrique une banque
    def from_masks(  # construit la banque a partir d'images et de leurs masques de segmentation
        cls,  # la classe DefectBank elle-meme
        images: Sequence[Any],  # images d'entrainement
        masks: Sequence[Any],  # un masque par image : numero de classe par pixel, 0 = fond
        *,  # les parametres suivants doivent etre nommes a l'appel
        min_area: int = 16,  # surface minimale d'un defaut en pixels (plus petit : ignore, souvent du bruit)
        pad: int = 3,  # marge de fond gardee autour du defaut, en pixels
        feather: float = 1.5,  # flou du bord de l'opacite, en pixels
    ) -> DefectBank:  # renvoie une banque remplie
        """Un defaut par composante connexe de chaque classe > 0 du masque (indices de classe, 0 = fond).

        pad : marge de fond gardee autour du defaut (sert a accorder la luminosite au collage) ;
        feather : flou du bord de l'opacite (un collage a bord franc se reconnait).
        """
        instances = []  # defauts decoupes
        for image, mask in zip(images, masks, strict=True):  # pour chaque image et son masque (strict : meme nombre exige)
            x = _as_tensor_image(image)  # image en tenseur (C, H, W) dans [0, 1]
            # masque en tableau numpy d'entiers, a la taille de l'image (ramene sur cpu si c'est un tenseur)
            m = np.asarray(mask.cpu() if isinstance(mask, torch.Tensor) else mask).reshape(x.shape[-2:]).astype(np.int64)
            height, width = m.shape  # taille de l'image
            for label in [int(v) for v in np.unique(m) if v > 0]:  # pour chaque classe de defaut presente (0 = fond, ignore)
                # composantes connexes : groupes de pixels qui se touchent (8 voisins) = un defaut chacun ;
                # count = nombre de groupes (fond compris), components = numero de groupe par pixel, stats = rectangle et surface
                count, components, stats, _ = cv2.connectedComponentsWithStats((m == label).astype(np.uint8), connectivity=8)
                for k in range(1, count):  # pour chaque groupe (le 0 est le fond)
                    left, top, w, h, area = stats[k]  # rectangle englobant et surface du groupe
                    if area < min_area:  # si le defaut est trop petit
                        continue  # on l'ignore
                    y0, x0 = max(0, top - pad), max(0, left - pad)  # coin haut-gauche avec marge, sans sortir de l'image
                    y1, x1 = min(height, top + h + pad), min(width, left + w + pad)  # coin bas-droit avec marge
                    region = (components[y0:y1, x0:x1] == k).astype(np.float32)  # 1 sur les pixels de CE defaut, 0 ailleurs
                    alpha = cv2.GaussianBlur(region, (0, 0), feather) if feather > 0 else region  # bords adoucis par un flou
                    alpha = np.maximum(alpha, region * 0.999)  # le defaut lui-meme reste opaque
                    # garde une copie des pixels, l'opacite et la classe
                    instances.append(DefectInstance(x[:, y0:y1, x0:x1].clone(), torch.from_numpy(alpha), label))
        return cls(instances)  # cree la banque avec tous les defauts trouves

    @classmethod  # methode appelee sur la classe : DefectBank.from_boxes(...) fabrique une banque
    def from_boxes(  # construit la banque a partir d'images et de leurs boites (detection)
        cls,  # la classe DefectBank elle-meme
        images: Sequence[Any],  # images d'entrainement
        boxes: Sequence[Any],  # boites de chaque image, format (x1, y1, x2, y2)
        labels: Sequence[Any],  # classes des boites de chaque image
        *,  # les parametres suivants doivent etre nommes a l'appel
        pad: int = 2,  # marge gardee autour de la boite, en pixels
        feather: float = 3.0,  # largeur du bord adouci, en pixels
        min_size: int = 4,  # cote minimal (marge comprise) ; plus petit : ignore
    ) -> DefectBank:  # renvoie une banque remplie
        """Un defaut par boite (x1, y1, x2, y2) ; l'opacite est un rectangle aux bords adoucis."""
        instances = []  # defauts decoupes
        # pour chaque image avec ses boites et ses classes (strict : meme nombre exige)
        for image, image_boxes, image_labels in zip(images, boxes, labels, strict=True):
            x = _as_tensor_image(image)  # image en tenseur (C, H, W) dans [0, 1]
            height, width = x.shape[-2:]  # taille de l'image
            for box, label in zip(  # pour chaque boite et sa classe
                np.asarray(image_boxes, dtype=np.float32).reshape(-1, 4),  # boites en tableau (N, 4)
                np.asarray(image_labels).reshape(-1),  # classes en tableau (N,)
                strict=True,  # erreur si les deux listes n'ont pas la meme longueur
            ):
                # coin haut-gauche arrondi vers le bas, moins la marge, sans sortir de l'image
                x0, y0 = max(0, int(math.floor(box[0])) - pad), max(0, int(math.floor(box[1])) - pad)
                # coin bas-droit arrondi vers le haut, plus la marge, sans sortir de l'image
                x1, y1 = min(width, int(math.ceil(box[2])) + pad), min(height, int(math.ceil(box[3])) + pad)
                if x1 - x0 < min_size or y1 - y0 < min_size:  # si la zone est trop petite
                    continue  # on l'ignore
                alpha = torch.from_numpy(_feathered_box(y1 - y0, x1 - x0, feather))  # opacite rectangulaire a bords adoucis
                instances.append(DefectInstance(x[:, y0:y1, x0:x1].clone(), alpha, int(label)))  # garde pixels, opacite, classe
        return cls(instances)  # cree la banque avec tous les defauts trouves

    def sample(self) -> DefectInstance:  # renvoie un defaut de la banque tire au hasard
        if not self.instances:  # si la banque est vide
            raise ValueError("banque de defauts vide")  # erreur claire
        return choice(self.instances)  # tirage avec le generateur torch (reproductible)


# transforme un defaut au hasard (miroir, quart de tour, echelle) ; renvoie (pixels, opacite)
def _transform_instance(inst: DefectInstance, scale: tuple[float, float], flip: bool, rot90: bool, max_hw: tuple[int, int]):
    patch, alpha = inst.patch, inst.alpha[None]  # pixels (C, h, w) et opacite mise en (1, h, w) pour la traiter pareil
    if flip and coin(0.5):  # retournement gauche-droite une fois sur deux
        patch, alpha = patch.flip(-1), alpha.flip(-1)  # inverse l'ordre des colonnes
    if flip and coin(0.5):  # retournement haut-bas une fois sur deux
        patch, alpha = patch.flip(-2), alpha.flip(-2)  # inverse l'ordre des lignes
    if rot90:  # si les quarts de tour sont permis
        k = randint(0, 3)  # 0 a 3 quarts de tour
        patch, alpha = torch.rot90(patch, k, (-2, -1)), torch.rot90(alpha, k, (-2, -1))  # tourne pixels et opacite ensemble
    s = log_uniform(*scale)  # facteur d'echelle tire en echelle log
    h, w = patch.shape[-2:]  # taille actuelle du defaut
    s = min(s, 0.9 * max_hw[0] / h, 0.9 * max_hw[1] / w)  # le defaut doit tenir dans l'image
    size = (max(2, int(round(h * s))), max(2, int(round(w * s))))  # nouvelle taille, au moins 2 x 2
    if size != (h, w):  # si la taille change
        # redimensionne par interpolation bilineaire (interpolate demande une forme (N, C, h, w), d'ou [None] puis [0])
        patch = nnf.interpolate(patch[None], size=size, mode="bilinear", align_corners=False)[0]
        alpha = nnf.interpolate(alpha[None], size=size, mode="bilinear", align_corners=False)[0]  # meme chose pour l'opacite
    return patch, alpha[0].clamp(0, 1)  # renvoie les pixels et l'opacite (h, w) bornee a [0, 1]


def _match_channels(patch: torch.Tensor, channels: int) -> torch.Tensor:  # renvoie le defaut avec le bon nombre de canaux
    if patch.shape[0] == channels:  # si le nombre de canaux est deja le bon
        return patch  # rien a faire
    if channels == 1:  # si l'image d'accueil est grise
        return patch.mean(0, keepdim=True)  # moyenne des canaux couleur -> 1 canal
    return patch.expand(channels, -1, -1)  # sinon defaut gris -> canal repete pour une image couleur


def copy_paste(  # colle des defauts de la banque sur une image ; renvoie l'image et les cibles mises a jour
    image: torch.Tensor,  # image d'accueil (C, H, W) en decimaux
    bank: DefectBank,  # banque de defauts
    *,  # les parametres suivants doivent etre nommes a l'appel
    boxes: torch.Tensor | None = None,  # boites existantes (detection)
    labels: torch.Tensor | None = None,  # classes des boites existantes
    mask: torch.Tensor | None = None,  # masque existant (segmentation)
    count: Sequence[int] = (1, 3),  # nombre de defauts a coller (tire dans l'intervalle)
    scale: Sequence[float] = (0.8, 1.25),  # intervalle du facteur d'echelle des defauts
    flip: bool = True,  # permet les retournements du defaut
    rot90: bool = True,  # permet les quarts de tour du defaut
    match_brightness: bool = True,  # accorde la luminosite du defaut a celle de l'endroit ou il est colle
    avoid_overlap: bool = True,  # evite de coller sur un objet deja present
    max_tries: int = 20,  # nombre d'essais de position avant d'abandonner un defaut
) -> dict[str, Any]:  # renvoie un dictionnaire {image, boxes, labels, mask}
    """Colle 1 a 3 defauts de la banque sur l'image ; renvoie {image, boxes, labels, mask} mis a jour.

    image (C, H, W) float ; boxes (N, 4) XYXY (tenseur ou BoundingBoxes) et labels (N,) en detection ;
    mask (H, W) d'indices de classe en segmentation. match_brightness : decale le defaut pour que
    son fond (la marge `pad`) ait la luminosite locale de l'image d'accueil ; avoid_overlap : ne
    colle pas sur un objet existant. Les boites des defauts colles sont calculees sur l'opacite > 0,5.
    """
    out = image.clone()  # copie : l'image d'origine reste intacte
    channels, height, width = out.shape  # nombre de canaux et taille
    box_list = [] if boxes is None else boxes.as_subclass(torch.Tensor).reshape(-1, 4).tolist()  # boites en liste Python
    label_list = [] if labels is None else torch.as_tensor(labels).reshape(-1).tolist()  # classes en liste Python
    new_mask = None if mask is None else mask.as_subclass(torch.Tensor).clone()  # copie du masque (s'il y en a un)
    lo, hi = (int(v) for v in pair(count))  # bornes du nombre de defauts, en entiers
    for _ in range(randint(lo, hi) if len(bank) else 0):  # pour chaque defaut a coller (aucun si la banque est vide)
        inst = bank.sample()  # tire un defaut de la banque
        patch, alpha = _transform_instance(inst, pair(scale), flip, rot90, (height, width))  # le transforme au hasard
        patch = _match_channels(patch.to(out), channels)  # meme type, appareil et nombre de canaux que l'image
        alpha = alpha.to(out)  # opacite au meme type et sur le meme appareil
        ph, pw = patch.shape[-2:]  # taille du defaut
        solid = alpha > 0.5  # pixels qui appartiennent vraiment au defaut
        if not solid.any():  # si aucun pixel (defaut trop reduit)
            continue  # on passe au suivant
        ys, xs = torch.nonzero(solid, as_tuple=True)  # coordonnees des pixels du defaut dans le morceau
        for _try in range(max_tries):  # plusieurs essais de position
            top, left = randint(0, height - ph), randint(0, width - pw)  # position tiree pour que le morceau tienne dans l'image
            # boite du defaut a cette position (x1, y1, x2, y2), calculee sur ses pixels pleins
            new_box = [left + int(xs.min()), top + int(ys.min()), left + int(xs.max()) + 1, top + int(ys.max()) + 1]
            # vrai si la nouvelle boite chevauche une boite existante (intersection non vide en x ET en y)
            overlap = any(min(new_box[2], b[2]) > max(new_box[0], b[0]) and min(new_box[3], b[3]) > max(new_box[1], b[1]) for b in box_list)
            if not (avoid_overlap and overlap):  # si la position convient
                break  # on la garde
        else:  # ce "else" de boucle s'execute si aucun essai n'a donne de place libre
            continue  # pas de place libre : on n'insiste pas
        region = out[:, top : top + ph, left : left + pw]  # zone de l'image qui va recevoir le defaut
        if match_brightness:  # si on accorde la luminosite
            ring = alpha < 0.5  # pixels de la marge autour du defaut (le fond d'origine du defaut)
            if ring.any():  # s'il y a une marge
                # decale le defaut pour que sa marge ait la meme luminosite moyenne (par canal) que la zone d'accueil
                patch = patch + (region[:, ring].mean(dim=1) - patch[:, ring].mean(dim=1))[:, None, None]
            else:  # sinon (pas de marge)
                # on compare les moyennes de tout le morceau
                patch = patch + (region.mean(dim=(1, 2)) - patch.mean(dim=(1, 2)))[:, None, None]
        # collage par transparence : fond x (1 - opacite) + defaut x opacite
        out[:, top : top + ph, left : left + pw] = region * (1 - alpha) + patch.clamp(0, 1) * alpha
        box_list.append([float(v) for v in new_box])  # ajoute la boite du defaut colle
        label_list.append(inst.label)  # et sa classe
        if new_mask is not None:  # si on a un masque
            new_mask[..., top : top + ph, left : left + pw][..., solid] = inst.label  # dessine le defaut dans le masque
    result: dict[str, Any] = {"image": out, "mask": new_mask}  # resultat : image et masque mis a jour
    box_tensor = torch.tensor(box_list, dtype=torch.float32).reshape(-1, 4)  # boites en tenseur (N, 4), meme si N = 0
    if isinstance(boxes, tv_tensors.BoundingBoxes):  # si les boites d'entree etaient des BoundingBoxes torchvision
        # on renvoie le meme type : format XYXY et taille de l'image (canvas_size)
        box_tensor = tv_tensors.BoundingBoxes(box_tensor, format="XYXY", canvas_size=(height, width))
    result["boxes"] = box_tensor  # ajoute les boites au resultat
    result["labels"] = torch.tensor(label_list, dtype=torch.int64)  # ajoute les classes au resultat
    return result  # renvoie le dictionnaire


class CopyPaste:  # copy_paste applique seulement avec une probabilite p (une fois sur deux par defaut)
    """copy_paste applique avec la probabilite p ; a appeler dans __getitem__, AVANT le Compose d'augmentation.

    paste = CopyPaste(DefectBank.from_masks(train_images, train_masks), p=0.5)
    sample = paste(image, mask=mask)          # segmentation
    sample = paste(image, boxes=b, labels=l)  # detection
    """

    # bank : banque de defauts ; p : probabilite ; **options : autres reglages transmis a copy_paste (count, scale...)
    def __init__(self, bank: DefectBank, p: float = 0.5, **options: Any) -> None:
        self.bank, self.p, self.options = bank, float(p), options  # garde les reglages

    def __call__(self, image: torch.Tensor, **targets: Any) -> dict[str, Any]:  # targets : boxes, labels et/ou mask
        if not coin(self.p) or not len(self.bank):  # si le tirage dit "pas cette fois", ou si la banque est vide
            boxes = targets.get("boxes")  # boites d'entree (None s'il n'y en a pas)
            return {  # renvoie les entrees inchangees, dans le meme format que copy_paste
                "image": image,  # image intacte
                "mask": targets.get("mask"),  # masque intact (ou None)
                "boxes": boxes if boxes is not None else torch.zeros(0, 4),  # boites intactes, ou tableau vide (0, 4)
                # classes intactes, ou tableau vide d'entiers
                "labels": targets.get("labels") if targets.get("labels") is not None else torch.zeros(0, dtype=torch.int64),
            }
        return copy_paste(image, self.bank, **targets, **self.options)  # sinon : colle des defauts


# --- mosaique (detection) -------------------------------------------------------------------------


def mosaic(  # assemble 4 exemples en une seule image ; renvoie {image, boxes, labels, mask}
    samples: Sequence[dict[str, Any]],  # exactement 4 exemples {image, boxes, labels, mask}
    *,  # les parametres suivants doivent etre nommes a l'appel
    size: Sequence[int] | None = None,  # taille de sortie (H, W) ; None = taille de la premiere image
    center: Sequence[float] = (0.3, 0.7),  # position du point de jonction, en fraction de la taille
    min_visible: float = 0.25,  # part minimale de la surface d'une boite qui doit rester visible
    min_size: float = 2.0,  # cote minimal d'une boite apres decoupe, en pixels
) -> dict[str, Any]:  # renvoie le dictionnaire de la mosaique
    """Assemble 4 images autour d'un centre aleatoire (mosaique facon YOLO) ; boites et masques suivent.

    samples : 4 dicts {image (C, H, W), boxes (N, 4) XYXY, labels (N,), mask (H, W) optionnel}.
    Chaque image est redimensionnee a la taille de sortie puis coupee par son quadrant ; une boite
    dont il reste moins de `min_visible` de la surface est retiree (un bout de defaut n'en est pas un).
    """
    if len(samples) != 4:  # si on n'a pas exactement 4 exemples
        raise ValueError("mosaic demande exactement 4 echantillons")  # erreur claire
    first = samples[0]["image"]  # premiere image : sert de reference (canaux, type, taille par defaut)
    channels = first.shape[0]  # nombre de canaux de la mosaique
    # taille de sortie : celle demandee, sinon celle de la premiere image
    height, width = (int(size[0]), int(size[1])) if size is not None else (int(first.shape[-2]), int(first.shape[-1]))
    xc, yc = int(width * uniform(*pair(center))), int(height * uniform(*pair(center)))  # point de jonction tire au hasard
    canvas = torch.zeros(channels, height, width, dtype=first.dtype)  # toile vide de la mosaique
    with_mask = all(s.get("mask") is not None for s in samples)  # vrai si les 4 exemples ont un masque
    mask = torch.zeros(height, width, dtype=torch.int64) if with_mask else None  # masque vide de la mosaique (ou None)
    # les 4 quadrants : (x0, y0, x1, y1) de la zone de la toile, puis decalage (ox, oy) de l'image ; "# fmt: skip" : ruff ne
    # reformate pas ces deux lignes (garde la disposition en tableau)
    quadrants = [(0, 0, xc, yc, xc - width, yc - height), (xc, 0, width, yc, xc, yc - height),  # ligne du haut : haut-gauche, haut-droite
                 (0, yc, xc, height, xc - width, yc), (xc, yc, width, height, xc, yc)]  # fmt: skip
    all_boxes, all_labels = [], []  # boites et classes gardees, un morceau par quadrant
    # pour chaque exemple et son quadrant, dans l'ordre : haut-gauche, haut-droite, bas-gauche, bas-droite
    for sample, (qx0, qy0, qx1, qy1, ox, oy) in zip(samples, quadrants, strict=True):
        image = sample["image"]  # image de l'exemple
        sy, sx = height / image.shape[-2], width / image.shape[-1]  # facteurs d'echelle vers la taille de sortie
        # redimensionne l'image a la taille de sortie (interpolation bilineaire)
        resized = nnf.interpolate(image[None].float(), size=(height, width), mode="bilinear", align_corners=False)[0]
        resized = _match_channels(resized, channels).to(canvas.dtype)  # meme nombre de canaux et meme type que la toile
        # le quadrant [qx0, qx1) x [qy0, qy1) du canevas recoit la partie [qx0 - ox, qx1 - ox) de l'image
        canvas[:, qy0:qy1, qx0:qx1] = resized[:, qy0 - oy : qy1 - oy, qx0 - ox : qx1 - ox]
        if mask is not None:  # si on construit un masque
            m = sample["mask"].as_subclass(torch.Tensor).reshape(1, 1, *image.shape[-2:]).float()  # masque en forme (1, 1, H, W)
            # redimensionne au plus proche voisin (garde des numeros de classe entiers), puis revient en (H, W) entiers
            m = nnf.interpolate(m, size=(height, width), mode="nearest")[0, 0].long()
            mask[qy0:qy1, qx0:qx1] = m[qy0 - oy : qy1 - oy, qx0 - ox : qx1 - ox]  # meme decoupe que pour l'image
        boxes = sample.get("boxes")  # boites de l'exemple
        if boxes is None or len(boxes) == 0:  # s'il n'y en a pas
            continue  # on passe a l'exemple suivant
        # boites mises a l'echelle de la sortie (x par sx, y par sy) puis decalees dans la toile
        b = boxes.as_subclass(torch.Tensor).reshape(-1, 4).float() * torch.tensor([sx, sy, sx, sy]) + torch.tensor([ox, oy, ox, oy])
        # boites coupees aux limites du quadrant
        clipped = torch.stack([b[:, 0].clamp(qx0, qx1), b[:, 1].clamp(qy0, qy1), b[:, 2].clamp(qx0, qx1), b[:, 3].clamp(qy0, qy1)], 1)
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])  # surface d'origine de chaque boite
        visible = (clipped[:, 2] - clipped[:, 0]) * (clipped[:, 3] - clipped[:, 1])  # surface qui reste dans le quadrant
        keep = (  # vrai pour les boites a garder
            # assez visibles ET assez larges ET assez hautes
            (visible >= min_visible * area) & ((clipped[:, 2] - clipped[:, 0]) >= min_size) & ((clipped[:, 3] - clipped[:, 1]) >= min_size)
        )
        all_boxes.append(clipped[keep])  # garde les boites retenues
        # garde leurs classes (classe 0 pour toutes si l'exemple n'a pas de "labels")
        all_labels.append(torch.as_tensor(sample.get("labels", torch.zeros(len(b))), dtype=torch.int64).reshape(-1)[keep])
    boxes_out = torch.cat(all_boxes) if all_boxes else torch.zeros(0, 4)  # toutes les boites en un tenseur (ou vide)
    labels_out = torch.cat(all_labels) if all_labels else torch.zeros(0, dtype=torch.int64)  # toutes les classes (ou vide)
    return {"image": canvas, "boxes": boxes_out, "labels": labels_out, "mask": mask}  # renvoie la mosaique et ses cibles
