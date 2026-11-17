"""Operations d'augmentation ecrites pour la toolbox (torchvision v2.Transform).

Convention : l'image est un tenseur float (C, H, W) dans [0, 1], C = 1 (gris) ou 3 (RGB). Les
cibles sont des tv_tensors (BoundingBoxes, Mask) passees dans le meme appel :

    out = op(image)                    # classification
    out, boxes = op(image, boxes)      # detection
    out, mask = op(image, mask)        # segmentation

Trois genres d'operations :
- PHOTOMETRIQUES (ImageOnly) : ne touchent que l'image (lumiere, bruit, flou, compression) ; les
  boites et masques passent tels quels, ils restent justes ;
- GEOMETRIQUES (RandomRot90, SafeCrop) : deplacent l'image ET les cibles, par les fonctions
  torchvision (F.rotate, F.resized_crop) qui savent transformer boites et masques ;
- CONJOINTES (SyntheticDefect) : modifient l'image et le masque ensemble (on ajoute un defaut).

Chaque operation tire ses parametres une fois par appel (make_params), avec le generateur torch,
et ne s'applique qu'avec la probabilite p.
"""

from __future__ import annotations  # permet d'ecrire les types (ex. "X | None") sans qu'ils soient evalues au chargement

import math  # fonctions mathematiques de base (cos, sin, sqrt, radians) sur des nombres simples
from collections.abc import Sequence  # Sequence : type "liste ou tuple"
from typing import Any  # Any : "n'importe quel type"

import cv2  # OpenCV : traitement d'images (CLAHE, compression JPEG, dessin de segments)
import numpy as np  # numpy : calcul sur des tableaux de nombres (format attendu par OpenCV)
import torch  # PyTorch : tenseurs (tableaux de nombres multidimensionnels), calcul sur cpu ou gpu
import torch.nn.functional as nnf  # fonctions de reseaux de neurones : ici convolution (conv2d) et remplissage des bords (pad)

# tree_flatten : met a plat une structure imbriquee (tuple, dict...) en liste ; tree_unflatten : la reconstruit
from torch.utils._pytree import tree_flatten, tree_unflatten

# tv_tensors : tenseurs etiquetes de torchvision (Image, BoundingBoxes = boites, Mask = masque) que les transformations reconnaissent
from torchvision import tv_tensors
from torchvision.transforms import v2  # v2 : transformations torchvision ; v2.Transform est la classe de base de nos operations
from torchvision.transforms.v2 import functional as F  # F : fonctions de transformation qui savent traiter boites et masques

from . import synthetic  # synthetic.py : fabrication de defauts synthetiques (tache, morceau deplace, rayure)
from ._base import (  # petites aides communes (voir _base.py)
    choice,  # choisit un element au hasard dans une liste
    coin,  # vrai avec une probabilite donnee (pile ou face truque)
    find_boxes,  # retrouve les boites parmi les entrees
    find_image,  # retrouve l'image parmi les entrees
    from_hwc,  # tableau numpy (H, W, C) -> tenseur torch (C, H, W)
    image_size,  # (hauteur, largeur) de l'image
    is_image,  # vrai si l'entree est l'image (et pas des boites ou un masque)
    log_uniform,  # tirage au hasard en echelle logarithmique (pour des facteurs multiplicatifs)
    numpy_rng,  # generateur aleatoire numpy derive de torch (reproductible)
    pair,  # 3 -> (3, 3) ; (1, 5) -> (1, 5) : valeur fixe ou intervalle de tirage
    randint,  # entier au hasard, bornes comprises
    to_hwc,  # tenseur torch (C, H, W) -> tableau numpy (H, W, C) pour OpenCV
    uniform,  # nombre au hasard dans un intervalle
)

# --- bases ----------------------------------------------------------------------------------------


class ImageOnly(v2.Transform):  # classe de base des operations qui ne modifient que l'image (pas les boites ni les masques)
    """Operation photometrique : parametres tires une fois (sample), appliquee a l'image seulement."""

    def __init__(self, p: float = 0.5) -> None:  # p : probabilite d'appliquer l'operation (0.5 = une fois sur deux)
        super().__init__()  # initialise la partie torchvision de l'objet
        self.p = float(p)  # garde la probabilite

    def sample(self, height: int, width: int, channels: int) -> dict[str, Any]:  # tire les parametres aleatoires de l'operation
        return {}  # par defaut aucun parametre ; chaque operation fille redefinit cette methode

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:  # appelee par torchvision une fois par appel de l'operation
        if not coin(self.p):  # si le tirage dit "ne pas appliquer cette fois"
            return {"apply": False}  # on le note : transform laissera tout passer
        image = find_image(flat_inputs)  # retrouve l'image parmi les entrees (image, boites, masque...)
        # tire les parametres une seule fois, en connaissant hauteur, largeur et nombre de canaux (1 si image 2D)
        params = self.sample(int(image.shape[-2]), int(image.shape[-1]), int(image.shape[-3]) if image.ndim >= 3 else 1)
        params["apply"] = True  # on applique l'operation
        return params  # torchvision passera ces parametres a transform pour chaque entree

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:  # appelee par torchvision pour CHAQUE entree avec les memes params
        if not params["apply"] or not is_image(inpt):  # si on n'applique pas, ou si l'entree n'est pas l'image (boites, masque)
            return inpt  # on la renvoie telle quelle
        return self.apply(inpt, params)  # sinon on modifie l'image

    def apply(self, image: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:  # modifie l'image ; a ecrire dans chaque operation
        raise NotImplementedError  # erreur si une classe fille oublie de la definir


class Clamp01(v2.Transform):  # derniere etape des augmentations : remet les pixels entre 0 et 1
    """Ramene l'image dans [0, 1] apres les augmentations photometriques ; boites et masques inchanges."""

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:  # appelee pour chaque entree
        return inpt.clamp(0.0, 1.0) if is_image(inpt) else inpt  # clamp : < 0 devient 0, > 1 devient 1 ; le reste passe


def _depthwise(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:  # renvoie l'image filtree (meme forme)
    """Convolution du meme noyau 2D sur chaque canal, bords en miroir (pas de cadre sombre)."""
    # convolution : chaque pixel est remplace par une moyenne ponderee de ses voisins (les poids forment le noyau)
    channels = image.shape[-3]  # nombre de canaux (1 ou 3)
    # noyau au bon type et appareil, mis en forme (canaux, 1, kh, kw) : une copie du meme noyau par canal
    k = kernel.to(image)[None, None].expand(channels, 1, -1, -1)
    pad_h, pad_w = kernel.shape[0] // 2, kernel.shape[1] // 2  # marge a ajouter sur chaque bord : moitie du noyau
    batch = image.reshape(-1, channels, *image.shape[-2:])  # forme (N, C, H, W) exigee par conv2d (N = 1 pour une image)
    padded = nnf.pad(batch, (pad_w, pad_w, pad_h, pad_h), mode="reflect")  # agrandit l'image par effet miroir sur les bords
    # groups=channels : chaque canal est filtre seul ; la taille revient a celle d'origine grace a la marge
    return nnf.conv2d(padded, k, groups=channels).reshape(image.shape)


# renvoie deux tableaux (hauteur x largeur) : la coordonnee y et la coordonnee x de chaque pixel, entre 0 et 1
def _grid(height: int, width: int, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Coordonnees normalisees (y, x) dans [0, 1] (centres des pixels)."""
    ys = (torch.arange(height, device=like.device, dtype=like.dtype) + 0.5) / height  # 0, 1, 2... + 0.5 (centre), divise par H
    xs = (torch.arange(width, device=like.device, dtype=like.dtype) + 0.5) / width  # idem pour les colonnes
    return torch.meshgrid(ys, xs, indexing="ij")  # repete ys sur les colonnes et xs sur les lignes : deux grilles H x W


# --- geometrie ------------------------------------------------------------------------------------


class RandomRot90(v2.Transform):  # rotation aleatoire de 90, 180 ou 270 degres
    """Rotation d'un multiple de 90 degres (sans interpolation, sans coin vide) ; boites et masques suivent.

    A eviter quand la direction compte (direction de laminage d'une bande, texte, fleche).
    """

    def __init__(self, p: float = 0.5) -> None:  # p : probabilite de tourner l'image
        super().__init__()  # initialise la partie torchvision de l'objet
        self.p = float(p)  # garde la probabilite

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:  # tire le nombre de quarts de tour, une fois par appel
        return {"k": randint(1, 3) if coin(self.p) else 0}  # k = 1, 2 ou 3 quarts de tour ; 0 = pas de rotation

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:  # appelee pour chaque entree avec le meme k
        if params["k"] == 0:  # si pas de rotation cette fois
            return inpt  # entree renvoyee telle quelle
        if is_image(inpt) or isinstance(inpt, (tv_tensors.BoundingBoxes, tv_tensors.Mask)):  # image, boites ou masque
            # tourne de 90 x k degres ; expand=True : l'image s'agrandit pour tout contenir (H et W echangees si k impair)
            return F.rotate(inpt, angle=90.0 * params["k"], expand=True)
        return inpt  # autre entree (ex. etiquette de classe) : inchangee


class SafeCrop(v2.Transform):  # recadrage aleatoire qui ne coupe aucun objet
    """Recadrage aleatoire puis retour a la taille d'origine, qui GARDE tous les objets (keep_objects=True).

    area : fraction de la surface gardee ; aspect : variation du rapport largeur / hauteur autour de
    celui de l'image. Avec des boites (ou un masque), la fenetre contient toujours leur union :
    aucun defaut n'est coupe, contrairement a RandomResizedCrop. En classification, sans cible, la
    fenetre est libre : garder area >= 0,7 pour ne pas sortir le defaut de l'image.
    """

    def __init__(  # enregistre les reglages du recadrage
        self,  # l'objet lui-meme
        area: Sequence[float] = (0.5, 1.0),  # fraction de la surface gardee (tiree dans l'intervalle)
        aspect: Sequence[float] = (0.8, 1.25),  # variation du rapport largeur / hauteur
        keep_objects: bool = True,  # True : la fenetre englobe toujours tous les objets
        p: float = 0.5,  # probabilite de recadrer
    ) -> None:  # ne renvoie rien
        super().__init__()  # initialise la partie torchvision de l'objet
        self.area, self.aspect, self.keep_objects, self.p = pair(area), pair(aspect), keep_objects, float(p)  # garde les reglages

    # renvoie le rectangle (x min, y min, x max, y max) qui contient tous les objets, ou None s'il n'y en a pas
    def _union(self, flat_inputs: list[Any]) -> tuple[float, float, float, float] | None:
        boxes = find_boxes(flat_inputs)  # boites englobantes, s'il y en a (detection)
        if boxes is not None and boxes.shape[0] > 0:  # si on a au moins une boite
            # convertit au format XYXY (x gauche, y haut, x droite, y bas), en tenseur simple pour calculer
            xyxy = F.convert_bounding_box_format(boxes, new_format=tv_tensors.BoundingBoxFormat.XYXY).as_subclass(torch.Tensor)
            # union : le plus petit x gauche, le plus petit y haut, le plus grand x droite, le plus grand y bas
            return float(xyxy[:, 0].min()), float(xyxy[:, 1].min()), float(xyxy[:, 2].max()), float(xyxy[:, 3].max())
        for inpt in flat_inputs:  # sinon on cherche un masque (segmentation)
            if isinstance(inpt, tv_tensors.Mask):  # si l'entree est un masque
                # pixels non nuls du masque (tous canaux confondus) : leurs coordonnees y et x
                ys, xs = torch.nonzero(inpt.as_subclass(torch.Tensor).reshape(-1, *inpt.shape[-2:]).amax(0) > 0, as_tuple=True)
                if len(ys):  # si le masque contient au moins un pixel d'objet
                    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)  # rectangle qui les contient
        return None  # ni boite ni objet dans le masque : fenetre libre

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:  # tire la fenetre de recadrage, une fois par appel
        height, width = image_size(flat_inputs)  # taille de l'image
        if not coin(self.p):  # si le tirage dit "pas de recadrage cette fois"
            return {"crop": None, "size": [height, width]}  # crop None : transform laissera tout passer
        area = uniform(*self.area) * height * width  # surface de la fenetre en pixels
        ratio = log_uniform(*self.aspect) * width / height  # rapport largeur / hauteur de la fenetre
        w = min(float(width), math.sqrt(area * ratio))  # largeur deduite de la surface et du rapport (pas plus que l'image)
        h = min(float(height), area / max(w, 1.0))  # hauteur = surface / largeur (pas plus que l'image)
        union = self._union(flat_inputs) if self.keep_objects else None  # zone a garder entierement, si demande
        if union is not None:  # s'il y a des objets a proteger
            # agrandit la fenetre si besoin pour qu'elle soit au moins aussi grande que l'union (sans depasser l'image)
            w, h = min(float(width), max(w, union[2] - union[0])), min(float(height), max(h, union[3] - union[1]))
        w, h = max(1, int(math.ceil(w))), max(1, int(math.ceil(h)))  # arrondit au pixel superieur, au moins 1 pixel
        lo_x, hi_x, lo_y, hi_y = 0.0, float(width - w), 0.0, float(height - h)  # positions possibles du coin haut-gauche
        if union is not None:  # avec des objets, on restreint les positions pour que la fenetre contienne l'union
            lo_x, hi_x = max(0.0, union[2] - w), min(union[0], float(width - w))  # x : bord gauche <= objets et bord droit >= objets
            lo_y, hi_y = max(0.0, union[3] - h), min(union[1], float(height - h))  # y : meme raisonnement en vertical
            hi_x, hi_y = max(lo_x, hi_x), max(lo_y, hi_y)  # securite : l'intervalle ne doit jamais etre a l'envers
        left = int(min(max(round(uniform(lo_x, hi_x)), 0), width - w))  # bord gauche tire au hasard, garde dans l'image
        top = int(min(max(round(uniform(lo_y, hi_y)), 0), height - h))  # bord haut tire au hasard, garde dans l'image
        return {"crop": (top, left, h, w), "size": [height, width]}  # fenetre et taille de sortie (celle d'origine)

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:  # appelee pour chaque entree avec la meme fenetre
        if params["crop"] is None:  # si pas de recadrage cette fois
            return inpt  # entree renvoyee telle quelle
        if is_image(inpt) or isinstance(inpt, (tv_tensors.BoundingBoxes, tv_tensors.Mask)):  # image, boites ou masque
            top, left, h, w = params["crop"]  # fenetre tiree
            # decoupe la fenetre puis la remet a la taille d'origine ; boites et masque sont deplaces de la meme facon
            return F.resized_crop(inpt, top, left, h, w, size=params["size"], antialias=True)
        return inpt  # autre entree : inchangee


# --- photometrie ----------------------------------------------------------------------------------


class RandomGamma(ImageOnly):  # correction gamma aleatoire : change les tons moyens sans toucher au noir ni au blanc
    """Gamma aleatoire tire en echelle logarithmique dans [gamma_min, gamma_max] (< 1 eclaircit les sombres)."""

    def __init__(self, gamma: Sequence[float] = (0.8, 1.25), p: float = 0.5) -> None:  # gamma : intervalle de tirage
        super().__init__(p)  # la classe de base garde p
        self.gamma = pair(gamma)  # (min, max) du gamma

    def sample(self, height, width, channels):  # tire le gamma de cet appel
        return {"gamma": log_uniform(*self.gamma)}  # echelle log : 0.8 et 1.25 (inverses) aussi probables

    def apply(self, image, params):  # applique la correction
        return image.clamp(0.0, 1.0) ** params["gamma"]  # pixel puissance gamma (apres clamp : pas de puissance d'un negatif)


class RandomHueSaturation(ImageOnly):  # variation aleatoire des couleurs
    """Teinte et saturation (images couleur seulement ; une image grise passe telle quelle)."""

    # hue : decalage de teinte maximal ; saturation : variation maximale de la saturation (vivacite des couleurs)
    def __init__(self, hue: float = 0.05, saturation: float = 0.3, p: float = 0.5) -> None:
        super().__init__(p)  # la classe de base garde p
        self.hue, self.saturation = float(hue), float(saturation)  # garde les reglages

    def sample(self, height, width, channels):  # tire les deux parametres de cet appel
        # decalage de teinte dans [-hue, hue] et facteur de saturation dans [1 - saturation, 1 + saturation]
        return {"hue": uniform(-self.hue, self.hue), "saturation": uniform(1 - self.saturation, 1 + self.saturation)}

    def apply(self, image, params):  # applique les changements de couleur
        if image.shape[-3] != 3:  # si l'image n'a pas 3 canaux (image grise)
            return image  # rien a faire : pas de couleur
        # change d'abord la teinte puis la saturation (pixels ramenes dans [0, 1] avant)
        return F.adjust_saturation(F.adjust_hue(image.clamp(0, 1), params["hue"]), params["saturation"])


class RandomSharpness(ImageOnly):  # nettete aleatoire
    """Nettete : facteur < 1 adoucit, > 1 accentue (tire en echelle logarithmique)."""

    def __init__(self, factor: Sequence[float] = (0.5, 2.0), p: float = 0.3) -> None:  # factor : intervalle du facteur
        super().__init__(p)  # la classe de base garde p
        self.factor = pair(factor)  # (min, max) du facteur

    def sample(self, height, width, channels):  # tire le facteur de cet appel
        return {"factor": log_uniform(*self.factor)}  # echelle log : 0.5 et 2 aussi probables

    def apply(self, image, params):  # applique le changement de nettete
        return F.adjust_sharpness(image.clamp(0, 1), params["factor"])  # fonction torchvision (1 = image inchangee)


class RandomCLAHE(ImageOnly):  # egalisation d'histogramme locale aleatoire
    """Egalisation d'histogramme locale (CLAHE d'OpenCV, en 16 bits pour ne pas perdre de dynamique).

    En couleur, seule la luminance (L de Lab) est egalisee : les teintes ne bougent pas.
    """

    # clip_limit : limite du renforcement de contraste ; tile_grid : nombre de tuiles par cote
    def __init__(self, clip_limit: Sequence[float] = (1.0, 4.0), tile_grid: int = 8, p: float = 0.3) -> None:
        super().__init__(p)  # la classe de base garde p
        self.clip_limit, self.tile_grid = pair(clip_limit), int(tile_grid)  # garde les reglages

    def sample(self, height, width, channels):  # tire la limite de cet appel
        return {"clip": uniform(*self.clip_limit)}  # plus grand = contraste local plus fort

    def apply(self, image, params):  # applique CLAHE
        # cree l'outil CLAHE d'OpenCV : l'image est decoupee en 8 x 8 tuiles egalisees chacune
        clahe = cv2.createCLAHE(clipLimit=params["clip"], tileGridSize=(self.tile_grid, self.tile_grid))
        array = np.clip(to_hwc(image), 0, 1)  # tableau numpy (H, W) ou (H, W, 3) dans [0, 1]
        if array.ndim == 2:  # si image grise
            # passe en entiers 16 bits (0-65535), egalise, puis revient en decimal dans [0, 1]
            out = clahe.apply((array * 65535 + 0.5).astype(np.uint16)).astype(np.float32) / 65535
        else:  # sinon image couleur
            lab = cv2.cvtColor(array, cv2.COLOR_RGB2Lab)  # espace Lab : L = luminosite (0-100), a et b = couleur
            light = (lab[..., 0] / 100 * 65535 + 0.5).astype(np.uint16)  # luminosite en entiers 16 bits
            lab[..., 0] = clahe.apply(light).astype(np.float32) / 65535 * 100  # egalise la luminosite seule, remise en 0-100
            out = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)  # revient en RGB : les couleurs n'ont pas bouge
        return from_hwc(out, image)  # tenseur torch de meme forme et meme type que l'image d'entree


# --- capteur et eclairage -------------------------------------------------------------------------


class IlluminationGradient(ImageOnly):  # eclairage non uniforme : l'image est multipliee par un gain qui varie dans l'espace
    """Eclairage non uniforme : gain lineaire (rampe) ou radial (point chaud / coins sombres) multiplie a l'image.

    strength : amplitude du gain (0,3 -> le gain varie de 0,7 a 1,3 sur l'image) ; mode : "linear",
    "radial" ou "both" (tire au hasard) ; angle : direction de la rampe en degres (0 = gauche-droite ;
    (-10, 10) pour une rampe en travers d'une bande qui defile verticalement). Le sens est tire au hasard.
    C'est le defaut d'acquisition le plus frequent en ligne : le modele doit l'ignorer.
    """

    def __init__(  # enregistre les reglages
        self,  # l'objet lui-meme
        strength: Sequence[float] = (0.1, 0.4),  # intervalle de l'amplitude du gain
        mode: str = "both",  # "linear", "radial" ou "both"
        angle: Sequence[float] = (0.0, 360.0),  # intervalle de la direction de la rampe, en degres
        p: float = 0.5,  # probabilite d'appliquer l'operation
    ) -> None:  # ne renvoie rien
        super().__init__(p)  # la classe de base garde p
        if mode not in ("linear", "radial", "both"):  # si le mode est inconnu (faute de frappe)
            raise ValueError(f"mode inconnu : {mode} (linear, radial, both)")  # erreur claire des la construction
        self.strength, self.mode, self.angle = pair(strength), mode, pair(angle)  # garde les reglages

    def sample(self, height, width, channels):  # tire les parametres de cet appel
        mode = self.mode if self.mode != "both" else choice(["linear", "radial"])  # "both" : on tire l'un des deux modes
        return {  # parametres tires
            "mode": mode,  # rampe ou radial
            "strength": uniform(*self.strength),  # amplitude du gain
            "theta": math.radians(uniform(*self.angle)),  # direction de la rampe, convertie en radians pour cos et sin
            "sign": choice([-1.0, 1.0]),  # sens : plus clair d'un cote ou de l'autre (ou centre plus clair / plus sombre)
            "center": (uniform(0.0, 1.0), uniform(0.0, 1.0)),  # centre du mode radial (x, y), n'importe ou dans l'image
        }

    def apply(self, image, params):  # applique le gain a l'image
        yy, xx = _grid(image.shape[-2], image.shape[-1], image)  # coordonnees y et x de chaque pixel, entre 0 et 1
        s = params["strength"]  # amplitude du gain
        if params["mode"] == "linear":  # si rampe lineaire
            # position de chaque pixel le long de la direction theta, par rapport au centre de l'image
            ramp = (xx - 0.5) * math.cos(params["theta"]) + (yy - 0.5) * math.sin(params["theta"])
            # rampe ramenee entre -1 et 1, puis gain = 1 +/- s (division protegee contre 0)
            gain = 1.0 + params["sign"] * s * ramp / max(float(ramp.abs().max()), 1e-6)
        else:  # sinon mode radial
            cy, cx = params["center"][1], params["center"][0]  # centre du point chaud (y, x)
            r2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / 2.0  # distance au carre, ~[0, 1]
            gain = 1.0 + params["sign"] * s * (1.0 - 2.0 * r2.clamp(0, 1))  # gain maximal au centre, decroissant avec la distance
        return image * gain  # multiplie chaque pixel par son gain


class Vignetting(ImageOnly):  # vignettage : bords de l'image plus sombres que le centre
    """Vignettage : assombrissement radial des bords (objectif, eclairage annulaire), centre legerement deplace."""

    # strength : assombrissement maximal dans les coins ; center_jitter : deplacement maximal du centre
    def __init__(self, strength: Sequence[float] = (0.2, 0.5), center_jitter: float = 0.1, p: float = 0.3) -> None:
        super().__init__(p)  # la classe de base garde p
        self.strength, self.center_jitter = pair(strength), float(center_jitter)  # garde les reglages

    def sample(self, height, width, channels):  # tire les parametres de cet appel
        j = self.center_jitter  # deplacement maximal du centre
        # force du vignettage et centre (x, y) autour du milieu de l'image (0.5, 0.5)
        return {"strength": uniform(*self.strength), "center": (0.5 + uniform(-j, j), 0.5 + uniform(-j, j))}

    def apply(self, image, params):  # assombrit les bords
        yy, xx = _grid(image.shape[-2], image.shape[-1], image)  # coordonnees y et x de chaque pixel, entre 0 et 1
        cx, cy = params["center"]  # centre du vignettage
        r = (((xx - cx) ** 2 + (yy - cy) ** 2) / 0.5).sqrt().clamp(0, 1)  # distance au centre, ~1 dans les coins
        return image * (1.0 - params["strength"] * r**2)  # gain 1 au centre, 1 - strength dans les coins


class Exposure(ImageOnly):  # exposition aleatoire : image plus claire ou plus sombre
    """Gain et decalage (temps d'exposition, gain camera, puissance d'eclairage) : x * gain + offset."""

    # gain : facteur multiplicatif ; offset : valeur ajoutee a chaque pixel
    def __init__(self, gain: Sequence[float] = (0.75, 1.3), offset: Sequence[float] = (-0.05, 0.05), p: float = 0.5) -> None:
        super().__init__(p)  # la classe de base garde p
        self.gain, self.offset = pair(gain), pair(offset)  # garde les reglages

    def sample(self, height, width, channels):  # tire les parametres de cet appel
        return {"gain": log_uniform(*self.gain), "offset": uniform(*self.offset)}  # gain en echelle log, decalage uniforme

    def apply(self, image, params):  # applique l'exposition
        return image * params["gain"] + params["offset"]  # pixel x gain + decalage


class LineStripes(ImageOnly):  # stries de camera lineaire : chaque colonne a son propre gain
    """Stries d'une camera lineaire : chaque colonne (ou ligne) a son propre gain (non-uniformite des pixels).

    strength : ecart-type relatif du gain par colonne ; axis : "columns" (stries verticales, camera
    lineaire horizontale sur une bande qui defile verticalement) ou "rows".
    """

    # strength : intervalle de l'ecart-type du gain ; axis : "columns" ou "rows"
    def __init__(self, strength: Sequence[float] = (0.01, 0.05), axis: str = "columns", p: float = 0.3) -> None:
        super().__init__(p)  # la classe de base garde p
        if axis not in ("columns", "rows"):  # si l'axe est inconnu
            raise ValueError(f"axis inconnu : {axis} (columns, rows)")  # erreur claire des la construction
        self.strength, self.axis = pair(strength), axis  # garde les reglages

    def sample(self, height, width, channels):  # tire un gain par colonne (ou par ligne)
        n = width if self.axis == "columns" else height  # nombre de gains a tirer
        # gain = 1 + ecart-type x bruit gaussien (torch.randn : nombres centres sur 0, ecart-type 1)
        return {"gain": 1.0 + uniform(*self.strength) * torch.randn(n)}

    def apply(self, image, params):  # applique les gains
        gain = params["gain"].to(image)  # gains au meme type et sur le meme appareil que l'image
        # colonnes : gain de forme (1, W) repete sur toutes les lignes ; lignes : forme (H, 1) repete sur les colonnes
        return image * (gain[None, :] if self.axis == "columns" else gain[:, None])


class RandomGaussianNoise(ImageOnly):  # bruit gaussien additif : petit nombre aleatoire ajoute a chaque pixel
    """Bruit gaussien additif (bruit de lecture du capteur) ; sigma fixe ou intervalle de tirage."""

    def __init__(self, sigma: float | Sequence[float] = 0.02, p: float = 0.5) -> None:  # sigma : ecart-type du bruit
        super().__init__(p)  # la classe de base garde p
        self.sigma = pair(sigma)  # (min, max) de l'ecart-type

    def sample(self, height, width, channels):  # tire l'ecart-type de cet appel
        return {"sigma": uniform(*self.sigma)}  # tirage uniforme dans l'intervalle

    def apply(self, image, params):  # ajoute le bruit
        return image + params["sigma"] * torch.randn_like(image)  # randn_like : bruit gaussien de meme forme que l'image


class PoissonNoise(ImageOnly):  # bruit de photons : la lumiere arrive par grains, le compte varie au hasard
    """Bruit de photons (Poisson) : plus fort dans les zones claires ; peak = nombre de photons pour le blanc.

    Petit peak (30) : image tres bruitee (faible lumiere, temps d'exposition court).
    """

    def __init__(self, peak: Sequence[float] = (30.0, 300.0), p: float = 0.3) -> None:  # peak : photons pour un pixel blanc
        super().__init__(p)  # la classe de base garde p
        self.peak = pair(peak)  # (min, max) du nombre de photons

    def sample(self, height, width, channels):  # tire le nombre de photons de cet appel
        return {"peak": log_uniform(*self.peak)}  # echelle log : 30, 100 et 300 couvrent bien la plage

    def apply(self, image, params):  # applique le bruit
        peak = params["peak"]  # nombre de photons pour le blanc
        # nombre moyen de photons par pixel -> tirage de Poisson (compte aleatoire) -> retour dans [0, 1]
        return torch.poisson(image.clamp(0, 1) * peak) / peak


class SpeckleNoise(ImageOnly):  # bruit multiplicatif : plus fort sur les pixels clairs
    """Bruit multiplicatif (granularite) : x * (1 + sigma * n)."""

    def __init__(self, sigma: Sequence[float] = (0.02, 0.08), p: float = 0.3) -> None:  # sigma : ecart-type relatif du bruit
        super().__init__(p)  # la classe de base garde p
        self.sigma = pair(sigma)  # (min, max) de l'ecart-type

    def sample(self, height, width, channels):  # tire l'ecart-type de cet appel
        return {"sigma": uniform(*self.sigma)}  # tirage uniforme dans l'intervalle

    def apply(self, image, params):  # applique le bruit
        return image * (1.0 + params["sigma"] * torch.randn_like(image))  # chaque pixel multiplie par 1 + petit bruit


class SaltPepper(ImageOnly):  # pixels morts (noirs) et chauds (blancs) isoles
    """Pixels morts et chauds : une fraction `amount` des pixels mise a 0 ou 1."""

    def __init__(self, amount: Sequence[float] = (0.0005, 0.005), p: float = 0.2) -> None:  # amount : fraction de pixels touches
        super().__init__(p)  # la classe de base garde p
        self.amount = pair(amount)  # (min, max) de la fraction

    def sample(self, height, width, channels):  # tire la fraction de cet appel
        return {"amount": uniform(*self.amount)}  # tirage uniforme dans l'intervalle

    def apply(self, image, params):  # place les pixels morts et chauds
        draw = torch.rand(image.shape[-2:], device=image.device)  # un nombre entre 0 et 1 par position de pixel (H, W)
        out = image.clone()  # copie : on ne modifie pas l'image d'origine
        a = params["amount"]  # fraction de pixels a toucher
        out[..., draw < a / 2] = 0.0  # la moitie de la fraction (tirages les plus bas) : pixels noirs, sur tous les canaux
        out[..., draw > 1 - a / 2] = 1.0  # l'autre moitie (tirages les plus hauts) : pixels blancs
        return out  # renvoie l'image modifiee


def motion_kernel(length: int, angle_deg: float) -> torch.Tensor:  # renvoie le noyau (petite image de poids) du flou de mouvement
    """Noyau de flou de mouvement : un segment de `length` pixels a `angle_deg` (0 = horizontal, 90 = vertical)."""
    length = max(1, int(length))  # longueur entiere d'au moins 1 pixel
    size = length if length % 2 == 1 else length + 1  # taille impaire : le noyau a un pixel central
    canvas = np.zeros((size * 4, size * 4), np.float32)  # dessine a 4x puis reduit : segment anti-crenele
    c = size * 2  # centre de la toile agrandie
    # demi-segment en x et en y a l'echelle 4x (longueur totale 4 x length, soit length apres reduction)
    dx, dy = math.cos(math.radians(angle_deg)) * length * 2, math.sin(math.radians(angle_deg)) * length * 2
    # trace le segment centre (valeur 1, epaisseur 4 = 1 pixel apres reduction, LINE_AA = bords adoucis)
    cv2.line(canvas, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1.0, 4, cv2.LINE_AA)
    kernel = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)  # reduit a la taille finale (moyenne des pixels)
    return torch.from_numpy(kernel / max(kernel.sum(), 1e-6))  # poids divises par leur somme : la luminosite moyenne ne change pas


class MotionBlur(ImageOnly):  # flou de mouvement : trainee dans une direction
    """Flou de mouvement (piece ou bande qui defile pendant l'exposition) ; angle en degres, 90 = vertical."""

    # length : longueur de la trainee en pixels ; angle : direction en degres
    def __init__(self, length: Sequence[int] = (3, 9), angle: Sequence[float] = (0.0, 180.0), p: float = 0.3) -> None:
        super().__init__(p)  # la classe de base garde p
        self.length, self.angle = pair(length), pair(angle)  # garde les reglages

    def sample(self, height, width, channels):  # tire longueur et angle de cet appel
        return {"length": randint(int(self.length[0]), int(self.length[1])), "angle": uniform(*self.angle)}  # longueur entiere

    def apply(self, image, params):  # applique le flou
        return _depthwise(image, motion_kernel(params["length"], params["angle"]))  # convolution par le segment, canal par canal


class DefocusBlur(ImageOnly):  # flou de mise au point
    """Flou de mise au point : noyau en disque (plus realiste qu'une gaussienne pour une optique)."""

    def __init__(self, radius: Sequence[float] = (1.0, 3.0), p: float = 0.2) -> None:  # radius : rayon du disque en pixels
        super().__init__(p)  # la classe de base garde p
        self.radius = pair(radius)  # (min, max) du rayon

    def sample(self, height, width, channels):  # tire le rayon de cet appel
        return {"radius": uniform(*self.radius)}  # tirage uniforme dans l'intervalle

    def apply(self, image, params):  # applique le flou
        r = params["radius"]  # rayon du disque
        half = int(math.ceil(r))  # demi-taille du noyau (arrondie au-dessus)
        yy, xx = np.mgrid[-half : half + 1, -half : half + 1].astype(np.float32)  # coordonnees de chaque case du noyau
        disk = np.clip(r + 0.5 - np.sqrt(xx**2 + yy**2), 0, 1)  # bord anti-crenele
        return _depthwise(image, torch.from_numpy(disk / disk.sum()))  # convolution par le disque normalise (somme des poids = 1)


class JpegCompression(ImageOnly):  # artefacts de compression JPEG (blocs, halos autour des contours)
    """Artefacts de compression JPEG (images archivees, cameras IP) ; quality 100 = sans perte visible."""

    def __init__(self, quality: Sequence[int] = (30, 90), p: float = 0.3) -> None:  # quality : intervalle de la qualite JPEG
        super().__init__(p)  # la classe de base garde p
        self.quality = pair(quality)  # (min, max) de la qualite

    def sample(self, height, width, channels):  # tire la qualite de cet appel
        return {"quality": randint(int(self.quality[0]), int(self.quality[1]))}  # qualite entiere

    def apply(self, image, params):  # compresse puis decompresse l'image
        array = (np.clip(to_hwc(image), 0, 1) * 255 + 0.5).astype(np.uint8)  # image en entiers 8 bits (0-255), format OpenCV
        color = array.ndim == 3  # vrai si image couleur
        ok, encoded = cv2.imencode(  # encode en JPEG en memoire ; ok dit si l'encodage a reussi
            ".jpg",  # format de compression voulu
            cv2.cvtColor(array, cv2.COLOR_RGB2BGR) if color else array,  # image, en BGR (ordre des couleurs d'OpenCV) si couleur
            [cv2.IMWRITE_JPEG_QUALITY, params["quality"]],  # reglage : qualite JPEG tiree
        )
        if not ok:  # si l'encodage a echoue
            return image  # on renvoie l'image intacte
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)  # decode : l'image porte les artefacts
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB) if color else decoded  # remet les couleurs en ordre RGB
        return from_hwc(decoded.astype(np.float32) / 255.0, image)  # retour en tenseur dans [0, 1], meme forme que l'entree


class LowResolution(ImageOnly):  # basse resolution : les details fins disparaissent
    """Basse resolution : reduction (moyenne) puis agrandissement (bilineaire ou plus proche voisin)."""

    def __init__(self, scale: Sequence[float] = (0.3, 0.7), p: float = 0.2) -> None:  # scale : facteur de reduction
        super().__init__(p)  # la classe de base garde p
        self.scale = pair(scale)  # (min, max) du facteur

    def sample(self, height, width, channels):  # tire la taille reduite et le mode d'agrandissement
        s = uniform(*self.scale)  # facteur de reduction
        # taille reduite (au moins 2 x 2) ; nearest vrai 3 fois sur 10 : agrandissement "pixelise"
        return {"small": [max(2, int(height * s)), max(2, int(width * s))], "nearest": coin(0.3)}

    def apply(self, image, params):  # reduit puis agrandit
        size = list(image.shape[-2:])  # taille d'origine (H, W)
        small = F.resize(image, params["small"], antialias=True)  # reduction avec lissage (moyenne des pixels)
        # mode d'agrandissement : plus proche voisin (gros pixels) ou bilineaire (interpolation douce)
        mode = v2.InterpolationMode.NEAREST if params["nearest"] else v2.InterpolationMode.BILINEAR
        return F.resize(small, size, interpolation=mode, antialias=False)  # agrandit a la taille d'origine


class SpecularHighlight(ImageOnly):  # reflet speculaire : tache claire allongee sur metal poli
    """Reflet speculaire : une tache claire allongee (surface metallique polie sous eclairage direct).

    intensity : luminance ajoutee au centre ; size : ecart-type du reflet en fraction du petit cote ;
    elongation : rapport des deux axes (1 = rond, 4 = trainee).
    """

    def __init__(  # enregistre les reglages
        self,  # l'objet lui-meme
        intensity: Sequence[float] = (0.2, 0.6),  # luminosite ajoutee au centre
        size: Sequence[float] = (0.05, 0.25),  # taille du reflet en fraction du petit cote
        elongation: Sequence[float] = (1.0, 4.0),  # allongement (1 = rond)
        p: float = 0.2,  # probabilite d'appliquer l'operation
    ) -> None:  # ne renvoie rien
        super().__init__(p)  # la classe de base garde p
        self.intensity, self.size, self.elongation = pair(intensity), pair(size), pair(elongation)  # garde les reglages

    def sample(self, height, width, channels):  # tire les parametres du reflet
        return {  # parametres tires
            "intensity": uniform(*self.intensity),  # luminosite au centre
            "sigma": uniform(*self.size),  # taille (ecart-type de la tache gaussienne)
            "elongation": uniform(*self.elongation),  # allongement
            "theta": uniform(0.0, math.pi),  # orientation du grand axe, en radians (0 a 180 degres)
            "center": (uniform(0.1, 0.9), uniform(0.1, 0.9)),  # centre (x, y), pas trop pres des bords
        }

    def apply(self, image, params):  # ajoute le reflet
        height, width = image.shape[-2:]  # taille de l'image
        yy, xx = _grid(height, width, image)  # coordonnees y et x de chaque pixel, entre 0 et 1
        aspect = width / height  # rapport largeur / hauteur
        # ecart au centre ; x multiplie par aspect pour mesurer x et y dans la meme unite (la tache reste ronde)
        dx, dy = (xx - params["center"][0]) * aspect, yy - params["center"][1]
        c, s = math.cos(params["theta"]), math.sin(params["theta"])  # cosinus et sinus de l'orientation
        u, v = dx * c + dy * s, -dx * s + dy * c  # coordonnees tournees : u le long du grand axe, v en travers
        sigma = params["sigma"] * min(1.0, aspect)  # taille rapportee au petit cote de l'image
        # tache gaussienne : 1 au centre, decroissante ; plus etiree le long de u (sigma x elongation)
        blob = torch.exp(-0.5 * ((u / (sigma * params["elongation"])) ** 2 + (v / sigma) ** 2))
        return image + params["intensity"] * blob  # ajoute la tache claire a l'image


class DustSpots(ImageOnly):  # poussieres : petits disques flous plus sombres ou plus clairs
    """Poussieres et taches sur l'optique ou la piece : petits disques flous plus sombres (ou plus clairs).

    ATTENTION : une tache peut ressembler a un defaut. A n'utiliser que si les poussieres ne sont PAS
    une classe de defaut, et jamais pour apprendre le "normal" en detection d'anomalies.
    """

    def __init__(  # enregistre les reglages
        self,  # l'objet lui-meme
        count: Sequence[int] = (1, 8),  # nombre de taches
        radius: Sequence[float] = (1.0, 4.0),  # rayon d'une tache en pixels
        contrast: Sequence[float] = (0.1, 0.4),  # variation de luminosite au centre d'une tache
        p: float = 0.2,  # probabilite d'appliquer l'operation
    ) -> None:  # ne renvoie rien
        super().__init__(p)  # la classe de base garde p
        self.count, self.radius, self.contrast = pair(count), pair(radius), pair(contrast)  # garde les reglages

    def sample(self, height, width, channels):  # tire la position, la taille et le contraste de chaque tache
        spots = []  # une entree (x, y, rayon, contraste signe) par tache
        for _ in range(randint(int(self.count[0]), int(self.count[1]))):  # pour chaque tache (nombre tire au hasard)
            sign = -1.0 if coin(0.75) else 1.0  # la plupart des poussieres sont sombres
            # position (x, y) n'importe ou dans l'image, rayon, et contraste (negatif = tache sombre)
            spots.append((uniform(0, width), uniform(0, height), uniform(*self.radius), sign * uniform(*self.contrast)))
        return {"spots": spots}  # liste des taches tirees

    def apply(self, image, params):  # dessine les taches
        height, width = image.shape[-2:]  # taille de l'image
        ys = torch.arange(height, device=image.device, dtype=image.dtype)[:, None]  # numeros de ligne en colonne (H, 1)
        xs = torch.arange(width, device=image.device, dtype=image.dtype)[None, :]  # numeros de colonne en ligne (1, W)
        delta = torch.zeros(height, width, device=image.device, dtype=image.dtype)  # variation de luminosite, nulle au depart
        for x, y, r, amount in params["spots"]:  # pour chaque tache
            delta += amount * torch.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * r * r))  # ajoute une tache gaussienne floue
        return image + delta  # applique les taches a tous les canaux


# --- occlusion ------------------------------------------------------------------------------------


# renvoie la valeur de remplissage d'un trou : moyenne de l'image, bruit, ou nombre fixe
def _fill_value(image: torch.Tensor, fill: Any, shape: Sequence[int]) -> torch.Tensor | float:
    if fill == "mean":  # si on remplit avec la moyenne
        return image.mean(dim=(-2, -1), keepdim=True)  # moyenne de chaque canal, forme (C, 1, 1) qui s'etend au trou
    if fill == "noise":  # si on remplit avec du bruit
        return torch.rand(*shape, device=image.device, dtype=image.dtype)  # valeurs au hasard entre 0 et 1, de la forme du trou
    return float(fill)  # sinon : nombre fixe (ex. 0.0 = noir)


class CoarseDropout(ImageOnly):  # plusieurs rectangles caches au hasard
    """Plusieurs rectangles masques (Cutout generalise) : force le modele a regarder toute l'image.

    holes : nombre de trous ; size : cote en fraction de l'image ; fill : 0.0, "mean" ou "noise".
    Les cibles ne changent pas : en detection un trou peut cacher un petit objet qui reste annote,
    d'ou des trous petits et une probabilite faible hors classification.
    """

    def __init__(  # enregistre les reglages
        self,  # l'objet lui-meme
        holes: Sequence[int] = (1, 6),  # nombre de trous
        size: Sequence[float] = (0.05, 0.15),  # cote d'un trou en fraction de l'image
        fill: Any = "mean",  # remplissage : nombre, "mean" ou "noise"
        p: float = 0.3,  # probabilite d'appliquer l'operation
    ) -> None:  # ne renvoie rien
        super().__init__(p)  # la classe de base garde p
        self.holes, self.size, self.fill = pair(holes), pair(size), fill  # garde les reglages

    def sample(self, height, width, channels):  # tire la position et la taille de chaque trou
        rects = []  # une entree (haut, gauche, hauteur, largeur) par trou
        for _ in range(randint(int(self.holes[0]), int(self.holes[1]))):  # pour chaque trou (nombre tire au hasard)
            h, w = max(1, int(uniform(*self.size) * height)), max(1, int(uniform(*self.size) * width))  # taille, au moins 1 pixel
            rects.append((randint(0, height - h), randint(0, width - w), h, w))  # position tiree pour que le trou reste dans l'image
        return {"rects": rects}  # liste des trous tires

    def apply(self, image, params):  # remplit les trous
        out = image.clone()  # copie : on ne modifie pas l'image d'origine
        for top, left, h, w in params["rects"]:  # pour chaque trou
            # remplace le rectangle (tous canaux) par la valeur de remplissage
            out[..., top : top + h, left : left + w] = _fill_value(image, self.fill, (*image.shape[:-2], h, w))
        return out  # renvoie l'image trouee


class GridMask(ImageOnly):  # grille reguliere de carres caches
    """Grille reguliere de carres masques (GridMask) ; spacing : periode en fraction du petit cote ; ratio : part masquee."""

    # spacing : pas de la grille ; ratio : part de la surface masquee ; fill : remplissage
    def __init__(self, spacing: Sequence[float] = (0.15, 0.35), ratio: float = 0.5, fill: Any = 0.0, p: float = 0.2) -> None:
        super().__init__(p)  # la classe de base garde p
        self.spacing, self.ratio, self.fill = pair(spacing), float(ratio), fill  # garde les reglages

    def sample(self, height, width, channels):  # tire le pas et le decalage de la grille
        period = max(4, int(uniform(*self.spacing) * min(height, width)))  # pas de la grille en pixels (au moins 4)
        return {"period": period, "offset": (randint(0, period - 1), randint(0, period - 1))}  # decalage (y, x) de la grille

    def apply(self, image, params):  # cache les carres de la grille
        height, width = image.shape[-2:]  # taille de l'image
        # cote d'un carre cache : racine du ratio x pas (un carre couvre ratio de la surface d'une cellule)
        period, side = params["period"], max(1, int(params["period"] * math.sqrt(self.ratio)))
        ys = (torch.arange(height, device=image.device) + params["offset"][0]) % period < side  # lignes dans une bande cachee
        xs = (torch.arange(width, device=image.device) + params["offset"][1]) % period < side  # colonnes dans une bande cachee
        hole = ys[:, None] & xs[None, :]  # pixel cache si sa ligne ET sa colonne sont dans une bande : des carres
        # torch.where : valeur de remplissage la ou hole est vrai, pixel d'origine ailleurs
        return torch.where(hole, torch.as_tensor(_fill_value(image, self.fill, image.shape), dtype=image.dtype), image)


# --- defauts synthetiques (image + masque ensemble) -----------------------------------------------

# nom du type de defaut -> fonction qui le fabrique (voir synthetic.py)
SYNTHETIC_KINDS = {"perlin": synthetic.perlin_anomaly, "cutpaste": synthetic.cutpaste, "scratch": synthetic.synthetic_scratch}


# ajoute un defaut a une image saine ; renvoie (image modifiee, masque du defaut)
def synthesize_defect(image: torch.Tensor, kind: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Ajoute un defaut synthetique a une image saine (C, H, W) : renvoie (image, masque uint8 (H, W)).

    kind : "perlin" (tache organique, facon DRAEM), "cutpaste" (morceau deplace), "scratch" (rayure),
    ou None (tire au hasard). Pour l'auto-supervision (apprendre a segmenter des defauts simules sur
    des images saines) ; la performance finale se mesure toujours sur de vrais defauts.
    """
    kind = kind or choice(list(SYNTHETIC_KINDS))  # type demande, ou tire au hasard si aucun
    if kind not in SYNTHETIC_KINDS:  # si le type est inconnu
        raise ValueError(f"defaut synthetique inconnu : {kind} ; disponibles : {sorted(SYNTHETIC_KINDS)}")  # erreur claire
    out, mask = SYNTHETIC_KINDS[kind](to_hwc(image), rng=numpy_rng())  # fabrique le defaut sur la version numpy de l'image
    # image ramenee dans [0, 1] et reconvertie en tenseur ; masque numpy converti en tenseur
    return from_hwc(np.clip(out, 0, 1).astype(np.float32), image), torch.from_numpy(mask)


class SyntheticDefect(v2.Transform):  # ajoute un defaut dans l'image ET le dessine dans le masque (segmentation)
    """Colle un defaut synthetique dans l'image ET l'ecrit dans le masque (classe `class_index`).

    Change la verite terrain : ce n'est pas une augmentation "invariante" mais un generateur de
    defauts, reserve a la segmentation (le masque dit ou est le defaut ajoute).
    """

    # kinds : types de defauts possibles ; class_index : numero de classe ecrit dans le masque ; p : probabilite
    def __init__(self, kinds: Sequence[str] = ("perlin", "cutpaste", "scratch"), class_index: int = 1, p: float = 0.5) -> None:
        super().__init__()  # initialise la partie torchvision de l'objet
        unknown = set(kinds) - set(SYNTHETIC_KINDS)  # types demandes qui n'existent pas
        if unknown:  # s'il y en a
            # erreur claire des la construction
            raise ValueError(f"defauts synthetiques inconnus : {sorted(unknown)} ; disponibles : {sorted(SYNTHETIC_KINDS)}")
        self.kinds, self.class_index, self.p = list(kinds), int(class_index), float(p)  # garde les reglages

    def forward(self, *inputs: Any) -> Any:  # appelee par op(image, mask) : traite l'image et le masque ensemble
        sample = inputs if len(inputs) > 1 else inputs[0]  # plusieurs entrees : tuple ; une seule : l'entree elle-meme
        if not coin(self.p):  # si le tirage dit "pas de defaut cette fois"
            return sample  # entrees renvoyees telles quelles
        flat, spec = tree_flatten(sample)  # liste a plat des entrees + plan pour reconstruire la meme structure
        i_image = next((i for i, x in enumerate(flat) if is_image(x)), None)  # position de l'image dans la liste (None si absente)
        i_mask = next((i for i, x in enumerate(flat) if isinstance(x, tv_tensors.Mask)), None)  # position du masque
        if i_image is None or i_mask is None:  # s'il manque l'image ou le masque
            raise ValueError("synthetic_defect demande une image et un masque (tache segmentation)")  # erreur claire
        image, defect = synthesize_defect(flat[i_image], choice(self.kinds))  # ajoute un defaut d'un type tire au hasard
        mask = flat[i_mask].as_subclass(torch.Tensor).clone()  # copie du masque en tenseur simple
        mask[..., defect.bool().to(mask.device)] = self.class_index  # pixels du defaut : classe class_index dans le masque
        flat[i_image], flat[i_mask] = image, tv_tensors.Mask(mask)  # remplace image et masque dans la liste
        return tree_unflatten(flat, spec)  # reconstruit la structure d'origine (meme ordre, meme forme)
