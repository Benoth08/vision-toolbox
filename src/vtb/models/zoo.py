"""Catalogue des modeles de classification : choisir un modele par son nom dans le YAML.

    model:
      name: resnet18          # ou efficientnet_b0, convnext_tiny, mobilenet_v3_large, timm:<nom>, simple_cnn ...
      pretrained: true
      in_channels: 1          # images grises : la premiere convolution est adaptee aux poids RGB

Pour chaque modele, le catalogue connait ce qui change d'une famille a l'autre et qu'on ne veut
pas retenir : le nom de sa tete (fc, classifier.1, classifier.3, classifier.2), sa premiere
convolution, ses blocs dans l'ordre (pour degeler du haut vers le bas), sa licence. Le reste
du code ne manipule que des noms.

Poids pre-entraines : torchvision les telecharge depuis download.pytorch.org (cache dans
TORCH_HOME, par defaut ~/.cache/torch). Sur un reseau ferme, `tools/fetch_weights.py` les
recupere ailleurs, puis `weights_path` (ou TORCH_HOME) les donne au modele sans internet.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (ex. str | None)

import logging  # journal : messages horodates sur la console et dans un fichier
import time  # mesure du temps (latence)
from dataclasses import dataclass, field  # @dataclass : classe de donnees ; field : valeur par defaut creee a la demande
from typing import Any  # Any = n'importe quel type

import pandas as pd  # pandas : tableaux de donnees (ici le catalogue)
import torch  # PyTorch : tenseurs (tableaux de nombres sur CPU ou GPU) et reseaux de neurones
import torch.nn as nn  # nn : briques de reseaux de neurones (couches, modules)
from torchvision import models as tv_models  # modeles de vision de torchvision (ResNet, EfficientNet...) et leurs poids

logger = logging.getLogger(__name__)  # journal propre a ce module


@dataclass(frozen=True)  # frozen=True : une fiche ne peut plus etre modifiee apres creation
class ModelSpec:  # fiche d'une architecture : tout ce qu'il faut savoir pour la reutiliser
    """Ce qu'il faut savoir d'une architecture pour la reutiliser."""

    name: str  # nom utilise dans le YAML (ex. "resnet18")
    source: str  # torchvision | timm | custom
    builder: str  # fonction torchvision.models, nom timm ou nom de classe maison
    weights: str | None  # enum de poids torchvision, ex. "ResNet18_Weights.DEFAULT"
    head: str  # chemin de la derniere couche lineaire (remplacee pour nos classes)
    first_conv: str  # chemin de la premiere convolution (adaptee aux images 1 ou N canaux)
    blocks: tuple[str, ...]  # blocs du backbone, de l'entree vers la sortie
    input_size: int = 224  # taille d'image (cote en pixels) prevue pour ce modele
    # licence du code et des poids (a verifier avant un usage commercial)
    licence: str = "BSD-3-Clause (code torchvision) ; poids ImageNet : usage de recherche usuel, a verifier pour un produit"
    notes: str = ""  # remarque libre affichee dans le catalogue
    extra: dict[str, Any] = field(default_factory=dict)  # options supplementaires (dict neuf pour chaque fiche)


# raccourci : fabrique la fiche d'un modele torchvision ; renvoie un ModelSpec
def _tv(name: str, weights: str, head: str, first_conv: str, blocks: tuple[str, ...], notes: str = "") -> ModelSpec:
    return ModelSpec(name, "torchvision", name, weights, head, first_conv, blocks, notes=notes)  # builder = meme nom que le modele


RESNET_BLOCKS = ("layer1", "layer2", "layer3", "layer4")  # les 4 etages d'un ResNet, de l'entree vers la sortie

ZOO: dict[str, ModelSpec] = {  # le catalogue : nom -> fiche
    spec.name: spec  # cle = nom du modele, valeur = sa fiche
    for spec in [  # pour chaque fiche de la liste ci-dessous
        _tv(  # ResNet18
            "resnet18",  # nom
            "ResNet18_Weights.DEFAULT",  # poids ImageNet
            "fc",  # tete : couche "fc"
            "conv1",  # premiere convolution
            RESNET_BLOCKS,  # blocs
            "le choix par defaut : rapide, robuste, 11,7 M de parametres",  # remarque
        ),
        _tv("resnet34", "ResNet34_Weights.DEFAULT", "fc", "conv1", RESNET_BLOCKS),  # ResNet34 : plus profond
        # ResNet50 : plus gros, plus precis
        _tv("resnet50", "ResNet50_Weights.DEFAULT", "fc", "conv1", RESNET_BLOCKS, "plus precis, 2 a 3 fois plus lent que resnet18"),
        _tv(  # EfficientNet-B0
            "efficientnet_b0",  # nom
            "EfficientNet_B0_Weights.DEFAULT",  # poids ImageNet
            "classifier.1",  # tete : 2e couche du classifier (la 1re est un dropout)
            "features.0.0",  # premiere convolution
            tuple(f"features.{i}" for i in range(1, 9)),  # blocs features.1 a features.8
            "tres bon rapport precision / calcul",  # remarque
        ),
        _tv(  # MobileNetV3 small
            "mobilenet_v3_small",  # nom
            "MobileNet_V3_Small_Weights.DEFAULT",  # poids ImageNet
            "classifier.3",  # tete : 4e couche du classifier
            "features.0.0",  # premiere convolution
            tuple(f"features.{i}" for i in range(1, 13)),  # blocs features.1 a features.12
            "le plus leger : PC industriel, embarque",  # remarque
        ),
        _tv(  # MobileNetV3 large
            "mobilenet_v3_large",  # nom
            "MobileNet_V3_Large_Weights.DEFAULT",  # poids ImageNet
            "classifier.3",  # tete
            "features.0.0",  # premiere convolution
            tuple(f"features.{i}" for i in range(1, 17)),  # blocs features.1 a features.16
        ),
        _tv(  # ConvNeXt tiny
            "convnext_tiny",  # nom
            "ConvNeXt_Tiny_Weights.DEFAULT",  # poids ImageNet
            "classifier.2",  # tete : 3e couche du classifier
            "features.0.0",  # premiere convolution
            tuple(f"features.{i}" for i in range(1, 8)),  # blocs features.1 a features.7
            "architecture moderne, plus lente sur CPU",  # remarque
        ),
        _tv(  # RegNetY 800MF
            "regnet_y_800mf",  # nom
            "RegNet_Y_800MF_Weights.DEFAULT",  # poids ImageNet
            "fc",  # tete
            "stem.0",  # premiere convolution
            tuple(f"trunk_output.block{i}" for i in range(1, 5)),  # blocs trunk_output.block1 a block4
            "famille RegNet : bon compromis sur CPU",  # remarque
        ),
        ModelSpec(  # CNN simple ecrit dans le depot (vtb.models.custom)
            "simple_cnn",  # nom
            "custom",  # source : code maison
            "SimpleCNN",  # classe a construire
            None,  # pas de poids pre-entraines
            "classifier.2",  # tete
            "features.0.block.0",  # premiere convolution
            ("features.0", "features.1", "features.2", "features.3"),  # les 4 blocs
            128,  # taille d'entree prevue
            "MIT (code du depot)",  # licence
            "CNN de zero (cours, module 7)",  # remarque
        ),
        ModelSpec(  # petit ResNet ecrit dans le depot
            "small_resnet",  # nom
            "custom",  # source : code maison
            "SmallResNet",  # classe a construire
            None,  # pas de poids pre-entraines
            "classifier.2",  # tete
            "stem.0",  # premiere convolution
            ("stages.0", "stages.1", "stages.2", "stages.3"),  # les 4 etages
            128,  # taille d'entree prevue
            "MIT (code du depot)",  # licence
            "blocs residuels de zero (modele custom)",  # remarque
        ),
    ]
}


def get_spec(name: str) -> ModelSpec:  # renvoie la fiche d'un modele a partir de son nom
    """Spec d'un modele du catalogue, ou spec generique pour timm:<nom> (tete et blocs lus sur le modele)."""
    if name.startswith("timm:"):  # modele de la bibliotheque timm (des centaines de modeles)
        return ModelSpec(  # fiche generique : tete, convolution et blocs vides (trouves plus tard sur le modele)
            name,  # nom complet "timm:<nom>"
            "timm",  # source
            name.split(":", 1)[1],  # nom timm (ce qui suit "timm:")
            None,  # poids geres par timm
            "",  # tete inconnue a ce stade
            "",  # premiere convolution inconnue
            (),  # blocs inconnus
            licence="Apache-2.0 (code timm) ; poids : voir la fiche du modele",  # licence
        )
    if name not in ZOO:  # nom absent du catalogue
        raise ValueError(f"modele inconnu : {name} ; catalogue : {sorted(ZOO)} ou timm:<nom>")  # erreur avec la liste des noms
    return ZOO[name]  # renvoie la fiche


def get_module(model: nn.Module, path: str) -> nn.Module:  # renvoie la couche designee par un chemin pointe
    """Sous-module par chemin pointe ("classifier.3") ; les indices de Sequential sont des noms comme les autres."""
    module = model  # on part du modele entier
    for part in path.split("."):  # pour chaque morceau du chemin ("classifier", puis "3")
        module = getattr(module, part)  # descend d'un niveau
    return module  # renvoie la couche trouvee


def set_module(model: nn.Module, path: str, new: nn.Module) -> None:  # remplace la couche designee par une nouvelle
    parent_path, _, last = path.rpartition(".")  # coupe au dernier point : "classifier" et "3"
    parent = get_module(model, parent_path) if parent_path else model  # module parent (le modele lui-meme si pas de point)
    setattr(parent, last, new)  # remplace l'enfant par la nouvelle couche


def _torchvision_weights(spec: ModelSpec):  # renvoie l'objet de poids torchvision (ex. ResNet18_Weights.DEFAULT)
    # "ResNet18_Weights.DEFAULT" -> ("ResNet18_Weights", "DEFAULT") ; "type: ignore" car weights peut etre None pour le verificateur
    enum_name, member = spec.weights.split(".")  # type: ignore[union-attr]
    return getattr(getattr(tv_models, enum_name), member)  # tv_models.ResNet18_Weights.DEFAULT


def imagenet_top1(spec: ModelSpec) -> float | None:  # renvoie la precision ImageNet annoncee (ou None)
    """Precision ImageNet annoncee par torchvision (lue dans les metadonnees, sans telecharger les poids)."""
    if spec.source != "torchvision" or spec.weights is None:  # pas de metadonnees hors torchvision
        return None  # inconnue
    # metadonnees des poids -> "_metrics" -> "ImageNet-1K" -> "acc@1" (part des images bien classees au 1er choix)
    return _torchvision_weights(spec).meta.get("_metrics", {}).get("ImageNet-1K", {}).get("acc@1")


# construit le reseau decrit par la fiche ; renvoie le modele (avec sa tete d'origine)
def instantiate(spec: ModelSpec, pretrained: bool, extra: dict[str, Any] | None = None) -> nn.Module:
    """Construit l'architecture (avec sa tete d'origine) ; les poids ImageNet seulement si pretrained."""
    extra = extra or {}  # options supplementaires (dict vide par defaut)
    if spec.source == "torchvision":  # modele torchvision
        weights = _torchvision_weights(spec) if pretrained else None  # poids ImageNet si demande, sinon poids aleatoires
        try:  # le telechargement des poids peut echouer (pas d'internet)
            return getattr(tv_models, spec.builder)(weights=weights)  # appelle tv_models.resnet18(weights=...) par exemple
        except (OSError, RuntimeError) as error:  # URLError herite d'OSError : telechargement impossible
            raise RuntimeError(  # erreur claire qui explique la marche a suivre hors ligne
                # 1re partie du message : quels poids manquent et quel outil lancer
                f"poids {spec.weights} introuvables hors ligne : lancer tools/fetch_weights.py sur une machine connectee, "
                # 2e partie : comment donner le fichier au modele, puis l'erreur d'origine
                f"puis donner le fichier via model.weights_path ou la variable TORCH_HOME ({error})"
            ) from error  # "from error" garde l'erreur d'origine comme cause (visible dans la trace)
    if spec.source == "timm":  # modele timm
        import timm  # timm : grande bibliotheque de modeles de vision (importee seulement si besoin)

        return timm.create_model(spec.builder, pretrained=pretrained, **extra)  # construit le modele timm
    from vtb.models import custom  # modeles maison (SimpleCNN, SmallResNet)

    return getattr(custom, spec.builder)(num_classes=1000, **extra)  # tete remplacee ensuite, comme pour les autres


def count_params(model: nn.Module) -> int:  # renvoie le nombre total de parametres (poids) du modele
    return sum(p.numel() for p in model.parameters())  # somme du nombre de valeurs de chaque tenseur de poids


@torch.no_grad()  # pas de calcul de gradient : plus rapide, on ne fait que predire
# mesure le temps d'une prediction ; renvoie la duree moyenne en millisecondes
def measure_latency(model: nn.Module, size: int, runs: int = 10, batch: int = 1, channels: int = 3) -> float:
    """Latence moyenne en millisecondes sur CPU (une image, apres deux passages de chauffe)."""
    model.eval()  # mode evaluation (dropout coupe, BatchNorm figee)
    x = torch.zeros(batch, channels, size, size)  # image factice remplie de zeros (N, C, H, W)
    for _ in range(2):  # deux passages de chauffe (les premiers sont plus lents : allocations, caches)
        model(x)  # passage avant, resultat ignore
    start = time.perf_counter()  # depart du chronometre
    for _ in range(runs):  # passages mesures
        model(x)  # passage avant
    return 1000 * (time.perf_counter() - start) / runs  # duree moyenne d'un passage, en millisecondes


def list_models(with_latency: bool = False, size: int = 224) -> pd.DataFrame:  # renvoie le catalogue sous forme de tableau
    """Le catalogue en tableau : parametres, precision ImageNet, latence CPU mesuree sur CETTE machine (option)."""
    rows = []  # une ligne par modele
    for spec in ZOO.values():  # pour chaque fiche du catalogue
        model = instantiate(spec, pretrained=False)  # construit le modele sans telecharger de poids
        row = {  # ligne du tableau
            "name": spec.name,  # nom
            "source": spec.source,  # torchvision ou custom
            "params_M": round(count_params(model) / 1e6, 2),  # nombre de parametres en millions
            "imagenet_top1": imagenet_top1(spec),  # precision ImageNet annoncee
            "notes": spec.notes,  # remarque
        }
        if with_latency:  # si la mesure de latence est demandee
            row[f"cpu_ms_{size}px"] = round(measure_latency(model, size), 1)  # latence CPU mesuree ici, en ms
        rows.append(row)  # ajoute la ligne
    return pd.DataFrame(rows)  # renvoie le tableau
