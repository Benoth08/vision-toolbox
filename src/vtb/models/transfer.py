"""Transfer learning : reutiliser un modele pre-entraine, du simple gel au fine-tuning complet.

Vocabulaire : le transfer learning (reutiliser ce qu'un modele a appris ailleurs) recouvre
deux familles. L'extraction de caracteristiques garde le reseau gele et n'entraine que ce qui
est au-dessus ; le fine-tuning degele une partie ou la totalite du reseau. Strategies :

    features     rien ne bouge dans le reseau : vecteurs + regression logistique (secondes)
    head         la tete seule, backbone gele                       (quelques centaines d'images)
    partial      la tete + les N derniers blocs, taux plus faible   (quelques milliers : le cas courant)
    full         tout, a faible taux                                (beaucoup d'images, domaine loin d'ImageNet)
    progressive  la tete, puis un bloc de plus toutes les k epoques (optimiseur recree a chaque etape)

Deux regles qui valent pour toutes : l'optimiseur ne recoit que ce qui apprend, et ce qui etait
deja appris apprend plus lentement que la tete neuve (sinon les premiers gradients, bruites,
de la tete aleatoire detruisent les poids pre-entraines).
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (ex. int | None)

import logging  # journal : messages horodates sur la console et dans un fichier
from typing import Literal  # Literal : liste fermee de valeurs permises

import torch  # PyTorch : tenseurs (tableaux de nombres sur CPU ou GPU) et reseaux de neurones
import torch.nn as nn  # nn : briques de reseaux de neurones (couches, modules)
from pydantic import Field  # Field : valeur par defaut d'un champ de configuration

from vtb.core.config import StrictModel  # base des sections de configuration (refuse les cles inconnues)
from vtb.models.zoo import ModelSpec, get_module, get_spec, instantiate, set_module  # catalogue des modeles et acces aux couches
from vtb.preprocess.tensor import IMAGENET_MEAN, IMAGENET_STD  # moyenne et ecart-type RGB d'ImageNet

logger = logging.getLogger(__name__)  # journal propre a ce module
Strategy = Literal["features", "head", "partial", "full", "progressive"]  # les 5 strategies possibles


class TransferConfig(StrictModel):  # reglages du transfer learning (section model.transfer du YAML)
    # strategie : que degeler (voir le tableau en haut du fichier) ; backbone = partie pre-entrainee qui extrait les caracteristiques
    strategy: Strategy = "partial"
    unfreeze_last: int = 1  # partial : nombre de blocs degeles, en partant de la sortie
    # taux d'apprentissage (taille des pas de correction des poids) du backbone = taux de la tete x ce facteur
    backbone_lr_factor: float = 0.1
    layer_decay: float = 1.0  # < 1 : chaque bloc plus profond apprend encore plus lentement (taux discriminants)
    progressive_every: int = 2  # progressive : un bloc de plus toutes les k epoques (epoque = un passage sur tout le train)
    freeze_batchnorm: bool = True  # BatchNorm des blocs geles en mode eval (statistiques ImageNet conservees)
    logistic_c: float = 1.0  # features : inverse de la regularisation de la regression logistique


class ModelConfig(StrictModel):  # section "model" du YAML : quel reseau et comment l'adapter
    name: str = "resnet18"  # nom du modele dans le catalogue (ou timm:<nom>)
    pretrained: bool = True  # True : partir des poids appris sur ImageNet
    weights_path: str | None = None  # fichier de poids local (reseau ferme) ; prioritaire sur le telechargement
    in_channels: int = 3  # 1 pour des images grises sans les tripler
    extra: dict = Field(default_factory=dict)  # parametres propres a un modele maison (channels, dropout) ou timm
    transfer: TransferConfig = Field(default_factory=TransferConfig)  # reglages du transfer learning


# --- construction -----------------------------------------------------------------------------


def _resolve_timm(model: nn.Module, spec: ModelSpec) -> ModelSpec:  # complete la fiche d'un modele timm ; renvoie une nouvelle fiche
    """Pour un modele timm : tete, premiere convolution et blocs lus sur le modele lui-meme."""
    # tete = derniere couche lineaire du modele (on parcourt les couches a l'envers)
    head = next(name for name, m in reversed(list(model.named_modules())) if isinstance(m, nn.Linear))
    first = next(name for name, m in model.named_modules() if isinstance(m, nn.Conv2d))  # premiere convolution rencontree
    blocks = tuple(info["module"] for info in getattr(model, "feature_info", []))  # blocs annonces par timm (vide si absent)
    return ModelSpec(spec.name, "timm", spec.builder, None, head, first, blocks, licence=spec.licence)  # fiche complete


def load_backbone_weights(model: nn.Module, path: str) -> None:  # charge des poids depuis un fichier local
    """Charge un state_dict local ; seule la tete a le droit de ne pas correspondre (1000 classes ImageNet)."""
    # state_dict = dict nom de couche -> tenseur de poids ; weights_only=True : ne charge que des tenseurs (securite)
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("model_state", state)  # certains fichiers rangent les poids sous la cle "model_state"
    state = {k.removeprefix("module."): v for k, v in state.items()}  # poids sauves depuis DataParallel
    result = model.load_state_dict(state, strict=False)  # charge ce qui correspond ; strict=False : tolere des cles en trop ou en moins
    if result.unexpected_keys or len(result.missing_keys) > 4:  # cles inconnues, ou trop de cles manquantes (plus que la tete)
        # erreur avec quelques exemples de cles fautives
        raise ValueError(f"poids {path} incompatibles : manquants {result.missing_keys[:6]} ; inattendus {result.unexpected_keys[:6]}")
    logger.info("poids locaux charges : %s", path)  # trace dans le journal


def replace_head(model: nn.Module, spec: ModelSpec, num_classes: int) -> None:  # remplace la tete pour nos classes
    """Remplace la derniere couche lineaire par une couche a num_classes sorties (la seule couche neuve)."""
    old = get_module(model, spec.head)  # ancienne tete (1000 sorties pour ImageNet)
    if not isinstance(old, nn.Linear):  # la tete doit etre une couche lineaire
        raise TypeError(f"{spec.name} : {spec.head} n'est pas une couche lineaire ({type(old).__name__})")  # erreur claire
    set_module(model, spec.head, nn.Linear(old.in_features, num_classes))  # nouvelle couche : memes entrees, nos classes en sortie


def adapt_input_channels(model: nn.Module, spec: ModelSpec, channels: int) -> None:  # adapte la 1re convolution au nombre de canaux
    """Premiere convolution a 1 ou N canaux, initialisee a partir des poids RGB.

    Gris : somme des trois noyaux (une image grise recopiee en RGB donnerait exactement la meme
    sortie). N canaux : moyenne RGB recopiee et remise a l'echelle par 3 / N, pour garder la
    meme amplitude d'activation.
    """
    old = get_module(model, spec.first_conv)  # premiere convolution actuelle (3 canaux RGB)
    if not isinstance(old, nn.Conv2d) or old.in_channels == channels:  # pas une convolution, ou deja le bon nombre de canaux
        return  # rien a faire
    # nouvelle convolution : memes reglages que l'ancienne, seul le nombre de canaux d'entree change
    new = nn.Conv2d(channels, old.out_channels, old.kernel_size, old.stride, old.padding, old.dilation, old.groups, old.bias is not None)
    with torch.no_grad():  # copie de poids : pas de gradient a calculer
        if channels == 1:  # image grise
            new.weight.copy_(old.weight.sum(dim=1, keepdim=True))  # somme des noyaux R, G, B
        else:  # N canaux
            new.weight.copy_(old.weight.mean(dim=1, keepdim=True).repeat(1, channels, 1, 1) * 3.0 / channels)  # moyenne recopiee, x 3/N
        if old.bias is not None:  # si l'ancienne convolution avait un biais
            new.bias.copy_(old.bias)  # on le recopie tel quel
    set_module(model, spec.first_conv, new)  # met la nouvelle convolution a la place de l'ancienne


# construit le modele pret a entrainer ; renvoie (modele, fiche)
def build_classifier(config: ModelConfig, num_classes: int) -> tuple[nn.Module, ModelSpec]:
    """Le modele pret a entrainer (tete remplacee, entree adaptee) et sa spec."""
    spec = get_spec(config.name)  # fiche du modele demande
    pretrained = config.pretrained and spec.source != "custom" and config.weights_path is None  # telecharger les poids ImageNet ?
    model = instantiate(spec, pretrained=pretrained, extra=config.extra)  # construit le reseau
    if spec.source == "timm":  # modele timm : la fiche est completee en lisant le modele
        spec = _resolve_timm(model, spec)  # tete, premiere convolution et blocs trouves
    if config.weights_path:  # fichier de poids local fourni
        load_backbone_weights(model, config.weights_path)  # charge ces poids
    replace_head(model, spec, num_classes)  # nouvelle tete a num_classes sorties
    adapt_input_channels(model, spec, config.in_channels)  # premiere convolution adaptee (gris, multicanal)
    return model, spec  # renvoie le modele et sa fiche


# normalisation a utiliser avec ces poids ; renvoie (moyennes, ecarts-types) ou None
def default_normalization(config: ModelConfig) -> tuple[list[float], list[float]] | None:
    """Normalisation imposee par les poids pre-entraines (ImageNet) ; None = a calculer sur le train."""
    if not (config.pretrained or config.weights_path) or get_spec(config.name).source == "custom":  # pas de poids pre-entraines
        return None  # a calculer sur nos images d'entrainement
    if config.in_channels == 1:  # moyenne des trois canaux, coherente avec la somme des noyaux RGB
        return [sum(IMAGENET_MEAN) / 3], [sum(IMAGENET_STD) / 3]  # une seule valeur moyenne et un seul ecart-type
    if config.in_channels == 3:  # RGB
        return list(IMAGENET_MEAN), list(IMAGENET_STD)  # statistiques ImageNet telles quelles
    return [sum(IMAGENET_MEAN) / 3] * config.in_channels, [sum(IMAGENET_STD) / 3] * config.in_channels  # N canaux : moyenne recopiee


# --- gel et degel -----------------------------------------------------------------------------


def _parameters_of(model: nn.Module, prefix: str) -> list[nn.Parameter]:  # renvoie les poids d'un bloc designe par son nom
    return [p for name, p in model.named_parameters() if name == prefix or name.startswith(prefix + ".")]  # "layer4" et "layer4.*"


# gele / degele les poids selon la strategie (requires_grad)
def apply_strategy(model: nn.Module, spec: ModelSpec, transfer: TransferConfig, unfrozen_blocks: int | None = None) -> None:
    """Fixe requires_grad selon la strategie. unfrozen_blocks force le nombre de blocs degeles (progressive)."""
    # requires_grad = True : le poids sera modifie a l'entrainement ; False : il est gele
    strategy = transfer.strategy  # strategie choisie
    if spec.source == "custom" or strategy == "full":  # modele maison (pas pre-entraine) ou strategie full : tout apprend
        for p in model.parameters():  # pour chaque poids
            p.requires_grad = True  # degele
        return  # fini
    for p in model.parameters():  # sinon on commence par tout geler
        p.requires_grad = False  # gele
    if strategy == "features":  # extraction de caracteristiques : rien n'apprend dans le reseau
        return  # tout reste gele
    for p in _parameters_of(model, spec.head):  # poids de la tete
        p.requires_grad = True  # la tete apprend toujours
    n = {"head": 0, "partial": transfer.unfreeze_last, "progressive": unfrozen_blocks or 0}[strategy]  # nombre de blocs a degeler
    for block in spec.blocks[len(spec.blocks) - min(n, len(spec.blocks)) :]:  # les n derniers blocs (les plus proches de la sortie)
        for p in _parameters_of(model, block):  # pour chaque poids du bloc
            p.requires_grad = True  # degele


def freeze_frozen_batchnorm(model: nn.Module) -> None:  # remet en mode eval les BatchNorm gelees
    """Remet en mode eval les BatchNorm dont les poids sont geles (a appeler apres chaque model.train()).

    Geler requires_grad ne gele pas les statistiques glissantes d'une BatchNorm : en mode train
    elles continuent de suivre tes images. Sur un domaine loin d'ImageNet, garder celles
    d'ImageNet dans les blocs geles est en general plus stable.
    """
    for module in model.modules():  # pour chaque couche du modele
        # si c'est une BatchNorm et qu'aucun de ses poids n'apprend
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and not any(p.requires_grad for p in module.parameters()):
            module.eval()  # mode eval : elle utilise ses statistiques enregistrees sans les mettre a jour


# groupes de poids avec leur taux d'apprentissage, pour l'optimiseur ; renvoie une liste de dicts
def param_groups(model: nn.Module, spec: ModelSpec, lr: float, transfer: TransferConfig) -> list[dict]:
    """Groupes pour l'optimiseur : la tete au taux lr, chaque bloc degele a lr x facteur x decroissance^profondeur."""
    # optimiseur = algorithme (SGD, Adam...) qui corrige les poids a partir des gradients ; il accepte des groupes a taux differents
    head_params = [p for p in _parameters_of(model, spec.head) if p.requires_grad] if spec.head else []  # poids de la tete qui apprennent
    if spec.source == "custom" or not head_params:  # modele maison, ou tete introuvable
        return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr}]  # un seul groupe, un seul taux
    groups = [{"params": head_params, "lr": lr, "name": "head"}]  # 1er groupe : la tete au taux plein
    seen = {id(p) for p in head_params}  # poids deja ranges (id = identifiant unique de l'objet)
    for depth, block in enumerate(reversed(spec.blocks)):  # pour chaque bloc, de la sortie vers l'entree (depth = 0, 1, 2...)
        params = [p for p in _parameters_of(model, block) if p.requires_grad and id(p) not in seen]  # ses poids degeles pas encore ranges
        if params:  # s'il y en a
            # taux = lr x facteur du backbone x decroissance^profondeur : plus on descend, plus on apprend lentement
            groups.append({"params": params, "lr": lr * transfer.backbone_lr_factor * transfer.layer_decay**depth, "name": block})
            seen.update(id(p) for p in params)  # marque ces poids comme ranges
    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in seen]  # tige, normalisations finales
    if rest:  # s'il reste des poids degeles hors des blocs
        # dernier groupe, au taux le plus faible
        groups.append({"params": rest, "lr": lr * transfer.backbone_lr_factor * transfer.layer_decay ** len(spec.blocks), "name": "rest"})
    return groups  # renvoie les groupes


class ProgressiveUnfreezer:  # gere le degel progressif au fil des epoques
    """Strategie progressive : un bloc de plus toutes les `every` epoques ; signale quand recreer l'optimiseur."""

    def __init__(self, model: nn.Module, spec: ModelSpec, transfer: TransferConfig) -> None:  # memorise le modele ; gele tout sauf la tete
        self.model, self.spec, self.transfer = model, spec, transfer  # garde les references pour plus tard
        self.unfrozen = 0  # nombre de blocs degeles pour l'instant
        apply_strategy(model, spec, transfer, unfrozen_blocks=0)  # depart : la tete seule apprend

    def update(self, epoch: int) -> bool:  # met a jour le degel pour cette epoque ; renvoie True si quelque chose a change
        """Appele au debut de chaque epoque (a partir de 1) ; True si un bloc vient d'etre degele."""
        # nombre de blocs a degeler : +1 toutes les progressive_every epoques, sans depasser le nombre de blocs
        target = min((epoch - 1) // max(self.transfer.progressive_every, 1), len(self.spec.blocks))
        if target == self.unfrozen:  # rien de nouveau
            return False  # l'optimiseur peut rester tel quel
        self.unfrozen = target  # nouveau nombre de blocs degeles
        apply_strategy(self.model, self.spec, self.transfer, unfrozen_blocks=target)  # applique le degel
        logger.info("progressive : %d bloc(s) degele(s) (%s)", target, ", ".join(self.spec.blocks[-target:]))  # journal
        return True  # signale qu'il faut recreer l'optimiseur (de nouveaux poids apprennent)
