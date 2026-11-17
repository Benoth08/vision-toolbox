"""Modeles de segmentation (segmentation_models_pytorch, MIT) avec la normalisation DANS le modele.

Le programme de production envoie une image dans [0, 1] et recoit des probabilites par pixel :
normalisation ImageNet et softmax / sigmoide sont dans le graphe exporte, rien a reecrire.

Poids pre-entraines : pour les encodeurs resnet*, ceux de torchvision (le meme cache que la
classification, utilisable hors ligne) ; pour les autres encodeurs, ceux de smp (telechargement).
"""

from __future__ import annotations  # permet les annotations de type modernes

import logging  # journal des messages

import torch  # PyTorch : tenseurs et reseaux de neurones
from torch import nn  # nn : briques des reseaux (modules, couches)

from vtb.preprocess.tensor import IMAGENET_MEAN, IMAGENET_STD  # moyenne et ecart-type par canal des images d'ImageNet
from vtb.tasks.segmentation.config import SegModelConfig  # reglages du modele

logger = logging.getLogger(__name__)  # journal propre a ce module

ARCHITECTURES = {  # nom dans la config -> nom de classe dans segmentation_models_pytorch (smp)
    "unet": "Unet",  # U-Net : encodeur + decodeur relies par des raccourcis a chaque echelle
    "unetplusplus": "UnetPlusPlus",  # U-Net++ : raccourcis plus denses
    "fpn": "FPN",  # pyramide de caracteristiques (Feature Pyramid Network)
    "deeplabv3plus": "DeepLabV3Plus",  # DeepLab v3+ : convolutions dilatees, bon contexte global
    "pspnet": "PSPNet",  # reseau a pooling pyramidal
    "linknet": "Linknet",  # decodeur leger
    "manet": "MAnet",  # U-Net avec attention
    "segformer": "Segformer",  # architecture a base de transformeur
}
TORCHVISION_RESNETS = {"resnet18", "resnet34", "resnet50", "resnet101", "resnet152"}  # encodeurs dont on prend les poids torchvision


# poids ImageNet de torchvision pour un encodeur resnet, adaptes au nombre de canaux d'entree ; renvoie un state_dict
def _torchvision_encoder_weights(encoder: str, in_channels: int) -> dict[str, torch.Tensor]:
    """Poids ImageNet torchvision d'un resnet ; N canaux (N != 3) : moyenne des noyaux RGB repartie sur les N canaux."""
    from torchvision.models import get_model_weights  # retrouve les poids disponibles d'un modele torchvision par son nom

    state = get_model_weights(encoder).DEFAULT.get_state_dict(progress=False)  # dictionnaire nom de couche -> poids (cache local)
    if in_channels != 3:  # si l'entree n'a pas 3 canaux
        weight = state["conv1.weight"]  # poids de la premiere convolution : (sorties, 3, k, k)
        # moyenne sur les 3 canaux RGB, recopiee sur N canaux, mise a l'echelle 3/N pour garder la meme reponse moyenne
        state["conv1.weight"] = weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1) * 3 / in_channels
    return state  # les poids prets a charger


class Segmenter(nn.Module):  # enveloppe autour du reseau smp : normalisation a l'entree, softmax a la sortie
    """Image (N, C, H, W) dans [0, 1] -> probabilites (N, K, H, W) ; K = nombre de classes, fond compris.

    repeat = 3 pour une image grise et un encodeur pre-entraine en RGB : l'image est copiee sur
    3 canaux DANS le modele, puis normalisee canal par canal. La reponse est exactement celle des
    poids ImageNet (une somme des noyaux ne l'est pas, car la normalisation differe par canal),
    et le programme de production envoie toujours une image a 1 canal.
    """

    # net : reseau smp ; mean / std : normalisation par canal ; repeat : copies du canal gris
    def __init__(self, net: nn.Module, mean: list[float], std: list[float], num_classes: int, repeat: int = 1) -> None:
        super().__init__()  # initialise nn.Module
        self.net, self.num_classes, self.repeat = net, num_classes, repeat  # garde le reseau et les reglages
        # buffers : ranges dans le modele et exportes avec lui, mais non appris ; forme (1, C, 1, 1) pour agir canal par canal
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))  # ecart-type par canal, meme forme

    def logits(self, x: torch.Tensor) -> torch.Tensor:  # renvoie les logits : scores bruts par classe et par pixel, avant softmax
        if self.repeat > 1:  # si l'image grise doit etre recopiee
            x = x.repeat(1, self.repeat, 1, 1)  # 1 canal -> 3 canaux identiques
        return self.net((x - self.mean) / self.std)  # normalisation (centrer, reduire) puis reseau

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # image -> probabilites par classe et par pixel
        return torch.softmax(self.logits(x), dim=1)  # softmax sur les classes (dim 1) : probabilites positives qui somment a 1


# cree le reseau decrit par la config, avec ou sans poids pre-entraines ; renvoie un Segmenter
def build_segmenter(cfg: SegModelConfig, num_classes: int, in_channels: int, load_weights: bool = True) -> Segmenter:
    """load_weights=False : meme architecture, sans telecharger ni charger de poids (rechargement d'un modele entraine)."""
    import segmentation_models_pytorch as smp  # import ici : dependance facultative, chargee seulement si on segmente

    pretrained = cfg.encoder_weights == "imagenet"  # vrai si on part de poids appris sur ImageNet
    repeat = 3 if pretrained and in_channels == 1 else 1  # image grise + encodeur RGB pre-entraine : recopie sur 3 canaux
    net_channels = in_channels * repeat  # canaux reellement recus par le reseau
    use_torchvision = pretrained and cfg.encoder in TORCHVISION_RESNETS  # poids torchvision pour les resnet (cache local)
    smp_weights = "imagenet" if pretrained and not use_torchvision and load_weights else None  # sinon smp telecharge les siens
    # dropout du decodeur : seulement si demande et si l'architecture l'accepte (segformer)
    kwargs = {"decoder_dropout": cfg.decoder_dropout} if cfg.decoder_dropout and cfg.architecture in ("segformer",) else {}
    net = getattr(smp, ARCHITECTURES[cfg.architecture])(  # recupere la classe smp par son nom et la cree
        encoder_name=cfg.encoder,  # nom de l'encodeur (ex. resnet18)
        encoder_weights=smp_weights,  # poids smp a telecharger, ou None
        in_channels=net_channels,  # canaux d'entree du reseau
        classes=num_classes,  # nombre de classes en sortie (une carte de scores par classe)
        **kwargs,  # dropout eventuel
    )
    if use_torchvision and load_weights:  # si les poids viennent de torchvision
        net.encoder.load_state_dict(_torchvision_encoder_weights(cfg.encoder, net_channels))  # les charge dans l'encodeur
        # message : encodeur, canaux d'origine, nombre de copies
        logger.info("encodeur %s : poids ImageNet torchvision (%d canal/canaux, repetition x%d)", cfg.encoder, in_channels, repeat)
    if pretrained:  # poids pre-entraines : il faut la normalisation qui a servi a les apprendre
        # 3 canaux : valeurs ImageNet par canal ; autre nombre : moyenne des 3 canaux ImageNet sur chaque canal
        mean, std = (IMAGENET_MEAN, IMAGENET_STD) if net_channels == 3 else ([0.449] * net_channels, [0.226] * net_channels)
    else:  # sinon (poids aleatoires)
        mean, std = [0.5] * net_channels, [0.25] * net_channels  # depart de zero : simple recentrage
    return Segmenter(net, mean, std, num_classes, repeat)  # le modele complet, normalisation et softmax inclus


# groupes de parametres pour l'optimiseur : encodeur a pas plus petits ; renvoie une liste de dictionnaires
def param_groups(model: Segmenter, lr: float, encoder_lr_factor: float, weight_decay: float) -> list[dict]:
    """Encodeur (pre-entraine) a lr x facteur, decodeur et tete (neufs) a lr."""
    encoder = list(model.net.encoder.parameters())  # parametres de l'encodeur
    ids = {id(p) for p in encoder}  # leurs identifiants (pour les reconnaitre rapidement)
    rest = [p for p in model.parameters() if id(p) not in ids]  # tous les autres parametres (decodeur, tete)
    return [  # deux groupes, chacun avec son taux d'apprentissage
        {"params": encoder, "lr": lr * encoder_lr_factor, "weight_decay": weight_decay, "name": "encoder"},  # encodeur : pas reduits
        {"params": rest, "lr": lr, "weight_decay": weight_decay, "name": "decoder"},  # decodeur et tete : pas normaux
    ]
