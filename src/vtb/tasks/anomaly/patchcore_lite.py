"""PatchCore simplifie, ecrit en clair : la methode de reference en inspection, sans entrainement.

Un reseau pre-entraine gele decrit chaque zone de l'image par un vecteur (couches intermediaires,
voisinage 3x3) ; on memorise les vecteurs des images normales ; une zone dont le vecteur est loin
de tous les vecteurs memorises est anormale. La memoire est reduite par un coreset k-center
glouton (comme le papier) ou par tirage aleatoire.

Le modele entier (extraction, distance au plus proche voisin) est un nn.Module : il s'exporte en
ONNX avec sa memoire, et la distance est ecrite en produits matriciels (|a|^2 - 2ab + |b|^2),
ce que tous les moteurs d'inference savent faire.
"""

from __future__ import annotations  # permet les annotations de type modernes (list[str], X | None)

import logging  # journal des messages d'avancement

import torch  # PyTorch : calcul sur des tenseurs (tableaux de nombres multi-dimensions)
import torch.nn as nn  # nn : briques des reseaux de neurones (modules, couches)
import torch.nn.functional as F  # F : fonctions sans parametres (interpolation, moyenne locale...)
from torch.utils.data import DataLoader  # DataLoader : fournit les images par lots (batchs)
from torchvision import models  # models : reseaux connus (ResNet...) avec leurs poids pre-entraines

from vtb.preprocess.tensor import IMAGENET_MEAN, IMAGENET_STD  # moyenne et ecart-type par canal des images d'ImageNet

logger = logging.getLogger(__name__)  # journal propre a ce module


class PatchFeatures(nn.Module):  # extrait un vecteur de caracteristiques par zone de l'image avec un ResNet18 gele
    """Caracteristiques locales d'un ResNet18 gele : couches demandees, remontees a la resolution de la premiere, concatenees."""

    def __init__(self, layers: list[str], pretrained: bool = True) -> None:  # layers : noms des couches a lire (ex. layer2, layer3)
        super().__init__()  # initialise nn.Module
        # ResNet18 avec poids appris sur ImageNet (ou aleatoires pour les tests) ; eval() = mode inference (BatchNorm fige)
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None).eval()
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)  # le debut du ResNet (avant les blocs)
        # les 4 blocs du ResNet, ranges par nom pour pouvoir s'arreter a la couche voulue
        self.blocks = nn.ModuleDict({name: getattr(backbone, name) for name in ("layer1", "layer2", "layer3", "layer4")})
        self.layers = layers  # couches dont on garde la sortie
        # buffer : tenseur range dans le modele (sauve et exporte avec lui) mais non appris ; forme (1, 3, 1, 1) pour agir par canal
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))  # ecart-type ImageNet, meme forme
        for p in self.parameters():  # pour chaque poids du reseau
            p.requires_grad = False  # gele : pas de gradient a calculer, le reseau ne sera jamais entraine

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # images (N, C, S, S) -> vecteurs par zone (N, D, h, w)
        if x.shape[1] == 1:  # image grise : recopiee sur 3 canaux pour le backbone ImageNet
            x = x.repeat(1, 3, 1, 1)  # recopie le canal unique 3 fois
        x = self.stem((x - self.mean) / self.std)  # entree en [0, 1] : la normalisation ImageNet est DANS le modele exporte
        outputs = {}  # sorties des blocs, par nom
        for name, block in self.blocks.items():  # pour chaque bloc du ResNet, dans l'ordre
            x = block(x)  # passe dans le bloc
            outputs[name] = x  # garde sa sortie
            if name == self.layers[-1]:  # si c'est la derniere couche demandee
                break  # inutile de calculer les blocs suivants
        target = outputs[self.layers[0]].shape[-2:]  # taille (h, w) de la premiere couche demandee (la plus fine)
        # premiere couche telle quelle, puis les suivantes agrandies a sa taille (interpolation bilineaire : moyenne ponderee des voisins)
        feats = (  # liste des cartes de caracteristiques, toutes a la meme taille
            [outputs[self.layers[0]]]  # la premiere couche demandee, telle quelle
            + [  # suivie des autres couches...
                F.interpolate(outputs[n], size=target, mode="bilinear", align_corners=False)  # sortie de la couche n, agrandie a target
                for n in self.layers[1:]  # pour chaque couche demandee apres la premiere
            ]
        )
        # colle les canaux de toutes les couches puis moyenne sur un voisinage 3x3
        return F.avg_pool2d(torch.cat(feats, dim=1), 3, stride=1, padding=1)  # voisinage 3x3 : chaque vecteur decrit une zone


# coreset : petit sous-ensemble de vecteurs qui couvre bien tous les autres ; renvoie les indices choisis
def greedy_coreset(vectors: torch.Tensor, n: int, seed: int, projection_dim: int = 128) -> torch.Tensor:
    """Indices d'un coreset k-center glouton : a chaque pas, le vecteur le plus loin de ceux deja choisis.

    Une projection aleatoire (Johnson-Lindenstrauss) reduit la dimension pour la selection seule :
    c'est ce qui rend le calcul abordable sur CPU, avec un resultat quasi identique.
    """
    generator = torch.Generator().manual_seed(seed)  # generateur aleatoire a graine fixe : meme resultat a chaque execution
    if vectors.shape[1] > projection_dim:  # si les vecteurs sont plus longs que la dimension de projection
        # matrice aleatoire qui projette en projection_dim dimensions (divisee pour garder des distances comparables)
        projection = torch.randn(vectors.shape[1], projection_dim, generator=generator) / projection_dim**0.5
        reduced = vectors @ projection  # @ = produit matriciel : vecteurs raccourcis, distances a peu pres conservees
    else:  # sinon les vecteurs sont deja courts
        reduced = vectors  # on les garde tels quels
    # Distances au carre par |x|^2 - 2 x.c + |c|^2 : un produit matrice-vecteur par pas, environ 20 fois
    # plus rapide que torch.cdist pour cette forme (N x 1) ; l'ordre des distances, donc le choix, est le meme.
    squared = (reduced * reduced).sum(dim=1)  # |x|^2 de chaque vecteur, calcule une seule fois

    def distance_to(i: int) -> torch.Tensor:  # distance au carre de tous les vecteurs au vecteur numero i
        return (squared - 2 * (reduced @ reduced[i]) + squared[i]).clamp_min_(0)  # clamp_min_ : pas de negatif du aux arrondis

    selected = [int(torch.randint(len(reduced), (1,), generator=generator))]  # premier vecteur tire au hasard
    distances = distance_to(selected[0])  # distance de chaque vecteur au plus proche vecteur choisi (un seul pour l'instant)
    step = max(1, n // 10)  # frequence des messages d'avancement (tous les 10 %)
    for k in range(n - 1):  # pour chacun des n - 1 vecteurs restant a choisir
        if k % step == 0:  # tous les 10 % environ
            logger.info("coreset : %d / %d vecteurs choisis parmi %d", k, n, len(reduced))  # long sur CPU : on montre l'avancement
        index = int(distances.argmax())  # le vecteur le plus loin de tous ceux deja choisis (le moins bien couvert)
        selected.append(index)  # on le choisit
        torch.minimum(distances, distance_to(index), out=distances)  # met a jour : distance au plus proche choisi, nouveau inclus
    return torch.tensor(selected)  # indices des vecteurs gardes


class PatchCoreLite(nn.Module):  # le detecteur complet : extraction + distance au plus proche vecteur normal memorise
    """Image (N, C, S, S) dans [0, 1] -> carte brute (N, 1, h, w) : distance de chaque zone au plus proche vecteur normal."""

    def __init__(self, layers: list[str], pretrained: bool = True) -> None:  # cree l'extracteur et une memoire vide
        super().__init__()  # initialise nn.Module
        self.features = PatchFeatures(layers, pretrained)  # extracteur de vecteurs par zone
        self.register_buffer("memory", torch.zeros(0))  # memoire des vecteurs normaux (vide tant que fit n'a pas tourne)
        self.register_buffer("memory_sq", torch.zeros(0))  # |b|^2 de chaque vecteur memorise, precalcule pour aller plus vite

    @torch.no_grad()  # pas de calcul de gradient ici : on ne fait que lire des caracteristiques
    def fit(self, loader: DataLoader, fraction: float, coreset: str, seed: int, device: torch.device) -> None:  # remplit la memoire
        """Memorise une fraction des vecteurs des images normales du loader."""
        self.eval().to(device)  # mode evaluation et envoi sur CPU ou GPU
        chunks = []  # vecteurs extraits, lot par lot
        for images, _, _ in loader:  # pour chaque lot d'images normales
            f = self.features(images.to(device))  # vecteurs par zone (N, D, h, w)
            # permute + reshape : une ligne par zone (N*h*w lignes de D valeurs) ; cpu() libere la memoire du GPU
            chunks.append(f.permute(0, 2, 3, 1).reshape(-1, f.shape[1]).cpu())
        bank = torch.cat(chunks)  # tous les vecteurs normaux dans un seul tableau
        n = max(1, int(fraction * len(bank)))  # nombre de vecteurs a garder (au moins 1)
        if coreset == "greedy":  # si on veut le coreset k-center du papier
            keep = greedy_coreset(bank, n, seed)  # indices choisis pour bien couvrir l'espace
        else:  # sinon tirage aleatoire
            keep = torch.randperm(len(bank), generator=torch.Generator().manual_seed(seed))[:n]  # n indices au hasard
        self.memory = bank[keep].contiguous().to(device)  # la memoire : vecteurs gardes, ranges d'un bloc en memoire
        self.memory_sq = (self.memory**2).sum(dim=1).to(device)  # |b|^2 de chaque vecteur memorise
        # bilan : taille de la memoire, dimension des vecteurs, nombre total et methode
        logger.info("memoire PatchCore : %d vecteurs de dimension %d (sur %d, coreset %s)", n, bank.shape[1], len(bank), coreset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # images -> carte d'anomalie brute (distance par zone)
        f = self.features(x)  # vecteurs par zone (N, C, h, w)
        n, c, h, w = f.shape  # taille du lot, longueur des vecteurs, hauteur et largeur de la grille de zones
        flat = f.permute(0, 2, 3, 1).reshape(n, h * w, c)  # une ligne par zone : (N, h*w, C)
        # |a - b|^2 = |a|^2 - 2 a.b + |b|^2, puis minimum sur la memoire : pas de cdist, exportable partout
        sq = (flat**2).sum(dim=2, keepdim=True) - 2 * flat @ self.memory.t() + self.memory_sq.view(1, 1, -1)
        nearest = sq.amin(dim=2).clamp_min(0).sqrt()  # distance au plus proche vecteur normal (racine du carre, jamais negatif)
        return nearest.view(n, 1, h, w)  # remise en forme d'image : (N, 1, h, w)
