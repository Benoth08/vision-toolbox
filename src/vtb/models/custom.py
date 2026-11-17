"""Modeles ecrits de zero : le CNN du cours et un petit reseau residuel.

Ils servent de point de comparaison et de modele quand le domaine est trop loin d'ImageNet
(spectres, radiographies) ou que la taille compte. Meme contrat que les modeles du zoo : la
derniere couche s'appelle classifier.2, la premiere convolution est la premiere du reseau.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes

import torch  # PyTorch : tenseurs (tableaux de nombres sur CPU ou GPU) et reseaux de neurones
import torch.nn as nn  # nn : briques de reseaux de neurones (couches, modules)


class ConvBlock(nn.Module):  # bloc de base : convolution, normalisation, activation, reduction de taille
    """Conv 3x3 -> BatchNorm -> ReLU -> MaxPool 2x2 : la taille spatiale est divisee par 2."""

    def __init__(self, in_channels: int, out_channels: int) -> None:  # construit le bloc (canaux d'entree -> canaux de sortie)
        super().__init__()  # initialise la partie nn.Module (obligatoire)
        self.block = nn.Sequential(  # Sequential : enchaine les couches dans l'ordre
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),  # pas de biais : la BatchNorm le porte
            nn.BatchNorm2d(out_channels),  # BatchNorm : normalise chaque canal sur le lot (apprentissage plus stable)
            nn.ReLU(inplace=True),  # ReLU : garde les valeurs positives, met les negatives a 0 (non-linearite)
            nn.MaxPool2d(2),  # garde le maximum de chaque carre 2x2 : hauteur et largeur divisees par 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # passage avant : renvoie la sortie du bloc
        return self.block(x)  # applique les 4 couches dans l'ordre


class SimpleCNN(nn.Module):  # reseau convolutif simple : 4 blocs puis un classifieur
    """Quatre blocs de convolution, moyenne spatiale (toute taille d'entree), une couche de classification."""

    def __init__(  # construit le reseau
        self,  # l'objet en construction
        num_classes: int,  # nombre de classes a predire
        channels: tuple[int, ...] = (32, 64, 128, 256),  # nombre de canaux en sortie de chaque bloc
        dropout: float = 0.3,  # dropout : eteint au hasard 30 % des valeurs a l'entrainement (limite le sur-apprentissage)
        in_channels: int = 3,  # canaux de l'image d'entree (3 = RGB, 1 = gris)
    ) -> None:  # un constructeur ne renvoie rien
        super().__init__()  # initialise la partie nn.Module
        blocks, cin = [], in_channels  # liste des blocs ; nombre de canaux courant
        for cout in channels:  # pour chaque nombre de canaux de sortie
            blocks.append(ConvBlock(cin, cout))  # ajoute un bloc cin -> cout
            cin = cout  # la sortie de ce bloc est l'entree du suivant
        self.features = nn.Sequential(*blocks)  # extracteur de caracteristiques : les blocs a la suite
        self.pool = nn.AdaptiveAvgPool2d(1)  # moyenne de chaque canal sur toute l'image : marche quelle que soit la taille d'entree
        # tete de classification : aplatit, dropout, couche lineaire -> un logit par classe (classifier.2 = la couche lineaire)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(cin, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # passage avant : renvoie les logits (N, num_classes)
        return self.classifier(self.pool(self.features(x)))  # caracteristiques -> moyenne spatiale -> classification


class ResidualBlock(nn.Module):  # bloc residuel : sortie = transformation(x) + x
    """Deux convolutions et un raccourci (y = F(x) + x) : le bloc apprend une correction, le gradient passe par le raccourci."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:  # construit le bloc ; stride 2 = taille divisee par 2
        super().__init__()  # initialise la partie nn.Module
        self.body = nn.Sequential(  # chemin principal F(x)
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),  # convolution 3x3 (stride = pas de deplacement)
            nn.BatchNorm2d(out_channels),  # normalisation par canal
            nn.ReLU(inplace=True),  # activation
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),  # seconde convolution 3x3
            nn.BatchNorm2d(out_channels),  # normalisation (le ReLU vient apres l'addition)
        )
        self.shortcut: nn.Module = nn.Identity()  # raccourci : par defaut x passe tel quel
        if stride != 1 or in_channels != out_channels:  # le raccourci doit avoir la forme de la sortie
            # convolution 1x1 (+ normalisation) pour ajuster le nombre de canaux et la taille
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # passage avant : renvoie la sortie du bloc
        return torch.relu(self.body(x) + self.shortcut(x))  # F(x) + raccourci, puis ReLU


class SmallResNet(nn.Module):  # petit reseau residuel : tige, 4 etages, moyenne, tete
    """Tige de convolution, quatre etages residuels qui divisent la taille par 2, moyenne spatiale, tete."""

    def __init__(  # construit le reseau
        self,  # l'objet en construction
        num_classes: int,  # nombre de classes a predire
        channels: tuple[int, ...] = (32, 64, 128, 256),  # canaux en sortie de chaque etage
        dropout: float = 0.3,  # part de valeurs eteintes au hasard avant la derniere couche
        in_channels: int = 3,  # canaux de l'image d'entree
    ) -> None:  # un constructeur ne renvoie rien
        super().__init__()  # initialise la partie nn.Module
        self.stem = nn.Sequential(  # tige : premiere convolution qui passe de l'image aux premiers canaux
            nn.Conv2d(in_channels, channels[0], 3, padding=1, bias=False),  # convolution 3x3
            nn.BatchNorm2d(channels[0]),  # normalisation par canal
            nn.ReLU(inplace=True),  # activation
        )
        stages, cin = [], channels[0]  # liste des etages ; nombre de canaux courant
        for cout in channels:  # pour chaque etage
            stages.append(ResidualBlock(cin, cout, stride=2))  # bloc residuel qui divise la taille par 2
            cin = cout  # la sortie devient l'entree de l'etage suivant
        self.stages = nn.Sequential(*stages)  # les etages a la suite
        self.pool = nn.AdaptiveAvgPool2d(1)  # moyenne spatiale : un nombre par canal
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(cin, num_classes))  # tete : un logit par classe

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # passage avant : renvoie les logits (N, num_classes)
        return self.classifier(self.pool(self.stages(self.stem(x))))  # tige -> etages -> moyenne -> classification
