"""Auto-encodeur convolutif : apprendre a reconstruire le normal ; ce qu'il reconstruit mal est anormal.

Limite de principe (module 19) : un auto-encodeur generalise aussi aux anomalies faciles a
reconstruire (une tache lisse, un corps etranger uniforme) et les rate. Les methodes a
caracteristiques pre-entrainees (PatchCore, PaDiM) n'ont pas ce defaut ; l'auto-encodeur reste
utile loin d'ImageNet (radiographie, spectres) ou quand la taille du modele compte.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[int], X | None) partout

import logging  # journal des messages (infos d'avancement) plutot que des print
import time  # mesure de la duree de chaque epoque
from pathlib import Path  # Path : represente un chemin de fichier, sous Windows comme sous Linux

import pandas as pd  # pandas : tableaux de donnees (ici l'historique d'entrainement)
import torch  # PyTorch : calcul sur des tenseurs (tableaux de nombres multi-dimensions) et reseaux de neurones
import torch.nn as nn  # nn : les briques des reseaux (couches, modules)
import torch.nn.functional as F  # F : fonctions sans parametres (pertes, interpolation...)
from torch.utils.data import DataLoader  # DataLoader : decoupe un jeu de donnees en lots (batchs) d'images

logger = logging.getLogger(__name__)  # journal propre a ce module (son nom apparait dans les messages)


def _down(cin: int, cout: int) -> nn.Sequential:  # bloc qui divise hauteur et largeur par 2 ; renvoie la suite de couches
    """Conv 4x4 pas 2 : divise la resolution par 2."""
    # convolution 4x4 pas 2 (filtre glissant qui saute 1 pixel sur 2) -> BatchNorm (stabilise les valeurs) -> LeakyReLU (non-linearite)
    return nn.Sequential(nn.Conv2d(cin, cout, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, inplace=True))


def _up(cin: int, cout: int) -> nn.Sequential:  # bloc qui multiplie hauteur et largeur par 2 ; renvoie la suite de couches
    """Convolution transposee 4x4 pas 2 : multiplie la resolution par 2."""
    # convolution transposee (l'inverse d'une convolution a pas 2 : agrandit l'image) -> BatchNorm -> ReLU
    return nn.Sequential(nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class ConvAutoencoder(nn.Module):  # auto-encodeur : compresse l'image (encodeur) puis la reconstruit (decodeur)
    """(N, C, S, S) dans [0, 1] -> reconstruction de meme forme dans [0, 1] (sigmoide). S multiple de 2^len(widths)."""

    # construit le reseau : widths = nombre de canaux a chaque etage, channels = 3 (couleur) ou 1 (gris)
    def __init__(self, latent_channels: int = 32, widths: list[int] | tuple[int, ...] = (32, 64, 128, 256), channels: int = 3) -> None:
        super().__init__()  # initialise la partie nn.Module (obligatoire avant d'ajouter des couches)
        layers, previous = [], channels  # liste des blocs de l'encodeur ; previous = canaux en entree du bloc suivant
        for width in widths:  # pour chaque etage de l'encodeur
            layers.append(_down(previous, width))  # ajoute un bloc qui divise la taille par 2 et passe a width canaux
            previous = width  # le bloc suivant recoit width canaux
        layers.append(nn.Conv2d(previous, latent_channels, 1))  # le goulot : convolution 1x1 qui reduit a latent_channels canaux
        self.encoder = nn.Sequential(*layers)  # encodeur = tous les blocs enchaines dans l'ordre
        decoder = [nn.Conv2d(latent_channels, widths[-1], 1), nn.ReLU(inplace=True)]  # sortie du goulot : on remonte a widths[-1] canaux
        for cin, cout in zip(list(widths)[::-1], list(widths)[::-1][1:]):  # pour chaque paire d'etages en sens inverse (256->128...)
            decoder.append(_up(cin, cout))  # bloc qui double la taille et reduit les canaux
        # dernier agrandissement vers le nombre de canaux de l'image, puis sigmoide pour ramener chaque pixel dans [0, 1]
        decoder += [nn.ConvTranspose2d(widths[0], channels, 4, stride=2, padding=1), nn.Sigmoid()]
        self.decoder = nn.Sequential(*decoder)  # decodeur = tous ces blocs enchaines

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # passage avant : image -> image reconstruite
        return self.decoder(self.encoder(x))  # compresse puis reconstruit


class ReconstructionMap(nn.Module):  # transforme l'auto-encodeur en detecteur : renvoie l'erreur de reconstruction par pixel
    """Image -> carte brute d'erreur de reconstruction (N, 1, S, S), moyenne sur les canaux."""

    def __init__(self, autoencoder: ConvAutoencoder) -> None:  # garde l'auto-encodeur entraine
        super().__init__()  # initialise nn.Module
        self.autoencoder = autoencoder  # l'auto-encodeur devient un sous-module (exporte avec lui)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # image -> carte d'erreur (grande la ou l'image est mal reconstruite)
        # ecart au carre entre image et reconstruction, moyenne sur les canaux (dim=1) ; keepdim garde la forme (N, 1, S, S)
        return ((x - self.autoencoder(x)) ** 2).mean(dim=1, keepdim=True)


# une epoque = un passage complet sur les donnees ; entraine si un optimiseur est donne, sinon evalue ; renvoie la perte moyenne
def _epoch(model: nn.Module, loader: DataLoader, device: torch.device, optimizer: torch.optim.Optimizer | None = None) -> float:
    model.train(optimizer is not None)  # mode entrainement (BatchNorm apprend) ou evaluation (BatchNorm fige)
    total, seen = 0.0, 0  # somme des pertes et nombre d'images vues
    with torch.set_grad_enabled(optimizer is not None):  # calcule les gradients (directions de correction) seulement en entrainement
        for images, _, _ in loader:  # pour chaque lot d'images (masque et etiquette ignores : on n'apprend que le normal)
            images = images.to(device)  # envoie le lot sur le processeur de calcul (CPU ou GPU)
            loss = F.mse_loss(model(images), images)  # MSE : ecart moyen au carre entre la reconstruction et l'image d'origine
            if optimizer is not None:  # si on est en entrainement
                optimizer.zero_grad(set_to_none=True)  # efface les gradients du lot precedent
                loss.backward()  # calcule les gradients de la perte par rapport a chaque poids
                optimizer.step()  # l'optimiseur corrige les poids dans le sens qui reduit la perte
            total += loss.item() * len(images)  # perte du lot ponderee par sa taille (le dernier lot peut etre plus petit)
            seen += len(images)  # compte les images traitees
    return total / max(seen, 1)  # perte moyenne par image (max evite une division par zero)


def fit_autoencoder(  # entraine l'auto-encodeur avec arret anticipe ; renvoie l'historique epoque par epoque
    model: ConvAutoencoder,  # le reseau a entrainer
    train_loader: DataLoader,  # lots d'images normales d'entrainement
    val_loader: DataLoader,  # lots d'images normales de validation (pour choisir le meilleur modele)
    epochs: int,  # nombre maximal d'epoques
    lr: float,  # taux d'apprentissage : taille des pas de correction des poids
    weight_decay: float,  # penalite qui garde les poids petits (limite le sur-apprentissage)
    patience: int,  # nombre d'epoques sans amelioration avant d'arreter
    device: torch.device,  # CPU ou GPU
    checkpoint: Path,  # fichier ou sauver le meilleur modele
) -> pd.DataFrame:  # renvoie un tableau : une ligne par epoque
    """Meilleur modele selon la MSE de reconstruction des normales de validation, arret anticipe."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)  # optimiseur AdamW : regle de mise a jour des poids
    # divise le taux d'apprentissage par 2 si la perte de validation stagne 4 epoques
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)
    history, best, bad = [], float("inf"), 0  # historique, meilleure perte vue (infinie au depart), epoques sans progres
    for epoch in range(1, epochs + 1):  # pour chaque epoque, numerotee a partir de 1
        start = time.time()  # heure de debut pour mesurer la duree
        # une epoque d'entrainement (avec optimiseur) puis une epoque d'evaluation sur la validation (sans)
        train_loss, val_loss = _epoch(model, train_loader, device, optimizer), _epoch(model, val_loader, device)
        scheduler.step(val_loss)  # le scheduler regarde la perte de validation pour ajuster le taux d'apprentissage
        improved = val_loss < best  # vrai si c'est la meilleure validation jusqu'ici
        if improved:  # si le modele s'est ameliore
            best, bad = val_loss, 0  # retient le record et remet le compteur d'echecs a zero
            # sauvegarde les poids (state_dict) avec l'epoque et la perte : on rechargera ce meilleur modele
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, checkpoint)
        else:  # sinon (pas de progres)
            bad += 1  # une epoque de plus sans amelioration
        # ajoute une ligne a l'historique : epoque, pertes et taux d'apprentissage actuel
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"]})
        logger.info(  # affiche le bilan de l'epoque
            "epoque %2d | train MSE %.5f | val MSE %.5f | %.0fs%s",  # modele du message : chaque %... recoit une valeur ci-dessous
            epoch,  # numero d'epoque
            train_loss,  # perte d'entrainement
            val_loss,  # perte de validation
            time.time() - start,  # duree de l'epoque en secondes
            " | meilleur" if improved else "",  # mention si nouveau meilleur modele
        )
        if bad >= patience:  # si trop d'epoques sans progres
            logger.info("arret anticipe : %d epoques sans amelioration", patience)  # on previent
            break  # arret anticipe : inutile de continuer, le modele ne progresse plus
    return pd.DataFrame(history)  # convertit la liste de dictionnaires en tableau pandas
