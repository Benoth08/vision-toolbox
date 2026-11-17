"""Le moteur d'entrainement de la classification : les cinq lignes, l'evaluation, la boucle complete.

Les fonctions ne connaissent ni les images ni le projet : un modele, des DataLoader, une perte,
un optimiseur. Les deux crochets de `fit` (debut d'epoque, passage en mode train) servent au
degel progressif et au gel des BatchNorm sans compliquer la boucle.
"""

from __future__ import annotations  # les annotations de type restent du texte : syntaxe moderne "X | None" partout

import logging  # journal des messages (info, avertissements) au lieu de print
import math  # fonctions mathematiques (cos, pi) pour le calendrier du taux d'apprentissage
import time  # mesure du temps de chaque epoque
from collections.abc import Callable  # type "fonction appelable" pour les annotations (crochets passes a fit)
from dataclasses import dataclass  # dataclass : classe qui sert juste a ranger quelques valeurs nommees
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import numpy as np  # numpy : calcul sur des tableaux de nombres
import pandas as pd  # pandas : tableaux de donnees ; ici l'historique epoque par epoque
import torch  # PyTorch : tenseurs (tableaux de nombres, sur CPU ou GPU), optimiseurs, sauvegarde
import torch.nn as nn  # nn : briques de reseau et fonctions de perte
from torch.utils.data import DataLoader  # DataLoader : distribue les images par lots (batch = groupe d'images traitees ensemble)

logger = logging.getLogger(__name__)  # journal propre a ce module


@dataclass  # genere automatiquement le constructeur a partir des champs ci-dessous
class EvalResult:  # resultat d'une evaluation : perte, accuracy et tableaux detailles par image
    loss: float  # perte moyenne (erreur mesuree par la fonction de perte) sur le jeu evalue
    accuracy: float  # part des images bien classees (entre 0 et 1)
    preds: np.ndarray  # classe predite pour chaque image
    targets: np.ndarray  # vraie classe de chaque image
    probs: np.ndarray  # probabilites par classe pour chaque image (N, nb classes)


def train_one_epoch(  # une epoque d'entrainement (un passage sur tout le train) ; renvoie (perte moyenne, accuracy)
    model: nn.Module,  # le reseau a entrainer
    loader: DataLoader,  # les lots d'images du train
    criterion: nn.Module,  # fonction de perte : mesure l'ecart entre prediction et vraie classe
    optimizer: torch.optim.Optimizer,  # optimiseur : corrige les poids a partir du gradient
    device: torch.device,  # CPU ou GPU : ou faire les calculs
    scaler: torch.amp.GradScaler | None = None,  # outil de precision mixte (float16 sur GPU) ; None = calcul normal float32
    grad_clip: float | None = None,  # plafond de la norme du gradient ; None = pas de plafond
    on_train_mode: Callable[[nn.Module], None] | None = None,  # crochet appele juste apres model.train() (optionnel)
) -> tuple[float, float]:  # renvoie deux nombres : perte moyenne et accuracy
    """Une passe sur le train ; renvoie (perte moyenne, accuracy)."""
    model.train()  # mode entrainement : dropout actif, batchnorm met a jour ses statistiques
    if on_train_mode is not None:  # si un crochet a ete fourni
        on_train_mode(model)  # ex. remettre en eval les BatchNorm des blocs geles
    total, correct, seen = 0.0, 0, 0  # cumul des pertes, nombre de bonnes reponses, nombre d'images vues
    for images, targets in loader:  # pour chaque lot d'images et leurs vraies classes
        # envoie le lot sur le device ; non_blocking : copie vers le GPU en parallele des calculs
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)  # efface les gradients du lot precedent (sinon ils s'additionnent)
        with torch.autocast(device_type=device.type, enabled=scaler is not None):  # calcul en float16 si precision mixte active
            logits = model(images)  # passe avant : logits = scores bruts par classe (avant softmax)
            loss = criterion(logits, targets)  # perte du lot : plus elle est grande, plus le modele se trompe
        if scaler is not None:  # si precision mixte
            scaler.scale(loss).backward()  # multiplie la perte avant de calculer les gradients (evite qu'ils deviennent 0 en float16)
            scaler.unscale_(optimizer)  # remet les gradients a leur vraie echelle (necessaire avant de les plafonner)
        else:  # sinon calcul normal
            loss.backward()  # retropropagation : calcule le gradient (direction de correction) de chaque poids
        if grad_clip:  # si un plafond de gradient est demande
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # reduit les gradients trop grands (evite un pas geant)
        if scaler is not None:  # si precision mixte
            scaler.step(optimizer)  # pas d'optimisation, saute si les gradients contiennent des valeurs infinies
            scaler.update()  # ajuste le facteur d'echelle pour le prochain lot
        else:  # sinon calcul normal
            optimizer.step()  # pas d'optimisation : corrige les poids dans le sens qui reduit la perte
        total += loss.item() * len(targets)  # .item() : un tenseur accumule garderait tout le graphe ; x taille du lot
        correct += (logits.argmax(1) == targets).sum().item()  # argmax = classe au plus grand score ; compte les bonnes
        seen += len(targets)  # nombre d'images traitees jusqu'ici
    return total / max(seen, 1), correct / max(seen, 1)  # moyennes sur l'epoque ; max(., 1) evite une division par 0


@torch.no_grad()  # pas de calcul de gradient pendant l'evaluation : plus rapide, moins de memoire
# evalue le modele sur un jeu (val ou test) sans apprendre ; renvoie un EvalResult
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> EvalResult:
    """Une passe sans apprentissage ; renvoie perte, accuracy, predictions, cibles et probabilites."""
    model.eval()  # mode evaluation : dropout coupe, batchnorm fige
    total, preds, targets_all, probs = 0.0, [], [], []  # cumul de perte et listes a remplir lot par lot
    for images, targets in loader:  # pour chaque lot du jeu evalue
        images, targets = images.to(device), targets.to(device)  # envoie le lot sur CPU ou GPU
        logits = model(images)  # scores bruts par classe
        total += criterion(logits, targets).item() * len(targets)  # cumule la perte du lot (ponderee par sa taille)
        probs.append(torch.softmax(logits.float(), 1).cpu())  # softmax : transforme les scores en probabilites qui somment a 1
        preds.append(logits.argmax(1).cpu())  # classe predite = celle au plus grand score
        targets_all.append(targets.cpu())  # garde les vraies classes
    p, t = torch.cat(preds).numpy(), torch.cat(targets_all).numpy()  # colle tous les lots en deux tableaux numpy
    # perte moyenne, accuracy (0 si jeu vide), predictions, cibles, probabilites
    return EvalResult(total / max(len(t), 1), float((p == t).mean()) if len(t) else 0.0, p, t, torch.cat(probs).numpy())


# cree l'optimiseur demande ; groups = groupes de parametres (chacun avec son taux) ; renvoie l'optimiseur
def build_optimizer(groups: list[dict], name: str, weight_decay: float) -> torch.optim.Optimizer:
    if name == "sgd":  # si on demande SGD (descente de gradient classique)
        # momentum : garde une partie de la direction precedente ; nesterov : variante qui anticipe ce pas
        return torch.optim.SGD(groups, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    return torch.optim.AdamW(groups, weight_decay=weight_decay)  # sinon AdamW : adapte le pas de chaque poids, bon choix par defaut


# cree le calendrier du taux d'apprentissage (scheduler) ; renvoie le scheduler ou None
def build_scheduler(optimizer: torch.optim.Optimizer, name: str, epochs: int, warmup_epochs: int):
    """plateau : divise le taux par 2 quand la perte de val stagne ; cosine : decroissance douce, avec montee lineaire optionnelle."""
    if name == "plateau":  # si on veut baisser le taux seulement quand ca stagne
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)  # /2 apres 2 epoques sans progres
    if name == "cosine":  # si on veut une baisse douce en forme de cosinus
        # multiplicateur du taux pour une epoque donnee (1 = taux plein, 0 = taux nul)
        def factor(epoch: int) -> float:
            if epoch < warmup_epochs:  # pendant la montee lineaire du debut
                return (epoch + 1) / max(warmup_epochs, 1)  # le taux monte de 1/warmup jusqu'a 1
            progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)  # avancement apres la montee, de 0 a 1
            return 0.5 * (1 + math.cos(math.pi * progress))  # demi-cosinus : passe doucement de 1 a 0

        return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)  # applique factor(epoque) au taux de depart
    return None  # "none" : taux constant, pas de scheduler


def fit(  # boucle d'entrainement complete ; renvoie l'historique (une ligne par epoque) dans un DataFrame
    model: nn.Module,  # le reseau a entrainer
    train_loader: DataLoader,  # lots d'entrainement
    val_loader: DataLoader,  # lots de validation (pour choisir le meilleur modele et arreter a temps)
    criterion: nn.Module,  # fonction de perte
    make_optimizer: Callable[[], torch.optim.Optimizer],  # fonction qui fabrique un optimiseur neuf
    scheduler_name: str,  # "plateau", "cosine" ou "none"
    epochs: int,  # nombre maximal d'epoques
    patience: int,  # epoques sans amelioration avant l'arret anticipe
    device: torch.device,  # CPU ou GPU
    checkpoint_path: Path,  # fichier ou sauver le meilleur modele
    warmup_epochs: int = 0,  # epoques de montee lineaire du taux (cosine seulement)
    amp: bool = True,  # precision mixte si GPU CUDA
    grad_clip: float | None = None,  # plafond de la norme du gradient
    on_epoch_start: Callable[[int], bool] | None = None,  # crochet de debut d'epoque ; True = recreer l'optimiseur
    on_train_mode: Callable[[nn.Module], None] | None = None,  # crochet apres model.train() (gel des BatchNorm)
) -> pd.DataFrame:  # renvoie un tableau : epoque, pertes, accuracies, taux
    """Boucle complete : meilleur modele selon la perte de validation, arret anticipe, historique.

    make_optimizer est appele au depart et chaque fois que on_epoch_start renvoie True (un bloc
    vient d'etre degele) : un optimiseur ne connait que les parametres qu'on lui a donnes.
    """
    optimizer = make_optimizer()  # optimiseur de depart
    scheduler = build_scheduler(optimizer, scheduler_name, epochs, warmup_epochs)  # calendrier du taux associe
    scaler = torch.amp.GradScaler("cuda") if amp and device.type == "cuda" else None  # precision mixte seulement sur GPU CUDA
    history, best, bad = [], float("inf"), 0  # historique, meilleure perte de val (infinie au depart), epoques sans progres
    for epoch in range(1, epochs + 1):  # pour chaque epoque, numerotee a partir de 1
        if on_epoch_start is not None and on_epoch_start(epoch):  # si le crochet signale un changement (bloc degele)
            optimizer = make_optimizer()  # nouvel optimiseur qui connait les parametres nouvellement degeles
            scheduler = build_scheduler(optimizer, scheduler_name, epochs, warmup_epochs)  # nouveau calendrier pour lui
        start = time.time()  # heure de debut de l'epoque
        # une epoque d'apprentissage sur le train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, grad_clip, on_train_mode)
        val = evaluate(model, val_loader, criterion, device)  # mesure sur la validation (images jamais apprises)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):  # si scheduler "plateau"
            scheduler.step(val.loss)  # il a besoin de la perte de val pour savoir si ca stagne
        elif scheduler is not None:  # sinon, s'il y a un scheduler (cosine)
            scheduler.step()  # passe a l'epoque suivante du calendrier
        lr = optimizer.param_groups[0]["lr"]  # taux d'apprentissage actuel (premier groupe), pour l'historique
        improved = val.loss < best  # vrai si la perte de val est la meilleure vue jusqu'ici
        if improved:  # si le modele s'est ameliore
            best, bad = val.loss, 0  # retient le nouveau record et remet le compteur d'echecs a 0
            # sauve les poids (state_dict = dictionnaire nom de couche -> valeurs), l'epoque et la perte
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_loss": val.loss}, checkpoint_path)
        else:  # sinon pas d'amelioration
            bad += 1  # une epoque de plus sans progres
        history.append(  # ajoute une ligne a l'historique
            # epoque, perte et accuracy du train et de la val, taux d'apprentissage
            {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val.loss, "val_acc": val.accuracy, "lr": lr}
        )
        logger.info(  # affiche un resume de l'epoque dans le journal
            "epoque %2d | train perte %.4f acc %.3f | val perte %.4f acc %.3f | lr %.1e | %.0fs%s",  # gabarit du message
            epoch,  # numero d'epoque
            train_loss,  # perte du train
            train_acc,  # accuracy du train
            val.loss,  # perte de validation
            val.accuracy,  # accuracy de validation
            lr,  # taux d'apprentissage
            time.time() - start,  # duree de l'epoque en secondes
            " | meilleur" if improved else "",  # marque l'epoque si elle a donne le meilleur modele
        )
        if bad >= patience:  # si trop d'epoques sans amelioration
            logger.info("arret anticipe : %d epoques sans amelioration de la perte de validation", patience)  # previent
            break  # sort de la boucle : inutile de continuer, le modele sur-apprend
    return pd.DataFrame(history)  # historique sous forme de tableau (une ligne par epoque)
