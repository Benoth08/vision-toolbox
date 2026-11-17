"""Outils d'entrainement : les diagnostics a lancer AVANT un long entrainement, et les petites briques de boucle.

    from vtb.toolkit import training as tt
    curve, lr = tt.lr_find(model, train_loader, nn.CrossEntropyLoss())    # quel taux d'apprentissage
    check = tt.overfit_one_batch(model, next(iter(train_loader)), nn.CrossEntropyLoss())
    print(check.attrs["verdict"])                                          # le modele PEUT-il apprendre ?
    info, layers = tt.model_summary(model, (1, 3, 224, 224))               # parametres, taille, GFLOPs
    print(tt.benchmark(model, torch.randn(1, 3, 224, 224)))                 # latence

Convention des lots : (entrees, cibles) comme un DataLoader de classification ou de segmentation.
Pour un autre format (detection, dictionnaires), passer `compute_loss(model, batch) -> perte`.
lr_find et overfit_one_batch remettent le modele (et l'optimiseur) dans l'etat d'avant : ce sont
des diagnostics, ils ne doivent pas entrainer le modele dans votre dos.
"""

from __future__ import annotations  # permet d'ecrire les types "float | None" partout sans les evaluer a l'execution

import copy  # copie profonde d'objets (pour sauvegarder les poids et les restaurer ensuite)
import logging  # journal de messages plutot que des print
import math  # fonctions mathematiques de base (infini, test "nombre fini")
import statistics  # moyenne rapide d'une liste de nombres
import time  # chronometre precis (perf_counter)
from collections.abc import Callable, Iterator, Sequence  # types pour annoter : fonction, iterateur, liste
from typing import Any  # Any : "n'importe quel type" (un lot peut etre un tuple, un dict...)

import numpy as np  # numpy : calcul sur des tableaux de nombres
import pandas as pd  # pandas : tableaux avec des colonnes nommees, pour les courbes et les resumes
import torch  # PyTorch : tenseurs (tableaux de nombres qui peuvent aller sur GPU) et calcul des gradients
from torch import nn  # nn : briques des reseaux de neurones (couches, fonctions de perte, Module)

from vtb.core.runtime import count_parameters, seed_everything  # noqa: F401 - reexporte pour la boite a outils

logger = logging.getLogger(__name__)  # journal propre a ce fichier

# Perte (loss) : nombre qui mesure a quel point les predictions sont fausses ; l'entrainement la fait baisser.
LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]  # type : fonction (sorties, cibles) -> perte
ComputeLoss = Callable[[nn.Module, Any], torch.Tensor]  # type : fonction (modele, lot) -> perte, pour les lots d'autre format


def _to_device(batch: Any, device: torch.device | str) -> Any:  # envoie un lot sur le CPU ou le GPU ; renvoie le lot deplace
    """Deplace recursivement tenseurs, listes, tuples et dictionnaires."""
    # Lot (batch) : groupe d'exemples traites ensemble en une seule passe du modele.
    if isinstance(batch, torch.Tensor):  # si c'est un tenseur
        return batch.to(device)  # on le copie sur l'appareil voulu ("cpu" ou "cuda" = GPU)
    if isinstance(batch, dict):  # si c'est un dictionnaire
        return {k: _to_device(v, device) for k, v in batch.items()}  # on deplace chaque valeur (appel recursif)
    if isinstance(batch, (list, tuple)):  # si c'est une liste ou un tuple
        return type(batch)(_to_device(v, device) for v in batch)  # on deplace chaque element et on garde le meme type
    return batch  # autre chose (texte, nombre) : rien a deplacer


# choisit la facon de calculer la perte d'un lot ; renvoie une fonction (modele, lot) -> perte
def _loss_function(loss_fn: LossFn | None, compute_loss: ComputeLoss | None) -> ComputeLoss:
    """(entrees, cibles) + loss_fn par defaut ; compute_loss pour tout autre format de lot."""
    if compute_loss is not None:  # si l'utilisateur a donne sa propre fonction
        return compute_loss  # on l'utilise telle quelle
    if loss_fn is None:  # si on n'a ni l'une ni l'autre
        raise ValueError("donner loss_fn (lot = (entrees, cibles)) ou compute_loss(model, batch)")  # on ne sait pas calculer la perte

    def default(model: nn.Module, batch: Any) -> torch.Tensor:  # cas standard : le lot est (entrees, cibles)
        inputs, targets = batch[0], batch[1]  # separe les images et les reponses attendues
        return loss_fn(model(inputs), targets)  # passe les images dans le modele et compare aux cibles

    return default  # renvoie la fonction construite


def _cycle(loader: Any) -> Iterator[Any]:  # parcourt le DataLoader en boucle ; produit les lots un par un, sans fin
    """Itere sans fin sur un DataLoader (lr_find peut demander plus de lots qu'une epoque)."""
    # Epoque : un passage complet sur toutes les images d'entrainement.
    while True:  # boucle infinie : on recommence le DataLoader a chaque fin d'epoque
        yielded = False  # pour detecter un DataLoader vide
        for batch in loader:  # pour chaque lot du DataLoader
            yielded = True  # au moins un lot a ete produit
            yield batch  # donne ce lot a l'appelant et attend qu'il demande le suivant
        if not yielded:  # si aucun lot n'a ete produit sur toute l'epoque
            raise ValueError("le DataLoader est vide")  # on arrete, sinon la boucle tournerait pour rien


def _default_optimizer(model: nn.Module, lr: float) -> torch.optim.Optimizer:  # cree l'optimiseur par defaut (AdamW) ; le renvoie
    # Optimiseur : algorithme qui modifie les poids du modele a partir des gradients pour faire baisser la perte.
    params = [p for p in model.parameters() if p.requires_grad]  # seulement les poids entrainables (pas les poids geles)
    if not params:  # si tout est gele
        raise ValueError("aucun parametre entrainable (tout est gele ?)")  # rien a entrainer : erreur claire
    return torch.optim.AdamW(params, lr=lr, weight_decay=0.01)  # AdamW : optimiseur courant ; weight_decay freine les poids trop grands


# --------------------------------------------------------------------------- recherche du taux d'apprentissage


def lr_find(  # test du taux d'apprentissage ; renvoie (courbe lr / perte, lr suggere)
    model: nn.Module,  # le modele a tester
    loader: Any,  # le DataLoader d'entrainement (fournit les lots)
    loss_fn: LossFn | None = None,  # fonction de perte (ex. nn.CrossEntropyLoss())
    *,  # les arguments suivants doivent etre nommes
    optimizer: torch.optim.Optimizer | None = None,  # optimiseur a utiliser (None = AdamW par defaut)
    compute_loss: ComputeLoss | None = None,  # ou bien : fonction (modele, lot) -> perte pour un lot d'autre format
    start_lr: float = 1e-7,  # premier taux d'apprentissage essaye (tres petit)
    end_lr: float = 1.0,  # dernier taux essaye (tres grand)
    steps: int = 100,  # nombre de pas (un lot et un taux par pas)
    device: torch.device | str = "cpu",  # appareil de calcul : "cpu" ou "cuda" (GPU)
    smooth: float = 0.05,  # force du lissage de la courbe (petit = courbe plus lisse)
    diverge: float = 4.0,  # on arrete quand la perte lissee depasse 4 fois son meilleur niveau
) -> tuple[pd.DataFrame, float]:  # (courbe, taux suggere)
    """Test du taux d'apprentissage (Smith 2015) : lr croissant exponentiellement, un lot par pas.

    Renvoie la courbe (lr, loss, smoothed) et un lr suggere : la pente la plus forte de la perte
    lissee AVANT son minimum (la ou le modele apprend le plus vite). Regle pratique : choisir entre
    ce lr et un dixieme du lr du minimum ; avec un modele pre-entraine en fine-tuning complet,
    prendre plutot 3 a 10 fois moins. Le modele et l'optimiseur sont restaures a la fin.
    Arret anticipe quand la perte lissee depasse `diverge` fois son meilleur niveau.
    """
    # Taux d'apprentissage (lr) : taille du pas fait a chaque mise a jour des poids ; trop petit = lent, trop grand = diverge.
    model.to(device)  # place les poids du modele sur l'appareil de calcul
    compute = _loss_function(loss_fn, compute_loss)  # fonction qui calcule la perte d'un lot
    optimizer = optimizer or _default_optimizer(model, start_lr)  # optimiseur donne, sinon AdamW
    model_state = copy.deepcopy(model.state_dict())  # copie de tous les poids (state_dict) pour les restaurer a la fin
    optim_state = copy.deepcopy(optimizer.state_dict())  # copie de l'etat interne de l'optimiseur, pour la meme raison
    was_training = model.training  # retient si le modele etait en mode entrainement ou evaluation
    factor = (end_lr / start_lr) ** (1.0 / max(steps - 1, 1))  # multiplicateur du lr a chaque pas pour aller de start_lr a end_lr
    rows, average, best = [], 0.0, math.inf  # lignes de la courbe, moyenne mobile de la perte, meilleure perte lissee
    batches = _cycle(loader)  # source de lots sans fin
    model.train()  # mode entrainement (dropout et batch norm se comportent comme a l'entrainement)
    try:  # le bloc finally restaurera le modele meme en cas d'erreur
        for step in range(steps):  # pour chaque pas du test
            lr = start_lr * factor**step  # taux de ce pas : croissance exponentielle
            for group in optimizer.param_groups:  # pour chaque groupe de poids de l'optimiseur
                group["lr"] = lr  # on impose le nouveau taux
            batch = _to_device(next(batches), device)  # prend le lot suivant et l'envoie sur l'appareil
            optimizer.zero_grad(set_to_none=True)  # efface les gradients du pas precedent (sinon ils s'additionnent)
            loss = compute(model, batch)  # passe avant : calcule la perte sur ce lot
            value = float(loss.detach())  # valeur de la perte en nombre Python (detach : sans suivi des gradients)
            if not math.isfinite(value):  # si la perte vaut NaN ou l'infini : le lr est beaucoup trop grand
                logger.info("lr_find : perte non finie a lr=%.2e, arret", lr)  # on le note dans le journal
                break  # et on arrete le test
            # Gradient : pour chaque poids, dans quel sens et de combien le bouger pour faire baisser la perte.
            loss.backward()  # passe arriere : calcule les gradients
            optimizer.step()  # met a jour les poids avec ces gradients et le lr du pas
            average = smooth * value + (1 - smooth) * average  # moyenne mobile exponentielle : lisse le bruit d'un lot a l'autre
            smoothed = average / (1 - (1 - smooth) ** (step + 1))  # correction du biais de la moyenne mobile
            rows.append({"lr": lr, "loss": value, "smoothed": smoothed})  # garde le point de la courbe
            best = min(best, smoothed)  # meilleure perte lissee vue jusqu'ici
            if step > 5 and smoothed > diverge * best:  # apres quelques pas, si la perte explose
                break  # inutile de continuer : le lr est trop grand
    finally:  # dans tous les cas : on remet tout comme avant
        model.load_state_dict(model_state)  # restaure les poids d'origine
        optimizer.load_state_dict(optim_state)  # restaure l'etat de l'optimiseur
        model.train(was_training)  # remet le mode d'origine (entrainement ou evaluation)
    curve = pd.DataFrame(rows)  # la courbe en tableau : colonnes lr, loss, smoothed
    return curve, suggest_lr(curve)  # renvoie la courbe et le lr suggere


def suggest_lr(curve: pd.DataFrame, skip_start: float = 0.1) -> float:  # lit la courbe de lr_find ; renvoie un lr conseille
    """Pente la plus negative de la perte lissee en fonction de log10(lr), avant le minimum de la courbe."""
    if len(curve) < 5:  # si la courbe a trop peu de points pour calculer une pente
        return float(curve["lr"].iloc[0]) if len(curve) else float("nan")  # premier lr essaye, ou NaN si la courbe est vide
    lrs = np.log10(curve["lr"].to_numpy())  # lr en echelle logarithmique (1e-4 -> -4) : les pas y sont reguliers
    losses = curve["smoothed"].to_numpy()  # perte lissee
    stop = int(np.argmin(losses))  # position du minimum de la perte
    start = int(len(curve) * skip_start)  # on saute le debut (10 %) : courbe encore bruitee
    if stop - start < 3:  # si le minimum arrive trop tot pour mesurer une pente
        return float(curve["lr"].iloc[max(stop // 2, 0)])  # repli : le lr a mi-chemin avant le minimum
    slopes = np.gradient(losses[start : stop + 1], lrs[start : stop + 1])  # pente de la perte entre le debut et le minimum
    return float(10 ** lrs[start + int(np.argmin(slopes))])  # lr ou la perte descend le plus vite (repasse de log10 a lr)


# --------------------------------------------------------------------------- sur-apprendre un lot


def overfit_one_batch(  # verifie que le modele peut apprendre par coeur un lot ; renvoie l'historique (DataFrame) avec un verdict
    model: nn.Module,  # le modele a tester
    batch: Any,  # un seul lot, ex. next(iter(train_loader))
    loss_fn: LossFn | None = None,  # fonction de perte
    *,  # les arguments suivants doivent etre nommes
    compute_loss: ComputeLoss | None = None,  # ou bien : fonction (modele, lot) -> perte
    steps: int = 100,  # nombre de pas d'entrainement sur ce meme lot
    lr: float = 1e-3,  # taux d'apprentissage
    device: torch.device | str = "cpu",  # appareil de calcul
    target_ratio: float = 0.1,  # reussite si la perte finale < 10 % de la perte de depart
) -> pd.DataFrame:  # historique pas par pas ; verdict dans .attrs
    """Le premier diagnostic quand rien n'apprend : un modele correct DOIT apprendre par coeur un seul lot.

    Renvoie la perte (et l'exactitude pour une classification) a chaque pas ; df.attrs["ok"] et
    df.attrs["verdict"] disent si la perte finale est tombee sous target_ratio x la perte initiale.
    Si ce n'est pas le cas, le probleme est dans le code (cibles, perte, gel, lr), pas dans les
    donnees ni dans le nombre d'epoques. Les poids sont restaures a la fin.
    """
    model.to(device)  # place le modele sur l'appareil de calcul
    compute = _loss_function(loss_fn, compute_loss)  # fonction qui calcule la perte du lot
    batch = _to_device(batch, device)  # envoie le lot sur l'appareil, une fois pour toutes
    state = copy.deepcopy(model.state_dict())  # sauvegarde des poids, pour les restaurer a la fin
    was_training = model.training  # mode d'origine du modele
    optimizer = _default_optimizer(model, lr)  # optimiseur AdamW neuf
    # classification simple si : perte standard, lot (entrees, cibles), cibles = entiers (long) en 1 dimension (une classe par image)
    is_classif = compute_loss is None and isinstance(batch, (list, tuple)) and batch[1].dtype == torch.long and batch[1].ndim == 1
    rows = []  # une ligne d'historique par pas
    model.train()  # mode entrainement
    try:  # le bloc finally restaurera les poids meme en cas d'erreur
        for step in range(steps):  # pour chaque pas d'entrainement
            optimizer.zero_grad(set_to_none=True)  # efface les gradients du pas precedent
            loss = compute(model, batch)  # passe avant : perte sur le lot
            if not torch.isfinite(loss):  # si la perte vaut NaN ou l'infini
                rows.append({"step": step, "loss": float("nan")})  # on note l'echec
                break  # et on arrete
            loss.backward()  # passe arriere : calcule les gradients
            optimizer.step()  # met a jour les poids
            row = {"step": step, "loss": float(loss.detach())}  # ligne d'historique : numero du pas et perte
            if is_classif and (step % 10 == 0 or step == steps - 1):  # en classification, tous les 10 pas et au dernier
                with torch.no_grad():  # pas de calcul de gradient ici : juste une mesure
                    # exactitude : part des images dont la classe de plus haut score (argmax) est la bonne
                    row["accuracy"] = float((model(batch[0]).argmax(1) == batch[1]).float().mean())
            rows.append(row)  # garde la ligne
    finally:  # dans tous les cas
        model.load_state_dict(state)  # restaure les poids d'origine : ce test n'entraine pas le modele pour de vrai
        model.train(was_training)  # remet le mode d'origine
    history = pd.DataFrame(rows)  # l'historique en tableau
    first, last = history["loss"].iloc[0], history["loss"].iloc[-1]  # perte au premier et au dernier pas
    ok = bool(np.isfinite(last) and (last <= target_ratio * first or last < 1e-2))  # reussi si la perte a fondu (ou est tres basse)
    if not np.isfinite(last):  # si la perte a fini en NaN ou infini
        verdict = "ECHEC : la perte devient NaN/inf -> lr trop fort, log(0) dans la perte, entrees non normalisees."  # causes probables
    elif ok:  # si le test est reussi
        # message de reussite avec la perte de depart et d'arrivee
        verdict = f"OK : perte {first:.3f} -> {last:.4f} en {len(history)} pas ; le modele et la perte sont capables d'apprendre."
    else:  # la perte n'a pas assez baisse
        verdict = (  # message d'echec avec la liste des erreurs de code les plus frequentes
            f"ECHEC : perte {first:.3f} -> {last:.3f} (attendu < {target_ratio * first:.3f}). A verifier : cibles "
            "(indices 0..C-1, bon type), perte adaptee aux sorties (logits pour CrossEntropy, pas de softmax avant), "
            "parametres geles par erreur, lr (essayer x10 et /10), model.train() / no_grad oublie, augmentation trop forte."
        )
    history.attrs["ok"] = ok  # attrs : petit dictionnaire d'informations attache au tableau
    history.attrs["verdict"] = verdict  # le verdict en francais
    return history  # renvoie l'historique (avec ok et verdict)


# --------------------------------------------------------------------------- resume du modele


def model_summary(  # resume du modele ; renvoie (dict des chiffres globaux, tableau par sous-module)
    model: nn.Module,  # le modele a decrire
    input_size: Sequence[int] = (1, 3, 224, 224),  # forme de l'entree d'essai : (lot, canaux, hauteur, largeur)
    *,  # les arguments suivants doivent etre nommes
    device: torch.device | str = "cpu",  # appareil de calcul
    dtype: torch.dtype = torch.float32,  # type des nombres de l'entree d'essai
) -> tuple[dict, pd.DataFrame]:  # (infos, couches)
    """Parametres (total, entrainables, geles), taille en Mo, GFLOPs d'une passe avant, forme de sortie.

    Le tableau donne, pour chaque sous-module de premier niveau : type, parametres, entrainables et
    forme de sortie (hooks). GFLOPs : operations flottantes comptees par torch (une multiplication-
    addition = 2), pour la taille d'entree donnee ; la latence reelle se mesure avec benchmark().
    """
    from torch.utils.flop_counter import FlopCounterMode  # compteur d'operations de PyTorch (importe seulement ici)

    model.to(device)  # place le modele sur l'appareil
    was_training = model.training  # mode d'origine
    model.eval()  # mode evaluation : comportement d'inference (dropout coupe, batch norm fige)
    shapes: dict[str, str] = {}  # forme de sortie de chaque sous-module, remplie par les hooks

    # Hook : fonction "accrochee" a une couche, appelee automatiquement a chaque passage dans cette couche.
    def hook(name: str) -> Callable:  # fabrique un hook qui retient la forme de sortie sous ce nom ; renvoie le hook
        def record(_module: nn.Module, _inputs: Any, output: Any) -> None:  # appelee apres la couche, avec sa sortie
            first = output[0] if isinstance(output, (list, tuple)) and output else output  # sortie multiple : on regarde la premiere
            if isinstance(first, dict):  # si la sortie est un dictionnaire
                first = next(iter(first.values()), None)  # on prend sa premiere valeur
            # forme du tenseur (ex. "1x64x56x56"), ou le nom du type si ce n'est pas un tenseur
            shapes[name] = "x".join(map(str, first.shape)) if isinstance(first, torch.Tensor) else type(output).__name__

        return record  # renvoie la fonction a accrocher

    children = list(model.named_children())  # sous-modules de premier niveau (nom, module)
    handles = [child.register_forward_hook(hook(name)) for name, child in children]  # accroche un hook a chacun ; garde les poignees
    example = torch.zeros(tuple(input_size), device=device, dtype=dtype)  # entree d'essai remplie de zeros
    try:  # le bloc finally decrochera les hooks meme en cas d'erreur
        # no_grad : pas de gradients (plus rapide) ; FlopCounterMode compte les operations pendant le bloc
        with torch.no_grad(), FlopCounterMode(display=False) as counter:
            output = model(example)  # une passe avant (les hooks remplissent shapes)
        flops = counter.get_total_flops()  # nombre total d'operations flottantes de cette passe
    finally:  # dans tous les cas
        for handle in handles:  # pour chaque hook accroche
            handle.remove()  # on le decroche (sinon il resterait pour toujours sur le modele)
        model.train(was_training)  # remet le mode d'origine
    first = output[0] if isinstance(output, (list, tuple)) and output else output  # sortie principale du modele
    # taille memoire des poids (parametres) et des buffers (ex. moyennes de batch norm) : nombre d'elements x octets par element
    size_bytes = sum(p.numel() * p.element_size() for p in model.parameters()) + sum(b.numel() * b.element_size() for b in model.buffers())
    total = count_parameters(model, trainable_only=False)  # nombre total de poids
    trainable = count_parameters(model, trainable_only=True)  # nombre de poids entrainables
    info = {  # chiffres globaux
        "total_params": total,  # tous les poids
        "trainable_params": trainable,  # poids qui seront modifies par l'entrainement
        "frozen_params": total - trainable,  # poids geles (non modifies)
        "size_mb": round(size_bytes / 2**20, 2),  # taille en mega-octets (2**20 octets)
        "gflops": float(f"{flops / 1e9:.4g}"),  # milliards d'operations, 4 chiffres significatifs
        "flops": int(flops),  # nombre exact d'operations
        "input": "x".join(map(str, input_size)),  # forme de l'entree en texte
        "output": "x".join(map(str, first.shape)) if isinstance(first, torch.Tensor) else type(output).__name__,  # forme de sortie
    }
    layers = pd.DataFrame(  # tableau : une ligne par sous-module
        [  # liste de dictionnaires construite par la boucle ci-dessous
            {  # ligne d'un sous-module
                "module": name,  # son nom dans le modele
                "type": type(child).__name__,  # sa classe (Conv2d, Sequential...)
                "params": sum(p.numel() for p in child.parameters()),  # nombre de poids
                "trainable": sum(p.numel() for p in child.parameters() if p.requires_grad),  # dont entrainables
                "output": shapes.get(name, ""),  # forme de sortie notee par le hook ("" si pas appele)
            }
            for name, child in children  # pour chaque sous-module de premier niveau
        ]
    )
    return info, layers  # renvoie les chiffres globaux et le tableau


# --------------------------------------------------------------------------- taille de lot maximale


def max_batch_size(  # cherche le plus grand lot qui tient en memoire GPU ; renvoie un entier, ou None sur CPU
    model: nn.Module,  # le modele
    input_shape: Sequence[int],  # forme d'UNE entree, sans la dimension du lot, ex. (3, 224, 224)
    *,  # les arguments suivants doivent etre nommes
    device: torch.device | str = "cuda",  # GPU a tester
    start: int = 2,  # premiere taille de lot essayee
    limit: int = 1024,  # taille maximale essayee
    train: bool = True,  # True : teste passe avant + arriere (comme a l'entrainement)
    margin: float = 0.9,  # marge de securite : on renvoie 90 % du maximum trouve
) -> int | None:  # la taille de lot conseillee
    """Plus grand lot qui tient en memoire GPU (passe avant + arriere si train), par doublement puis dichotomie.

    input_shape sans la dimension du lot, ex. (3, 224, 224). Renvoie None sur CPU (la RAM n'est pas
    le facteur limitant de la meme facon). Le resultat est multiplie par `margin` (0.9) : la
    fragmentation et les augmentations prennent un peu de memoire en plus pendant le vrai entrainement.
    """
    device = torch.device(device)  # en objet device pour lire son type
    if device.type != "cuda" or not torch.cuda.is_available():  # si ce n'est pas un GPU CUDA disponible
        logger.info("max_batch_size : seulement sur GPU CUDA")  # on le signale
        return None  # rien a mesurer
    model.to(device)  # place le modele sur le GPU
    was_training = model.training  # mode d'origine
    model.train(train)  # mode entrainement si on teste l'entrainement, sinon evaluation

    def fits(batch: int) -> bool:  # essaie un lot de cette taille ; renvoie True s'il tient en memoire
        try:  # le GPU peut manquer de memoire
            x = torch.randn(batch, *input_shape, device=device)  # lot d'essai aleatoire directement sur le GPU
            if train:  # en mode entrainement
                out = model(x)  # passe avant
                out = out[0] if isinstance(out, (list, tuple)) else out  # sortie multiple : on prend la premiere
                out.float().mean().backward()  # fausse perte (moyenne) + passe arriere : alloue la memoire des gradients
                model.zero_grad(set_to_none=True)  # libere les gradients
            else:  # en mode inference
                with torch.no_grad():  # pas de gradients
                    model(x)  # passe avant seule
            return True  # pas d'erreur : ca tient
        except torch.cuda.OutOfMemoryError:  # si le GPU manque de memoire
            return False  # ca ne tient pas
        finally:  # dans tous les cas
            torch.cuda.empty_cache()  # rend au GPU la memoire liberee, pour que l'essai suivant parte de zero

    good, bad = 0, None  # plus grande taille qui tient, plus petite taille qui ne tient pas
    batch = start  # on commence petit
    while batch <= limit:  # doublement jusqu'a la limite
        if fits(batch):  # si cette taille tient
            good, batch = batch, batch * 2  # on la retient et on double
        else:  # sinon
            bad = batch  # premiere taille qui ne tient pas
            break  # on arrete de doubler
    if bad is not None:  # si on a trouve une taille trop grande
        while bad - good > 1:  # dichotomie : on coupe l'intervalle en deux tant qu'il reste des tailles a tester
            middle = (good + bad) // 2  # taille au milieu
            if fits(middle):  # si elle tient
                good = middle  # la limite est au-dessus
            else:  # sinon
                bad = middle  # la limite est en dessous
    model.train(was_training)  # remet le mode d'origine
    return max(1, int(good * margin)) if good else None  # applique la marge ; None si meme le plus petit lot ne tient pas


# --------------------------------------------------------------------------- briques de boucle


class EarlyStopping:  # arret anticipe : dit quand arreter l'entrainement parce que la validation ne progresse plus
    """Arret anticipe : stop apres `patience` epoques sans amelioration d'au moins min_delta.

    stopper = EarlyStopping(patience=8, mode="max")
    for epoch in ...:
        ...
        if stopper.step(val_f1):
            break
        if stopper.improved:
            torch.save(model.state_dict(), "best.pt")
    """

    def __init__(self, patience: int = 10, mode: str = "min", min_delta: float = 0.0) -> None:  # cree l'objet avec ses reglages
        if mode not in ("min", "max"):  # mode "min" (ex. perte) ou "max" (ex. F1) seulement
            raise ValueError("mode : min ou max")  # autre valeur : erreur claire
        self.patience, self.mode, self.min_delta = patience, mode, min_delta  # epoques de patience, sens, amelioration minimale
        self.best: float | None = None  # meilleure valeur vue (None au depart)
        self.best_epoch = -1  # epoque de cette meilleure valeur
        self.counter = 0  # nombre d'epoques de suite sans amelioration
        self.epoch = -1  # numero de la derniere epoque enregistree
        self.improved = False  # True si la derniere epoque a ameliore le record

    def step(self, value: float) -> bool:  # enregistre la valeur d'une epoque ; renvoie True s'il faut s'arreter
        """Enregistre la valeur de l'epoque ; True s'il faut s'arreter."""
        self.epoch += 1  # une epoque de plus
        better = (  # True si la valeur bat le record d'au moins min_delta
            self.best is None  # premiere epoque : forcement un record
            or (self.mode == "min" and value < self.best - self.min_delta)  # mode min : plus petit que le record
            or (self.mode == "max" and value > self.best + self.min_delta)  # mode max : plus grand que le record
        )
        self.improved = bool(better)  # retient s'il y a eu amelioration (pour sauvegarder le meilleur modele)
        if better:  # si c'est un nouveau record
            self.best, self.best_epoch, self.counter = float(value), self.epoch, 0  # on le note et on remet le compteur a zero
        else:  # pas d'amelioration
            self.counter += 1  # une epoque de plus sans progres
        return self.counter >= self.patience  # True quand la patience est epuisee


class AverageMeter:  # moyenne courante d'une valeur au fil des lots (ex. perte moyenne d'une epoque)
    """Moyenne courante ponderee (perte moyenne d'une epoque sur des lots de tailles differentes)."""

    def __init__(self) -> None:  # cree le compteur
        self.reset()  # demarre a zero

    def reset(self) -> None:  # remet le compteur a zero (au debut de chaque epoque)
        self.sum, self.count = 0.0, 0  # somme ponderee et nombre d'exemples

    def update(self, value: float, n: int = 1) -> None:  # ajoute une valeur moyenne calculee sur n exemples
        self.sum += float(value) * n  # ponderation : un lot de 32 compte 32 fois
        self.count += n  # n exemples de plus

    @property  # property : se lit comme un attribut (meter.avg), sans parentheses
    def avg(self) -> float:  # renvoie la moyenne courante
        return self.sum / self.count if self.count else float("nan")  # NaN si rien n'a encore ete ajoute


class Timer:  # chronometre ; utilisable avec "with"
    """Chronometre : `with Timer() as t: ...` puis t.seconds ; ou t = Timer().start() ... t.stop()."""

    def __init__(self) -> None:  # cree un chronometre arrete
        self.seconds = 0.0  # duree mesuree
        self._start: float | None = None  # instant de depart (None = arrete)

    def start(self) -> Timer:  # demarre le chronometre ; renvoie lui-meme (pour ecrire Timer().start())
        self._start = time.perf_counter()  # note l'instant de depart (horloge precise)
        return self  # renvoie l'objet

    def stop(self) -> float:  # arrete le chronometre ; renvoie la duree en secondes
        if self._start is not None:  # s'il tournait
            self.seconds = time.perf_counter() - self._start  # duree = maintenant - depart
            self._start = None  # le chronometre est arrete
        return self.seconds  # renvoie la duree

    def __enter__(self) -> Timer:  # appelee au debut d'un bloc "with Timer() as t"
        return self.start()  # demarre et renvoie le chronometre (c'est t)

    def __exit__(self, *_exc: Any) -> None:  # appelee a la sortie du bloc "with", meme en cas d'erreur
        self.stop()  # arrete le chronometre

    def __repr__(self) -> str:  # texte affiche quand on montre l'objet ; renvoie ex. "Timer(1.234 s)"
        return f"Timer({self.seconds:.3f} s)"  # duree avec 3 chiffres apres la virgule


# --------------------------------------------------------------------------- latence


# mesure la latence (temps de reponse) d'un modele ; renvoie un dict de durees en millisecondes et le debit
def benchmark(model_or_callable: nn.Module | Callable, example: Any, *, warmup: int = 5, runs: int = 30) -> dict[str, float]:
    """Latence d'une inference : moyenne, mediane, p95, min (ms) et debit (images / s).

    Les premiers passages (warmup) sont exclus : allocation memoire, choix des algorithmes. Sur GPU,
    synchronisation avant chaque mesure (sinon on mesure le lancement, pas le calcul). Un nn.Module
    est mis en eval et sous inference_mode ; une fonction (session ONNX, predicteur) est appelee telle quelle.
    """
    is_module = isinstance(model_or_callable, nn.Module)  # True si c'est un modele PyTorch
    cuda = isinstance(example, torch.Tensor) and example.is_cuda  # True si l'entree est sur GPU
    if is_module:  # pour un modele PyTorch
        was_training = model_or_callable.training  # mode d'origine
        model_or_callable.eval()  # mode evaluation (inference)

    def call() -> None:  # une inference ; ne renvoie rien (seul le temps compte)
        if is_module:  # modele PyTorch
            with torch.inference_mode():  # inference_mode : comme no_grad, mais encore plus leger
                model_or_callable(example)  # passe avant
        else:  # autre fonction (ex. session ONNX)
            model_or_callable(example)  # appel tel quel
        if cuda:  # sur GPU, les calculs partent en arriere-plan
            torch.cuda.synchronize()  # on attend qu'ils soient vraiment finis avant d'arreter le chrono

    try:  # le bloc finally remettra le mode d'origine
        for _ in range(warmup):  # quelques passages de chauffe, non mesures
            call()  # premiere allocation memoire, choix des algorithmes...
        times = []  # durees mesurees en millisecondes
        for _ in range(runs):  # passages mesures
            start = time.perf_counter()  # instant de depart
            call()  # une inference
            times.append((time.perf_counter() - start) * 1000)  # duree en millisecondes
    finally:  # dans tous les cas
        if is_module:  # pour un modele PyTorch
            model_or_callable.train(was_training)  # remet le mode d'origine
    batch = int(example.shape[0]) if hasattr(example, "shape") and len(example.shape) > 0 else 1  # nombre d'images par appel
    mean = statistics.fmean(times)  # duree moyenne
    return {  # renvoie les statistiques de temps
        "mean_ms": round(mean, 3),  # moyenne
        "p50_ms": round(float(np.percentile(times, 50)), 3),  # mediane : la moitie des appels sont plus rapides
        "p95_ms": round(float(np.percentile(times, 95)), 3),  # 95 % des appels sont plus rapides que cette valeur
        "min_ms": round(min(times), 3),  # le plus rapide
        "batch": batch,  # images par appel
        "images_per_s": round(1000 * batch / mean, 1),  # debit : images traitees par seconde
    }
