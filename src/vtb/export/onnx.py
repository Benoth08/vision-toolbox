"""Export ONNX generique, verification contre PyTorch, latence PyTorch / onnxruntime / OpenVINO.

Un export qui passe sans verification ne vaut rien : un operateur mal traduit ne leve aucune
erreur, il donne des sorties fausses. On compare donc toujours les sorties sur de vraies images,
avec un lot d'une autre taille que celui de l'export (le lot dynamique est ainsi teste aussi).
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[str], X | None) partout

import inspect  # inspection du code : ici, lire le nom du premier parametre de forward()
import logging  # messages d'information et d'avertissement, plus souples que print
import time  # mesure du temps (perf_counter : chronometre precis)
from collections.abc import Sequence  # Sequence : type "liste ou tuple", pour annoter les parametres
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import numpy as np  # numpy : calcul sur des tableaux de nombres
import onnxruntime as ort  # onnxruntime : moteur qui execute un modele ONNX, sans PyTorch
import torch  # PyTorch : tenseurs (tableaux de nombres, sur CPU ou GPU) et reseaux de neurones
import torch.nn as nn  # nn : briques de reseaux ; nn.Module = classe de base de tout modele

logger = logging.getLogger(__name__)  # journal propre a ce module


class OnnxVerificationError(RuntimeError):  # erreur levee quand le modele ONNX ne donne pas les memes sorties que PyTorch
    """Les sorties PyTorch et ONNX different au-dela de la tolerance."""


def _first_arg_name(model: nn.Module) -> str:  # renvoie le nom du premier parametre de model.forward (souvent "x")
    return next(iter(inspect.signature(model.forward).parameters))  # signature = liste des parametres ; on prend le premier


# ecrit le fichier ONNX avec l'un des deux exporteurs de PyTorch ; ne renvoie rien
def _export(
    model: nn.Module,  # modele PyTorch a exporter
    example: torch.Tensor,  # lot d'exemple : l'exporteur fait passer ces donnees dans le modele pour tracer le calcul
    path: Path,  # fichier .onnx a ecrire
    input_name: str,  # nom donne a l'entree dans le fichier ONNX
    output_names: Sequence[str],  # noms donnes aux sorties
    dynamic: bool,  # True : taille de lot libre (dynamique) ; False : figee a celle de l'exemple
    dynamo: bool,  # True : nouvel exporteur (torch.export / dynamo) ; False : ancien exporteur (trace TorchScript)
    opset: int,  # version de l'ensemble d'operateurs ONNX (opset) a utiliser
) -> None:  # None : ne renvoie rien, ecrit un fichier
    if dynamo:  # nouvel exporteur
        # lot dynamique : la dimension 0 (le lot) du premier argument peut aller de 1 a 1024 ; sinon formes figees (None)
        shapes = {_first_arg_name(model): {0: torch.export.Dim("batch", min=1, max=1024)}} if dynamic else None
        torch.onnx.export(  # ecrit le fichier ONNX
            model,  # modele a exporter
            (example,),  # arguments d'exemple, en tuple
            str(path),  # fichier de sortie
            input_names=[input_name],  # nom de l'entree
            output_names=list(output_names),  # noms des sorties
            dynamic_shapes=shapes,  # dimensions libres (ou None)
            dynamo=True,  # utilise le nouvel exporteur
            external_data=False,  # tous les poids dans le meme fichier (pas de fichier .data a cote)
            opset_version=opset,  # version des operateurs ONNX
        )
    else:  # ancien exporteur
        # lot dynamique : la dimension 0 de l'entree et de chaque sortie s'appelle "batch" (taille libre)
        axes = {input_name: {0: "batch"}, **{n: {0: "batch"} for n in output_names}} if dynamic else None
        torch.onnx.export(  # ecrit le fichier ONNX
            model,  # modele a exporter
            (example,),  # arguments d'exemple, en tuple
            str(path),  # fichier de sortie
            input_names=[input_name],  # nom de l'entree
            output_names=list(output_names),  # noms des sorties
            dynamic_axes=axes,  # dimensions libres (ou None)
            opset_version=opset,  # version des operateurs ONNX
            dynamo=False,  # utilise l'ancien exporteur
        )


def batch_size_of(path: str | Path) -> int | None:  # renvoie la taille de lot figee du modele, ou None si elle est libre
    """Taille de lot imposee par le modele ONNX, None si le lot est dynamique."""
    dim = create_session(path).get_inputs()[0].shape[0]  # premiere dimension declaree de l'entree (le lot)
    return dim if isinstance(dim, int) else None  # un entier = lot fige ; un nom ("batch") = lot dynamique


def _accepts_batch(path: Path, example: torch.Tensor, n: int) -> bool:  # True si le modele ONNX tourne vraiment avec un lot de n
    """Le modele tourne-t-il vraiment avec un lot de n ? (une forme declaree dynamique peut cacher un reshape fige)"""
    try:  # toute erreur pendant l'essai veut dire "non"
        session = create_session(path)  # charge le modele ONNX dans onnxruntime
        if batch_size_of(path) not in (None, n):  # lot fige a une autre valeur que n
            return False  # inutile d'essayer
        # lot de n copies de la premiere image (repeat : n fois sur l'axe 0, 1 fois sur les autres), en tableau numpy
        x = example[:1].repeat(n, *([1] * (example.ndim - 1))).numpy()
        outputs = session.run(None, {session.get_inputs()[0].name: x})  # execute le modele (None = toutes les sorties)
        # un reshape fige peut tourner avec un autre lot et rendre une sortie de mauvaise taille :
        # les sorties de la toolbox commencent toutes par le lot, on le verifie
        return all(o.ndim > 0 and o.shape[0] == n for o in outputs)
    except Exception:  # noqa: BLE001 - tout echec veut dire "lot non dynamique"
        return False  # le modele n'accepte pas ce lot


# exporte un modele PyTorch en ONNX (lot dynamique si possible) ; renvoie le chemin du fichier ecrit
def export_onnx(
    model: nn.Module,  # modele PyTorch entraine
    example: torch.Tensor,  # lot d'exemple (forme lot x canaux x hauteur x largeur)
    path: str | Path,  # fichier .onnx a ecrire
    output_names: Sequence[str],  # noms des sorties (ex. ["logits"])
    input_name: str = "input",  # nom de l'entree
    dynamic_batch: bool = True,  # True : on veut pouvoir envoyer des lots de n'importe quelle taille
    opset: int = 18,  # version des operateurs ONNX
) -> Path:  # renvoie le chemin du fichier
    """Exporte sur CPU, en mode eval. Lot dynamique demande : exporteur dynamo, puis ancien exporteur,
    puis, en dernier recours, lot fixe de 1 (certains modeles figent la taille du lot dans un reshape).
    Chaque tentative est verifiee en executant le modele avec un AUTRE lot que l'exemple.
    `batch_size_of(path)` dit ensuite si le lot est dynamique (a recopier dans la carte du modele).
    """
    path = Path(path)  # convertit en Path
    path.parent.mkdir(parents=True, exist_ok=True)  # cree le dossier de destination s'il n'existe pas
    model = model.eval().cpu()  # mode evaluation (dropout coupe, batchnorm fige) et poids sur le CPU
    example = example.cpu()  # l'exemple aussi sur le CPU
    if not dynamic_batch:  # lot fixe demande
        _export(model, example, path, input_name, output_names, False, True, opset)  # export direct, nouvel exporteur
        logger.info("ONNX ecrit : %s (%.1f Mo, lot fixe)", path, path.stat().st_size / 1e6)  # taille du fichier en Mo
        return path  # termine
    for dynamo in (True, False):  # essaie le nouvel exporteur, puis l'ancien
        try:  # l'export peut echouer selon le modele
            _export(model, example, path, input_name, output_names, True, dynamo, opset)  # export avec lot dynamique
        except Exception as error:  # certains modeles ne passent pas encore par torch.export
            # previent (premiere ligne du message d'erreur, 200 caracteres au plus)
            logger.warning("exporteur %s en echec (%s)", "dynamo" if dynamo else "ancien", str(error).splitlines()[0][:200])
            continue  # essaie l'exporteur suivant
        if _accepts_batch(path, example, len(example) + 1):  # verifie avec un lot d'une autre taille que l'exemple
            logger.info("ONNX ecrit : %s (%.1f Mo, lot dynamique)", path, path.stat().st_size / 1e6)  # succes
            return path  # termine
        logger.warning("exporteur %s : le lot reste fige dans le graphe", "dynamo" if dynamo else "ancien")  # lot pas vraiment libre
    _export(model, example[:1], path, input_name, output_names, False, False, opset)  # dernier recours : lot fixe de 1
    # previent que le programme de production devra envoyer les images une par une
    logger.warning("ONNX ecrit avec un lot FIXE de 1 : %s (le programme de production envoie les images une par une)", path)
    return path  # renvoie le chemin du fichier


def create_session(path: str | Path) -> ort.InferenceSession:  # charge un modele ONNX ; renvoie une session prete a executer
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])  # execution sur CPU (le meme partout)


@torch.no_grad()  # pas de calcul de gradient (derivees pour l'entrainement) : plus rapide, moins de memoire
# compare les sorties PyTorch et ONNX sur un vrai lot ; renvoie un rapport des ecarts (ou leve OnnxVerificationError)
def verify_onnx(model: nn.Module, path: str | Path, batch: torch.Tensor, atol: float = 1e-4, rtol: float = 1e-3) -> dict[str, float]:
    """Compare chaque sortie PyTorch et ONNX ; echec si l'ecart max depasse atol + rtol x (valeur max de la sortie).

    La tolerance est relative a l'ECHELLE de la sortie et non a chaque valeur : une racine carree
    pres de zero (distance PatchCore) ou un logit de 10 donnent des ecarts absolus differents pour
    une meme precision relative. Un vrai defaut d'export donne des ecarts de l'ordre de la sortie
    elle-meme, des milliers de fois au-dessus de ce seuil.
    """
    model = model.eval().cpu()  # mode evaluation, sur CPU
    torch_out = model(batch.cpu())  # sorties de reference calculees par PyTorch
    torch_outputs = [torch_out] if isinstance(torch_out, torch.Tensor) else list(torch_out)  # toujours une liste de sorties
    session = create_session(path)  # charge le modele ONNX
    name, fixed = session.get_inputs()[0].name, batch_size_of(path)  # nom de l'entree ; taille de lot figee (ou None)
    if fixed is None:  # lot dynamique : tout le lot d'un coup
        onnx_outputs = session.run(None, {name: batch.cpu().numpy()})  # execute avec le lot converti en numpy
    else:  # lot fige : par paquets de la taille imposee, puis recollage
        # execute le modele sur chaque paquet de "fixed" images
        chunks = [session.run(None, {name: batch[i : i + fixed].cpu().numpy()}) for i in range(0, len(batch), fixed)]
        onnx_outputs = [np.concatenate([c[k] for c in chunks]) for k in range(len(chunks[0]))]  # recolle chaque sortie k
    if len(torch_outputs) != len(onnx_outputs):  # une sortie en trop ou manquante : zip la masquerait
        raise RuntimeError(f"ONNX rend {len(onnx_outputs)} sortie(s), PyTorch {len(torch_outputs)}")  # on arrete l'export
    report = {}  # rapport des ecarts
    for i, (t, o) in enumerate(zip(torch_outputs, onnx_outputs)):  # pour chaque sortie : version PyTorch t, version ONNX o
        t = t.numpy()  # tenseur PyTorch -> tableau numpy
        if t.shape != o.shape:  # formes differentes : export faux
            raise OnnxVerificationError(f"sortie {i} : forme {o.shape} au lieu de {t.shape}")  # arret avec message clair
        diff = float(np.abs(t - o).max())  # plus grand ecart absolu entre les deux sorties
        scale = float(np.abs(t).max())  # echelle de la sortie : sa plus grande valeur absolue
        report[f"output_{i}_max_abs_diff"] = diff  # note l'ecart
        report[f"output_{i}_scale"] = scale  # note l'echelle
        if diff > atol + rtol * scale:  # ecart au-dela de la tolerance (absolue + relative a l'echelle)
            # arret : l'export ne reproduit pas le modele
            raise OnnxVerificationError(f"sortie {i} : ecart max {diff:.2e} pour une echelle {scale:.2e} (lot de {len(batch)})")
    report["batch_size"] = int(len(batch))  # taille du lot verifie
    # message de succes, nombres en notation scientifique courte (ex. 3.20e-06)
    logger.info("verification ONNX OK : %s", {k: f"{v:.2e}" if isinstance(v, float) else v for k, v in report.items()})
    return report  # renvoie le rapport


def to_openvino(onnx_path: str | Path) -> Path:  # convertit un ONNX au format OpenVINO ; renvoie le chemin du .xml
    """Convertit en OpenVINO IR (.xml + .bin) : plus rapide sur CPU Intel. Necessite l'extra export."""
    import openvino as ov  # OpenVINO : moteur d'inference d'Intel (import a la demande : paquet optionnel)

    xml = Path(onnx_path).with_suffix(".xml")  # meme nom que le .onnx, extension .xml (le .bin des poids est ecrit a cote)
    ov.save_model(ov.convert_model(str(onnx_path)), str(xml))  # convertit puis enregistre
    logger.info("OpenVINO ecrit : %s", xml)  # message d'information
    return xml  # renvoie le chemin du .xml


@torch.no_grad()  # pas de calcul de gradient : on mesure seulement la prediction
# mesure la latence (temps de calcul d'un lot) des differents moteurs ; renvoie un dict en millisecondes
def measure_latency(
    model: nn.Module,  # modele PyTorch
    onnx_path: str | Path | None,  # fichier ONNX a mesurer ; None = ne pas mesurer onnxruntime
    sample: torch.Tensor,  # lot d'entree utilise pour la mesure
    runs: int = 20,  # nombre de passages chronometres (on en fait la moyenne)
    openvino_xml: str | Path | None = None,  # modele OpenVINO a mesurer ; None = ne pas mesurer
) -> dict[str, float]:  # renvoie un dictionnaire moteur -> millisecondes
    """Millisecondes par lot (sample) pour PyTorch, onnxruntime et OpenVINO, apres deux passages de chauffe."""

    def timed(fn) -> float:  # chronometre la fonction fn ; renvoie la duree moyenne d'un appel en millisecondes
        for _ in range(2):  # deux passages de chauffe (non chronometres) : allocations memoire, caches
            fn()  # appel de chauffe
        start = time.perf_counter()  # top depart du chronometre (en secondes)
        for _ in range(runs):  # passages chronometres
            fn()  # appel mesure
        return round(1000 * (time.perf_counter() - start) / runs, 2)  # moyenne en millisecondes, 2 decimales

    model = model.eval().cpu()  # mode evaluation, sur CPU (comparaison equitable avec onnxruntime CPU)
    x = sample.cpu()  # lot sur CPU
    result = {"torch_cpu_ms": timed(lambda: model(x)), "batch": int(len(x))}  # latence PyTorch et taille du lot
    if onnx_path is not None:  # si un fichier ONNX est donne
        session = create_session(onnx_path)  # charge le modele ONNX
        name, array = session.get_inputs()[0].name, x.numpy()  # nom de l'entree ; lot en numpy (prepare une seule fois)
        result["onnxruntime_cpu_ms"] = timed(lambda: session.run(None, {name: array}))  # latence onnxruntime
    if openvino_xml is not None:  # si un modele OpenVINO est donne
        import openvino as ov  # import a la demande (paquet optionnel)

        compiled = ov.Core().compile_model(str(openvino_xml), "CPU")  # charge et prepare le modele pour le CPU
        array = x.numpy()  # lot en numpy
        result["openvino_cpu_ms"] = timed(lambda: compiled(array))  # latence OpenVINO
    logger.info("latence : %s", result)  # affiche le bilan
    return result  # renvoie les latences
