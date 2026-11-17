"""Telecharge a l'avance les poids pre-entraines dans weights/, pour travailler ensuite sans acces a Internet.

    python tools/fetch_weights.py                                   # le jeu par defaut (resnet18, YOLO11n x3, YOLOX-tiny, RF-DETR nano)
    python tools/fetch_weights.py --torchvision resnet18 resnet50 efficientnet_b0 --yolox tiny s \\
        --ultralytics yolo11n.pt yolo11s.pt --hf timm/wide_resnet50_2.tv_in1k

Ou vont les fichiers, et comment la toolbox les retrouve :
    weights/torch/hub/checkpoints/   poids torchvision   -> definir TORCH_HOME=weights/torch
    weights/yolo11n.pt ...           poids Ultralytics   -> ultralytics.weights: weights/yolo11n.pt
    weights/yolox_tiny.pth ...       poids YOLOX         -> yolox.weights: weights/yolox_tiny.pth
    weights/rf-detr-nano.pth ...     poids RF-DETR       -> rfdetr.weights: weights/rf-detr-nano.pth
    weights/hf/                      poids Hugging Face  -> definir HF_HOME=weights/hf (anomalib, timm)
Un manifeste weights/manifest.json donne taille et SHA-256 de chaque fichier (verifier une copie).
Les poids ne vont jamais dans le depot git (.gitignore) : licences et taille.
"""

# Poids pre-entraines : les nombres d'un reseau deja entraine (souvent sur ImageNet ou COCO), qu'on
# reprend comme point de depart au lieu de partir de valeurs au hasard.
from __future__ import annotations  # permet d'ecrire les types "list[str]" sans les evaluer a l'execution

import argparse  # lecture des options de la ligne de commande
import hashlib  # empreintes de fichiers (SHA-256) pour verifier qu'une copie est intacte
import json  # ecriture du manifeste au format JSON
import logging  # journal de messages plutot que des print
import os  # variables d'environnement (TORCH_HOME)
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

logger = logging.getLogger("fetch_weights")  # journal nomme d'apres le script

ULTRALYTICS_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/{name}"  # adresse des poids YOLO11 ({name} = fichier)
YOLOX_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_{variant}.pth"  # poids YOLOX par taille
RFDETR_URLS = {  # stockage de Roboflow (meme adresses que la librairie rfdetr)
    "nano": "https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth",  # RF-DETR nano (le plus petit)
    "small": "https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth",  # RF-DETR small
    "medium": "https://storage.googleapis.com/rfdetr/medium_coco/checkpoint_best_regular.pth",  # RF-DETR medium
    "base": "https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth",  # RF-DETR base (le plus gros ici)
}


def sha256(path: Path) -> str:  # calcule l'empreinte SHA-256 d'un fichier ; renvoie 64 caracteres hexadecimaux
    digest = hashlib.sha256()  # calculateur d'empreinte vide
    with path.open("rb") as handle:  # ouvre le fichier en lecture binaire (octets bruts), ferme a la fin du bloc
        for block in iter(lambda: handle.read(1 << 20), b""):  # lit des blocs de 1 Mo (1 << 20 octets) jusqu'a la fin (b"")
            digest.update(block)  # ajoute le bloc au calcul (pas besoin de charger tout le fichier en memoire)
    return digest.hexdigest()  # renvoie l'empreinte en texte


def download(url: str, target: Path) -> Path:  # telecharge url vers target s'il n'y est pas deja ; renvoie le chemin
    import torch  # PyTorch (importe ici : seulement si on telecharge vraiment)

    if target.exists():  # si le fichier est deja la
        logger.info("deja present : %s", target)  # on le dit
        return target  # et on ne retelecharge pas
    target.parent.mkdir(parents=True, exist_ok=True)  # cree le dossier de destination
    logger.info("telechargement %s", url)  # annonce le telechargement
    torch.hub.download_url_to_file(url, str(target))  # telecharge avec la fonction de PyTorch (barre de progression)
    return target  # renvoie le chemin du fichier


def fetch_torchvision(names: list[str], out: Path) -> list[Path]:  # telecharge des poids torchvision ; renvoie les chemins
    """Poids ImageNet par defaut de torchvision, dans le cache que torchvision relit (TORCH_HOME)."""
    os.environ["TORCH_HOME"] = str(out / "torch")  # dit a PyTorch de ranger son cache dans weights/torch
    from torchvision.models import get_model_weights  # donne la liste des poids disponibles pour un modele torchvision

    files = []  # chemins des fichiers telecharges
    for name in names:  # pour chaque modele demande (ex. "resnet18")
        weights = get_model_weights(name).DEFAULT  # ses poids par defaut (les meilleurs sur ImageNet)
        weights.get_state_dict(progress=True)  # telecharge les poids dans le cache (ou les relit s'ils y sont)
        files.append(out / "torch" / "hub" / "checkpoints" / Path(weights.url).name)  # chemin du fichier dans le cache
    return files  # renvoie la liste des chemins


def fetch_hf(repos: list[str], out: Path) -> list[Path]:  # telecharge des depots Hugging Face ; renvoie leurs dossiers
    from huggingface_hub import snapshot_download  # telechargement d'un depot complet de Hugging Face (site de modeles)

    # telecharge chaque depot dans weights/hf/hub et renvoie le dossier obtenu
    return [Path(snapshot_download(repo_id=repo, cache_dir=str(out / "hf" / "hub"))) for repo in repos]


def main() -> None:  # lit les options, telecharge tout, ecrit le manifeste ; ne renvoie rien
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")  # messages avec l'heure
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # aide = premiere ligne de la docstring
    parser.add_argument("--out", type=Path, default=Path("weights"))  # dossier de destination
    parser.add_argument("--torchvision", nargs="*", default=["resnet18"])  # modeles torchvision (plusieurs possibles)
    parser.add_argument("--ultralytics", nargs="*", default=["yolo11n.pt", "yolo11n-obb.pt", "yolo11n-seg.pt"])  # fichiers YOLO11
    parser.add_argument("--yolox", nargs="*", default=["tiny"])  # tailles YOLOX (tiny, s, m...)
    parser.add_argument("--rfdetr", nargs="*", default=["nano"], choices=sorted(RFDETR_URLS))  # tailles RF-DETR
    parser.add_argument("--hf", nargs="*", default=[], help="depots Hugging Face (ex. timm/wide_resnet50_2.tv_in1k)")  # depots HF
    args = parser.parse_args()  # lit les options tapees
    files: list[Path] = []  # tous les fichiers telecharges, pour le manifeste
    files += fetch_torchvision(args.torchvision, args.out)  # poids torchvision
    files += [download(ULTRALYTICS_URL.format(name=n), args.out / n) for n in args.ultralytics]  # poids Ultralytics (YOLO11)
    files += [download(YOLOX_URL.format(variant=v), args.out / f"yolox_{v}.pth") for v in args.yolox]  # poids YOLOX
    files += [download(RFDETR_URLS[s], args.out / f"rf-detr-{s}.pth") for s in args.rfdetr]  # poids RF-DETR
    if args.hf:  # si des depots Hugging Face sont demandes
        fetch_hf(args.hf, args.out)  # les telecharge
        # ajoute tous les vrais fichiers du cache HF (pas les liens symboliques, qui pointent vers ces memes fichiers)
        files += [p for p in (args.out / "hf").rglob("*") if p.is_file() and not p.is_symlink()]
    # manifeste : pour chaque fichier present, sa taille et son empreinte (pour verifier une copie plus tard)
    manifest = {str(p.relative_to(args.out)): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in files if p.exists()}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")  # ecrit le manifeste en JSON lisible
    total = sum(v["bytes"] for v in manifest.values()) / 1e6  # taille totale en mega-octets
    logger.info("%d fichiers, %.0f Mo -> %s (manifeste : manifest.json)", len(manifest), total, args.out)  # bilan


if __name__ == "__main__":  # vrai seulement quand le script est lance directement
    main()  # lance le telechargement
