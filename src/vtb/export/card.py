"""La carte du modele (model_card.json) : le contrat entre l'entrainement et le programme de production.

Un programme C#, C++ ou Python qui charge le fichier ONNX n'a besoin de rien d'autre que cette
carte pour preparer l'image et lire les sorties : taille, canaux, pretraitement OpenCV dans
l'ordre, mise a l'echelle, normalisation, noms et formes des sorties, classes, seuils.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[str], X | None) partout

import json  # ecriture de fichiers JSON (texte structure en dictionnaires et listes)
import platform  # informations sur la machine ; ici la version de Python
from datetime import UTC, datetime  # date et heure ; UTC = temps universel (pas de fuseau horaire local)
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux
from typing import Any, Literal  # Any : n'importe quel type ; Literal : quelques valeurs texte precises

from pydantic import Field  # pydantic : valide les champs a la creation ; Field fixe une valeur par defaut

from vtb.core.config import StrictModel  # modele pydantic qui refuse les champs inconnus (faute de frappe = erreur)

Task = Literal["classification", "detection", "segmentation", "anomaly"]  # taches possibles d'un modele


class InputSpec(StrictModel):  # description de l'entree du reseau (section "input" de la carte)
    """Ce que le reseau attend en entree, apres le pretraitement OpenCV."""

    name: str = "input"  # nom de l'entree dans le fichier ONNX
    layout: Literal["NCHW"] = "NCHW"  # ordre des dimensions : lot (N), canaux (C), hauteur (H), largeur (W)
    height: int  # hauteur de l'image attendue, en pixels
    width: int  # largeur de l'image attendue, en pixels
    channels: int  # 1 (gris) ou 3 (RGB)
    color: Literal["RGB", "GRAY", "BGR"]  # BGR : YOLOX (convention OpenCV)
    dtype: Literal["float32"] = "float32"  # type des nombres : flottants 32 bits
    scale: str = "divide by 255 (8 bits) or 65535 (16 bits)"  # mise a l'echelle des pixels vers [0, 1], ecrite en clair
    # normalisation : (pixel - moyenne) / ecart-type, canal par canal, comme a l'entrainement
    mean: list[float] = Field(default_factory=list)  # vide = pas de normalisation
    std: list[float] = Field(default_factory=list)  # ecarts-types par canal (vide = pas de normalisation)
    # stretch = etirer a la taille ; letterbox = reduire sans deformer puis completer par des bandes de couleur unie
    resize: Literal["stretch", "letterbox", "letterbox_top_left", "none"] = "stretch"  # none : Faster R-CNN redimensionne lui-meme
    letterbox_pad_value: int | None = None  # valeur des bandes de remplissage ; detection YOLO : 114
    dynamic_batch: bool = True  # True : le modele accepte des lots de n'importe quelle taille
    read_color: Literal["rgb", "gray"] | None = None  # couleur de LECTURE du fichier, avant le pretraitement


class OutputSpec(StrictModel):  # description d'une sortie du reseau
    name: str  # nom de la sortie dans le fichier ONNX (ex. "logits")
    shape: list[int | str]  # forme ; un texte (ex. "batch") marque une dimension libre
    description: str  # explication en clair de ce que contient la sortie


class ModelCard(StrictModel):  # la carte complete, ecrite a cote du fichier ONNX
    """Tout ce qu'il faut savoir pour utiliser le modele exporte, sans lire le code Python."""

    task: Task  # tache du modele
    model_name: str  # nom de l'architecture (ex. "resnet18")
    # date de creation en temps universel, format ISO (ex. 2026-01-31T14:05:00Z), calculee a la creation de la carte
    created_utc: str = Field(default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"))
    framework: dict[str, str] = Field(default_factory=dict)  # versions des bibliotheques (voir framework_versions)
    input: InputSpec  # description de l'entree
    preprocess: list[dict[str, Any]] = Field(default_factory=list)  # Pipeline.describe() : a appliquer AVANT la mise a l'echelle
    outputs: list[OutputSpec]  # description de chaque sortie
    class_names: list[str] = Field(default_factory=list)  # noms des classes, dans l'ordre des indices de sortie
    thresholds: dict[str, float] = Field(default_factory=dict)  # seuils de decision (ex. score d'anomalie)
    postprocess: dict[str, Any] = Field(default_factory=dict)  # reglages du post-traitement (ex. NMS pour la detection)
    metrics: dict[str, Any] = Field(default_factory=dict)  # scores obtenus sur le test
    latency_ms: dict[str, float] = Field(default_factory=dict)  # temps de calcul mesures, en millisecondes
    files: dict[str, str] = Field(default_factory=dict)  # fichiers associes (onnx, openvino...)
    notes: str = ""  # remarques libres


def framework_versions() -> dict[str, str]:  # renvoie les versions de Python et des bibliotheques d'export
    """Versions utiles pour reproduire l'export (et diagnostiquer un ecart de resultats)."""
    import onnx  # bibliotheque du format ONNX (import a la demande)
    import onnxruntime  # moteur d'execution ONNX
    import torch  # PyTorch

    return {  # dictionnaire nom -> version
        "python": platform.python_version(),  # ex. "3.11.9"
        "torch": torch.__version__,  # version de PyTorch
        "onnx": onnx.__version__,  # version d'onnx
        "onnxruntime": onnxruntime.__version__,  # version d'onnxruntime
    }


def write_card(card: ModelCard, path: str | Path) -> None:  # ecrit la carte en JSON ; ne renvoie rien
    Path(path).parent.mkdir(parents=True, exist_ok=True)  # cree le dossier de destination s'il n'existe pas
    # model_dump(mode="json") : objet pydantic -> dictionnaire de types JSON simples ; ecrit indente
    Path(path).write_text(json.dumps(card.model_dump(mode="json"), indent=2), encoding="utf-8")


def read_card(path: str | Path) -> ModelCard:  # relit une carte JSON ; renvoie un ModelCard valide
    return ModelCard.model_validate_json(Path(path).read_text(encoding="utf-8"))  # lit, decode et verifie chaque champ
