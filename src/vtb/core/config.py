"""Configuration : un YAML par experience, valide par pydantic.

Chaque recette definit sa classe de configuration en assemblant les sections communes
ci-dessous (donnees, decoupage, pretraitement, execution). `extra="forbid"` fait d'une faute de
frappe dans le YAML une erreur claire au demarrage, au lieu d'un reglage ignore en silence.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type (ex. str | None) sans les evaluer tout de suite

from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux
from typing import Any, Literal, TypeVar  # Any = n'importe quel type ; Literal = liste fermee de valeurs ; TypeVar = type generique

import yaml  # PyYAML : lit et ecrit les fichiers YAML (fichiers de reglages lisibles par un humain)

# pydantic : verifie automatiquement les types et valeurs des reglages lus dans le YAML
from pydantic import BaseModel, ConfigDict, Field

# type generique : "une classe de configuration qui herite de StrictModel" (sert a typer load_config)
ConfigT = TypeVar("ConfigT", bound="StrictModel")

IMAGE_EXTENSIONS = [".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"]  # extensions de fichier reconnues comme des images


class StrictModel(BaseModel):  # classe de base de toutes les sections de configuration : refuse les cles inconnues
    """Base de toutes les sections : refuse les cles inconnues."""

    # extra="forbid" : une cle mal orthographiee dans le YAML provoque une erreur ;
    # protected_namespaces=() : autorise les champs dont le nom commence par "model_"
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class RuntimeConfig(StrictModel):  # section "runtime" : ou ecrire les resultats, graine, materiel de calcul
    """Reglages d'execution, communs a toutes les taches."""

    output_dir: str = "outputs/default"  # dossier ou seront ecrits les resultats (modele, journaux, figures)
    seed: int = 42  # graine aleatoire : meme valeur = memes tirages au hasard = resultats reproductibles
    device: str = "auto"  # materiel de calcul : auto | cpu | cuda | cuda:0 ... (cuda = carte graphique NVIDIA)
    # nombre de processus qui chargent les images en parallele ; 0 sous Windows dans un notebook, 2 a 4 dans un script
    num_workers: int = 0
    deterministic: bool = False  # True : resultats identiques sur GPU, un peu plus lent

    @property  # @property : la methode s'utilise comme un attribut (config.output_path, sans parentheses)
    def output_path(self) -> Path:  # renvoie le dossier de sortie sous forme de Path (plus pratique qu'une chaine)
        return Path(self.output_dir)  # convertit la chaine en objet chemin


class DataConfig(StrictModel):  # section "data" : ou sont les images et comment les lire
    """Ou sont les images et comment elles sont rangees."""

    root: str  # dossier racine du jeu de donnees (obligatoire : pas de valeur par defaut)
    # rangement des fichiers : un dossier par classe, par split, un CSV d'etiquettes, a plat, format MVTec, image + masque, annotations
    layout: Literal["class_folders", "split_folders", "csv", "flat", "mvtec", "image_mask", "annotations"] = "class_folders"
    labels_csv: str | None = None  # layout csv : chemin du CSV, avec les colonnes path_column et label_column
    path_column: str = "filename"  # layout csv : nom de la colonne qui contient le chemin de l'image
    label_column: str = "label"  # layout csv : nom de la colonne qui contient la classe de l'image
    masks_dir: str | None = None  # layout image_mask : dossier des masques (un masque par image)
    mask_suffix: str = ""  # ex. "_mask" si le masque de a.png s'appelle a_mask.png
    annotations_dir: str | None = None  # layout annotations : dossier ou fichier d'annotations
    # format des annotations (boites, polygones ou masques) : chaque logiciel d'annotation a le sien
    annotation_format: Literal["voc", "yolo", "coco", "labelme", "labelstudio", "cvat", "masks", "vtb"] = "voc"
    # options passees au lecteur d'annotations (masks : kind, class_names, min_area...) ; default_factory = nouveau dict vide a chaque fois
    annotation_options: dict[str, Any] = Field(default_factory=dict)
    # d'ou vient la classe d'un objet : du fichier d'annotation, ou du 1er sous-dossier (cas des masques binaires)
    object_label_from: Literal["annotation", "folder"] = "annotation"
    extensions: list[str] = Field(default_factory=lambda: list(IMAGE_EXTENSIONS))  # extensions acceptees (copie de la liste par defaut)
    color: Literal["rgb", "gray"] = "rgb"  # lecture en couleur ou en niveaux de gris (l'acier est souvent en gris)
    group_regex: str | None = None  # expression reguliere qui extrait un identifiant de groupe (piece, bobine) du nom de fichier
    groups_csv: str | None = None  # CSV path,group (ex. groups.csv de recipes/explore : quasi-doublons regroupes)
    timestamp_regex: str | None = None  # expression reguliere qui extrait une date du nom de fichier (decoupage temporel)
    max_images: int | None = None  # essai rapide : sous-echantillon aleatoire reproductible (jamais en production)


class SplitConfig(StrictModel):  # section "split" : comment couper les images en entrainement / validation / test
    """Decoupage train / val / test, calcule une fois et enregistre."""

    file: str = "outputs/split.csv"  # CSV ou le decoupage est enregistre (reutilise d'une execution a l'autre)
    # stratified = memes proportions de classes partout ; grouped = un groupe reste entier ; temporal = par date
    strategy: Literal["stratified", "grouped", "temporal", "random", "provided"] = "stratified"
    val_fraction: float = Field(0.15, ge=0.0, lt=1.0)  # part des images en validation (15 %) ; ge/lt = bornes verifiees
    test_fraction: float = Field(0.15, ge=0.0, lt=1.0)  # part des images en test final (15 %), jamais vues a l'entrainement
    n_folds: int | None = None  # ajoute une colonne fold (validation croisee : k decoupages successifs) sur train + val


class PreprocessStep(StrictModel):  # une etape de pretraitement decrite dans le YAML (nom + parametres libres)
    """Une operation de pretraitement : son nom et ses parametres (valides par le registre)."""

    model_config = ConfigDict(extra="allow")  # ici on accepte des cles en plus : ce sont les parametres de l'operation

    name: str  # nom de l'operation (ex. "clahe", "resize"), cherche ensuite dans le registre

    def params(self) -> dict[str, Any]:  # renvoie les parametres de l'operation (tout sauf le nom) dans un dict
        return {k: v for k, v in self.model_dump().items() if k != "name"}  # model_dump = tous les champs ; on retire "name"


class AugmentStep(StrictModel):  # une augmentation (deformation aleatoire de l'image pour varier les exemples)
    """Une augmentation aleatoire, appliquee seulement a l'entrainement."""

    model_config = ConfigDict(extra="allow")  # accepte des cles en plus : les parametres de l'augmentation

    name: str  # nom de l'augmentation (ex. "hflip" = retournement horizontal)

    def params(self) -> dict[str, Any]:  # renvoie les parametres de l'augmentation (tout sauf le nom)
        return {k: v for k, v in self.model_dump().items() if k != "name"}  # tous les champs sauf "name"


class ExportConfig(StrictModel):  # section "export" : formats de sortie du modele entraine
    """Export du modele final."""

    onnx: bool = True  # export ONNX : format standard de modele, lisible sans PyTorch (C#, C++, navigateur...)
    openvino: bool = False  # conversion supplementaire pour CPU Intel
    opset: int = 18  # version du jeu d'operations ONNX utilisee a l'export
    atol: float = 1e-4  # tolerance absolue : ecart maximal permis entre sorties PyTorch et ONNX
    rtol: float = 1e-3  # tolerance relative : ecart permis en proportion de la valeur
    latency_runs: int = 20  # nombre de passages pour mesurer la latence (temps de calcul d'une prediction)


def load_config(path: str | Path, config_class: type[ConfigT]) -> ConfigT:  # lit un YAML et renvoie l'objet de config valide
    """Lit un YAML et le valide ; l'erreur pydantic donne la cle et la raison."""
    # safe_load : lit le YAML en dict sans executer de code ; "or {}" si le fichier est vide
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return config_class.model_validate(raw)  # verifie chaque cle et chaque type ; leve une erreur claire sinon


def save_config(config: StrictModel, path: str | Path) -> None:  # ecrit la configuration complete dans un YAML
    """Copie la configuration effective (valeurs par defaut comprises) a cote des resultats."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)  # cree le dossier parent s'il n'existe pas encore
    # model_dump(mode="json") : config -> dict de types simples ; sort_keys=False garde l'ordre des champs
    Path(path).write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False, allow_unicode=False), encoding="utf-8")
