"""Pre-annotation : les predictions d'un modele deviennent des propositions a corriger.

La boucle qui fait gagner le plus de temps d'etiquetage :
1. annoter a la main 100 a 200 images ; entrainer un premier modele ;
2. predire le reste, exporter les predictions vers l'outil (Label Studio, CVAT, X-AnyLabeling) ;
3. faire corriger en commencant par les images ou le modele HESITE (order_by_uncertainty) :
   ce sont celles qui apprennent le plus au modele suivant ;
4. reimporter les corrections, reentrainer, recommencer.
Regle : une proposition est relue par un humain avant de servir ; le jeu de TEST reste annote
entierement a la main (sinon on mesure l'accord du modele avec lui-meme).
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[str], X | None) partout

from collections.abc import Sequence  # Sequence : type "liste ou tuple", pour annoter les parametres
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux
from typing import Literal  # Literal : type qui n'accepte que quelques valeurs texte precises

import numpy as np  # numpy : calcul sur des tableaux de nombres

from vtb.annotate.cvat import write_cvat_xml  # ecrivain CVAT (XML)
from vtb.annotate.labelme import write_labelme_folder  # ecrivain labelme / X-AnyLabeling (un JSON par image)
from vtb.annotate.labelstudio import write_labelstudio_tasks  # ecrivain Label Studio (JSON de taches)
from vtb.annotate.schema import ImageAnnotation  # format pivot : toutes les etiquettes d'une image

Tool = Literal["labelstudio", "cvat", "labelme", "xanylabeling"]  # outils d'annotation vers lesquels on sait exporter


# retire les objets peu surs de chaque image ; renvoie une nouvelle liste (l'originale n'est pas modifiee)
def filter_predictions(predictions: list[ImageAnnotation], min_score: float = 0.25) -> list[ImageAnnotation]:
    """Retire les objets de score trop faible : une proposition presque toujours fausse coute plus qu'elle ne rapporte."""
    kept = []  # annotations filtrees
    for ann in predictions:  # pour chaque image predite
        # petite fonction : garde les objets sans score (humains) ou de score >= min_score ; noqa = ignorer la regle E731
        keep = lambda objs: [o for o in objs if o.score is None or o.score >= min_score]  # noqa: E731
        kept.append(  # ajoute une copie filtree de l'image
            ImageAnnotation(  # nouvelle annotation, champs dans l'ordre de la dataclass
                ann.image_path,  # meme image
                ann.width,  # meme largeur
                ann.height,  # meme hauteur
                list(ann.labels),  # copie des etiquettes d'image
                keep(ann.boxes),  # boites filtrees
                keep(ann.rotated),  # boites orientees filtrees
                keep(ann.polygons),  # polygones filtres
                ann.mask_path,  # meme masque
            )
        )
    return kept  # renvoie les annotations filtrees


# ecrit les pre-annotations dans le format d'import de l'outil choisi ; renvoie le chemin ecrit
def export_predictions(
    predictions: list[ImageAnnotation],  # predictions du modele, au format pivot
    tool: Tool,  # outil cible : "labelstudio", "cvat", "labelme" ou "xanylabeling"
    out: str | Path,  # fichier (Label Studio, CVAT) ou dossier (labelme) de sortie
    class_names: list[str] | None = None,  # ordre des classes (utilise pour CVAT)
    document_root: str | Path | None = None,  # dossier racine local-files (Label Studio)
    images_dir: str | Path | None = None,  # dossier des images, pour des chemins relatifs (CVAT)
    min_score: float = 0.25,  # confiance minimale pour proposer un objet
) -> Path:  # renvoie le chemin ecrit
    """Ecrit les pre-annotations au format d'import de l'outil ; renvoie le fichier (ou dossier) ecrit.

    labelstudio : un fichier de taches (champ predictions, avec score), a importer dans le projet ;
    cvat        : un XML CVAT 1.1 a charger par Actions > Upload annotations ;
    labelme / xanylabeling : un JSON par image (avec score), a placer a cote des images.
    """
    kept = filter_predictions(predictions, min_score)  # retire d'abord les objets trop peu surs
    out = Path(out)  # convertit en Path
    if tool == "labelstudio":  # Label Studio
        write_labelstudio_tasks(kept, out, document_root=document_root, as_predictions=True)  # ecrit en "predictions"
    elif tool == "cvat":  # CVAT
        write_cvat_xml(kept, out, class_names, images_dir)  # ecrit un XML CVAT 1.1
    elif tool in ("labelme", "xanylabeling"):  # labelme ou X-AnyLabeling (meme format)
        write_labelme_folder(kept, out)  # un JSON par image
    else:  # outil non prevu
        raise ValueError(f"outil inconnu : {tool}")  # arret avec message clair
    return out  # renvoie le chemin ecrit


# incertitude de classification par image ; renvoie un tableau de N valeurs (grand = le modele hesite)
def classification_uncertainty(probabilities: np.ndarray, method: Literal["margin", "least_confident", "entropy"] = "margin") -> np.ndarray:
    """Incertitude par image a partir des probabilites (N, C) : plus la valeur est grande, plus le modele hesite.

    margin : 1 - (p1 - p2), ecart entre les deux classes les plus probables (le plus utilise) ;
    least_confident : 1 - p1 ; entropy : entropie normalisee dans [0, 1].
    """
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0)  # bornes : evite log(0) pour l'entropie
    top = np.sort(p, axis=1)[:, ::-1]  # probabilites de chaque image triees de la plus grande a la plus petite
    if method == "margin":  # marge : faible ecart entre les deux premieres classes = hesitation
        return 1.0 - (top[:, 0] - (top[:, 1] if p.shape[1] > 1 else 0.0))  # 1 - (p1 - p2) ; une seule classe : p2 = 0
    if method == "least_confident":  # moindre confiance : premiere classe peu probable = hesitation
        return 1.0 - top[:, 0]  # 1 - p1
    if method == "entropy":  # entropie : mesure de dispersion des probabilites
        return -(p * np.log(p)).sum(axis=1) / np.log(p.shape[1])  # divisee par log(C) pour rester entre 0 et 1
    raise ValueError(f"methode inconnue : {method}")  # aucune methode connue : arret avec message clair


# incertitude de detection par image ; renvoie un tableau de N valeurs entre 0 et 1
def detection_uncertainty(predictions: list[ImageAnnotation], empty_value: float = 0.0) -> np.ndarray:
    """Incertitude d'une image detectee : le score le plus proche de 0,5 parmi ses objets (1 - |2s - 1|).

    Une image sans aucune detection vaut empty_value : la mettre a 1 force la relecture des
    images ou le modele ne voit rien (utile quand les defauts sont rares).
    """
    values = []  # une valeur par image
    for ann in predictions:  # pour chaque image predite
        scores = [o.score for o in (*ann.boxes, *ann.rotated, *ann.polygons) if o.score is not None]  # scores de ses objets
        # 1 - |2s - 1| vaut 1 pour s = 0,5 (hesitation maximale) et 0 pour s = 0 ou 1 ; on garde la pire
        values.append(max(1.0 - abs(2.0 * s - 1.0) for s in scores) if scores else empty_value)
    return np.asarray(values, dtype=np.float64)  # renvoie un tableau numpy


# ordre de relecture : renvoie les numeros d'images de la plus incertaine a la plus sure
def order_by_uncertainty(
    items: list[ImageAnnotation] | np.ndarray | Sequence[float],  # predictions, probabilites (N, C) ou scores deja calcules
    method: Literal["margin", "least_confident", "entropy"] = "margin",  # mesure d'incertitude pour la classification
    empty_value: float = 0.0,  # incertitude donnee aux images sans detection
) -> list[int]:  # renvoie une liste d'indices
    """Indices tries de l'image la plus incertaine a la plus sure.

    items : predictions de detection / segmentation (list[ImageAnnotation]), matrice de
    probabilites (N, C) de classification, ou directement un score d'incertitude par image.
    """
    if isinstance(items, list) and items and isinstance(items[0], ImageAnnotation):  # si ce sont des predictions d'objets
        # incertitude de chaque image d'apres les scores de ses objets ("type: ignore" calme le verificateur de types)
        scores = detection_uncertainty(items, empty_value)  # type: ignore[arg-type]
    else:  # sinon : nombres (probabilites ou scores)
        array = np.asarray(items, dtype=np.float64)  # convertit en tableau numpy
        scores = classification_uncertainty(array, method) if array.ndim == 2 else array  # 2D = probabilites ; 1D = deja des scores
    return [int(i) for i in np.argsort(-scores, kind="stable")]  # tri decroissant (signe moins) ; "stable" garde l'ordre a egalite
