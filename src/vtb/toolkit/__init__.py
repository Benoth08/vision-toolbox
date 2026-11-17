"""Boite a outils de travail : ce qu'on refait sur chaque projet, ecrit une fois et teste.

    dataset   inventaire d'un dossier, moyenne / ecart-type, poids de classes, echantillon equilibre,
              export en dossiers train/val/test, redimensionnement et conversion 16 -> 8 bits en lot,
              images d'une video, patchs (avec masques)
    training  lr_find, overfit_one_batch, model_summary, max_batch_size, EarlyStopping,
              AverageMeter, Timer, benchmark
    stats     bootstrap_ci, wilson_ci, mcnemar, threshold_for, operating_points,
              expected_calibration_error, worst_errors

En ligne de commande : python -m vtb.toolkit --help
"""

# On remonte ici les fonctions des trois sous-modules : l'utilisateur ecrit "from vtb.toolkit import mean_std"
# sans avoir a savoir dans quel fichier elle est rangee.
from vtb.toolkit.dataset import (  # outils sur les dossiers d'images (fichier dataset.py)
    class_weights,  # poids par classe pour compenser les classes rares dans la perte
    convert_folder_8bit,  # convertit tout un dossier d'images 16 bits en 8 bits
    export_imagefolder,  # copie les images en dossiers train/val/test (un sous-dossier par classe)
    extract_frames,  # extrait des images d'une video a intervalle regulier
    folder_summary,  # inventaire d'un dossier : nombre d'images par classe, tailles, formats
    make_patches,  # decoupe une image (et son masque) en petits carres = patchs
    mean_std,  # moyenne et ecart-type des pixels, pour normaliser les images
    patch_folder,  # decoupe en patchs toutes les images d'un dossier
    resize_folder,  # redimensionne toutes les images d'un dossier
    resize_image,  # redimensionne une seule image
    sample_per_class,  # tire le meme nombre d'images dans chaque classe (echantillon equilibre)
    sample_weights,  # poids par image, pour tirer plus souvent les classes rares
    summarize,  # resume chiffre d'un inventaire (tableau)
    to_8bit,  # convertit une image 16 bits en 8 bits (valeurs 0-255)
)
from vtb.toolkit.stats import (  # outils statistiques pour juger un resultat (fichier stats.py)
    Interval,  # petit objet (valeur, borne basse, borne haute) = intervalle de confiance
    bootstrap_ci,  # intervalle de confiance d'une metrique par bootstrap (retirages au hasard)
    expected_calibration_error,  # mesure si la confiance annoncee par le modele est honnete
    mcnemar,  # test pour savoir si deux modeles different vraiment
    operating_points,  # tableau rappel / fausses alarmes / precision pour chaque seuil
    threshold_for,  # choisit le seuil de decision qui respecte une contrainte (ex. rappel >= 0.95)
    wilson_ci,  # intervalle de confiance d'une proportion (ex. 48 detectes sur 79)
    worst_errors,  # indices des erreurs ou le modele etait le plus sur de lui
)
from vtb.toolkit.training import (  # outils pour l'entrainement (fichier training.py)
    AverageMeter,  # moyenne courante d'une valeur (ex. la perte) au fil des lots
    EarlyStopping,  # arrete l'entrainement quand la validation ne progresse plus
    Timer,  # chronometre simple
    benchmark,  # mesure la vitesse d'un modele (images par seconde)
    lr_find,  # cherche un bon taux d'apprentissage (learning rate)
    max_batch_size,  # plus grande taille de lot qui tient en memoire
    model_summary,  # resume d'un modele : couches, nombre de parametres
    overfit_one_batch,  # verifie que le modele peut apprendre par coeur un seul lot (test de sante)
    seed_everything,  # fixe tous les germes aleatoires pour des resultats reproductibles
    suggest_lr,  # propose un taux d'apprentissage a partir de la courbe de lr_find
)

# __all__ : la liste officielle des noms exportes (ce que "from vtb.toolkit import *" importe)
__all__ = [
    "AverageMeter",  # moyenne courante
    "EarlyStopping",  # arret anticipe
    "Interval",  # intervalle de confiance
    "Timer",  # chronometre
    "benchmark",  # mesure de vitesse
    "bootstrap_ci",  # intervalle par bootstrap
    "class_weights",  # poids par classe
    "convert_folder_8bit",  # conversion 16 -> 8 bits d'un dossier
    "expected_calibration_error",  # erreur de calibration (ECE)
    "export_imagefolder",  # export train/val/test
    "extract_frames",  # images d'une video
    "folder_summary",  # inventaire d'un dossier
    "lr_find",  # recherche du taux d'apprentissage
    "make_patches",  # decoupe en patchs
    "max_batch_size",  # plus grand lot possible
    "mcnemar",  # comparaison de deux modeles
    "mean_std",  # moyenne et ecart-type
    "model_summary",  # resume du modele
    "operating_points",  # tableau des seuils
    "overfit_one_batch",  # test d'apprentissage par coeur
    "patch_folder",  # patchs d'un dossier
    "resize_folder",  # redimensionnement d'un dossier
    "resize_image",  # redimensionnement d'une image
    "sample_per_class",  # echantillon equilibre
    "sample_weights",  # poids par image
    "seed_everything",  # reproductibilite
    "suggest_lr",  # taux d'apprentissage propose
    "summarize",  # resume d'inventaire
    "threshold_for",  # choix du seuil
    "to_8bit",  # conversion 16 -> 8 bits
    "wilson_ci",  # intervalle d'une proportion
    "worst_errors",  # pires erreurs
]
