"""Presets : des listes d'augmentations pretes, chacune avec son raisonnement.

    augment:
      - {name: preset, preset: steel_strip}
      - {name: preset, preset: medium, exclude: [rot90]}     # retirer une operation du preset
      - {name: jpeg, p: 0.3}                                 # et ajouter les siennes apres

Un preset est expanse en etapes ordinaires au moment de build_augmentation. Les etapes non prevues
pour la tache sont retirees en silence (une politique RandAugment n'existe qu'en classification ;
`_tasks` restreint une etape a certaines taches). Les parametres sont ceux d'un capteur industriel
courant : a ajuster apres avoir REGARDE le resultat (tools.preview) sur ses propres images.
"""

from __future__ import annotations  # permet d'ecrire les types (ex. "list[str] | None") sans qu'ils soient evalues au chargement

from dataclasses import dataclass, field  # dataclass : petite classe "fiche" ; field : valeur par defaut fabriquee a chaque objet
from typing import Any  # Any : "n'importe quel type"

CLASSIF_ONLY = ["classification"]  # valeur de "_tasks" pour une etape reservee a la classification


@dataclass(frozen=True)  # frozen : un preset ne peut plus etre modifie apres sa creation
class Preset:  # un preset = une explication + une liste d'etapes d'augmentation
    summary: str  # pourquoi ce preset, et quand l'utiliser
    steps: list[dict[str, Any]] = field(default_factory=list)  # les etapes {name: ..., parametres} ; liste neuve par defaut


# p = probabilite d'appliquer chaque etape a une image ; les intervalles [a, b] sont tires au hasard a chaque image
_LIGHT = [  # base commune : variations faibles et sures
    {"name": "hflip", "p": 0.5},  # retournement gauche-droite une fois sur deux
    {"name": "vflip", "p": 0.5},  # retournement haut-bas une fois sur deux
    {"name": "brightness_contrast", "brightness": 0.1, "contrast": 0.1, "p": 0.5},  # luminosite et contraste a +/- 10 %
    {"name": "gamma", "gamma": [0.9, 1.1], "p": 0.3},  # leger changement des tons moyens
]

PRESETS: dict[str, Preset] = {  # nom du preset -> preset
    "light": Preset(  # preset "light" : le plus prudent
        # 1er argument : le resume (pourquoi, quand), en morceaux de texte que Python colle bout a bout
        "Variations faibles et sures (retournements, un peu de lumiere) : le point de depart quand on ne sait pas, "
        "ou quand le jeu est grand. Suppose une vue de dessus sans haut ni bas.",  # fin du resume (1er argument de Preset)
        _LIGHT,  # ses etapes : la base commune
    ),
    "medium": Preset(  # preset "medium" : le reglage par defaut d'un petit jeu industriel
        # 1er argument : le resume (pourquoi, quand), affiche par available_presets()
        "Le reglage par defaut d'un petit jeu industriel : light + petite affine, bruit, flou et eclairage non uniforme.",
        [  # ses etapes
            *_LIGHT,  # toutes les etapes de light (l'etoile deplie la liste ici)
            # rotation de 5 degres au plus, decalage de 5 %, zoom de 0.9 a 1.1
            {"name": "affine", "degrees": 5.0, "translate": 0.05, "scale": [0.9, 1.1], "p": 0.5},
            {"name": "illumination_gradient", "strength": [0.05, 0.2], "p": 0.3},  # eclairage un peu inegal
            {"name": "gaussian_noise", "sigma": [0.005, 0.02], "p": 0.3},  # bruit du capteur leger
            {"name": "blur", "kernel_size": 5, "sigma": [0.1, 1.0], "p": 0.2},  # leger flou
        ],
    ),
    "strong": Preset(  # preset "strong" : pour un tres petit jeu ou un modele qui apprend par coeur
        "Pour un tres petit jeu ou un modele qui sur-apprend vite : rot90, perspective, exposition, flou de mouvement, "
        # fin du resume (1er argument de Preset)
        "JPEG et trous (classification). A verifier visuellement : trop fort, il cree des images impossibles.",
        [  # ses etapes
            {"name": "hflip", "p": 0.5},  # retournement gauche-droite
            {"name": "vflip", "p": 0.5},  # retournement haut-bas
            {"name": "rot90", "p": 0.5},  # quarts de tour
            {"name": "brightness_contrast", "brightness": 0.25, "contrast": 0.25, "p": 0.8},  # lumiere et contraste a +/- 25 %
            {"name": "gamma", "gamma": [0.7, 1.4], "p": 0.5},  # tons moyens nettement changes
            # rotation de 10 degres au plus, decalage de 10 %, zoom de 0.8 a 1.2
            {"name": "affine", "degrees": 10.0, "translate": 0.1, "scale": [0.8, 1.2], "p": 0.7},
            {"name": "perspective", "distortion": 0.15, "p": 0.3},  # camera un peu inclinee
            {"name": "illumination_gradient", "strength": [0.1, 0.4], "p": 0.5},  # eclairage nettement inegal
            {"name": "exposure", "gain": [0.7, 1.4], "offset": [-0.05, 0.05], "p": 0.4},  # exposition variable
            {"name": "gaussian_noise", "sigma": [0.01, 0.04], "p": 0.4},  # bruit du capteur marque
            {"name": "motion_blur", "length": [3, 9], "p": 0.2},  # flou de mouvement
            {"name": "jpeg", "quality": [40, 90], "p": 0.2},  # artefacts de compression
            # trous dans l'image, en classification seulement (en detection ils pourraient cacher un objet annote)
            {"name": "coarse_dropout", "holes": [1, 4], "size": [0.05, 0.12], "p": 0.3, "_tasks": CLASSIF_ONLY},
        ],
    ),
    "steel_strip": Preset(  # preset "steel_strip" : bande d'acier vue par une camera lineaire
        "Bande d'acier qui defile VERTICALEMENT sous une camera lineaire : retournements oui, rot90 et rotations non "
        "(la direction de laminage reste verticale), eclairage variable en travers de la bande, stries de colonnes, "
        "flou de defilement vertical, petites variations d'echelle (la taille d'un defaut a un sens). "
        # derniere phrase du resume : comment adapter le preset a une bande qui defile horizontalement
        "Bande horizontale : changer motion_blur.angle en [0, 0], line_stripes.axis en rows et l'angle du gradient en [80, 100].",
        [  # ses etapes
            {"name": "hflip", "p": 0.5},  # retournement gauche-droite : la bande reste verticale
            {"name": "vflip", "p": 0.5},  # retournement haut-bas : la bande reste verticale
            # pas de rotation, decalage de 5 %, zoom faible (la taille d'un defaut compte)
            {"name": "affine", "degrees": 0.0, "translate": 0.05, "scale": [0.95, 1.05], "p": 0.5},
            # rampe d'eclairage en travers de la bande (angle proche de 0 = variation gauche-droite)
            {"name": "illumination_gradient", "strength": [0.05, 0.3], "mode": "linear", "angle": [-10.0, 10.0], "p": 0.5},
            {"name": "exposure", "gain": [0.8, 1.25], "offset": [-0.03, 0.03], "p": 0.5},  # exposition variable
            {"name": "line_stripes", "strength": [0.005, 0.03], "axis": "columns", "p": 0.3},  # stries verticales de la camera
            {"name": "motion_blur", "length": [3, 7], "angle": [90.0, 90.0], "p": 0.2},  # flou vertical : la bande defile
            {"name": "gaussian_noise", "sigma": [0.005, 0.02], "p": 0.3},  # bruit du capteur leger
        ],
    ),
    "steel_texture": Preset(  # preset "steel_texture" : vignettes de texture sans direction
        "Vignettes de surface texturee sans direction privilegiee (type NEU-CLS, tuiles) : les huit symetries du carre, "
        "petite echelle, lumiere, bruit et flou legers.",  # fin du resume (1er argument de Preset)
        [  # ses etapes
            {"name": "hflip", "p": 0.5},  # retournement gauche-droite
            {"name": "vflip", "p": 0.5},  # retournement haut-bas
            {"name": "rot90", "p": 0.5},  # quarts de tour : avec les retournements, les 8 symetries du carre
            # pas de rotation libre, decalage de 5 %, zoom de 0.9 a 1.1
            {"name": "affine", "degrees": 0.0, "translate": 0.05, "scale": [0.9, 1.1], "p": 0.5},
            {"name": "brightness_contrast", "brightness": 0.15, "contrast": 0.15, "p": 0.6},  # lumiere et contraste a +/- 15 %
            {"name": "gamma", "gamma": [0.85, 1.2], "p": 0.4},  # tons moyens un peu changes
            {"name": "illumination_gradient", "strength": [0.05, 0.25], "p": 0.3},  # eclairage un peu inegal
            {"name": "gaussian_noise", "sigma": [0.005, 0.02], "p": 0.3},  # bruit du capteur leger
            {"name": "blur", "kernel_size": 5, "sigma": [0.1, 1.0], "p": 0.2},  # leger flou
        ],
    ),
    "metal_part": Preset(  # preset "metal_part" : piece usinee toujours posee de la meme facon
        "Piece usinee en position fixe (bridage, robot) : PAS de retournement (une piece en miroir n'existe pas), "
        # fin du resume (1er argument de Preset)
        "petits decalages de pose, exposition, reflets, vignettage, bruit et legere defocalisation.",
        [  # ses etapes
            # rotation de 3 degres au plus, decalage de 3 %, zoom tres faible : petits ecarts de pose
            {"name": "affine", "degrees": 3.0, "translate": 0.03, "scale": [0.97, 1.03], "p": 0.5},
            {"name": "exposure", "gain": [0.8, 1.25], "offset": [-0.03, 0.03], "p": 0.5},  # exposition variable
            {"name": "specular", "intensity": [0.1, 0.4], "p": 0.3},  # reflets sur le metal
            {"name": "vignetting", "strength": [0.1, 0.4], "p": 0.3},  # bords plus sombres
            {"name": "gaussian_noise", "sigma": [0.005, 0.02], "p": 0.3},  # bruit du capteur leger
            {"name": "defocus_blur", "radius": [1.0, 2.0], "p": 0.2},  # legere perte de mise au point
        ],
    ),
    "anomaly_normals": Preset(  # preset "anomaly_normals" : pour apprendre a quoi ressemble une piece normale
        "Pour apprendre le NORMAL (auto-encodeur, STFPM...) : seulement la variabilite d'acquisition qui n'est pas "
        "une anomalie (exposition, gamma, bruit, decalage de 2 %). Ni taches, ni trous, ni stries : ce seraient des "
        # fin du resume (1er argument de Preset)
        "anomalies apprises comme normales. Retournements a ajouter soi-meme pour une texture isotrope.",
        [  # ses etapes : uniquement des variations qui ne ressemblent pas a un defaut
            {"name": "exposure", "gain": [0.9, 1.1], "offset": [-0.02, 0.02], "p": 0.5},  # exposition a +/- 10 %
            {"name": "gamma", "gamma": [0.95, 1.05], "p": 0.3},  # tons moyens a peine changes
            {"name": "gaussian_noise", "sigma": [0.002, 0.01], "p": 0.3},  # bruit tres faible
            {"name": "translate", "fraction": 0.02, "p": 0.3},  # decalage de 2 % au plus
        ],
    ),
}


# renvoie les etapes d'un preset pour une tache, sans les operations exclues
def preset_steps(name: str, task: str, exclude: list[str] | tuple[str, ...] = ()) -> list[dict[str, Any]]:
    """Etapes d'un preset pour une tache (copie : on peut la modifier), sans les operations exclues."""
    if name not in PRESETS:  # si le preset demande n'existe pas
        raise ValueError(f"preset inconnu : {name} ; disponibles : {sorted(PRESETS)}")  # erreur claire avec la liste des presets
    unknown = set(exclude) - {s["name"] for s in PRESETS[name].steps}  # operations a exclure qui ne sont pas dans le preset
    if unknown:  # s'il y en a (faute de frappe probable)
        raise ValueError(f"exclude : {sorted(unknown)} absent(s) du preset {name}")  # erreur claire
    steps = []  # etapes retenues
    for step in PRESETS[name].steps:  # pour chaque etape du preset
        if step["name"] in exclude:  # si l'utilisateur l'a exclue
            continue  # on la saute
        if task not in step.get("_tasks", [task]):  # si l'etape est reservee a d'autres taches (sans "_tasks" : toutes)
            continue  # on la saute
        # copie de l'etape sans la cle "_tasks" ; les listes sont copiees aussi pour que l'appelant ne modifie pas le preset
        steps.append({k: (list(v) if isinstance(v, list) else v) for k, v in step.items() if k != "_tasks"})
    return steps  # renvoie la liste d'etapes
