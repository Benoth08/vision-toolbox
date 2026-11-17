"""Enchaine prepare (si besoin), train, evaluate, export pour une configuration.

python recipes/detection/scripts/run.py --config recipes/detection/configs/quick_yolo.yaml
python recipes/detection/scripts/run.py --config ... --steps evaluate,export
"""

from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

from vtb.core.config import load_config  # lit le fichier YAML de l'experience et verifie chaque reglage
from vtb.core.experiment import parse_config_path, start  # lecture de --config, puis preparation (dossier, journal, graines)
from vtb.tasks.detection import workflow  # les etapes de la recette detection (prepare, train, evaluate, predict, export)
from vtb.tasks.detection.config import DetectionConfig  # le modele de configuration : la liste des cles autorisees dans le YAML


def main() -> None:  # point d'entree : lance les etapes demandees l'une apres l'autre ; ne renvoie rien
    # lit --config et --steps (liste des etapes separees par des virgules ; par defaut toutes)
    args = parse_config_path(__doc__.splitlines()[0], [("--steps", {"default": "prepare,train,evaluate,export"})])
    config = load_config(args.config, DetectionConfig)  # charge le YAML ; une cle inconnue ou mal typee arrete tout ici
    steps = args.steps.split(",")  # "evaluate,export" -> ["evaluate", "export"]
    output, device = start(config, "run")  # dossier de sortie, journal run.log, copie de la config, graines, CPU ou GPU
    if "prepare" in steps and not Path(config.split.file).exists():  # prepare seulement si le decoupage n'existe pas encore
        workflow.prepare(config)  # annotations au format pivot + decoupage (le meme pour tous les moteurs : comparables)
    if "train" in steps:  # si l'etape train est demandee
        workflow.train(config, output, device)  # entrainement sur train, meilleure epoque choisie sur val
    if "evaluate" in steps:  # si l'etape evaluate est demandee
        workflow.evaluate(config, output, device)  # mesures sur le jeu de test
    if "export" in steps:  # si l'etape export est demandee
        workflow.export(config, output)  # export ONNX (format standard de modele) verifie + carte du modele


if __name__ == "__main__":  # vrai seulement quand on lance le fichier directement (pas quand on l'importe)
    main()  # lance le script
