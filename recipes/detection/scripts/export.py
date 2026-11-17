"""Script 5 sur 5 : ONNX verifie (sorties brutes ET post-traitement de production), OpenVINO, latence, model_card.json.

python recipes/detection/scripts/export.py --config recipes/detection/configs/yolo11n_neudet.yaml
"""

from vtb.core.config import load_config  # lit le fichier YAML de l'experience et verifie chaque reglage
from vtb.core.experiment import parse_config_path, start  # lecture de --config, puis preparation (dossier, journal, graines)
from vtb.tasks.detection import workflow  # les etapes de la recette detection (prepare, train, evaluate, predict, export)
from vtb.tasks.detection.config import DetectionConfig  # le modele de configuration : la liste des cles autorisees dans le YAML


def main() -> None:  # point d'entree : exporte le detecteur en ONNX et le verifie ; ne renvoie rien
    args = parse_config_path(__doc__.splitlines()[0])  # lit --config ; l'aide reprend la 1re ligne de la docstring
    config = load_config(args.config, DetectionConfig)  # charge le YAML ; une cle inconnue ou mal typee arrete tout ici
    output, _ = start(config, "export", save=False)  # dossier de sortie + journal export.log ; le device est ignore (_) : export sur CPU
    # ONNX : format de fichier standard pour un modele, lisible sans PyTorch ; verifie les sorties brutes contre PyTorch
    # et les boites finales apres post-traitement (NMS...), mesure la latence et ecrit model_card.json
    workflow.export(config, output)


if __name__ == "__main__":  # vrai seulement quand on lance le fichier directement (pas quand on l'importe)
    main()  # lance le script
