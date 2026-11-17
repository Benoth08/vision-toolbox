"""Fichiers de reference pour le programme de production : quelques images + les reponses du predicteur Python sans PyTorch.

    python tools/make_golden.py --model outputs/classification/resnet18_partial --images data/raw/neu_cls_flat/crazing --n 5

Ecrit <model>/golden/ (copies des images) et <model>/golden.json : pour chaque image, la somme du
tenseur d'entree (verifie le pretraitement) et la reponse attendue (verifie le post-traitement).
Le programme C# les relit avec `--golden <model>/golden.json`.
"""

# "Golden" = reponses de reference : si le programme C# trouve les memes nombres que Python sur les
# memes images, son pretraitement et son post-traitement sont corrects.
from __future__ import annotations  # permet d'ecrire les types modernes sans les evaluer a l'execution

import argparse  # lecture des options de la ligne de commande
import json  # ecriture du fichier golden.json
import logging  # journal de messages plutot que des print
import shutil  # copie de fichiers
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux

import numpy as np  # numpy : calcul sur des tableaux de nombres

from vtb.core.runtime import setup_logging  # reglage standard du journal de la boite a outils
from vtb.io.images import list_images, read_image  # liste et lecture d'images
from vtb.serve.predictor import OnnxPredictor  # predicteur qui execute un modele ONNX sans PyTorch

logger = logging.getLogger(__name__)  # journal propre a ce fichier


def jsonable(value):  # rend une valeur ecrivable en JSON (types numpy -> types Python) ; renvoie la valeur convertie
    if isinstance(value, dict):  # si c'est un dictionnaire
        return {k: jsonable(v) for k, v in value.items() if k != "mask"}  # convertit chaque valeur ; retire le masque (trop gros)
    if isinstance(value, (list, tuple)):  # si c'est une liste ou un tuple
        return [jsonable(v) for v in value]  # convertit chaque element
    if isinstance(value, (np.floating, np.integer, np.bool_)):  # si c'est un nombre numpy (JSON ne le connait pas)
        return value.item()  # le transforme en nombre Python ordinaire
    return value  # deja ecrivable tel quel


def main() -> None:  # cree les images et le fichier de reference ; ne renvoie rien
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # aide = premiere ligne de la docstring
    parser.add_argument("--model", type=Path, required=True, help="dossier contenant model.onnx et model_card.json")  # dossier modele
    parser.add_argument("--images", type=Path, required=True, help="dossier d'images representatives")  # images a utiliser
    parser.add_argument("--n", type=int, default=5)  # nombre d'images de reference
    args = parser.parse_args()  # lit les options tapees
    setup_logging()  # active l'affichage des messages
    # ONNX : format de fichier standard pour un modele, executable sans PyTorch (ici avec onnxruntime, et en C#)
    predictor = OnnxPredictor(args.model)  # charge model.onnx et sa fiche (model_card.json)
    folder = args.model / "golden"  # dossier ou copier les images de reference
    folder.mkdir(exist_ok=True)  # le cree s'il n'existe pas
    items = []  # une entree de reference par image
    for path in list_images(args.images)[: args.n]:  # pour chacune des n premieres images du dossier
        target = folder / path.name  # chemin de la copie
        shutil.copy2(path, target)  # copie l'image a cote du modele (le programme C# la relira)
        image = read_image(target, color=predictor.color)  # relit la copie dans le mode couleur attendu par le modele
        tensor, _ = predictor.prepare(image)  # pretraitement : image -> tenseur d'entree du modele (tableau de nombres)
        items.append(  # ajoute l'entree de reference de cette image
            {  # dictionnaire : ce que le programme C# doit retrouver pour cette image
                "file": f"golden/{path.name}",  # chemin relatif de l'image
                "tensor_shape": list(tensor.shape),  # forme du tenseur (ex. [1, 3, 224, 224])
                "tensor_sum": float(tensor.astype(np.float64).sum()),  # somme de toutes ses valeurs : verifie le pretraitement
                "tensor_first_values": [round(float(v), 6) for v in tensor.ravel()[:8]],  # 8 premieres valeurs, pour chercher un ecart
                "result": jsonable(predictor(image)),  # reponse complete du predicteur Python : la reponse attendue
            }
        )
    payload = {"task": predictor.card.task, "model_name": predictor.card.model_name, "images": items}  # contenu du fichier
    (args.model / "golden.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")  # ecrit golden.json
    logger.info("%d images de reference -> %s", len(items), args.model / "golden.json")  # bilan


if __name__ == "__main__":  # vrai seulement quand le script est lance directement
    main()  # lance la creation des fichiers de reference
