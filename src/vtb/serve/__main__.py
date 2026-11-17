"""python -m vtb.serve --model <dossier exporte> [--host 0.0.0.0] [--port 8000]"""

import argparse  # lecture des options de la ligne de commande (--model, --host, --port)

from vtb.core.runtime import setup_logging  # configure l'affichage des messages du journal
from vtb.serve.app import create_app  # construit l'application web FastAPI


def main() -> None:  # point d'entree de "python -m vtb.serve" ; ne renvoie rien
    parser = argparse.ArgumentParser(description="Service HTTP de prediction (ONNX + carte du modele)")  # lecteur d'options
    parser.add_argument("--model", required=True, help="dossier contenant model.onnx et model_card.json")  # option obligatoire
    parser.add_argument("--host", default="127.0.0.1")  # adresse d'ecoute ; 127.0.0.1 = cette machine seulement
    parser.add_argument("--port", type=int, default=8000)  # port d'ecoute (http://localhost:8000)
    args = parser.parse_args()  # lit les options tapees
    setup_logging()  # active les messages du journal sur la console
    import uvicorn  # uvicorn : serveur web qui fait tourner l'application FastAPI (import a la demande)

    uvicorn.run(create_app(args.model), host=args.host, port=args.port)  # demarre le service (tourne jusqu'a Ctrl+C)


if __name__ == "__main__":  # vrai seulement si le fichier est lance directement (et pas importe)
    main()  # lance le programme
