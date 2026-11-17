"""Service HTTP generique (FastAPI) : un dossier de modele exporte -> une API /predict.

    python -m vtb.serve --model outputs/classification/resnet18_partial --port 8000
    curl -F "file=@piece.png" http://localhost:8000/predict

Le service ne connait que model.onnx et model_card.json : le meme code sert la classification,
les anomalies, la segmentation et la detection. Pas de PyTorch dans l'image Docker.
"""

from __future__ import annotations  # permet d'ecrire les annotations de type modernes (list[str], X | None) partout

import logging  # messages d'information et d'avertissement, plus souples que print
import time  # mesure du temps (perf_counter : chronometre precis)
from pathlib import Path  # Path : represente un chemin de fichier, fonctionne sous Windows comme sous Linux
from typing import Annotated  # Annotated : ajoute une information a un type (ici : "ce parametre est un fichier envoye")

import cv2  # OpenCV : decodage des octets d'une image
import numpy as np  # numpy : calcul sur des tableaux de nombres (une image = un tableau)

# FastAPI : cree une API web ; File / UploadFile = fichier envoye par le client ; HTTPException = reponse d'erreur
from fastapi import FastAPI, File, HTTPException, UploadFile  # extra serve

from vtb.serve.predictor import OnnxPredictor  # predicteur ONNX + carte du modele, sans PyTorch

logger = logging.getLogger(__name__)  # journal propre a ce module


def decode_image(data: bytes, color: str) -> np.ndarray:  # octets recus -> image (tableau RGB ou gris)
    """Octets d'un fichier image (PNG, JPEG, BMP, TIFF 8 ou 16 bits) -> tableau RGB ou gris, comme read_image."""
    flag = cv2.IMREAD_GRAYSCALE if color == "gray" else cv2.IMREAD_COLOR  # lecture en gris ou en couleur
    # frombuffer : octets -> tableau numpy sans copie ; ANYDEPTH garde les images 16 bits en 16 bits
    image = cv2.imdecode(np.frombuffer(data, np.uint8), flag | cv2.IMREAD_ANYDEPTH)
    if image is None:  # None = les octets ne forment pas une image lisible
        raise ValueError("fichier image illisible")  # arret avec un message clair
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV lit en BGR : on remet en RGB


def create_app(model_dir: str | Path) -> FastAPI:  # construit l'application web pour un modele ; renvoie l'objet FastAPI
    predictor = OnnxPredictor(model_dir)  # charge le modele une seule fois, au demarrage du service
    app = FastAPI(title=f"vtb - {predictor.card.task} - {predictor.card.model_name}")  # titre affiche dans la doc /docs

    @app.get("/health")  # route GET /health : verifier que le service repond
    def health() -> dict:  # renvoie l'etat du service
        return {"status": "ok", "task": predictor.card.task, "model": predictor.card.model_name}  # reponse JSON

    @app.get("/card")  # route GET /card : consulter la carte du modele
    def card() -> dict:  # renvoie la carte du modele
        return predictor.card.model_dump()  # objet pydantic -> dictionnaire (FastAPI le convertit en JSON)

    @app.post("/predict")  # route POST /predict : envoyer une image, recevoir la prediction
    async def predict(file: Annotated[UploadFile, File()]) -> dict:  # async : peut attendre la reception du fichier sans bloquer
        try:  # le fichier recu peut ne pas etre une image
            image = decode_image(await file.read(), predictor.color)  # lit les octets recus puis les decode en image
        except ValueError as error:  # image illisible
            raise HTTPException(status_code=400, detail=str(error)) from error  # reponse 400 = requete incorrecte
        start = time.perf_counter()  # top depart du chronometre
        result = predictor(image)  # prediction complete (pretraitement, ONNX, post-traitement)
        result.pop("mask", None)  # le masque complet ne part pas en JSON (trop gros)
        result["filename"] = file.filename  # rappelle le nom du fichier envoye
        result["ms"] = round(1000 * (time.perf_counter() - start), 2)  # duree de la prediction en millisecondes
        return result  # renvoye au client en JSON

    return app  # renvoie l'application prete a etre lancee par uvicorn
