<<<<<<< HEAD
"""Flask backend for LeafGuard plant disease detection

Endpoints:
- GET /health -> health check
- POST /predict -> accepts JSON {"image": "data:image/...;base64,..."}

Run locally:
  python -m venv venv
  venv\Scripts\activate  # Windows
  pip install -r backend/requirements.txt
  python -m backend.main
"""
from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io

# Define a flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Model saved with Keras model.save()
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'plant_disease_model.h5')

# Load your trained model
model = load_model(MODEL_PATH)




def model_predict(img, model):
    """Predict from PIL Image object"""
    img = img.resize((224, 224))
    
    # Preprocessing the image
    x = image.img_to_array(img)
    x = x / 255
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    class_idx = np.argmax(preds, axis=1)[0]
    confidence = float(preds[0][class_idx]) * 100
    
    disease_map = {
        0: "Pepper__bell___Bacterial_spot",
        1: "Pepper__bell___healthy",
        2: "Potato___Early_blight",
        3: "Potato___healthy",
        4: "Potato___Late_blight",
        5: "Tomato__Tomato_mosaic_virus",
        6: "Tomato__Tomato_YellowLeaf__Curl_Virus",
        7: "Tomato_Bacterial_spot",
        8: "Tomato_Early_blight",
    }
    
    disease_name = disease_map.get(class_idx, "Unknown")
    is_healthy = "healthy" in disease_name.lower()
    
    # Generate recommendations
    recommendations = []
    if not is_healthy:
        if "Bacterial_spot" in disease_name:
            recommendations = [
                "Remove infected leaves immediately",
                "Apply copper-based bactericide",
                "Avoid overhead watering"
            ]
        elif "Early_blight" in disease_name:
            recommendations = [
                "Remove affected foliage",
                "Apply fungicide treatment",
                "Improve air circulation"
            ]
        elif "Late_blight" in disease_name:
            recommendations = [
                "Remove infected plants immediately",
                "Apply fungicide as directed",
                "Ensure proper spacing between plants"
            ]
        elif "mosaic_virus" in disease_name:
            recommendations = [
                "Remove infected plants",
                "Control insect vectors",
                "Sanitize tools between uses"
            ]
        else:
            recommendations = ["Consult agricultural expert", "Monitor plant closely"]
    else:
        recommendations = ["Plant is healthy", "Continue regular care"]
    
    return {
        "is_healthy": is_healthy,
        "confidence": confidence,
        "disease": disease_name,
        "recommendations": recommendations
    }
=======
# backend/main_fastapi.py
"""
FastAPI backend for LeafGuard plant-disease detection
Endpoints
  GET  /health
  POST /predict   JSON {"image": "data:image/...;base64,..."}
Run:
  uvicorn backend.main_fastapi:app --host 0.0.0.0 --port 5000 --reload
"""
from __future__ import annotations
import base64, io, time
from typing import Dict, Any

import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# FastAPI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# TF config (same as before)
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
InteractiveSession(config=config)

# ------------------------------------------------------------------
# LOAD MODEL ONCE AT START-UP
# Resolve the model path relative to this file so uvicorn can find it
# even when started from different working directories.
# ------------------------------------------------------------------
import os

HERE = os.path.dirname(__file__)
DEFAULT_MODEL_FILENAME = "plant_disease_model.h5"
MODEL_PATH = os.environ.get("MODEL_PATH") or os.path.join(HERE, DEFAULT_MODEL_FILENAME)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: '{MODEL_PATH}'.\n"
        "Place 'plant_disease_model.h5' in the 'backend' folder or set MODEL_PATH env var to the correct file."
    )

model = load_model(MODEL_PATH)

# ------------------------------------------------------------------
# FASTAPI APP
# ------------------------------------------------------------------
app = FastAPI(title="LeafGuard API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# PREDICTION UTILS (unchanged logic)
# ------------------------------------------------------------------
DISEASE_MAP = {
    0: "Pepper__bell___Bacterial_spot",
    1: "Pepper__bell___healthy",
    2: "Potato___Early_blight",
    3: "Potato___healthy",
    4: "Potato___Late_blight",
    5: "Tomato__Tomato_mosaic_virus",
    6: "Tomato__Tomato_YellowLeaf__Curl_Virus",
    7: "Tomato_Bacterial_spot",
    8: "Tomato_Early_blight",
}

def model_predict(img: Image.Image) -> Dict[str, Any]:
    img = img.resize((224, 224))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    class_idx = int(np.argmax(preds, axis=1)[0])
    confidence = float(preds[0][class_idx]) * 100
>>>>>>> 9be11a8 (latestchanges)

    disease_name = DISEASE_MAP.get(class_idx, "Unknown")
    is_healthy = "healthy" in disease_name.lower()

<<<<<<< HEAD
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok", "model_loaded": True})
=======
    recommendations = []
    if not is_healthy:
        if "Bacterial_spot" in disease_name:
            recommendations = [
                "Remove infected leaves immediately",
                "Apply copper-based bactericide",
                "Avoid overhead watering",
            ]
        elif "Early_blight" in disease_name:
            recommendations = [
                "Remove affected foliage",
                "Apply fungicide treatment",
                "Improve air circulation",
            ]
        elif "Late_blight" in disease_name:
            recommendations = [
                "Remove infected plants immediately",
                "Apply fungicide as directed",
                "Ensure proper spacing between plants",
            ]
        elif "mosaic_virus" in disease_name:
            recommendations = [
                "Remove infected plants",
                "Control insect vectors",
                "Sanitize tools between uses",
            ]
        else:
            recommendations = ["Consult agricultural expert", "Monitor plant closely"]
    else:
        recommendations = ["Plant is healthy", "Continue regular care"]
>>>>>>> 9be11a8 (latestchanges)

    return {
        "is_healthy": is_healthy,
        "confidence": confidence,
        "disease": disease_name,
        "recommendations": recommendations,
    }

<<<<<<< HEAD
@app.route('/predict', methods=['POST'])
def predict():
    """Predict disease from base64 image"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image provided"}), 400
        
        # Decode base64 image
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        img_bytes = base64.b64decode(image_data)
        from PIL import Image
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Make prediction
        result = model_predict(img, model)
        result['timestamp'] = int(__import__('time').time() * 1000)
        
        return jsonify(result)
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
=======
# ------------------------------------------------------------------
# REQUEST/RESPONSE MODELS
# ------------------------------------------------------------------
class PredictIn(BaseModel):
    image: str  # base64 string

class PredictOut(BaseModel):
    is_healthy: bool
    confidence: float
    disease: str
    recommendations: list[str]
    timestamp: int

# ------------------------------------------------------------------
# ENDPOINTS
# ------------------------------------------------------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "model_loaded": True}

@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn) -> Dict[str, Any]:
    try:
        b64 = payload.image
        if "," in b64:  # strip data-uri prefix
            b64 = b64.split(",")[1]

        img_bytes = base64.b64decode(b64)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad image data: {e}")

    try:
        result = model_predict(pil_img)
        result["timestamp"] = int(time.time() * 1000)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
>>>>>>> 9be11a8 (latestchanges)
