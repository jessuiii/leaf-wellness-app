# backend/main_fastapi.py
"""
FastAPI backend for LeafGuard plant-disease detection
Endpoints
  GET  /health
  POST /predict   JSON {"image": "data:image/...;base64,..."}
  GET  /models    Get available models info
  POST /predict/{model_type}   Predict using specific model
Run:
  uvicorn backend.main_fastapi:app --host 0.0.0.0 --port 5000 --reload
"""
from __future__ import annotations
import base64, io, time
from typing import Dict, Any, Optional

import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# FastAPI
from fastapi import FastAPI, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import Bhargavi models
try:
    from models.bhargavi_models import BhargaviTomatoDiseaseClassifier, get_model_info
    BHARGAVI_MODELS_AVAILABLE = True
except ImportError:
    BHARGAVI_MODELS_AVAILABLE = False
    print("Warning: Bhargavi models not available")

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

# Initialize Bhargavi models dictionary
bhargavi_models = {}
if BHARGAVI_MODELS_AVAILABLE:
    # Lazy load models when needed
    pass

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
    """Original model prediction (legacy)"""
    img = img.resize((224, 224))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    class_idx = int(np.argmax(preds, axis=1)[0])
    confidence = float(preds[0][class_idx]) * 100

    disease_name = DISEASE_MAP.get(class_idx, "Unknown")
    is_healthy = "healthy" in disease_name.lower()

    recommendations = get_recommendations(disease_name, is_healthy)

    return {
        "is_healthy": is_healthy,
        "confidence": confidence,
        "disease": disease_name,
        "recommendations": recommendations,
        "model_type": "original"
    }

def get_bhargavi_model(model_type: str):
    """Get or create Bhargavi model instance"""
    if not BHARGAVI_MODELS_AVAILABLE:
        raise HTTPException(status_code=501, detail="Bhargavi models not available")
    
    if model_type not in bhargavi_models:
        bhargavi_models[model_type] = BhargaviTomatoDiseaseClassifier(model_type=model_type)
    
    return bhargavi_models[model_type]

def bhargavi_model_predict(img: Image.Image, model_type: str) -> Dict[str, Any]:
    """Predict using Bhargavi models"""
    classifier = get_bhargavi_model(model_type)
    
    # Use Bhargavi model prediction
    result = classifier.predict(img)
    
    # Add recommendations based on prediction
    disease_name = result['predicted_class']
    is_healthy = "healthy" in disease_name.lower()
    recommendations = get_recommendations(disease_name, is_healthy)
    
    return {
        "is_healthy": is_healthy,
        "confidence": result['confidence'] * 100,  # Convert to percentage
        "disease": disease_name,
        "recommendations": recommendations,
        "model_type": result['model_type'],
        "all_predictions": result['all_predictions']
    }

def get_recommendations(disease_name: str, is_healthy: bool) -> list[str]:
def get_recommendations(disease_name: str, is_healthy: bool) -> list[str]:
    """Get treatment recommendations based on disease"""
    if not is_healthy:
        if "Bacterial_spot" in disease_name:
            return [
                "Remove infected leaves immediately",
                "Apply copper-based bactericide",
                "Avoid overhead watering",
            ]
        elif "Early_blight" in disease_name:
            return [
                "Remove affected foliage",
                "Apply fungicide treatment",
                "Improve air circulation",
            ]
        elif "Late_blight" in disease_name:
            return [
                "Remove infected plants immediately",
                "Apply fungicide as directed",
                "Ensure proper spacing between plants",
            ]
        elif "mosaic_virus" in disease_name:
            return [
                "Remove infected plants",
                "Control insect vectors",
                "Sanitize tools between uses",
            ]
        elif "Leaf_Mold" in disease_name:
            return [
                "Improve ventilation",
                "Reduce humidity levels",
                "Apply appropriate fungicide",
            ]
        elif "Septoria_leaf_spot" in disease_name:
            return [
                "Remove infected leaves",
                "Apply fungicide treatment",
                "Avoid overhead watering",
            ]
        elif "Target_Spot" in disease_name:
            return [
                "Remove infected plant debris",
                "Apply fungicide treatment",
                "Improve air circulation",
            ]
        else:
            return ["Consult agricultural expert", "Monitor plant closely"]
    else:
        return ["Plant is healthy", "Continue regular care"]

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
    model_type: Optional[str] = None
    all_predictions: Optional[list] = None

class ModelsOut(BaseModel):
    original: dict
    bhargavi: Optional[dict] = None

# ------------------------------------------------------------------
# ENDPOINTS
# ------------------------------------------------------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok", 
        "model_loaded": True,
        "bhargavi_models_available": BHARGAVI_MODELS_AVAILABLE
    }

@app.get("/models", response_model=ModelsOut)
def get_available_models() -> Dict[str, Any]:
    """Get information about available models"""
    result = {
        "original": {
            "name": "Original Plant Disease Model",
            "description": "Legacy model for general plant disease detection",
            "classes": list(DISEASE_MAP.values())
        }
    }
    
    if BHARGAVI_MODELS_AVAILABLE:
        result["bhargavi"] = get_model_info()
    
    return result

@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn) -> Dict[str, Any]:
    """Original prediction endpoint (legacy)"""
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

@app.post("/predict/{model_type}", response_model=PredictOut)
def predict_with_model(
    payload: PredictIn, 
    model_type: str = Path(..., description="Model type: cnn, vgg16, or resnet50")
) -> Dict[str, Any]:
    """Predict using specific Bhargavi model"""
    if model_type not in ['cnn', 'vgg16', 'resnet50']:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid model type. Available: cnn, vgg16, resnet50"
        )
    
    try:
        b64 = payload.image
        if "," in b64:  # strip data-uri prefix
            b64 = b64.split(",")[1]

        img_bytes = base64.b64decode(b64)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad image data: {e}")

    try:
        result = bhargavi_model_predict(pil_img, model_type)
        result["timestamp"] = int(time.time() * 1000)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))