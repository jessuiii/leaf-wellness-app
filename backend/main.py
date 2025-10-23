# backend/main.py
"""
FastAPI backend for LeafGuard plant-disease detection - Marko Tomato-Only Models
Integrates PyTorch models from MarkoArsenovic/DeepLearning_PlantDiseases (Tomato subset only)

Endpoints:
  GET  /health
  POST /predict   JSON {"image": "data:image/...;base64,..."} - Original model
  GET  /models    Get available models info
  POST /predict/marko/{model_type}   Predict using Marko PyTorch models (tomato-only)
  GET  /dataset   Get tomato disease dataset information

Available Marko models: alexnet, densenet169, inception_v3, resnet34, vgg13, squeezenet1_1
All specialized for 10 tomato disease classes with 99%+ accuracy

Run:
  uvicorn backend.main:app --host 0.0.0.0 --port 5000 --reload
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

# Import Marko tomato models
try:
    from models.marko_tomato_models import MarkoTomatoDiseaseClassifier, get_marko_tomato_model_info, TomatoDatasetInfo
    MARKO_MODELS_AVAILABLE = True
except ImportError:
    MARKO_MODELS_AVAILABLE = False
    print("Warning: Marko tomato models not available")

# TF config for original model compatibility
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
InteractiveSession(config=config)

# ------------------------------------------------------------------
# LOAD ORIGINAL MODEL (for backward compatibility)
# ------------------------------------------------------------------
import os

HERE = os.path.dirname(__file__)
DEFAULT_MODEL_FILENAME = "plant_disease_model.h5"
MODEL_PATH = os.environ.get("MODEL_PATH") or os.path.join(HERE, DEFAULT_MODEL_FILENAME)

if not os.path.exists(MODEL_PATH):
    print(f"Warning: Original model file not found: '{MODEL_PATH}'. Only Marko models will be available.")
    model = None
else:
    model = load_model(MODEL_PATH)

# Initialize Marko models dictionary
marko_models = {}

# ------------------------------------------------------------------
# FASTAPI APP
# ------------------------------------------------------------------
app = FastAPI(title="LeafGuard API - Marko Tomato Models", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# PREDICTION UTILS
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
    """Original model prediction (legacy/backup)"""
    if model is None:
        raise HTTPException(status_code=503, detail="Original model not available")
        
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

def get_marko_model(model_type: str):
    """Get or create Marko tomato model instance"""
    if not MARKO_MODELS_AVAILABLE:
        raise HTTPException(status_code=501, detail="Marko tomato models not available")
    
    if model_type not in marko_models:
        marko_models[model_type] = MarkoTomatoDiseaseClassifier(model_type=model_type)
    
    return marko_models[model_type]

def marko_model_predict(img: Image.Image, model_type: str) -> Dict[str, Any]:
    """Predict using Marko PyTorch tomato models"""
    classifier = get_marko_model(model_type)
    
    # Use Marko tomato model prediction
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
        "model_type": f"marko_tomato_{result['model_type']}",
        "all_predictions": result['all_predictions']
    }

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
        elif "Powdery_mildew" in disease_name:
            return [
                "Improve air circulation",
                "Apply sulfur-based fungicide",
                "Remove affected plant parts",
            ]
        elif "Apple_scab" in disease_name:
            return [
                "Apply fungicide during wet periods",
                "Remove fallen leaves",
                "Prune for better air circulation",
            ]
        elif "Cedar_apple_rust" in disease_name:
            return [
                "Remove nearby juniper trees if possible",
                "Apply preventive fungicide",
                "Improve drainage around trees",
            ]
        elif "Esca" in disease_name or "Black_Measles" in disease_name:
            return [
                "Prune infected wood",
                "Apply wound sealant",
                "Improve vine nutrition",
            ]
        elif "Haunglongbing" in disease_name or "Citrus_greening" in disease_name:
            return [
                "Remove infected trees immediately",
                "Control citrus psyllid vectors",
                "Plant disease-free nursery stock",
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
    original: Optional[dict] = None
    marko: Optional[dict] = None

# ------------------------------------------------------------------
# ENDPOINTS
# ------------------------------------------------------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok", 
        "original_model_loaded": model is not None,
        "marko_tomato_models_available": MARKO_MODELS_AVAILABLE,
        "tomato_classes": 10
    }

@app.get("/models", response_model=ModelsOut)
def get_available_models() -> Dict[str, Any]:
    """Get information about available models"""
    result = {}
    
    if model is not None:
        result["original"] = {
            "name": "Original Plant Disease Model",
            "description": "Legacy model for general plant disease detection",
            "classes": list(DISEASE_MAP.values())
        }
    
    if MARKO_MODELS_AVAILABLE:
        result["marko"] = get_marko_tomato_model_info()
        result["marko"]["dataset_info"] = TomatoDatasetInfo.get_dataset_info()
    
    return result

@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn) -> Dict[str, Any]:
    """Original prediction endpoint (legacy/backup)"""
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

@app.post("/predict/marko/{model_type}", response_model=PredictOut)
def predict_with_marko_model(
    payload: PredictIn, 
    model_type: str = Path(..., description="Marko tomato model type: alexnet, densenet169, inception_v3, resnet34, vgg13, squeezenet1_1")
) -> Dict[str, Any]:
    """Predict using specific Marko PyTorch tomato model"""
    available_models = ['alexnet', 'densenet169', 'inception_v3', 'resnet34', 'vgg13', 'squeezenet1_1']
    if model_type not in available_models:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid Marko tomato model type. Available: {', '.join(available_models)}"
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
        result = marko_model_predict(pil_img, model_type)
        result["timestamp"] = int(time.time() * 1000)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dataset")
def get_marko_dataset_info():
    """Get tomato disease dataset information"""
    if not MARKO_MODELS_AVAILABLE:
        raise HTTPException(status_code=501, detail="Marko tomato models not available")
    
    return TomatoDatasetInfo.get_dataset_info()

@app.get("/diseases")
def get_marko_tomato_diseases():
    """Get detailed tomato disease information"""
    if not MARKO_MODELS_AVAILABLE:
        raise HTTPException(status_code=501, detail="Marko tomato models not available")
    
    return TomatoDatasetInfo.get_tomato_diseases()

@app.get("/tomato-model-info")
def get_tomato_model_info() -> Dict[str, Any]:
    """Get specific information about tomato disease detection models"""
    if not MARKO_MODELS_AVAILABLE:
        raise HTTPException(status_code=500, detail="Tomato models not available")
    
    classifier = MarkoTomatoDiseaseClassifier()
    return {
        "total_tomato_classes": classifier.num_classes,
        "tomato_class_names": classifier.class_names,
        "available_models": [
            "alexnet",
            "densenet169", 
            "inception_v3",
            "resnet34",
            "vgg13",
            "squeezenet1_1"
        ],
        "dataset": "PlantVillage (Tomato subset)",
        "framework": "PyTorch",
        "description": "Specialized models for tomato disease detection only",
        "focus": "Tomato crop diseases exclusively"
    }