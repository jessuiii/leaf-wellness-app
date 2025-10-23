# backend/main.py
"""
FastAPI backend for LeafGuard plant-disease detection with Azure Digital Twins integration
Endpoints:
  GET  /health
  POST /predict   (File upload with plant_id)
  GET  /plant/{plant_id}/history
  POST /plant/{plant_id}/create
  GET  /plants (get all plants)
  POST /plant/{plant_id}/treatment
Run:
  uvicorn main:app --host 0.0.0.0 --port 5000 --reload
"""
from __future__ import annotations
import base64, io, time, os
from typing import Dict, Any, List, Optional

import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# FastAPI
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Azure Digital Twins integration
try:
    from adt_client import ADTClient
    ADT_AVAILABLE = True
except ImportError:
    print("Azure Digital Twins packages not installed. Running in standalone mode.")
    ADT_AVAILABLE = False

# TF config (same as before)
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
InteractiveSession(config=config)

# ------------------------------------------------------------------
# LOAD MODEL ONCE AT START-UP
# ------------------------------------------------------------------
HERE = os.path.dirname(__file__)
DEFAULT_MODEL_FILENAME = "plant_disease_model.h5"
MODEL_PATH = os.environ.get("MODEL_PATH") or os.path.join(HERE, DEFAULT_MODEL_FILENAME)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: '{MODEL_PATH}'.\n"
        "Place 'plant_disease_model.h5' in the 'backend' folder or set MODEL_PATH env var to the correct file."
    )

model = load_model(MODEL_PATH)

# Initialize Azure Digital Twins client
adt_client = None
if ADT_AVAILABLE:
    try:
        adt_client = ADTClient()
        print("Azure Digital Twins client initialized successfully")
    except Exception as e:
        print(f"Failed to initialize Azure Digital Twins client: {e}")
        print("Running in standalone mode without digital twin integration")

# ------------------------------------------------------------------
# FASTAPI APP
# ------------------------------------------------------------------
app = FastAPI(title="LeafGuard API with Digital Twin Integration", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# PREDICTION UTILS - Updated disease mapping for tomato focus
# ------------------------------------------------------------------
DISEASE_MAP = {
    0: "Tomato___Bacterial_spot",
    1: "Tomato___Early_blight", 
    2: "Tomato___healthy",
    3: "Tomato___Late_blight",
    4: "Tomato___Leaf_Mold",
    5: "Tomato___Septoria_leaf_spot",
    6: "Tomato___Spider_mites_Two-spotted_spider_mite",
    7: "Tomato___Target_Spot",
    8: "Tomato___Tomato_mosaic_virus",
    9: "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
}

def model_predict(img: Image.Image) -> Dict[str, Any]:
    """Predict disease from image and return structured result."""
    img = img.resize((224, 224))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    class_idx = int(np.argmax(preds, axis=1)[0])
    confidence = float(preds[0][class_idx])

    disease_name = DISEASE_MAP.get(class_idx, "Unknown")
    is_healthy = "healthy" in disease_name.lower()

    # Enhanced recommendations based on tomato diseases
    recommendations = get_treatment_recommendations(disease_name, is_healthy)

    return {
        "is_healthy": is_healthy,
        "confidence": confidence,
        "disease": disease_name,
        "recommendations": recommendations,
    }

def get_treatment_recommendations(disease_name: str, is_healthy: bool) -> List[str]:
    """Get treatment recommendations based on disease type."""
    if is_healthy:
        return [
            "Plant appears healthy",
            "Continue regular watering and fertilization",
            "Monitor for early signs of disease",
            "Maintain good air circulation"
        ]
    
    recommendations_map = {
        "Bacterial_spot": [
            "Remove infected leaves immediately",
            "Apply copper-based bactericide",
            "Avoid overhead watering",
            "Improve air circulation around plants"
        ],
        "Early_blight": [
            "Remove affected foliage and destroy",
            "Apply fungicide containing chlorothalonil",
            "Mulch around plants to prevent soil splash",
            "Avoid overhead irrigation"
        ],
        "Late_blight": [
            "Remove infected plants immediately",
            "Apply protective fungicide (copper-based)",
            "Ensure proper plant spacing",
            "Monitor weather conditions (avoid wet periods)"
        ],
        "Leaf_Mold": [
            "Improve greenhouse ventilation",
            "Reduce humidity levels",
            "Apply appropriate fungicide",
            "Remove infected leaves"
        ],
        "Septoria_leaf_spot": [
            "Remove infected leaves from bottom up",
            "Apply fungicide early in season",
            "Mulch to prevent soil splash",
            "Rotate crops next season"
        ],
        "Spider_mites": [
            "Increase humidity around plants",
            "Apply miticide if infestation is severe",
            "Remove heavily infested leaves",
            "Introduce beneficial predators"
        ],
        "Target_Spot": [
            "Apply preventive fungicide",
            "Remove infected plant debris",
            "Ensure good air circulation",
            "Avoid overhead watering"
        ],
        "mosaic_virus": [
            "Remove infected plants immediately",
            "Control aphid and whitefly vectors",
            "Sanitize tools between plants",
            "Plant virus-resistant varieties"
        ],
        "Yellow_Leaf_Curl_Virus": [
            "Remove infected plants",
            "Control whitefly populations",
            "Use reflective mulches",
            "Plant resistant varieties when available"
        ]
    }
    
    # Find matching recommendations
    for key, recs in recommendations_map.items():
        if key in disease_name:
            return recs
    
    # Default recommendations for unknown diseases
    return [
        "Consult with agricultural extension service",
        "Take additional photos for expert analysis", 
        "Monitor plant closely for changes",
        "Consider isolation from healthy plants"
    ]

# ------------------------------------------------------------------
# REQUEST/RESPONSE MODELS
# ------------------------------------------------------------------
class PredictIn(BaseModel):
    image: str  # base64 string
    plant_id: Optional[str] = None  # Optional for backwards compatibility

class PredictOut(BaseModel):
    is_healthy: bool
    confidence: float
    disease: str
    recommendations: List[str]
    timestamp: int
    plant_id: Optional[str] = None
    scan_history: Optional[List[Dict[str, Any]]] = None

class PlantCreate(BaseModel):
    plant_id: str
    position: Dict[str, Any]
    location: Optional[str] = "Unknown"

class TreatmentRecord(BaseModel):
    treatment: str
    dosage: Optional[str] = ""
    notes: Optional[str] = ""

class PlantTwinResponse(BaseModel):
    plant_id: str
    current_health: str
    last_scan_date: Optional[str] = None
    visual_status: str
    position: Dict[str, Any]
    scan_history: List[Dict[str, Any]]
    treatment_history: List[Dict[str, Any]]

# ------------------------------------------------------------------
# ENDPOINTS
# ------------------------------------------------------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "ok", 
        "model_loaded": True,
        "adt_available": ADT_AVAILABLE,
        "adt_connected": adt_client is not None
    }

@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn) -> Dict[str, Any]:
    """Predict disease from base64 image (legacy endpoint)."""
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
        
        # If plant_id provided and ADT available, update digital twin
        if payload.plant_id and adt_client:
            try:
                # Upload image to blob storage
                image_url = adt_client.upload_image_to_blob(img_bytes, payload.plant_id)
                
                # Update digital twin
                adt_client.update_plant_scan(
                    payload.plant_id, 
                    result["disease"], 
                    result["confidence"], 
                    image_url
                )
                
                # Get recent history
                history = adt_client.get_plant_history(payload.plant_id, limit=5)
                result["scan_history"] = history
                result["plant_id"] = payload.plant_id
                
            except Exception as e:
                print(f"Error updating digital twin: {e}")
                # Continue without digital twin features
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-upload")
async def predict_upload(
    image: UploadFile = File(...),
    plant_id: str = Form(...)
) -> Dict[str, Any]:
    """Predict disease from uploaded image file with plant ID."""
    try:
        # Read image file
        img_bytes = await image.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad image file: {e}")

    try:
        result = model_predict(pil_img)
        result["timestamp"] = int(time.time() * 1000)
        result["plant_id"] = plant_id
        
        # Update digital twin if available
        if adt_client:
            try:
                # Upload image to blob storage
                image_url = adt_client.upload_image_to_blob(img_bytes, plant_id)
                
                # Update digital twin
                adt_client.update_plant_scan(
                    plant_id, 
                    result["disease"], 
                    result["confidence"], 
                    image_url
                )
                
                # Get recent history
                history = adt_client.get_plant_history(plant_id, limit=5)
                result["scan_history"] = history
                
            except Exception as e:
                print(f"Error updating digital twin: {e}")
                result["error"] = f"Prediction successful, but digital twin update failed: {str(e)}"
        else:
            result["note"] = "Digital twin not available - prediction only"
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/plant/{plant_id}/history")
async def get_plant_history(plant_id: str, limit: int = 10) -> Dict[str, Any]:
    """Get scan history for a specific plant."""
    if not adt_client:
        raise HTTPException(status_code=503, detail="Digital twin service not available")
    
    try:
        history = adt_client.get_plant_history(plant_id, limit)
        return {"plant_id": plant_id, "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/plant/{plant_id}")
async def get_plant(plant_id: str) -> PlantTwinResponse:
    """Get complete plant twin data."""
    if not adt_client:
        raise HTTPException(status_code=503, detail="Digital twin service not available")
    
    try:
        twin_data = adt_client.get_plant_twin(plant_id)
        if not twin_data:
            raise HTTPException(status_code=404, detail=f"Plant {plant_id} not found")
        
        return PlantTwinResponse(
            plant_id=twin_data.get("plantId", plant_id),
            current_health=twin_data.get("currentHealth", "Unknown"),
            last_scan_date=twin_data.get("lastScanDate"),
            visual_status=twin_data.get("visualStatus", "unknown"),
            position=twin_data.get("position", {}),
            scan_history=twin_data.get("scanHistory", []),
            treatment_history=twin_data.get("treatmentHistory", [])
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/plant/{plant_id}/create")
async def create_plant(plant_id: str, plant_data: PlantCreate) -> Dict[str, Any]:
    """Create a new plant twin."""
    if not adt_client:
        raise HTTPException(status_code=503, detail="Digital twin service not available")
    
    try:
        success = adt_client.create_plant_twin(
            plant_data.plant_id or plant_id, 
            plant_data.position, 
            plant_data.location
        )
        
        if success:
            return {"success": True, "plant_id": plant_id, "message": "Plant twin created successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to create plant twin")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/plants")
async def get_all_plants() -> Dict[str, Any]:
    """Get all plant twins."""
    if not adt_client:
        raise HTTPException(status_code=503, detail="Digital twin service not available")
    
    try:
        plants = adt_client.get_all_plants()
        return {"plants": plants, "count": len(plants)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/plant/{plant_id}/treatment")
async def add_treatment(plant_id: str, treatment: TreatmentRecord) -> Dict[str, Any]:
    """Add a treatment record to a plant."""
    if not adt_client:
        raise HTTPException(status_code=503, detail="Digital twin service not available")
    
    try:
        success = adt_client.add_treatment_record(
            plant_id, 
            treatment.treatment, 
            treatment.dosage, 
            treatment.notes
        )
        
        if success:
            return {"success": True, "plant_id": plant_id, "message": "Treatment recorded successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to record treatment")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/plants/status-summary")
async def get_plants_status_summary() -> Dict[str, Any]:
    """Get summary of all plants' health status for 3D visualization."""
    if not adt_client:
        raise HTTPException(status_code=503, detail="Digital twin service not available")
    
    try:
        plants = adt_client.get_all_plants()
        
        # Group plants by visual status for 3D display
        status_summary = {
            "healthy": [],
            "warning": [],
            "critical": [],
            "treatment": [],
            "unknown": []
        }
        
        for plant in plants:
            visual_status = plant.get("visualStatus", "unknown")
            plant_summary = {
                "plant_id": plant.get("plantId"),
                "position": plant.get("position", {}),
                "current_health": plant.get("currentHealth", "Unknown"),
                "last_scan_date": plant.get("lastScanDate"),
                "confidence": plant.get("confidence", 0)
            }
            
            if visual_status in status_summary:
                status_summary[visual_status].append(plant_summary)
            else:
                status_summary["unknown"].append(plant_summary)
        
        return {
            "summary": status_summary,
            "total_plants": len(plants),
            "last_updated": int(time.time() * 1000)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))