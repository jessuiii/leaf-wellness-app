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

# Import DenseNet tomato model
try:
    from models.densenet_tomato_model import DenseNetTomatoClassifier, get_densenet_model_info, TomatoDiseaseInfo
    DENSENET_MODELS_AVAILABLE = True
except ImportError:
    DENSENET_MODELS_AVAILABLE = False
    print("Warning: DenseNet tomato models not available")

# Import Marko tomato models (backup)
try:
    from models.marko_tomato_models import MarkoTomatoDiseaseClassifier, get_marko_tomato_model_info, TomatoDatasetInfo
    MARKO_MODELS_AVAILABLE = True
except ImportError:
    MARKO_MODELS_AVAILABLE = False
    print("Warning: Marko tomato models not available")

# Import visualization utilities
try:
    from visualization.torchvis_util import GradType
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: Visualization utilities not available")

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

# Initialize model dictionaries
marko_models = {}
densenet_model = None

# ------------------------------------------------------------------
# FASTAPI APP
# ------------------------------------------------------------------
app = FastAPI(title="LeafGuard API - DenseNet169 Tomato Detection", version="1.0")
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

def get_densenet_model():
    """Get or create DenseNet tomato model instance"""
    global densenet_model
    if not DENSENET_MODELS_AVAILABLE:
        raise HTTPException(status_code=501, detail="DenseNet tomato models not available")
    
    if densenet_model is None:
        densenet_model = DenseNetTomatoClassifier()
        densenet_model.load_model()  # Load with ImageNet weights
    
    return densenet_model

def densenet_model_predict(img: Image.Image, weights_path: str = None) -> Dict[str, Any]:
    """Predict using DenseNet169 tomato model"""
    if weights_path:
        # Create new instance with custom weights
        classifier = DenseNetTomatoClassifier()
        classifier.load_model(weights_path=weights_path)
    else:
        # Use cached model instance
        classifier = get_densenet_model()
    
    # Make prediction
    result = classifier.predict(img)
    
    # Add recommendations
    disease_name = result['predicted_class']
    is_healthy = "healthy" in disease_name.lower()
    recommendations = get_recommendations(disease_name, is_healthy)
    
    return {
        "is_healthy": is_healthy,
        "confidence": result['confidence'] * 100,
        "disease": disease_name,
        "recommendations": recommendations,
        "model_type": "densenet169_tomato",
        "all_predictions": result['all_predictions'],
        "custom_weights": weights_path is not None
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

class VisualizationRequest(BaseModel):
    image: str  # base64 string
    disease_class: str  # target disease class name
    method: Optional[str] = "guided"  # "naive" or "guided" for saliency
    output_format: Optional[str] = "base64"  # "base64" or "url"

class SaliencyRequest(VisualizationRequest):
    method: Optional[str] = "guided"  # "naive" or "guided"

class OcclusionRequest(VisualizationRequest):
    occlusion_size: Optional[int] = 50  # size of occlusion window
    stride: Optional[int] = 10  # stride for sliding window

class VisualizationResponse(BaseModel):
    success: bool
    image_original: Optional[str] = None  # base64 encoded
    image_visualization: Optional[str] = None  # base64 encoded  
    image_overlay: Optional[str] = None  # base64 encoded
    method: str
    target_class: str
    confidence: float
    predicted_class: str
    processing_time: float
    error: Optional[str] = None

# ------------------------------------------------------------------
# ENDPOINTS
# ------------------------------------------------------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok", 
        "original_model_loaded": model is not None,
        "densenet_models_available": DENSENET_MODELS_AVAILABLE,
        "marko_tomato_models_available": MARKO_MODELS_AVAILABLE,
        "primary_model": "DenseNet169",
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

@app.post("/predict/densenet", response_model=PredictOut)
def predict_with_densenet(payload: PredictIn) -> Dict[str, Any]:
    """Predict using DenseNet169 tomato model (primary endpoint)"""
    try:
        b64 = payload.image
        if "," in b64:  # strip data-uri prefix
            b64 = b64.split(",")[1]

        img_bytes = base64.b64decode(b64)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad image data: {e}")

    try:
        result = densenet_model_predict(pil_img)
        result["timestamp"] = int(time.time() * 1000)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/densenet/custom", response_model=PredictOut)
def predict_with_custom_densenet(
    payload: PredictIn,
    weights_path: str = Path(..., description="Path to custom trained DenseNet169 weights (.pth file)")
) -> Dict[str, Any]:
    """Predict using custom trained DenseNet169 model"""
    
    # Check if weights file exists
    if not os.path.exists(weights_path):
        raise HTTPException(
            status_code=404,
            detail=f"Custom weights file not found: {weights_path}"
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
        result = densenet_model_predict(pil_img, weights_path=weights_path)
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

@app.get("/densenet-info")
def get_densenet_model_info() -> Dict[str, Any]:
    """Get information about DenseNet169 tomato model"""
    if not DENSENET_MODELS_AVAILABLE:
        raise HTTPException(status_code=500, detail="DenseNet models not available")
    
    info = get_densenet_model_info()
    return {
        "model_name": info['model_name'],
        "architecture": info['architecture'],
        "num_classes": info['num_classes'],
        "input_size": info['input_size'],
        "framework": info['framework'],
        "expected_accuracy": info['expected_accuracy'],
        "class_names": info['class_names'],
        "training_support": True,
        "custom_weights_support": True,
        "primary_model": True
    }

@app.get("/densenet-diseases")
def get_densenet_disease_info():
    """Get detailed tomato disease information for DenseNet model"""
    if not DENSENET_MODELS_AVAILABLE:
        raise HTTPException(status_code=501, detail="DenseNet models not available")
    
    return TomatoDiseaseInfo.get_disease_info()

@app.get("/densenet-dataset")
def get_densenet_dataset_info():
    """Get dataset information for DenseNet training"""
    if not DENSENET_MODELS_AVAILABLE:
        raise HTTPException(status_code=501, detail="DenseNet models not available")
    
    return TomatoDiseaseInfo.get_dataset_info()

# ------------------------------------------------------------------
# VISUALIZATION ENDPOINTS
# ------------------------------------------------------------------

@app.post("/visualize/saliency", response_model=VisualizationResponse)
def generate_saliency_map(request: SaliencyRequest) -> Dict[str, Any]:
    """
    Generate saliency map for DenseNet169 prediction
    
    Uses guided backpropagation or naive backpropagation to show
    which parts of the image the model focuses on for classification.
    """
    if not DENSENET_MODELS_AVAILABLE:
        raise HTTPException(status_code=501, detail="DenseNet models not available")
    
    if not VISUALIZATION_AVAILABLE:
        raise HTTPException(status_code=501, detail="Visualization utilities not available")
    
    try:
        import torch
        import torchvision.transforms as transforms
        from PIL import Image
        import base64
        import io
        import time
        import numpy as np
        import matplotlib.pyplot as plt
        from visualization.torchvis_util import GradType, augment_module
        
        start_time = time.time()
        
        # Decode base64 image
        img_data = base64.b64decode(request.image.split(',')[1])
        img = Image.open(io.BytesIO(img_data)).convert('RGB')
        
        # Get DenseNet model
        model_instance = get_densenet_model()
        model = model_instance.model
        device = model_instance.device
        
        # Preprocess image
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        image_tensor = preprocess(img).unsqueeze(0).to(device)
        
        # Get prediction first
        model.eval()
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class_idx = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0, predicted_class_idx].item()
        
        # Find target class index
        class_names = model_instance.class_names
        if request.disease_class not in class_names:
            raise HTTPException(status_code=400, detail=f"Disease class '{request.disease_class}' not found")
        
        target_class_idx = class_names.index(request.disease_class)
        predicted_class = class_names[predicted_class_idx]
        
        # Generate saliency map
        method = GradType.GUIDED if request.method == "guided" else GradType.NAIVE
        
        # Set up hooks
        vis_param_dict, reset_state, remove_handles = augment_module(model)
        vis_param_dict['method'] = method
        
        # Prepare input for gradients
        image_tensor.requires_grad_(True)
        if image_tensor.grad is not None:
            image_tensor.grad.zero_()
        
        model.zero_grad()
        outputs = model(image_tensor)
        target_score = outputs[0, target_class_idx]
        target_score.backward()
        
        # Extract saliency
        gradients = image_tensor.grad.data
        saliency = torch.abs(gradients).max(dim=1)[0].squeeze().cpu().numpy()
        
        # Clean up
        remove_handles()
        
        # Create visualizations
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Saliency map
        axes[1].imshow(saliency, cmap='hot', interpolation='nearest')
        axes[1].set_title(f'Saliency Map ({request.method})')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(img)
        axes[2].imshow(saliency, cmap='hot', alpha=0.4, interpolation='nearest')
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        viz_b64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # Original image as base64
        orig_buffer = io.BytesIO()
        img.save(orig_buffer, format='PNG')
        orig_buffer.seek(0)
        orig_b64 = base64.b64encode(orig_buffer.getvalue()).decode()
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "image_original": f"data:image/png;base64,{orig_b64}",
            "image_visualization": f"data:image/png;base64,{viz_b64}",
            "image_overlay": None,  # Already included in visualization
            "method": request.method,
            "target_class": request.disease_class,
            "confidence": confidence,
            "predicted_class": predicted_class,
            "processing_time": processing_time,
            "error": None
        }
        
    except Exception as e:
        return {
            "success": False,
            "image_original": None,
            "image_visualization": None,
            "image_overlay": None,
            "method": request.method,
            "target_class": request.disease_class,
            "confidence": 0.0,
            "predicted_class": "Error",
            "processing_time": 0.0,
            "error": str(e)
        }

@app.post("/visualize/occlusion", response_model=VisualizationResponse)
def generate_occlusion_map(request: OcclusionRequest) -> Dict[str, Any]:
    """
    Generate occlusion heatmap for DenseNet169 prediction
    
    Shows which regions of the image are most important for classification
    by systematically masking different parts of the image.
    """
    if not DENSENET_MODELS_AVAILABLE:
        raise HTTPException(status_code=501, detail="DenseNet models not available")
    
    try:
        import torch
        import torchvision.transforms as transforms
        from PIL import Image
        import base64
        import io
        import time
        import numpy as np
        import matplotlib.pyplot as plt
        import copy
        import math
        
        start_time = time.time()
        
        # Decode base64 image
        img_data = base64.b64decode(request.image.split(',')[1])
        img = Image.open(io.BytesIO(img_data)).convert('RGB')
        
        # Get DenseNet model
        model_instance = get_densenet_model()
        model = model_instance.model
        device = model_instance.device
        
        # Find target class
        class_names = model_instance.class_names
        if request.disease_class not in class_names:
            raise HTTPException(status_code=400, detail=f"Disease class '{request.disease_class}' not found")
        
        target_class_idx = class_names.index(request.disease_class)
        
        # Create occlusion masks
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        
        output_height = int(math.ceil((height - request.occlusion_size) / request.stride + 1))
        output_width = int(math.ceil((width - request.occlusion_size) / request.stride + 1))
        
        # Preprocess
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Generate occluded images
        occluded_images = []
        for h in range(output_height):
            for w in range(output_width):
                h_start = h * request.stride
                w_start = w * request.stride
                h_end = min(height, h_start + request.occlusion_size)
                w_end = min(width, w_start + request.occlusion_size)
                
                input_image = copy.deepcopy(img_array)
                input_image[h_start:h_end, w_start:w_end] = 0
                
                pil_img = Image.fromarray(input_image.astype(np.uint8))
                occluded_images.append(preprocess(pil_img))
        
        # Batch process
        batch_size = 8
        confidences = []
        
        model.eval()
        with torch.no_grad():
            for i in range(0, len(occluded_images), batch_size):
                batch = occluded_images[i:i+batch_size]
                batch_tensor = torch.stack(batch).to(device)
                
                outputs = model(batch_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                target_confidences = probabilities[:, target_class_idx].cpu().numpy()
                confidences.extend(target_confidences)
        
        # Create heatmap
        heatmap = np.array(confidences[:output_height * output_width])
        heatmap = heatmap.reshape((output_height, output_width))
        
        # Get original prediction
        original_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(original_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class_idx = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0, predicted_class_idx].item()
        
        predicted_class = class_names[predicted_class_idx]
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Heatmap
        im = axes[1].imshow(heatmap, cmap='YlOrRd', interpolation='bilinear')
        axes[1].set_title(f'Occlusion Heatmap\n(size={request.occlusion_size}, stride={request.stride})')
        plt.colorbar(im, ax=axes[1], label='Class Confidence')
        
        # Overlay
        axes[2].imshow(img, alpha=0.7)
        axes[2].imshow(heatmap, cmap='YlOrRd', alpha=0.5, interpolation='bilinear')
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        viz_b64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # Original image as base64
        orig_buffer = io.BytesIO()
        img.save(orig_buffer, format='PNG')
        orig_buffer.seek(0)
        orig_b64 = base64.b64encode(orig_buffer.getvalue()).decode()
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "image_original": f"data:image/png;base64,{orig_b64}",
            "image_visualization": f"data:image/png;base64,{viz_b64}",
            "image_overlay": None,
            "method": f"occlusion_s{request.occlusion_size}_st{request.stride}",
            "target_class": request.disease_class,
            "confidence": probabilities[0, target_class_idx].item(),
            "predicted_class": predicted_class,
            "processing_time": processing_time,
            "error": None
        }
        
    except Exception as e:
        return {
            "success": False,
            "image_original": None,
            "image_visualization": None,
            "image_overlay": None,
            "method": f"occlusion_s{request.occlusion_size}_st{request.stride}",
            "target_class": request.disease_class,
            "confidence": 0.0,
            "predicted_class": "Error",
            "processing_time": 0.0,
            "error": str(e)
        }

@app.get("/visualize/methods")
def get_visualization_methods():
    """Get information about available visualization methods"""
    return {
        "saliency_methods": {
            "naive": {
                "name": "Naive Backpropagation",
                "description": "Standard gradient backpropagation showing pixel importance",
                "reference": "Simonyan et al. (2013)"
            },
            "guided": {
                "name": "Guided Backpropagation", 
                "description": "Enhanced gradient method that only backpropagates positive gradients",
                "reference": "Springenberg et al. (2014)"
            }
        },
        "occlusion_methods": {
            "systematic": {
                "name": "Systematic Occlusion",
                "description": "Systematically mask image regions to find important areas",
                "parameters": {
                    "occlusion_size": "Size of masking window (default: 50)",
                    "stride": "Step size for sliding window (default: 10)"
                }
            }
        },
        "supported_formats": ["base64"],
        "available_when": "DenseNet169 model loaded"
    }