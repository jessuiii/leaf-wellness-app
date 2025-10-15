r"""FastAPI backend for LeafGuard

Endpoints:
- GET /health -> simple health check
- POST /predict -> accepts JSON {"image": "data:image/...;base64,..."}

This file will try to load a Keras model from the path configured in MODEL_PATH
or from the same directory as this file named `plant_disease_model.h5`.

If the model can't be loaded, the server will still run and return a mock response
so the frontend can function during development.

Run locally (PowerShell):
  python -m venv .venv; .\.venv\Scripts\Activate.ps1
  pip install fastapi uvicorn pillow numpy
  # optionally install tensorflow if you have a model: pip install tensorflow
  uvicorn backend.main:app --reload --port 5000

Note: place your Keras `.h5` model next to this file or set the MODEL_PATH env var.
"""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from PIL import Image
except Exception as e:  # pragma: no cover - pillow required
    raise RuntimeError("Pillow is required. Install with `pip install pillow`") from e

MODEL: Optional[Any] = None
LABELS: Optional[List[str]] = None
# Default to a model file next to this script; allow overriding with MODEL_PATH env var
MODEL_PATH = os.environ.get("MODEL_PATH") or (Path(__file__).parent / "plant_disease_model.h5")
# Try to import numpy/tensorflow. If missing, we'll report a helpful error later.
try:
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except Exception:
    np = None  # type: ignore
    tf = None  # type: ignore
    load_model = None  # type: ignore
    TF_AVAILABLE = False


def _load_model_and_labels() -> None:
    global MODEL, LABELS
    model_path = Path(os.environ.get("MODEL_PATH", MODEL_PATH))

    # If configured MODEL_PATH doesn't exist, try to pick any .h5 file in the backend dir
    if not model_path.exists():
        backend_dir = Path(__file__).parent
        h5_files = list(backend_dir.glob("*.h5"))
        if h5_files:
            model_path = h5_files[0]
            print(f"Auto-detected model file: {model_path}")

    # attempt to load model
    if TF_AVAILABLE and model_path.exists():
        try:
            # Register any expected custom layers here. If your training code defines custom
            # layers, paste their class definitions below and add them to custom_objects.
            custom_objects = {}
            try:
                # Example safe stub for a 'CustomScaleLayer' that scales inputs by a factor.
                # Replace this with your real implementation from training for accurate results.
                class CustomScaleLayer(tf.keras.layers.Layer):
                    def __init__(self, scale=1.0, **kwargs):
                        super().__init__(**kwargs)
                        # store as tensor-like value when used in call
                        self.scale = float(scale)

                    def call(self, inputs):
                        # Accept single tensor or a list/tuple of tensors. If a sequence
                        # is provided, combine them with `tf.add_n` which works with
                        # KerasTensors so output shape/dtype can be inferred.
                        if isinstance(inputs, (list, tuple)):
                            tensors = [tf.convert_to_tensor(x) for x in inputs]
                            x = tf.add_n(tensors)
                        else:
                            x = tf.convert_to_tensor(inputs)
                        return x * tf.cast(self.scale, x.dtype)

                    def get_config(self):
                        cfg = super().get_config()
                        cfg.update({"scale": float(self.scale)})
                        return cfg

                custom_objects["CustomScaleLayer"] = CustomScaleLayer
            except Exception:
                custom_objects = {}

            # load without compiling (we only need inference). Passing compile=False
            # avoids issues when the saved model references unavailable optimizer objects.
            MODEL = load_model(str(model_path), custom_objects=custom_objects, compile=False)
            print(f"Loaded model from {model_path}")
        except Exception as exc:
            print("Failed to load model:", exc)
            MODEL = None
    else:
        MODEL = None

    # attempt to load labels from labels.json or labels.txt
    labels_json = model_path.with_suffix(".labels.json")
    labels_txt = model_path.with_suffix(".labels.txt")
    if labels_json.exists():
        try:
            LABELS = json.loads(labels_json.read_text())
            print(f"Loaded labels from {labels_json}")
        except Exception:
            LABELS = None
    elif labels_txt.exists():
        try:
            LABELS = [l.strip() for l in labels_txt.read_text().splitlines() if l.strip()]
            print(f"Loaded labels from {labels_txt}")
        except Exception:
            LABELS = None
    else:
        LABELS = None


# If your training dataset used these classes (common tomato dataset naming),
# this list is a helpful fallback. If your model was trained with a different
# label ordering, provide a `plant_disease_model.labels.json` or `.labels.txt`
# next to the .h5 file where each line is a label in model class order.
FALLBACK_LABELS = [
    "Tomato__Bacterial_spot",
    "Tomato__Early_blight",
    "Tomato__healthy",
    "Tomato__Late_blight",
    "Tomato__Leaf_Mold",
    "Tomato__Septoria_leaf_spot",
    "Tomato__Spider_mites",
    "Tomato__Target_Spot",
    "Tomato__Tomato_mosaic_virus",
    "Tomato__Tomato_Yellow_Leaf_Curl_Virus",
]


# Human-friendly recommendations per disease keyword. The code will try to
# match the predicted label to one of these keywords and return the matching
# recommendations. You should review and edit these recommendations to match
# agronomic best practices for your plants.
DISEASE_RECOMMENDATIONS = {
    "bacterial": [
        "Remove infected leaves and debris",
        "Avoid overhead irrigation; water at soil level",
        "Apply recommended bactericide if appropriate",
    ],
    "early_blight": [
        "Remove and destroy affected leaves",
        "Improve air circulation and reduce leaf wetness",
        "Use fungicide sprays labeled for early blight",
    ],
    "late_blight": [
        "Immediately remove infected plants to limit spread",
        "Apply appropriate fungicides and follow local guidance",
        "Avoid moving infected plant material between beds",
    ],
    "leaf_mold": [
        "Remove affected foliage; increase airflow",
        "Reduce humidity and spacing between plants",
        "Use fungicides where recommended",
    ],
    "septoria": [
        "Remove lower leaves and destroy infected tissue",
        "Rotate crops and avoid planting in same soil",
        "Apply appropriate fungicides if necessary",
    ],
    "spider": [
        "Rinse leaves with water to remove mites",
        "Introduce or conserve predatory insects",
        "Use miticides if infestation is severe",
    ],
    "target": [
        "Remove and destroy infected tissue",
        "Avoid wetting foliage and improve airflow",
        "Consider fungicide applications as recommended",
    ],
    "mosaic": [
        "Remove infected plants and control insect vectors",
        "Sanitize tools and avoid reusing contaminated material",
    ],
    "yellow_leaf": [
        "Investigate viral causes and insect vectors",
        "Remove severely infected plants and control vectors",
    ],
    "healthy": ["No action needed. Continue normal care."],
}


def _get_recommendations_for_label(label: Optional[str]) -> List[str]:
    if not label:
        return []
    lab = label.lower()
    # direct keyword matching
    for key, recs in DISEASE_RECOMMENDATIONS.items():
        if key in lab:
            return recs
    # try partial matches by tokenizing label
    tokens = [t for t in lab.replace('-', ' ').replace('_', ' ').split() if t]
    for t in tokens:
        for key, recs in DISEASE_RECOMMENDATIONS.items():
            if key in t:
                return recs
    # default: empty
    return []


_load_model_and_labels()

app = FastAPI(title="LeafGuard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ImageRequest(BaseModel):
    image: str


def decode_data_url(data_url: str) -> Image.Image:
    if not data_url or "," not in data_url:
        raise ValueError("Invalid data URL")
    header, encoded = data_url.split(",", 1)
    try:
        decoded = base64.b64decode(encoded)
    except Exception as exc:
        raise ValueError("Image decoding failed") from exc
    img = Image.open(io.BytesIO(decoded)).convert("RGB")
    return img


def _get_target_size() -> tuple[int, int]:
    # Try to infer model input size from MODEL, otherwise default to 224x224
    if MODEL is not None:
        try:
            shape = MODEL.input_shape
            # shape can be (None, H, W, C) (channels-last) or (None, C, H, W) (channels-first)
            if isinstance(shape, tuple):
                # common case: (None, H, W, C)
                if len(shape) == 4:
                    # channels-last: last dim is channel count (1 or 3)
                    if shape[-1] in (1, 3):
                        H = int(shape[1])
                        W = int(shape[2])
                        return (H, W)
                    # channels-first: second dim is channel count
                    if shape[1] in (1, 3):
                        H = int(shape[2])
                        W = int(shape[3])
                        return (H, W)
                # fallback: pick first two integer dims after batch dim
                ints = [s for s in shape if isinstance(s, int)]
                if len(ints) >= 2:
                    return (int(ints[0]), int(ints[1]))
        except Exception:
            pass
    return (224, 224)


def preprocess_image(img: Image.Image, target_size: tuple[int, int]) -> Any:
    # Use PIL + numpy preprocessing similar to training (rescale 1./255)
    # PIL expects size as (width, height)
    H, W = target_size
    img = img.resize((W, H))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def interpret_model_output(pred: Any) -> Dict[str, Any]:
    try:
        arr = np.asarray(pred)
        if arr.ndim == 0 or arr.shape == ():
            prob = float(arr) * 100.0
            is_healthy = prob < 50.0
            disease = None if is_healthy else "disease"
            return {"is_healthy": is_healthy, "confidence": prob, "disease": disease, "recommendations": []}

        if arr.ndim == 2:
            # (1, num_classes)
            probs = arr[0]
            idx = int(np.argmax(probs))
            conf = float(probs[idx]) * 100.0
            label = LABELS[idx] if LABELS and idx < len(LABELS) else (FALLBACK_LABELS[idx] if idx < len(FALLBACK_LABELS) else f"class_{idx}")
            lab = label.lower() if isinstance(label, str) else ""
            # detect healthy by token
            is_healthy = "healthy" in lab or lab in ("no_disease", "healthy_leaf")
            # derive recommendations using our keyword mapping helper
            recs: List[str] = []
            if not is_healthy:
                recs = _get_recommendations_for_label(label)
                if not recs:
                    # fallback generic suggestion
                    recs = [f"Recommended treatment for {label}"]
            return {"is_healthy": is_healthy, "confidence": conf, "disease": label, "recommendations": recs}

        # fallback
        return {"is_healthy": False, "confidence": 0.0, "disease": None, "recommendations": []}
    except Exception:
        return {"is_healthy": False, "confidence": 0.0, "disease": None, "recommendations": []}


def mock_prediction() -> Dict[str, Any]:
    return {
        "is_healthy": False,
        "confidence": 78.3,
        "disease": "Leaf Blight (mock)",
        "recommendations": [
            "Isolate the plant",
            "Apply an approved fungicide",
            "Remove affected leaves",
        ],
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "model_loaded": MODEL is not None, "tensorflow": TF_AVAILABLE}


@app.post("/predict")
def predict(req: ImageRequest, mock: bool = Query(False, description="Return a mock response instead of using the model"), debug: bool = Query(False, description="Include raw model probabilities and input info in the response (debug only)")):
    """Predict disease from a data-URL image. Set `?mock=true` to force a mock response.

    Returns an object compatible with the frontend's `AnalysisResult` mapping.
    """
    # If caller requests a mock response, return it immediately without decoding the image
    if mock:
        res = mock_prediction()
        res.update({"timestamp": int(__import__("time").time() * 1000)})
        return res

    try:
        img = decode_data_url(req.image)
    except ValueError as exc:
        # Bad image payload
        raise HTTPException(status_code=400, detail=str(exc))

    if not TF_AVAILABLE:
        raise HTTPException(status_code=503, detail="TensorFlow is not installed on the server. Install tensorflow to enable real predictions or call /predict?mock=true for a mock response.")

    if MODEL is None:
        raise HTTPException(status_code=503, detail=f"Model not found. Set MODEL_PATH env var or place plant_disease_model.h5 next to backend/main.py ({MODEL_PATH})")

    # Preprocess and predict
    try:
        target_size = _get_target_size()
        x = preprocess_image(img, target_size=target_size)
        pred = MODEL.predict(x)
        result = interpret_model_output(pred)
        result.update({"timestamp": int(__import__("time").time() * 1000)})

        if debug:
            try:
                arr = np.asarray(pred)
                if arr.ndim == 2:
                    probs = [float(p) for p in arr[0]]
                else:
                    probs = arr.tolist()
            except Exception:
                probs = None

            # include model metadata and preprocessing shape
            model_shape = None
            try:
                model_shape = getattr(MODEL, 'input_shape', None)
            except Exception:
                model_shape = None

            preprocessed_shape = None
            try:
                preprocessed_shape = getattr(x, 'shape', None)
                # convert numpy shape to list for JSON
                if hasattr(preprocessed_shape, '__iter__'):
                    preprocessed_shape = list(preprocessed_shape)
            except Exception:
                preprocessed_shape = None

            result['debug'] = {
                'raw_probs': probs,
                'labels_used': LABELS if LABELS is not None else FALLBACK_LABELS,
                'model_input_shape': model_shape,
                'preprocessed_shape': preprocessed_shape,
            }

        return result
    except Exception as exc:
        # Log error for server-side debugging
        print("Prediction error:", exc)
        raise HTTPException(status_code=500, detail="Prediction failed on the server")
