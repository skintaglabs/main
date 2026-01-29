"""FastAPI backend for SkinTag triage web application.

Provides upload endpoint for skin lesion images, runs SigLIP embedding +
classifier inference, and returns triage assessment results.
"""

import sys
from pathlib import Path

# Add project root
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import io
import pickle
import yaml
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse

from src.model.embeddings import EmbeddingExtractor
from src.model.triage import TriageSystem

app = FastAPI(title="SkinTag", description="AI-powered skin lesion triage screening tool")

# Global state (loaded on startup)
_state = {
    "extractor": None,
    "classifier": None,
    "e2e_model": None,  # End-to-end fine-tuned model (if available)
    "triage": None,
    "config": None,
    "inference_mode": None,  # "e2e" or "embedding+head"
}


@app.on_event("startup")
async def load_models():
    """Load models and config on server startup.

    Prefers fine-tuned end-to-end model if available (better accuracy),
    falls back to embedding extractor + classifier head.
    """
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path) as f:
        _state["config"] = yaml.safe_load(f)

    cache_dir = PROJECT_ROOT / "results" / "cache"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Try loading fine-tuned end-to-end model first
    e2e_dir = cache_dir / "finetuned_model"
    if (e2e_dir / "config.json").exists():
        try:
            from src.model.deep_classifier import EndToEndClassifier
            _state["e2e_model"] = EndToEndClassifier.load_for_inference(str(e2e_dir), device=device)
            _state["inference_mode"] = "e2e"
            print(f"Loaded fine-tuned end-to-end model from {e2e_dir}")
        except Exception as e:
            print(f"Failed to load e2e model: {e}, falling back to embedding+head")

    # Fall back to embedding extractor + pickled classifier
    if _state["inference_mode"] is None:
        for model_name in ["classifier_deep_mlp.pkl", "classifier_logistic_regression.pkl",
                            "classifier_deep.pkl", "classifier_logistic.pkl", "classifier.pkl"]:
            model_path = cache_dir / model_name
            if model_path.exists():
                with open(model_path, "rb") as f:
                    _state["classifier"] = pickle.load(f)
                print(f"Loaded classifier: {model_name}")
                break

        if _state["classifier"] is None:
            print("WARNING: No trained classifier found. Run train.py first.")

        _state["extractor"] = EmbeddingExtractor(device=device)
        _state["inference_mode"] = "embedding+head"
        print(f"Embedding extractor ready (device={device})")

    # Load triage system
    triage_config = _state["config"].get("triage", {})
    _state["triage"] = TriageSystem(triage_config)
    print(f"Triage system ready (inference mode: {_state['inference_mode']})")


@app.on_event("shutdown")
async def cleanup():
    if _state["extractor"] is not None:
        _state["extractor"].unload_model()


@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """Analyze an uploaded skin lesion image.

    Returns triage assessment with risk score, urgency tier, recommendation.
    """
    if _state["inference_mode"] == "e2e" and _state["e2e_model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if _state["inference_mode"] == "embedding+head" and _state["classifier"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")

    # Read and validate image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Classify — use end-to-end model or embedding+head
    if _state["inference_mode"] == "e2e":
        proba = _state["e2e_model"].predict_proba([image])
    else:
        extractor = _state["extractor"]
        embedding = extractor.extract([image])  # (1, 1152)
        clf = _state["classifier"]
        proba = clf.predict_proba(embedding.numpy())

    mal_prob = float(proba[0, 1]) if proba.ndim == 2 else float(proba[0])

    # Triage assessment
    triage = _state["triage"]
    result = triage.assess(mal_prob)

    return JSONResponse({
        "risk_score": round(result.risk_score, 4),
        "urgency_tier": result.urgency_tier,
        "recommendation": result.recommendation,
        "confidence": result.confidence,
        "disclaimer": result.disclaimer,
        "probabilities": {
            "benign": round(1 - mal_prob, 4),
            "malignant": round(mal_prob, 4),
        },
    })


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "inference_mode": _state["inference_mode"],
        "model_loaded": (_state["e2e_model"] is not None) or (_state["classifier"] is not None),
        "device": (
            _state["extractor"].device if _state["extractor"]
            else (_state["e2e_model"].device if _state["e2e_model"] else "unknown")
        ),
    }


# Serve static files and frontend
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = APP_DIR / "templates" / "index.html"
    if index_path.exists():
        return index_path.read_text()
    # Fallback: serve from static
    static_index = APP_DIR / "static" / "index.html"
    if static_index.exists():
        return static_index.read_text()
    return "<h1>SkinTag</h1><p>Frontend not found. Place index.html in app/templates/</p>"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
