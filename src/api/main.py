"""
FastAPI application for emotion classification inference.

Model loaded at startup, not per-request.
Routes are thin - they delegate to InferenceService.
"""
import json
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from .schemas import PredictRequest, PredictResponse, HealthResponse
from .services.inference import InferenceService

# Global service instance - initialized at startup
inference_service: InferenceService = None

# Model path - in production, this comes from environment variable
MODEL_PATH = os.getenv("MODEL_PATH", "models/distilbert-emotion-v1")


def load_model():
    """
    Load model and tokenizer from disk.
    
    Extracted as a function for testability - tests can mock this
    to avoid loading the full 265MB model.
    """
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_PATH, local_files_only=True
    )
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        MODEL_PATH, local_files_only=True
    )
    
    # Load label names from model metadata
    info_path = os.path.join(MODEL_PATH, "model_info.json")
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            info = json.load(f)
        label_names = info["label_names"]
    else:
        # Fallback - should not happen if model was saved correctly
        label_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    
    return model, tokenizer, label_names


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle handler.
    
    Startup: Load model into memory once.
    Shutdown: Clean up resources.
    """
    global inference_service
    
    print(f"Loading model from {MODEL_PATH}...")
    model, tokenizer, label_names = load_model()
    inference_service = InferenceService(
        model=model,
        tokenizer=tokenizer,
        label_names=label_names,
        model_name="distilbert-emotion-v1",
    )
    print(f"Model loaded. Device: {inference_service.device}")
    print(f"Labels: {inference_service.label_names}")
    
    # Warmup: absorb JIT compilation cost at startup, not on first user request
    print("Warming up model...")
    inference_service.predict("warmup")
    print("Model ready.")
    
    yield  # Application runs here
    
    # Shutdown cleanup
    print("Shutting down...")
    inference_service = None


app = FastAPI(
    title="ML Emotion Intelligence System",
    description="Emotion classification API powered by DistilBERT",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring and load balancers."""
    if inference_service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    info = inference_service.get_model_info()
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_name=info["model_name"],
        num_labels=info["num_labels"],
        label_names=info["label_names"],
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Predict emotion from text input.
    
    Returns predicted emotion with confidence scores for all classes.
    """
    if inference_service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = inference_service.predict(request.text)
        return PredictResponse(
            emotion=result["emotion"],
            confidence=result["confidence"],
            scores=result["scores"],
            processed_in_ms=result["processed_in_ms"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint - API information."""
    return {
        "name": "ML Emotion Intelligence System",
        "version": "0.1.0",
        "endpoints": {
            "/health": "GET - Health check",
            "/predict": "POST - Predict emotion from text",
        }
    }