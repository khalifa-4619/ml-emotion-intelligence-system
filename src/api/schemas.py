"""
Pydantic schemas for request/response validation.

These define the API contract. Changes here are breaking changes
and should be versioned carefully.
"""
from pydantic import BaseModel, Field
from typing import List, Dict


class PredictRequest(BaseModel):
    """
    Single-text prediction request.
    
    Validation:
    - text must be non-empty
    - text limited to 512 chars (tweets are short; prevents abuse)
    """
    text: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Input text to classify. Maximum 512 characters.",
        example="I am feeling great today!"
    )


class EmotionScore(BaseModel):
    """Confidence score for a single emotion label."""
    label: str = Field(..., description="Emotion label name")
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1"
    )


class PredictResponse(BaseModel):
    """
    Prediction response with top emotion and all scores.
    
    Returns:
    - emotion: the predicted label
    - confidence: confidence for the top prediction
    - scores: all class probabilities (for UI display)
    - processed_in_ms: inference latency (for monitoring)
    """
    emotion: str = Field(..., description="Predicted emotion label")
    confidence: float = Field(..., ge=0.0, le=1.0)
    scores: List[EmotionScore] = Field(..., description="All class probabilities")
    processed_in_ms: float = Field(..., description="Inference time in milliseconds")


class HealthResponse(BaseModel):
    """Health check response for monitoring."""
    status: str = Field("healthy")
    model_loaded: bool = Field(True)
    model_name: str
    num_labels: int
    label_names: List[str]