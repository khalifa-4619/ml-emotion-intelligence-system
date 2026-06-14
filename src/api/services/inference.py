"""
Model inference service.

Decoupled from HTTP layer for testability.
Can be instantiated with any Hugging Face sequence classification model.
"""
import time
import torch
import numpy as np
from typing import Dict, List, Any
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast


class InferenceService:
    """
    Wraps a trained emotion classification model for single-text inference.
    
    Model and tokenizer are injected via constructor - no hard dependency
    on specific model paths or Hugging Face Hub. This enables:
    - Unit testing with mock models
    - Swapping models without changing service code
    - Loading from different sources (local, S3, Hub)
    """

    def __init__(
        self,
        model: DistilBertForSequenceClassification,
        tokenizer: DistilBertTokenizerFast,
        label_names: List[str],
        model_name: str = "distilbert-emotion",
        max_length: int = 128,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.label_names = label_names
        self.model_name = model_name
        self.max_length = max_length
        
        # Set to eval mode - disables dropout for deterministic inference
        self.model.eval()
        
        # Determine device (CPU or CUDA)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Run inference on a single text input.
        
        Args:
            text: Input text to classify.
        
        Returns:
            Dict with:
            - emotion: predicted label
            - confidence: softmax probability of top class
            - scores: list of {label, score} for all classes
            - processed_in_ms: inference latency
        
        Handles:
        - Tokenization (truncation, padding)
        - Device placement
        - Softmax conversion
        - Label mapping
        """
        start_time = time.perf_counter()
        
        # Tokenize with fixed-length padding (single sample, so dynamic padding
        # offers no benefit here - we pad to max_length for consistent tensor shape)
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # Move to same device as model
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # Inference - no gradient computation needed
        with torch.no_grad():
            outputs = self.model(**encoded)
            logits = outputs.logits
        
        # Softmax for probabilities
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        
        # Get top prediction
        top_idx = int(np.argmax(probs))
        
        # Build response
        scores = [
            {"label": label, "score": float(score)}
            for label, score in zip(self.label_names, probs)
        ]
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        return {
            "emotion": self.label_names[top_idx],
            "confidence": float(probs[top_idx]),
            "scores": scores,
            "processed_in_ms": round(elapsed_ms, 2),
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata for health checks."""
        return {
            "model_name": self.model_name,
            "num_labels": len(self.label_names),
            "label_names": self.label_names,
            "device": str(self.device),
        }