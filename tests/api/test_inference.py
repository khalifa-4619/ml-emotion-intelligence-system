"""Tests for inference service with a mock model."""
import pytest
import torch
import numpy as np
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from src.api.services.inference import InferenceService


class MockModelOutput:
    """Simulates Hugging Face model output."""
    def __init__(self, logits):
        self.logits = logits


class MockModel:
    """
    Minimal mock that returns controlled logits.
    Avoids loading the 265MB real model for unit tests.
    """
    def __init__(self, num_labels=6):
        self.num_labels = num_labels
        self._mode = "eval"
    
    def eval(self):
        self._mode = "eval"
    
    def to(self, device):
        return self
    
    def __call__(self, **kwargs):
        # Return logits that make "joy" (index 1) the highest
        batch_size = kwargs["input_ids"].shape[0]
        logits = torch.ones(batch_size, self.num_labels) * 0.1
        logits[:, 1] = 5.0  # "joy" gets highest score
        return MockModelOutput(logits=logits)


class MockTokenizer:
    """Minimal mock that returns fake token IDs."""
    def __call__(self, text, truncation=True, padding="max_length", 
                 max_length=128, return_tensors="pt"):
        # Create fake input_ids of correct shape
        batch_size = 1 if isinstance(text, str) else len(text)
        return {
            "input_ids": torch.ones(batch_size, max_length, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, max_length, dtype=torch.long),
        }


class TestInferenceService:
    """Test inference service with mock dependencies."""
    
    LABEL_NAMES = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    
    @pytest.fixture
    def service(self):
        """Create InferenceService with mock model and tokenizer."""
        return InferenceService(
            model=MockModel(),
            tokenizer=MockTokenizer(),
            label_names=self.LABEL_NAMES,
        )
    
    def test_predict_returns_expected_structure(self, service):
        result = service.predict("I am happy!")
        
        assert "emotion" in result
        assert "confidence" in result
        assert "scores" in result
        assert "processed_in_ms" in result
    
    def test_predict_returns_joy_for_mock(self, service):
        """Mock model always predicts joy (index 1)."""
        result = service.predict("any text")
        assert result["emotion"] == "joy"
    
    def test_confidence_is_float_between_0_and_1(self, service):
        result = service.predict("test")
        assert 0.0 <= result["confidence"] <= 1.0
        assert isinstance(result["confidence"], float)
    
    def test_scores_cover_all_labels(self, service):
        result = service.predict("test")
        assert len(result["scores"]) == len(self.LABEL_NAMES)
        returned_labels = [s["label"] for s in result["scores"]]
        assert returned_labels == self.LABEL_NAMES
    
    def test_scores_sum_to_approximately_one(self, service):
        result = service.predict("test")
        total = sum(s["score"] for s in result["scores"])
        assert abs(total - 1.0) < 0.01  # Softmax should sum to ~1
    
    def test_processed_in_ms_is_positive(self, service):
        result = service.predict("test")
        assert result["processed_in_ms"] > 0
    
    def test_get_model_info(self, service):
        info = service.get_model_info()
        assert info["num_labels"] == 6
        assert info["label_names"] == self.LABEL_NAMES