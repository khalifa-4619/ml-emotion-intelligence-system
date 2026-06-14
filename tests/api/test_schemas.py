"""Tests for request/response schema validation."""
import pytest
from pydantic import ValidationError
from src.api.schemas import PredictRequest, PredictResponse, EmotionScore


class TestPredictRequest:
    """Verify input validation."""
    
    def test_valid_request(self):
        request = PredictRequest(text="I am happy today!")
        assert request.text == "I am happy today!"
    
    def test_empty_text_rejected(self):
        with pytest.raises(ValidationError):
            PredictRequest(text="")
    
    def test_too_long_text_rejected(self):
        with pytest.raises(ValidationError):
            PredictRequest(text="a" * 513)
    
    def test_max_length_accepted(self):
        request = PredictRequest(text="a" * 512)
        assert len(request.text) == 512


class TestPredictResponse:
    """Verify response structure."""
    
    def test_valid_response(self):
        response = PredictResponse(
            emotion="joy",
            confidence=0.95,
            scores=[
                EmotionScore(label="sadness", score=0.01),
                EmotionScore(label="joy", score=0.95),
                EmotionScore(label="love", score=0.02),
                EmotionScore(label="anger", score=0.01),
                EmotionScore(label="fear", score=0.005),
                EmotionScore(label="surprise", score=0.005),
            ],
            processed_in_ms=45.2,
        )
        assert response.emotion == "joy"
        assert response.confidence == 0.95
        assert len(response.scores) == 6
    
    def test_confidence_bounds_enforced(self):
        with pytest.raises(ValidationError):
            EmotionScore(label="joy", score=1.5)
        
        with pytest.raises(ValidationError):
            EmotionScore(label="joy", score=-0.1)
    
    def test_scores_sum_not_validated(self):
        """
        Pydantic doesn't validate that scores sum to 1.0.
        That's the model's responsibility (softmax guarantees it).
        This test documents that the schema accepts any valid floats.
        """
        response = PredictResponse(
            emotion="joy",
            confidence=0.5,
            scores=[
                EmotionScore(label="joy", score=0.5),
                EmotionScore(label="sadness", score=0.5),
            ],
            processed_in_ms=10.0,
        )
        assert response.confidence == 0.5 