"""Tests for Pydantic schemas."""

import pytest
from pydantic import ValidationError

from cimba.schemas import (
    ExtractedEntity,
    ClassificationResult,
    EntityExtractionResponse,
    SentimentAnalysis,
    SummaryResponse,
)


class TestExtractedEntity:
    """Tests for ExtractedEntity model."""

    def test_valid_entity(self):
        """Test creating a valid entity."""
        entity = ExtractedEntity(
            name="John Doe",
            entity_type="PERSON",
            confidence=0.95,
            context="John Doe is the CEO of...",
        )
        assert entity.name == "John Doe"
        assert entity.entity_type == "PERSON"
        assert entity.confidence == 0.95
        assert entity.context == "John Doe is the CEO of..."

    def test_entity_without_context(self):
        """Test entity without optional context."""
        entity = ExtractedEntity(
            name="Acme Corp",
            entity_type="ORGANIZATION",
            confidence=0.88,
        )
        assert entity.context is None

    def test_confidence_bounds(self):
        """Test that confidence must be between 0 and 1."""
        with pytest.raises(ValidationError):
            ExtractedEntity(
                name="Test",
                entity_type="PERSON",
                confidence=1.5,  # Invalid: > 1
            )

        with pytest.raises(ValidationError):
            ExtractedEntity(
                name="Test",
                entity_type="PERSON",
                confidence=-0.1,  # Invalid: < 0
            )

    def test_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(ValidationError):
            ExtractedEntity(name="Test", confidence=0.5)  # Missing entity_type


class TestClassificationResult:
    """Tests for ClassificationResult model."""

    def test_valid_classification(self):
        """Test creating a valid classification."""
        result = ClassificationResult(
            label="positive",
            confidence=0.92,
            reasoning="The text contains positive sentiment words.",
        )
        assert result.label == "positive"
        assert result.confidence == 0.92
        assert result.reasoning is not None

    def test_classification_without_reasoning(self):
        """Test classification without optional reasoning."""
        result = ClassificationResult(label="negative", confidence=0.78)
        assert result.reasoning is None

    def test_model_serialization(self):
        """Test model can be serialized to dict/JSON."""
        result = ClassificationResult(label="neutral", confidence=0.65)
        data = result.model_dump()
        assert data["label"] == "neutral"
        assert data["confidence"] == 0.65
        assert data["reasoning"] is None


class TestEntityExtractionResponse:
    """Tests for EntityExtractionResponse model."""

    def test_response_with_entities(self):
        """Test response with multiple entities."""
        response = EntityExtractionResponse(
            text_analyzed="John works at Google in California.",
            entities=[
                ExtractedEntity(name="John", entity_type="PERSON", confidence=0.9),
                ExtractedEntity(name="Google", entity_type="ORGANIZATION", confidence=0.95),
                ExtractedEntity(name="California", entity_type="LOCATION", confidence=0.88),
            ],
        )
        assert len(response.entities) == 3
        assert response.text_analyzed == "John works at Google in California."

    def test_response_empty_entities(self):
        """Test response with no entities found."""
        response = EntityExtractionResponse(
            text_analyzed="Hello world.",
            entities=[],
        )
        assert len(response.entities) == 0


class TestSentimentAnalysis:
    """Tests for SentimentAnalysis model."""

    def test_positive_sentiment(self):
        """Test positive sentiment analysis."""
        analysis = SentimentAnalysis(
            sentiment="positive",
            score=0.85,
            aspects=["product quality", "customer service"],
        )
        assert analysis.sentiment == "positive"
        assert analysis.score == 0.85
        assert len(analysis.aspects) == 2

    def test_score_bounds(self):
        """Test that score must be between -1 and 1."""
        with pytest.raises(ValidationError):
            SentimentAnalysis(sentiment="positive", score=1.5)

        with pytest.raises(ValidationError):
            SentimentAnalysis(sentiment="negative", score=-1.5)


class TestSummaryResponse:
    """Tests for SummaryResponse model."""

    def test_valid_summary(self):
        """Test creating a valid summary."""
        summary = SummaryResponse(
            summary="This is a brief summary of the text.",
            key_points=["Point 1", "Point 2", "Point 3"],
            word_count=8,
        )
        assert summary.summary == "This is a brief summary of the text."
        assert len(summary.key_points) == 3
        assert summary.word_count == 8

    def test_summary_empty_key_points(self):
        """Test summary with no key points."""
        summary = SummaryResponse(
            summary="Short summary.",
            key_points=[],
            word_count=2,
        )
        assert len(summary.key_points) == 0
