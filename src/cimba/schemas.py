"""Pydantic models for structured LLM outputs."""

from pydantic import BaseModel, Field


class TruthSet(BaseModel):
    """Model for truth set creation."""

    documents: list[str] = Field(..., description="The documents in the truth set")
    


class ExtractedEntity(BaseModel):
    """Model for entity extraction from text."""

    name: str = Field(..., description="The name of the extracted entity")
    entity_type: str = Field(
        ..., description="The type of entity (e.g., PERSON, ORGANIZATION, LOCATION)"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score between 0 and 1"
    )
    context: str | None = Field(
        default=None, description="The surrounding context where the entity was found"
    )


class ClassificationResult(BaseModel):
    """Model for text classification results."""

    label: str = Field(..., description="The classification label")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score between 0 and 1"
    )
    reasoning: str | None = Field(
        default=None, description="Explanation for why this classification was chosen"
    )


class EntityExtractionResponse(BaseModel):
    """Response model containing multiple extracted entities."""

    entities: list[ExtractedEntity] = Field(
        default_factory=list, description="List of extracted entities"
    )
    text_analyzed: str = Field(..., description="The original text that was analyzed")


class SentimentAnalysis(BaseModel):
    """Model for sentiment analysis results."""

    sentiment: str = Field(
        ..., description="The sentiment (positive, negative, neutral)"
    )
    score: float = Field(
        ..., ge=-1.0, le=1.0, description="Sentiment score from -1 (negative) to 1 (positive)"
    )
    aspects: list[str] = Field(
        default_factory=list, description="Key aspects or topics identified"
    )


class SummaryResponse(BaseModel):
    """Model for text summarization results."""

    summary: str = Field(..., description="The generated summary")
    key_points: list[str] = Field(
        default_factory=list, description="Main key points from the text"
    )
    word_count: int = Field(..., description="Word count of the summary")
