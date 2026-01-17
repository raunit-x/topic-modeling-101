"""Pydantic models for topic taxonomy building."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class TermContext(BaseModel):
    """A sample context where a term appears."""

    sentence: str = Field(..., description="The sentence containing the term")
    source_file: str = Field(..., description="Source file path")
    char_start: int = Field(..., description="Character start position in the sentence")
    char_end: int = Field(..., description="Character end position in the sentence")


class ExtractedTerm(BaseModel):
    """A term extracted via TF-IDF with sample contexts."""

    term: str = Field(..., description="The extracted term (unigram, bigram, or trigram)")
    term_type: Literal["unigram", "bigram", "trigram"] = Field(
        ..., description="Type of n-gram"
    )
    tfidf_score: float = Field(..., description="TF-IDF score for this term")
    document_frequency: int = Field(..., description="Number of documents containing this term")
    sample_contexts: list[TermContext] = Field(
        default_factory=list, description="Sample sentences where this term appears"
    )
    is_valid_entity: bool | None = Field(
        default=None, description="Whether LLM validated this as a meaningful entity"
    )
    embedding: list[float] | None = Field(
        default=None, description="Embedding vector for this term"
    )


class TermValidation(BaseModel):
    """LLM response for validating if a term is a meaningful entity."""

    term: str = Field(..., description="The term being validated")
    is_valid_entity: bool = Field(..., description="Whether this is a meaningful entity")
    reasoning: str = Field(..., description="Explanation for the decision")
    suggested_canonical: str | None = Field(
        default=None, description="Suggested canonical form if applicable"
    )


class TermValidationResponse(BaseModel):
    """LLM response for batch term validation."""

    validations: list[TermValidation] = Field(..., description="Validation results for each term")


class TermSimilarity(BaseModel):
    """LLM response for term pair similarity."""

    term_a: str = Field(..., description="First term")
    term_b: str = Field(..., description="Second term")
    is_same_concept: bool = Field(..., description="Whether terms refer to the same concept")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    reasoning: str = Field(..., description="Explanation for the decision")
    canonical_name: str | None = Field(
        default=None, description="Suggested canonical name if same concept"
    )


class TermSimilarityResponse(BaseModel):
    """LLM response for batch term similarity checking."""

    similarities: list[TermSimilarity] = Field(
        ..., description="Similarity results for each pair"
    )


class TopicCluster(BaseModel):
    """A preliminary cluster of related terms before hierarchy building."""

    cluster_id: str = Field(..., description="Unique cluster identifier")
    terms: list[str] = Field(..., description="Terms in this cluster")
    sample_contexts: list[str] = Field(
        default_factory=list, description="Representative contexts from member terms"
    )
    suggested_name: str | None = Field(
        default=None, description="LLM-suggested name for this cluster"
    )


class Topic(BaseModel):
    """A topic in the taxonomy with hierarchical relationships."""

    id: str = Field(..., description="Unique topic identifier")
    canonical_name: str = Field(..., description="Canonical name for this topic")
    description: str = Field(default="", description="LLM-generated description")
    terms: list[str] = Field(default_factory=list, description="Terms belonging to this topic")
    parent_topic_id: str | None = Field(default=None, description="Parent topic ID if any")
    children: list[str] = Field(default_factory=list, description="Child topic IDs")
    embedding: list[float] = Field(default_factory=list, description="Average embedding of terms")
    sample_contexts: list[str] = Field(
        default_factory=list, description="Representative usage contexts"
    )


class TopicHierarchyItem(BaseModel):
    """LLM response for a single topic in hierarchy."""

    name: str = Field(..., description="Canonical topic name")
    description: str = Field(..., description="Brief description of the topic")
    terms: list[str] = Field(..., description="Terms belonging to this topic")
    children: list["TopicHierarchyItem"] = Field(
        default_factory=list, description="Child topics"
    )


class TopicHierarchyResponse(BaseModel):
    """LLM response for building topic hierarchy."""

    topics: list[TopicHierarchyItem] = Field(..., description="Root-level topics with children")


class Taxonomy(BaseModel):
    """Complete topic taxonomy."""

    topics: dict[str, Topic] = Field(default_factory=dict, description="All topics by ID")
    term_to_topic: dict[str, str] = Field(
        default_factory=dict, description="Mapping from term to topic ID"
    )
    root_topics: list[str] = Field(default_factory=list, description="Root-level topic IDs")
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Creation timestamp",
    )
    source_stats: dict = Field(
        default_factory=dict, description="Statistics about the source corpus"
    )

    def get_topic_by_term(self, term: str) -> Topic | None:
        """Get the topic containing a term."""
        topic_id = self.term_to_topic.get(term)
        if topic_id:
            return self.topics.get(topic_id)
        return None

    def get_all_terms(self) -> list[str]:
        """Get all terms in the taxonomy."""
        return list(self.term_to_topic.keys())

    def get_topic_path(self, topic_id: str) -> list[str]:
        """Get the path from root to a topic."""
        path = []
        current_id = topic_id
        while current_id:
            topic = self.topics.get(current_id)
            if not topic:
                break
            path.append(topic.canonical_name)
            current_id = topic.parent_topic_id
        return list(reversed(path))


class CorpusStats(BaseModel):
    """Statistics about the processed corpus."""

    total_documents: int = Field(..., description="Number of documents processed")
    total_tokens: int = Field(..., description="Total number of tokens")
    unique_unigrams: int = Field(..., description="Number of unique unigrams")
    unique_bigrams: int = Field(..., description="Number of unique bigrams")
    unique_trigrams: int = Field(..., description="Number of unique trigrams")
    terms_extracted: int = Field(..., description="Number of terms selected by TF-IDF")
    valid_entities: int = Field(..., description="Number of terms validated as entities")
    topics_created: int = Field(..., description="Number of topics in final taxonomy")
