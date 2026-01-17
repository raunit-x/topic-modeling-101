"""Pydantic models for semantic clustering."""

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A document to be clustered."""

    id: str = Field(..., description="Unique identifier for the document")
    text: str = Field(..., description="The text content of the document")
    metadata: str | None = Field(default=None, description="Optional metadata about the document")
    embedding: list[float] | None = Field(default=None, description="Pre-computed embedding vector")
    cluster_id: str | None = Field(default=None, description="Assigned cluster ID after clustering")


class SimilarityPair(BaseModel):
    """Result of comparing two documents for semantic similarity."""

    doc_id_a: str = Field(..., description="ID of the first document")
    doc_id_b: str = Field(..., description="ID of the second document")
    is_similar: bool = Field(..., description="Whether the documents are semantically equivalent")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    reasoning: str = Field(..., description="Explanation for the similarity decision")


class SimilarityCheckRequest(BaseModel):
    """Input for LLM similarity checking - a pair of texts to compare."""

    id_a: str = Field(..., description="ID of the first text")
    text_a: str = Field(..., description="First text to compare")
    id_b: str = Field(..., description="ID of the second text")
    text_b: str = Field(..., description="Second text to compare")


class SimilarityCheckResponse(BaseModel):
    """LLM response for batch similarity checking."""

    results: list[SimilarityPair] = Field(..., description="List of similarity results for each pair")


class ClusterVerification(BaseModel):
    """Result of verifying a proposed cluster."""

    cluster_id: str = Field(..., description="ID of the cluster being verified")
    is_valid: bool = Field(..., description="Whether the cluster is valid (all members are truly similar)")
    members_to_remove: list[str] = Field(
        default_factory=list, description="Document IDs that should be removed from this cluster"
    )
    reasoning: str = Field(..., description="Explanation for the verification decision")


class ClusterVerificationRequest(BaseModel):
    """Input for LLM cluster verification."""

    cluster_id: str = Field(..., description="ID of the cluster to verify")
    member_texts: list[tuple[str, str]] = Field(
        ..., description="List of (doc_id, text) tuples for cluster members"
    )


class ClusterVerificationResponse(BaseModel):
    """LLM response for cluster verification."""

    verifications: list[ClusterVerification] = Field(
        ..., description="Verification results for each cluster"
    )


class KnownTruthPair(BaseModel):
    """A known pair for sanity checking LLM responses."""

    text_a: str = Field(..., description="First text in the pair")
    text_b: str = Field(..., description="Second text in the pair")
    expected_similar: bool = Field(
        ..., description="True if texts should be similar, False if they should NOT be similar"
    )


class ClusteringResult(BaseModel):
    """Final clustering output."""

    documents: list[Document] = Field(..., description="All documents with assigned cluster IDs")
    cluster_count: int = Field(..., description="Total number of clusters created")
    sanity_check_stats: dict = Field(
        default_factory=dict, description="Statistics from sanity checks during clustering"
    )
