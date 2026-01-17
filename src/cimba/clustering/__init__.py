"""Semantic clustering module for grouping semantically equivalent statements."""

from .clusterer import SemanticClusterer
from .index import EmbeddingIndex
from .llm_layers import check_similarity_batch, check_similarity_single, verify_clusters_batch
from .sanity_checker import DEFAULT_CANARIES, SanityCheckError, SanityChecker
from .schemas import (
    ClusteringResult,
    ClusterVerification,
    ClusterVerificationRequest,
    Document,
    KnownTruthPair,
    SimilarityCheckRequest,
    SimilarityPair,
)

__all__ = [
    # Main classes
    "SemanticClusterer",
    "EmbeddingIndex",
    "SanityChecker",
    # Schemas
    "Document",
    "SimilarityPair",
    "SimilarityCheckRequest",
    "ClusterVerification",
    "ClusterVerificationRequest",
    "ClusteringResult",
    "KnownTruthPair",
    # Functions
    "check_similarity_batch",
    "check_similarity_single",
    "verify_clusters_batch",
    # Constants/Errors
    "DEFAULT_CANARIES",
    "SanityCheckError",
]
