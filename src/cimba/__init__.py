"""Cimba AI - Interview development environment with LLM integrations."""

import logging

# Suppress verbose HTTP logs from OpenAI client
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

from .embeddings import get_embedding, get_embeddings_batch
from .schemas import ExtractedEntity, ClassificationResult
from .openai_client import chat_to_schema as openai_chat_to_schema
from .similarity import semantic_similarity, semantic_similarity_batch

__all__ = [
    "get_embedding",
    "get_embeddings_batch",
    "ExtractedEntity",
    "ClassificationResult",
    "openai_chat_to_schema",
    "semantic_similarity",
    "semantic_similarity_batch",
]
