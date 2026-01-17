"""LLM layers for semantic similarity detection and cluster verification."""

import asyncio
from typing import Literal

from openai import AsyncOpenAI

from ..openai_client import chat_to_schema, get_client
from .schemas import (
    ClusterVerification,
    ClusterVerificationRequest,
    SimilarityCheckRequest,
    SimilarityPair,
)


# System prompts for LLM calls
SIMILARITY_SYSTEM_PROMPT = """You are an expert at determining semantic equivalence between statements.

Two statements are semantically equivalent if they ask for or describe the SAME thing, even if worded differently.

Examples of equivalent statements:
- "how many customers do I have" ≡ "customer count"
- "total revenue" ≡ "sum of all sales"
- "what is the average order value" ≡ "mean order amount"

Examples of NOT equivalent statements:
- "customer count" ≢ "product inventory" (different entities)
- "total revenue" ≢ "profit margin" (different metrics)

For each pair, determine if they are semantically equivalent and provide your reasoning."""

VERIFICATION_SYSTEM_PROMPT = """You are an expert at verifying semantic clusters.

A valid cluster contains statements that ALL refer to the same underlying concept or query.

Your job is to:
1. Review all members of each proposed cluster
2. Identify if any members do NOT belong (are not semantically equivalent to the others)
3. Flag members that should be removed

Be conservative: if unsure, recommend removal to maintain cluster purity."""


class SimilarityBatchResponse(SimilarityPair):
    """Extended response model for batch similarity checking."""

    pass


class SimilarityBatchResult:
    """Result container for batch similarity LLM response."""

    results: list[SimilarityPair]


from pydantic import BaseModel, Field


class _SimilarityLLMResponse(BaseModel):
    """Internal model for LLM similarity response."""

    pairs: list[SimilarityPair] = Field(..., description="Similarity results for each pair")


class _VerificationLLMResponse(BaseModel):
    """Internal model for LLM verification response."""

    verifications: list[ClusterVerification] = Field(
        ..., description="Verification results for each cluster"
    )


async def check_similarity_batch(
    pairs: list[SimilarityCheckRequest],
    client: AsyncOpenAI | None = None,
    model: str = "gpt-4o-mini",
) -> list[SimilarityPair]:
    """
    Check semantic similarity for a batch of text pairs.

    Args:
        pairs: List of text pairs to check
        client: Optional OpenAI client
        model: Model to use for similarity checking

    Returns:
        List of SimilarityPair results
    """
    if not pairs:
        return []

    if client is None:
        client = get_client()

    # Format pairs for the prompt
    pairs_text = "\n\n".join(
        f"Pair {i+1}:\n"
        f"  ID A: {p.id_a}\n"
        f"  Text A: \"{p.text_a}\"\n"
        f"  ID B: {p.id_b}\n"
        f"  Text B: \"{p.text_b}\""
        for i, p in enumerate(pairs)
    )

    user_message = f"""Analyze the following {len(pairs)} pairs of statements for semantic equivalence.

{pairs_text}

For each pair, determine:
1. Are they semantically equivalent? (do they mean the same thing?)
2. How confident are you? (0.0-1.0)
3. Brief reasoning

Return results for ALL pairs in order."""

    messages = [
        {"role": "system", "content": SIMILARITY_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    response = await chat_to_schema(
        messages=messages,
        schema=_SimilarityLLMResponse,
        model=model,
        client=client,
    )

    return response.pairs


async def verify_clusters_batch(
    clusters: list[ClusterVerificationRequest],
    client: AsyncOpenAI | None = None,
    model: str = "gpt-4o-mini",
) -> list[ClusterVerification]:
    """
    Verify that proposed clusters are semantically valid.

    Args:
        clusters: List of clusters to verify
        client: Optional OpenAI client
        model: Model to use for verification

    Returns:
        List of ClusterVerification results
    """
    if not clusters:
        return []

    if client is None:
        client = get_client()

    # Format clusters for the prompt
    clusters_text = "\n\n".join(
        f"Cluster {c.cluster_id}:\n"
        + "\n".join(f"  - [{doc_id}]: \"{text}\"" for doc_id, text in c.member_texts)
        for c in clusters
    )

    user_message = f"""Review the following {len(clusters)} proposed clusters.

Each cluster should contain statements that are ALL semantically equivalent (refer to the same thing).

{clusters_text}

For each cluster:
1. Are all members truly equivalent?
2. If not, which specific members (by ID) should be removed?
3. Explain your reasoning

Be strict: clusters should be pure. When in doubt, recommend removal."""

    messages = [
        {"role": "system", "content": VERIFICATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    response = await chat_to_schema(
        messages=messages,
        schema=_VerificationLLMResponse,
        model=model,
        client=client,
    )

    return response.verifications


async def check_similarity_single(
    text_a: str,
    text_b: str,
    id_a: str = "a",
    id_b: str = "b",
    client: AsyncOpenAI | None = None,
    model: str = "gpt-4o-mini",
) -> SimilarityPair:
    """
    Check semantic similarity for a single pair of texts.

    Convenience wrapper around check_similarity_batch for single pairs.

    Args:
        text_a: First text
        text_b: Second text
        id_a: ID for first text
        id_b: ID for second text
        client: Optional OpenAI client
        model: Model to use

    Returns:
        SimilarityPair result
    """
    request = SimilarityCheckRequest(
        id_a=id_a,
        text_a=text_a,
        id_b=id_b,
        text_b=text_b,
    )
    results = await check_similarity_batch([request], client=client, model=model)
    return results[0]
