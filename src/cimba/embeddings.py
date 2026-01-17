"""OpenAI embeddings module with async single and concurrent batch support."""

import asyncio
import os

from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# Default embedding model
DEFAULT_MODEL = "text-embedding-3-small"

# Maximum concurrent requests for batch embeddings
MAX_CONCURRENCY = 12


def get_client() -> AsyncOpenAI:
    """Get or create async OpenAI client."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    return AsyncOpenAI(api_key=api_key)


async def get_embedding(
    text: str,
    model: str = DEFAULT_MODEL,
    client: AsyncOpenAI | None = None,
) -> list[float]:
    """
    Get embedding for a single text.

    Args:
        text: The text to embed
        model: The embedding model to use (default: text-embedding-3-small)
        client: Optional AsyncOpenAI client (creates one if not provided)

    Returns:
        List of floats representing the embedding vector
    """
    if client is None:
        client = get_client()

    # Clean the text - replace newlines with spaces
    text = text.replace("\n", " ").strip()

    if not text:
        raise ValueError("Text cannot be empty")

    response = await client.embeddings.create(
        input=text,
        model=model,
    )

    return response.data[0].embedding


async def get_embeddings_batch(
    texts: list[str],
    model: str = DEFAULT_MODEL,
    client: AsyncOpenAI | None = None,
    max_concurrency: int = MAX_CONCURRENCY,
) -> list[list[float]]:
    """
    Get embeddings for multiple texts using concurrent async calls.

    Makes up to max_concurrency parallel API calls for efficiency.

    Args:
        texts: List of texts to embed
        model: The embedding model to use (default: text-embedding-3-small)
        client: Optional AsyncOpenAI client (creates one if not provided)
        max_concurrency: Maximum number of concurrent requests (default: 12)

    Returns:
        List of embedding vectors (same order as input texts)
    """
    if not texts:
        return []

    if client is None:
        client = get_client()

    # Clean all texts
    cleaned_texts = [text.replace("\n", " ").strip() for text in texts]

    # Validate no empty texts
    for i, text in enumerate(cleaned_texts):
        if not text:
            raise ValueError(f"Text at index {i} cannot be empty")

    # Semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrency)

    async def get_single_embedding(index: int, text: str) -> tuple[int, list[float]]:
        """Get embedding for a single text with index tracking."""
        async with semaphore:
            response = await client.embeddings.create(
                input=text,
                model=model,
            )
            return (index, response.data[0].embedding)

    # Create tasks for all texts
    tasks = [
        get_single_embedding(i, text) for i, text in enumerate(cleaned_texts)
    ]

    # Execute all tasks concurrently (limited by semaphore)
    results = await asyncio.gather(*tasks)

    # Sort by index to maintain original order
    sorted_results = sorted(results, key=lambda x: x[0])

    return [embedding for _, embedding in sorted_results]


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First embedding vector
        vec2: Second embedding vector

    Returns:
        Cosine similarity score between -1 and 1
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same length")

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)
