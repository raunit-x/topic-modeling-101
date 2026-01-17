"""Pytest configuration and fixtures."""

import os
import pytest
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (require API keys)"
    )


@pytest.fixture
def openai_api_key() -> str | None:
    """Get OpenAI API key from environment."""
    return os.getenv("OPENAI_API_KEY")


@pytest.fixture
def anthropic_api_key() -> str | None:
    """Get Anthropic API key from environment."""
    return os.getenv("ANTHROPIC_API_KEY")


@pytest.fixture
def skip_without_openai_key(openai_api_key):
    """Skip test if OpenAI API key is not available."""
    if not openai_api_key:
        pytest.skip("OPENAI_API_KEY not set")


@pytest.fixture
def skip_without_anthropic_key(anthropic_api_key):
    """Skip test if Anthropic API key is not available."""
    if not anthropic_api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")


@pytest.fixture
def sample_texts() -> list[str]:
    """Sample texts for embedding tests."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
    ]


@pytest.fixture
def classification_messages() -> list[dict]:
    """Sample messages for classification tests."""
    return [
        {
            "role": "system",
            "content": "You are a sentiment classifier. Classify the sentiment of the given text.",
        },
        {"role": "user", "content": "I absolutely love this product! It's amazing!"},
    ]
