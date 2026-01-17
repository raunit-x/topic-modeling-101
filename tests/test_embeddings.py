"""Tests for embeddings module."""

import pytest

from cimba.embeddings import cosine_similarity


class TestCosineSimilarity:
    """Tests for cosine_similarity function (no API required)."""

    def test_identical_vectors(self):
        """Test that identical vectors have similarity of 1."""
        vec = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert abs(cosine_similarity(vec, vec) - 1.0) < 1e-9

    def test_orthogonal_vectors(self):
        """Test that orthogonal vectors have similarity of 0."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        assert abs(cosine_similarity(vec1, vec2)) < 1e-9

    def test_opposite_vectors(self):
        """Test that opposite vectors have similarity of -1."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [-1.0, -2.0, -3.0]
        assert abs(cosine_similarity(vec1, vec2) - (-1.0)) < 1e-9

    def test_similar_vectors(self):
        """Test similar vectors have high positive similarity."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [1.1, 2.1, 3.1]
        similarity = cosine_similarity(vec1, vec2)
        assert similarity > 0.99

    def test_different_length_vectors_raises(self):
        """Test that vectors of different lengths raise ValueError."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [1.0, 2.0]
        with pytest.raises(ValueError, match="same length"):
            cosine_similarity(vec1, vec2)

    def test_zero_vector(self):
        """Test that zero vector returns 0 similarity."""
        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 2.0, 3.0]
        assert cosine_similarity(vec1, vec2) == 0.0


class TestEmbeddingsValidation:
    """Tests for embeddings input validation (no API required)."""

    async def test_get_embedding_empty_text_raises(self, skip_without_openai_key):
        """Test that empty text raises ValueError."""
        from cimba.embeddings import get_embedding

        with pytest.raises(ValueError, match="cannot be empty"):
            await get_embedding("")

    async def test_get_embeddings_batch_empty_list(self, skip_without_openai_key):
        """Test that empty list returns empty list."""
        from cimba.embeddings import get_embeddings_batch

        result = await get_embeddings_batch([])
        assert result == []


@pytest.mark.integration
class TestEmbeddingsIntegration:
    """Integration tests that require OpenAI API key."""

    async def test_get_single_embedding(self, skip_without_openai_key):
        """Test getting a single embedding."""
        from cimba.embeddings import get_embedding

        embedding = await get_embedding("Hello, world!")
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

    async def test_get_batch_embeddings(self, skip_without_openai_key, sample_texts):
        """Test getting batch embeddings."""
        from cimba.embeddings import get_embeddings_batch

        embeddings = await get_embeddings_batch(sample_texts)
        assert len(embeddings) == len(sample_texts)
        assert all(isinstance(e, list) for e in embeddings)

    async def test_similar_texts_have_high_similarity(self, skip_without_openai_key):
        """Test that semantically similar texts have high cosine similarity."""
        from cimba.embeddings import get_embeddings_batch

        texts = [
            "The cat sat on the mat.",
            "A cat was sitting on a mat.",
            "Quantum physics explains subatomic particles.",
        ]
        embeddings = await get_embeddings_batch(texts)

        # Similar sentences should have high similarity
        sim_similar = cosine_similarity(embeddings[0], embeddings[1])
        # Different topics should have lower similarity
        sim_different = cosine_similarity(embeddings[0], embeddings[2])

        assert sim_similar > sim_different
        assert sim_similar > 0.8  # Should be quite similar

    async def test_batch_concurrency(self, skip_without_openai_key):
        """Test that batch embeddings work with many concurrent requests."""
        from cimba.embeddings import get_embeddings_batch

        # Create 20 texts to test concurrency (more than MAX_CONCURRENCY=12)
        texts = [f"This is test sentence number {i}." for i in range(20)]

        embeddings = await get_embeddings_batch(texts)

        assert len(embeddings) == 20
        assert all(isinstance(e, list) for e in embeddings)
        assert all(len(e) > 0 for e in embeddings)
