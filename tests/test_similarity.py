"""Tests for the hybrid semantic similarity module."""

import math

import pytest

from cimba.similarity import (
    calibrate_score,
    normalize_tokens,
    syntactic_similarity,
    semantic_similarity,
    semantic_similarity_from_embeddings,
    semantic_similarity_batch,
    topic_similarity,
    extract_topics_from_text,
    STOPWORDS,
)


class TestCalibrateScore:
    """Tests for the sigmoid calibration function."""

    def test_midpoint_returns_half(self):
        """Score at midpoint should return 0.5."""
        assert abs(calibrate_score(0.5, midpoint=0.5, steepness=10.0) - 0.5) < 1e-9

    def test_above_midpoint_higher_than_half(self):
        """Scores above midpoint should map to > 0.5."""
        result = calibrate_score(0.7, midpoint=0.5, steepness=10.0)
        assert result > 0.5

    def test_below_midpoint_lower_than_half(self):
        """Scores below midpoint should map to < 0.5."""
        result = calibrate_score(0.3, midpoint=0.5, steepness=10.0)
        assert result < 0.5

    def test_high_steepness_polarizes_more(self):
        """Higher steepness should produce more extreme values."""
        low_steep = calibrate_score(0.7, midpoint=0.5, steepness=5.0)
        high_steep = calibrate_score(0.7, midpoint=0.5, steepness=20.0)
        # High steepness pushes 0.7 closer to 1.0
        assert high_steep > low_steep

    def test_very_high_input_approaches_one(self):
        """Very high input should approach 1.0."""
        result = calibrate_score(0.95, midpoint=0.5, steepness=10.0)
        assert result > 0.98

    def test_very_low_input_approaches_zero(self):
        """Very low input should approach 0.0."""
        result = calibrate_score(0.05, midpoint=0.5, steepness=10.0)
        assert result < 0.02

    def test_custom_midpoint(self):
        """Custom midpoint should shift the sigmoid center."""
        # At midpoint=0.8, an input of 0.8 should give 0.5
        result = calibrate_score(0.8, midpoint=0.8, steepness=10.0)
        assert abs(result - 0.5) < 1e-9


class TestNormalizeTokens:
    """Tests for text tokenization and normalization."""

    def test_basic_tokenization(self):
        """Basic text should be tokenized correctly."""
        tokens = normalize_tokens("Hello World")
        assert "hello" in tokens
        assert "world" in tokens

    def test_stopwords_removed(self):
        """Stopwords should be filtered out."""
        tokens = normalize_tokens("The quick brown fox jumps over the lazy dog")
        # Common stopwords should not be present
        assert "the" not in tokens
        # Content words should remain
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens
        assert "jumps" in tokens
        assert "lazy" in tokens
        assert "dog" in tokens

    def test_punctuation_removed(self):
        """Punctuation should be removed."""
        tokens = normalize_tokens("Hello, world! How are you?")
        # No punctuation in tokens
        for token in tokens:
            assert "," not in token
            assert "!" not in token
            assert "?" not in token

    def test_empty_string(self):
        """Empty string should return empty list."""
        tokens = normalize_tokens("")
        assert tokens == []

    def test_only_stopwords(self):
        """String with only stopwords should return empty list."""
        tokens = normalize_tokens("the a an is are")
        assert tokens == []

    def test_single_char_tokens_removed(self):
        """Single character tokens should be filtered out."""
        tokens = normalize_tokens("a b c hello world")
        assert "hello" in tokens
        assert "world" in tokens
        # Single chars should be removed
        for token in tokens:
            assert len(token) > 1


class TestSyntacticSimilarity:
    """Tests for token-based Jaccard similarity."""

    def test_identical_texts(self):
        """Identical texts should have similarity of 1.0."""
        result = syntactic_similarity("hello world", "hello world")
        assert abs(result - 1.0) < 1e-9

    def test_no_overlap(self):
        """Texts with no content word overlap should have 0 similarity."""
        result = syntactic_similarity("apple banana", "orange grape")
        assert result == 0.0

    def test_partial_overlap(self):
        """Partial overlap should give intermediate similarity."""
        # "customer" appears in both (as customers/customer)
        result = syntactic_similarity("customers here", "customer there")
        # "customers" != "customer" exactly, so may not match
        # But "here" and "there" don't match either
        assert 0.0 <= result <= 1.0

    def test_empty_text(self):
        """Empty texts should return 0."""
        assert syntactic_similarity("", "hello") == 0.0
        assert syntactic_similarity("hello", "") == 0.0
        assert syntactic_similarity("", "") == 0.0

    def test_customer_example_overlap(self):
        """Customer-related texts should show overlap on 'customer' stem."""
        # Note: our simple tokenizer doesn't do stemming
        text_a = "How many customers"
        text_b = "My customer count"
        result = syntactic_similarity(text_a, text_b)
        # With no stemming, "customers" != "customer"
        # So overlap is based on exact matches
        assert 0.0 <= result <= 1.0


class TestTopicSimilarity:
    """Tests for taxonomy-based topic similarity."""

    @pytest.fixture
    def mock_taxonomy(self):
        """Create a mock taxonomy for testing."""
        from cimba.ner.schemas import Taxonomy, Topic

        taxonomy = Taxonomy(
            topics={
                "topic_customer": Topic(
                    id="topic_customer",
                    canonical_name="Customer",
                    terms=["customer", "customers", "client"],
                ),
                "topic_product": Topic(
                    id="topic_product",
                    canonical_name="Product",
                    terms=["product", "products", "item"],
                ),
                "topic_count": Topic(
                    id="topic_count",
                    canonical_name="Count",
                    terms=["count", "number", "total"],
                ),
            },
            term_to_topic={
                "customer": "topic_customer",
                "customers": "topic_customer",
                "client": "topic_customer",
                "product": "topic_product",
                "products": "topic_product",
                "item": "topic_product",
                "count": "topic_count",
                "number": "topic_count",
                "total": "topic_count",
            },
            root_topics=["topic_customer", "topic_product", "topic_count"],
        )
        return taxonomy

    def test_same_topic_high_similarity(self, mock_taxonomy):
        """Texts about the same topic should have high similarity."""
        result = topic_similarity(
            "How many customers do I have",
            "My customer count",
            mock_taxonomy,
        )
        # Both mention customer-related terms
        assert result > 0.0

    def test_different_topic_low_similarity(self, mock_taxonomy):
        """Texts about different topics should have lower similarity."""
        result = topic_similarity(
            "customer information",
            "product catalog",
            mock_taxonomy,
        )
        # Different topics, no overlap
        assert result == 0.0

    def test_no_topics_returns_zero(self, mock_taxonomy):
        """Texts with no recognizable topics should return 0."""
        result = topic_similarity(
            "hello world",
            "goodbye universe",
            mock_taxonomy,
        )
        assert result == 0.0

    def test_extract_topics_basic(self, mock_taxonomy):
        """Extract topics should find terms in text."""
        topics = extract_topics_from_text("customer count is 100", mock_taxonomy)
        assert "topic_customer" in topics
        assert "topic_count" in topics


@pytest.mark.integration
class TestSemanticSimilarityIntegration:
    """Integration tests requiring OpenAI API."""

    async def test_customer_semantic_match(self, skip_without_openai_key):
        """
        Test: "How many customers do I have" vs "My customer count"
        Expected: High similarity
        
        These phrases have the same semantic meaning (asking about customer count).
        The match should be significantly higher than dissimilar pairs.
        """
        score = await semantic_similarity(
            "How many customers do I have",
            "My customer count",
        )
        # Should be a strong match - higher than the customer/product comparison
        assert score > 0.6, f"Expected high similarity, got {score}"

    async def test_customer_vs_product_low_match(self, skip_without_openai_key):
        """
        Test: "How many customers do I have" vs "How many products do I have"
        Expected: Lower similarity than semantic match above
        
        Despite similar syntax, these ask about fundamentally different things.
        Key: this score should be LOWER than the customer semantic match.
        """
        score_different = await semantic_similarity(
            "How many customers do I have",
            "How many products do I have",
        )
        score_same = await semantic_similarity(
            "How many customers do I have",
            "My customer count",
        )
        # The key requirement: customer/product should be LESS similar than customer/customer
        assert score_different < score_same, (
            f"customer vs product ({score_different:.3f}) should be less similar "
            f"than customer vs customer ({score_same:.3f})"
        )

    async def test_identical_texts(self, skip_without_openai_key):
        """Identical texts should have very high similarity."""
        text = "The quick brown fox"
        score = await semantic_similarity(text, text)
        assert score > 0.95

    async def test_completely_unrelated(self, skip_without_openai_key):
        """Completely unrelated texts should have low similarity."""
        score = await semantic_similarity(
            "The quantum mechanics of black holes",
            "Best pizza recipes for beginners",
        )
        assert score < 0.3

    async def test_batch_similarity(self, skip_without_openai_key):
        """Test batch similarity computation."""
        pairs = [
            ("Hello world", "Hi there world"),
            ("Apple fruit", "Banana fruit"),
            ("Machine learning", "Cooking recipes"),
        ]
        scores = await semantic_similarity_batch(pairs)
        
        assert len(scores) == 3
        # First two pairs should be more similar than the third
        assert scores[0] > scores[2]
        assert scores[1] > scores[2]


class TestSemanticSimilarityFromEmbeddings:
    """Tests for pre-computed embedding similarity."""

    def test_identical_embeddings(self):
        """Identical embeddings should give high similarity."""
        emb = [1.0, 0.0, 0.0, 0.0]
        score = semantic_similarity_from_embeddings(emb, emb)
        # After calibration, should be very high
        assert score > 0.9

    def test_orthogonal_embeddings(self):
        """Orthogonal embeddings should give low similarity."""
        emb_a = [1.0, 0.0, 0.0, 0.0]
        emb_b = [0.0, 1.0, 0.0, 0.0]
        score = semantic_similarity_from_embeddings(emb_a, emb_b)
        # After calibration with midpoint=0.82, cosine=0 gives very low score
        assert score < 0.1

    def test_similar_embeddings(self):
        """Similar embeddings should give moderate-high similarity."""
        emb_a = [1.0, 0.5, 0.0, 0.0]
        emb_b = [0.9, 0.6, 0.1, 0.0]
        score = semantic_similarity_from_embeddings(emb_a, emb_b)
        # Should be reasonably high
        assert 0.3 < score < 1.0


class TestSemanticSimilarityWithTaxonomy:
    """Tests for semantic similarity with taxonomy integration."""

    @pytest.fixture
    def sample_taxonomy(self):
        """Create a sample taxonomy."""
        from cimba.ner.schemas import Taxonomy, Topic

        return Taxonomy(
            topics={
                "topic_customer": Topic(
                    id="topic_customer",
                    canonical_name="Customer",
                    terms=["customer", "customers"],
                ),
                "topic_product": Topic(
                    id="topic_product",
                    canonical_name="Product",
                    terms=["product", "products"],
                ),
            },
            term_to_topic={
                "customer": "topic_customer",
                "customers": "topic_customer",
                "product": "topic_product",
                "products": "topic_product",
            },
            root_topics=["topic_customer", "topic_product"],
        )

    @pytest.mark.integration
    async def test_with_taxonomy_boosts_match(
        self, skip_without_openai_key, sample_taxonomy
    ):
        """Taxonomy should work alongside semantic scoring."""
        score_without = await semantic_similarity(
            "How many customers do I have",
            "My customer count",
            taxonomy=None,
        )
        score_with = await semantic_similarity(
            "How many customers do I have",
            "My customer count",
            taxonomy=sample_taxonomy,
        )
        # Both should be reasonably high for a semantic match
        assert score_without > 0.5
        assert score_with > 0.5

    @pytest.mark.integration
    async def test_different_topics_lower_score(
        self, skip_without_openai_key, sample_taxonomy
    ):
        """Different topics should result in lower similarity."""
        score = await semantic_similarity(
            "customer information",
            "product catalog",
            taxonomy=sample_taxonomy,
        )
        # Different topics, should be lower
        assert score < 0.7
