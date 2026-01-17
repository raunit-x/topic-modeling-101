"""Tests for the semantic clustering module."""

import pytest
import numpy as np

from src.cimba.clustering import (
    DEFAULT_CANARIES,
    Document,
    EmbeddingIndex,
    KnownTruthPair,
    SanityCheckError,
    SanityChecker,
    SemanticClusterer,
    SimilarityCheckRequest,
    SimilarityPair,
)


class TestDocument:
    """Tests for the Document schema."""

    def test_document_creation(self):
        """Test creating a document with all fields."""
        doc = Document(
            id="test-1",
            text="how many customers",
            metadata="test metadata",
            embedding=[0.1, 0.2, 0.3],
            cluster_id="cluster-1",
        )
        assert doc.id == "test-1"
        assert doc.text == "how many customers"
        assert doc.metadata == "test metadata"
        assert doc.embedding == [0.1, 0.2, 0.3]
        assert doc.cluster_id == "cluster-1"

    def test_document_minimal(self):
        """Test creating a document with only required fields."""
        doc = Document(id="test-1", text="sample text")
        assert doc.id == "test-1"
        assert doc.text == "sample text"
        assert doc.metadata is None
        assert doc.embedding is None
        assert doc.cluster_id is None


class TestEmbeddingIndex:
    """Tests for the FAISS embedding index wrapper."""

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample normalized embeddings."""
        # Create 5 sample embeddings of dimension 3
        embeddings = np.random.randn(5, 3).astype(np.float32)
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        return embeddings.tolist()

    def test_empty_index(self):
        """Test empty index behavior."""
        index = EmbeddingIndex()
        assert index.size == 0
        assert index.dimension is None
        results = index.search([0.1, 0.2, 0.3], k=5)
        assert results == []

    def test_add_single(self, sample_embeddings):
        """Test adding a single document."""
        index = EmbeddingIndex()
        index.add("doc-1", sample_embeddings[0])
        assert index.size == 1
        assert index.dimension == 3

    def test_add_batch(self, sample_embeddings):
        """Test adding multiple documents at once."""
        index = EmbeddingIndex()
        doc_ids = [f"doc-{i}" for i in range(5)]
        index.add_batch(doc_ids, sample_embeddings)
        assert index.size == 5

    def test_search(self, sample_embeddings):
        """Test searching for similar documents."""
        index = EmbeddingIndex()
        doc_ids = [f"doc-{i}" for i in range(5)]
        index.add_batch(doc_ids, sample_embeddings)

        # Search for the first embedding - should find itself and similar docs
        results = index.search(sample_embeddings[0], k=3)
        assert len(results) <= 3
        assert results[0][0] == "doc-0"  # First result should be exact match
        assert results[0][1] > 0.99  # Very high similarity

    def test_search_with_exclude(self, sample_embeddings):
        """Test searching with excluded IDs."""
        index = EmbeddingIndex()
        doc_ids = [f"doc-{i}" for i in range(5)]
        index.add_batch(doc_ids, sample_embeddings)

        results = index.search(sample_embeddings[0], k=3, exclude_ids={"doc-0"})
        result_ids = [r[0] for r in results]
        assert "doc-0" not in result_ids

    def test_search_with_threshold(self, sample_embeddings):
        """Test searching with similarity threshold."""
        index = EmbeddingIndex()
        doc_ids = [f"doc-{i}" for i in range(5)]
        index.add_batch(doc_ids, sample_embeddings)

        # Very high threshold should return few or no results
        results = index.search(sample_embeddings[0], k=10, threshold=0.999)
        assert len(results) <= 1  # Only exact match or nothing

    def test_search_by_doc_id(self, sample_embeddings):
        """Test searching by document ID."""
        index = EmbeddingIndex()
        doc_ids = [f"doc-{i}" for i in range(5)]
        index.add_batch(doc_ids, sample_embeddings)

        results = index.search_by_doc_id("doc-0", k=3)
        result_ids = [r[0] for r in results]
        assert "doc-0" not in result_ids  # Should exclude self

    def test_build_from_documents(self, sample_embeddings):
        """Test building index from Document objects."""
        docs = [
            Document(id=f"doc-{i}", text=f"text {i}", embedding=emb)
            for i, emb in enumerate(sample_embeddings)
        ]

        index = EmbeddingIndex.build_from_documents(docs)
        assert index.size == 5

    def test_get_embedding(self, sample_embeddings):
        """Test retrieving embeddings."""
        index = EmbeddingIndex()
        index.add("doc-0", sample_embeddings[0])

        embedding = index.get_embedding("doc-0")
        assert embedding is not None
        assert len(embedding) == 3

        # Non-existent doc
        assert index.get_embedding("doc-999") is None

    def test_duplicate_add_raises(self, sample_embeddings):
        """Test that adding duplicate ID raises error."""
        index = EmbeddingIndex()
        index.add("doc-0", sample_embeddings[0])

        with pytest.raises(ValueError, match="already exists"):
            index.add("doc-0", sample_embeddings[1])


class TestSanityChecker:
    """Tests for the sanity checker with canary injection."""

    def test_default_canaries(self):
        """Test that default canaries are loaded."""
        checker = SanityChecker()
        assert len(checker.canaries) == len(DEFAULT_CANARIES)

    def test_custom_canaries(self):
        """Test using custom canaries."""
        custom = [
            KnownTruthPair(text_a="a", text_b="b", expected_similar=True),
        ]
        checker = SanityChecker(canaries=custom)
        assert len(checker.canaries) == 1

    def test_inject_canaries(self):
        """Test injecting canaries into a batch."""
        checker = SanityChecker(canaries_per_batch=1)

        pairs = [
            SimilarityCheckRequest(id_a="1", text_a="text 1", id_b="2", text_b="text 2"),
        ]

        augmented, positions = checker.inject_canaries(pairs)
        assert len(augmented) == 2  # Original + 1 canary
        assert len(positions) == 1

    def test_validate_canaries_pass(self):
        """Test validating canaries that pass."""
        checker = SanityChecker(mode="strict")
        canary = KnownTruthPair(text_a="a", text_b="b", expected_similar=True)
        checker.canaries = [canary]

        # Create a result that matches expectation
        results = [
            SimilarityPair(
                doc_id_a="__canary_0_a__",
                doc_id_b="__canary_0_b__",
                is_similar=True,
                confidence=0.9,
                reasoning="Same",
            ),
        ]
        positions = [(0, canary)]

        filtered, passed = checker.validate_canaries(results, positions)
        assert passed
        assert len(filtered) == 0  # Canary should be filtered out

    def test_validate_canaries_fail_strict(self):
        """Test that strict mode raises on failed canary."""
        checker = SanityChecker(mode="strict")
        canary = KnownTruthPair(text_a="a", text_b="b", expected_similar=True)
        checker.canaries = [canary]

        # Create a result that does NOT match expectation
        results = [
            SimilarityPair(
                doc_id_a="__canary_0_a__",
                doc_id_b="__canary_0_b__",
                is_similar=False,  # Wrong!
                confidence=0.9,
                reasoning="Different",
            ),
        ]
        positions = [(0, canary)]

        with pytest.raises(SanityCheckError):
            checker.validate_canaries(results, positions)

    def test_validate_canaries_fail_warn(self):
        """Test that warn mode logs but doesn't raise."""
        checker = SanityChecker(mode="warn")
        canary = KnownTruthPair(text_a="a", text_b="b", expected_similar=True)
        checker.canaries = [canary]

        results = [
            SimilarityPair(
                doc_id_a="__canary_0_a__",
                doc_id_b="__canary_0_b__",
                is_similar=False,
                confidence=0.9,
                reasoning="Different",
            ),
        ]
        positions = [(0, canary)]

        filtered, passed = checker.validate_canaries(results, positions)
        assert not passed
        assert checker.stats["failed"] == 1

    def test_stats_tracking(self):
        """Test that statistics are tracked correctly."""
        checker = SanityChecker(mode="warn")
        canary = KnownTruthPair(text_a="a", text_b="b", expected_similar=True)
        checker.canaries = [canary]

        # Simulate a pass
        results = [
            SimilarityPair(
                doc_id_a="__canary_0_a__",
                doc_id_b="__canary_0_b__",
                is_similar=True,
                confidence=0.9,
                reasoning="Same",
            ),
        ]
        positions = [(0, canary)]
        checker.validate_canaries(results, positions)

        stats = checker.get_stats()
        assert stats["passed"] == 1
        assert stats["failed"] == 0
        assert stats["pass_rate"] == 1.0


class TestSemanticClusterer:
    """Tests for the main SemanticClusterer class."""

    def test_initialization(self):
        """Test clusterer initialization with default params."""
        clusterer = SemanticClusterer()
        assert clusterer.knn_k == 20
        assert clusterer.similarity_threshold == 0.7
        assert clusterer.batch_size == 10

    def test_initialization_custom(self):
        """Test clusterer initialization with custom params."""
        clusterer = SemanticClusterer(
            knn_k=10,
            similarity_threshold=0.5,
            sanity_mode="warn",
            batch_size=5,
        )
        assert clusterer.knn_k == 10
        assert clusterer.similarity_threshold == 0.5
        assert clusterer.sanity_checker.mode == "warn"
        assert clusterer.batch_size == 5

    def test_from_config(self):
        """Test creating clusterer from config dict."""
        config = {
            "knn_k": 15,
            "similarity_threshold": 0.6,
            "sanity_mode": "sample",
        }
        clusterer = SemanticClusterer.from_config(config)
        assert clusterer.knn_k == 15
        assert clusterer.similarity_threshold == 0.6


class TestUnionFind:
    """Tests for the Union-Find data structure."""

    def test_union_find_basic(self):
        """Test basic union-find operations."""
        from src.cimba.clustering.clusterer import UnionFind

        uf = UnionFind()

        # Initially each element is its own parent
        assert uf.find("a") == "a"
        assert uf.find("b") == "b"

        # Union a and b
        uf.union("a", "b")
        assert uf.find("a") == uf.find("b")

    def test_union_find_clusters(self):
        """Test getting clusters from union-find."""
        from src.cimba.clustering.clusterer import UnionFind

        uf = UnionFind()

        uf.union("a", "b")
        uf.union("b", "c")
        uf.union("d", "e")

        clusters = uf.get_clusters()
        assert len(clusters) == 2

        # a, b, c should be in one cluster
        root_abc = uf.find("a")
        assert set(clusters[root_abc]) == {"a", "b", "c"}

        # d, e should be in another
        root_de = uf.find("d")
        assert set(clusters[root_de]) == {"d", "e"}


@pytest.mark.integration
class TestClusteringIntegration:
    """Integration tests for the full clustering pipeline."""

    @pytest.fixture
    def sample_docs(self):
        """Sample documents for integration tests."""
        return [
            {"id": "1", "text": "how many customers do I have"},
            {"id": "2", "text": "customer count"},
            {"id": "3", "text": "product inventory"},
        ]

    @pytest.mark.asyncio
    async def test_full_clustering_pipeline(self, sample_docs, skip_without_openai_key):
        """Test the full clustering pipeline with real API calls."""
        clusterer = SemanticClusterer(
            knn_k=5,
            similarity_threshold=0.3,
            sanity_mode="warn",
            batch_size=3,
        )

        result = await clusterer.cluster(sample_docs, show_progress=False)

        assert result.cluster_count >= 1
        assert len(result.documents) == 3
        assert all(d.cluster_id is not None for d in result.documents)
        assert all(d.embedding is not None for d in result.documents)
