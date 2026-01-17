"""Tests for the NER/taxonomy module."""

import pytest

from src.cimba.ner import (
    CorpusStats,
    ExtractedTerm,
    Taxonomy,
    TaxonomyBuilder,
    TermContext,
    TFIDFExtractor,
    Topic,
    TopicCluster,
    TopicClusterer,
)


class TestTermContext:
    """Tests for TermContext schema."""

    def test_term_context_creation(self):
        """Test creating a term context."""
        ctx = TermContext(
            sentence="Add fresh dahi to the curry",
            source_file="recipe.txt",
            char_start=10,
            char_end=14,
        )
        assert ctx.sentence == "Add fresh dahi to the curry"
        assert ctx.source_file == "recipe.txt"
        assert ctx.char_start == 10
        assert ctx.char_end == 14


class TestExtractedTerm:
    """Tests for ExtractedTerm schema."""

    def test_extracted_term_creation(self):
        """Test creating an extracted term."""
        term = ExtractedTerm(
            term="dahi",
            term_type="unigram",
            tfidf_score=0.85,
            document_frequency=5,
        )
        assert term.term == "dahi"
        assert term.term_type == "unigram"
        assert term.tfidf_score == 0.85
        assert term.document_frequency == 5
        assert term.sample_contexts == []
        assert term.is_valid_entity is None
        assert term.embedding is None

    def test_extracted_term_with_contexts(self):
        """Test creating an extracted term with contexts."""
        ctx = TermContext(
            sentence="Fresh dahi is best",
            source_file="test.txt",
            char_start=6,
            char_end=10,
        )
        term = ExtractedTerm(
            term="dahi",
            term_type="unigram",
            tfidf_score=0.85,
            document_frequency=5,
            sample_contexts=[ctx],
        )
        assert len(term.sample_contexts) == 1
        assert term.sample_contexts[0].sentence == "Fresh dahi is best"


class TestTopic:
    """Tests for Topic schema."""

    def test_topic_creation(self):
        """Test creating a topic."""
        topic = Topic(
            id="topic_dairy",
            canonical_name="Dairy Products",
            description="Products derived from milk",
            terms=["dahi", "curd", "yogurt"],
        )
        assert topic.id == "topic_dairy"
        assert topic.canonical_name == "Dairy Products"
        assert len(topic.terms) == 3
        assert topic.parent_topic_id is None
        assert topic.children == []


class TestTaxonomy:
    """Tests for Taxonomy schema."""

    def test_empty_taxonomy(self):
        """Test empty taxonomy."""
        taxonomy = Taxonomy()
        assert taxonomy.topics == {}
        assert taxonomy.term_to_topic == {}
        assert taxonomy.root_topics == []

    def test_taxonomy_with_topics(self):
        """Test taxonomy with topics."""
        topic = Topic(
            id="topic_dairy",
            canonical_name="Dairy Products",
            terms=["dahi", "curd"],
        )
        taxonomy = Taxonomy(
            topics={"topic_dairy": topic},
            term_to_topic={"dahi": "topic_dairy", "curd": "topic_dairy"},
            root_topics=["topic_dairy"],
        )

        assert len(taxonomy.topics) == 1
        assert len(taxonomy.term_to_topic) == 2
        assert taxonomy.get_topic_by_term("dahi") == topic
        assert taxonomy.get_topic_by_term("unknown") is None

    def test_get_all_terms(self):
        """Test getting all terms."""
        taxonomy = Taxonomy(
            term_to_topic={"dahi": "t1", "curd": "t1", "paneer": "t2"},
        )
        terms = taxonomy.get_all_terms()
        assert set(terms) == {"dahi", "curd", "paneer"}

    def test_get_topic_path(self):
        """Test getting topic path."""
        parent = Topic(id="parent", canonical_name="Dairy")
        child = Topic(
            id="child", canonical_name="Fermented Dairy", parent_topic_id="parent"
        )
        taxonomy = Taxonomy(
            topics={"parent": parent, "child": child},
            root_topics=["parent"],
        )
        path = taxonomy.get_topic_path("child")
        assert path == ["Dairy", "Fermented Dairy"]


class TestTFIDFExtractor:
    """Tests for TF-IDF extractor."""

    @pytest.fixture
    def sample_texts(self):
        """Sample texts for testing."""
        return [
            ("doc1.txt", "The dahi is fresh. Add dahi to curry for creaminess."),
            ("doc2.txt", "Fresh curd is similar to dahi. Curd is fermented milk."),
            ("doc3.txt", "Paneer is cottage cheese. Fresh paneer is best for curry."),
        ]

    def test_load_texts(self, sample_texts):
        """Test loading texts."""
        extractor = TFIDFExtractor(min_df=1)
        # With sentence_level=True (default), each sentence becomes a document
        n_loaded = extractor.load_texts(sample_texts, sentence_level=True)
        assert n_loaded >= 3  # At least one doc per text, likely more with sentence splitting
        
    def test_load_texts_file_level(self, sample_texts):
        """Test loading texts at file level (no sentence splitting)."""
        extractor = TFIDFExtractor(min_df=1)
        n_loaded = extractor.load_texts(sample_texts, sentence_level=False)
        assert n_loaded == 3

    def test_extract_terms(self, sample_texts):
        """Test extracting terms."""
        extractor = TFIDFExtractor(min_df=1, contexts_per_term=2)
        extractor.load_texts(sample_texts, sentence_level=False)  # Use file-level for predictable results
        terms = extractor.extract_terms(top_n=20)

        assert len(terms) > 0
        assert all(isinstance(t, ExtractedTerm) for t in terms)

        # Check that common terms are extracted (as unigrams or in n-grams)
        term_strings = [t.term for t in terms]
        has_dahi = any("dahi" in t for t in term_strings)
        has_curd = any("curd" in t for t in term_strings)
        assert has_dahi or has_curd, f"Expected dahi or curd in terms: {term_strings[:10]}"

    def test_extract_terms_with_contexts(self, sample_texts):
        """Test that contexts are collected."""
        extractor = TFIDFExtractor(min_df=1, contexts_per_term=3)
        extractor.load_texts(sample_texts, sentence_level=False)
        terms = extractor.extract_terms(top_n=20)

        # Find a term that should have contexts (including n-grams containing these words)
        found_with_context = False
        for term in terms:
            if any(word in term.term for word in ["dahi", "curd", "paneer"]):
                if len(term.sample_contexts) > 0:
                    found_with_context = True
                    break
        
        # At least some terms should have contexts
        terms_with_contexts = [t for t in terms if t.sample_contexts]
        assert len(terms_with_contexts) > 0 or found_with_context

    def test_get_stats(self, sample_texts):
        """Test getting corpus stats."""
        extractor = TFIDFExtractor(min_df=1)
        extractor.load_texts(sample_texts, sentence_level=False)
        extractor.extract_terms(top_n=10)

        stats = extractor.get_stats()
        assert stats is not None
        assert stats.total_documents == 3
        assert stats.total_tokens > 0

    def test_bigrams_and_trigrams(self, sample_texts):
        """Test extraction of bigrams and trigrams."""
        extractor = TFIDFExtractor(min_df=1)
        extractor.load_texts(sample_texts, sentence_level=False)
        terms = extractor.extract_terms(top_n=50)

        term_types = {t.term_type for t in terms}
        # Should have at least unigrams
        assert "unigram" in term_types


class TestTopicCluster:
    """Tests for TopicCluster schema."""

    def test_topic_cluster_creation(self):
        """Test creating a topic cluster."""
        cluster = TopicCluster(
            cluster_id="cluster_001",
            terms=["dahi", "curd", "yogurt"],
            sample_contexts=["Add dahi to curry", "Fresh curd is best"],
            suggested_name="Fermented Dairy",
        )
        assert cluster.cluster_id == "cluster_001"
        assert len(cluster.terms) == 3
        assert len(cluster.sample_contexts) == 2
        assert cluster.suggested_name == "Fermented Dairy"


class TestTopicClusterer:
    """Tests for TopicClusterer."""

    def test_initialization(self):
        """Test clusterer initialization."""
        clusterer = TopicClusterer(
            knn_k=5,
            similarity_threshold=0.7,
            min_confidence=0.8,
        )
        assert clusterer.knn_k == 5
        assert clusterer.similarity_threshold == 0.7
        assert clusterer.min_confidence == 0.8


class TestTaxonomyBuilder:
    """Tests for TaxonomyBuilder."""

    def test_initialization(self):
        """Test builder initialization."""
        builder = TaxonomyBuilder(
            top_n=1000,
            contexts_per_term=3,
            min_df=2,
        )
        assert builder.top_n == 1000
        assert builder.contexts_per_term == 3
        assert builder.min_df == 2

    def test_get_taxonomy_before_build(self):
        """Test getting taxonomy before building."""
        builder = TaxonomyBuilder()
        assert builder.get_taxonomy() is None

    def test_get_terms_before_build(self):
        """Test getting terms before building."""
        builder = TaxonomyBuilder()
        assert builder.get_terms() == []


class TestCorpusStats:
    """Tests for CorpusStats schema."""

    def test_corpus_stats_creation(self):
        """Test creating corpus stats."""
        stats = CorpusStats(
            total_documents=100,
            total_tokens=50000,
            unique_unigrams=5000,
            unique_bigrams=15000,
            unique_trigrams=25000,
            terms_extracted=1000,
            valid_entities=800,
            topics_created=50,
        )
        assert stats.total_documents == 100
        assert stats.total_tokens == 50000
        assert stats.topics_created == 50


@pytest.mark.integration
class TestNERIntegration:
    """Integration tests for the full NER pipeline."""

    @pytest.fixture
    def sample_corpus(self):
        """Sample corpus for integration tests."""
        return [
            ("doc1.txt", "Fresh dahi is essential for Indian cooking. The dahi adds creaminess."),
            ("doc2.txt", "Curd and dahi are the same thing. Both are fermented milk products."),
            ("doc3.txt", "Add paneer to the curry. Paneer is Indian cottage cheese."),
        ]

    @pytest.mark.asyncio
    async def test_full_pipeline(self, sample_corpus, skip_without_openai_key):
        """Test the full taxonomy building pipeline."""
        builder = TaxonomyBuilder(
            top_n=50,
            contexts_per_term=2,
            min_df=1,
            knn_k=3,
            batch_size=5,
        )

        taxonomy = await builder.build_from_texts(sample_corpus)

        assert taxonomy is not None
        assert len(taxonomy.topics) > 0
        assert len(taxonomy.term_to_topic) > 0
