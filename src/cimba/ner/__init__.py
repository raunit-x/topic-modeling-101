"""Topic-based NER module for building taxonomies from text corpora."""

from .schemas import (
    CorpusStats,
    ExtractedTerm,
    Taxonomy,
    TermContext,
    TermSimilarity,
    TermValidation,
    Topic,
    TopicCluster,
    TopicHierarchyItem,
)
from .taxonomy import TaxonomyBuilder
from .tfidf_extractor import TFIDFExtractor
from .topic_clusterer import TopicClusterer

__all__ = [
    # Main classes
    "TaxonomyBuilder",
    "TFIDFExtractor",
    "TopicClusterer",
    # Schemas
    "ExtractedTerm",
    "TermContext",
    "TermValidation",
    "TermSimilarity",
    "Topic",
    "TopicCluster",
    "TopicHierarchyItem",
    "Taxonomy",
    "CorpusStats",
]
