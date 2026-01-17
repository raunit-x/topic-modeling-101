"""Taxonomy builder and manager for hierarchical topic organization."""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm

from ..embeddings import get_embeddings_batch
from ..openai_client import chat_to_schema, get_client as get_llm_client
from .prompts import HIERARCHY_PROMPT
from .schemas import (
    CorpusStats,
    ExtractedTerm,
    Taxonomy,
    Topic,
    TopicCluster,
    TopicHierarchyItem,
    TopicHierarchyResponse,
)
from .tfidf_extractor import TFIDFExtractor
from .topic_clusterer import TopicClusterer

logger = logging.getLogger(__name__)


class _HierarchyLLMResponse(BaseModel):
    """LLM response for hierarchy building."""

    topics: list[TopicHierarchyItem] = Field(..., description="Root-level topics")


class TaxonomyBuilder:
    """
    Build a hierarchical topic taxonomy from raw text files.

    Complete pipeline:
    1. Extract terms using TF-IDF
    2. Validate terms as entities using LLM
    3. Cluster similar terms
    4. Build hierarchical taxonomy
    5. Save to JSON
    """

    def __init__(
        self,
        top_n: int = 5000,
        contexts_per_term: int = 5,
        min_df: int = 2,
        knn_k: int = 10,
        similarity_threshold: float = 0.6,
        min_confidence: float = 0.7,
        batch_size: int = 10,
        llm_model: str = "gpt-4o-mini",
    ):
        """
        Initialize the taxonomy builder.

        Args:
            top_n: Number of top terms to extract via TF-IDF
            contexts_per_term: Sample contexts per term
            min_df: Minimum document frequency for terms
            knn_k: KNN candidates for clustering
            similarity_threshold: Embedding similarity threshold
            min_confidence: LLM confidence threshold
            batch_size: Batch size for LLM calls
            llm_model: OpenAI model for LLM calls
        """
        self.top_n = top_n
        self.contexts_per_term = contexts_per_term
        self.min_df = min_df
        self.knn_k = knn_k
        self.similarity_threshold = similarity_threshold
        self.min_confidence = min_confidence
        self.batch_size = batch_size
        self.llm_model = llm_model

        # Components
        self._extractor = TFIDFExtractor(
            min_df=min_df,
            contexts_per_term=contexts_per_term,
        )
        self._clusterer = TopicClusterer(
            knn_k=knn_k,
            similarity_threshold=similarity_threshold,
            min_confidence=min_confidence,
            batch_size=batch_size,
            llm_model=llm_model,
        )

        # State
        self._taxonomy: Taxonomy | None = None
        self._terms: list[ExtractedTerm] = []
        self._clusters: list[TopicCluster] = []

    async def build_from_files(
        self,
        directory: str,
        pattern: str = "*.txt",
        show_progress: bool = True,
    ) -> Taxonomy:
        """
        Build taxonomy from text files in a directory.

        Args:
            directory: Path to directory with text files
            pattern: Glob pattern for files
            show_progress: Show progress bars

        Returns:
            Complete Taxonomy object
        """
        logger.info(f"{'='*60}")
        logger.info(f"[TAXONOMY] Starting taxonomy build from {directory}")
        logger.info(f"{'='*60}")

        # Step 1: Load and extract terms
        logger.info(f"[TAXONOMY] STEP 1: Loading files and extracting terms")
        n_files = self._extractor.load_files(directory, pattern)
        if n_files == 0:
            raise ValueError(f"No files found in {directory} matching {pattern}")

        self._terms = self._extractor.extract_terms(self.top_n)
        logger.info(f"[TAXONOMY] STEP 1 COMPLETE: Extracted {len(self._terms)} terms from {n_files} files")

        # Step 2: Validate entities
        logger.info(f"[TAXONOMY] STEP 2: Validating entities with LLM")
        logger.info(f"[TAXONOMY] Passing {len(self._terms)} terms to validate_entities()")
        self._terms = await self._clusterer.validate_entities(self._terms, show_progress)
        valid_terms = [t for t in self._terms if t.is_valid_entity]
        logger.info(f"[TAXONOMY] STEP 2 COMPLETE: {len(valid_terms)} valid entities (out of {len(self._terms)} terms)")

        if not valid_terms:
            logger.error("[TAXONOMY] No valid entities found! Check LLM validation.")
            raise ValueError("No valid entities found after LLM validation")

        # Step 3: Generate embeddings
        logger.info(f"[TAXONOMY] STEP 3: Generating embeddings for {len(valid_terms)} valid terms")
        valid_terms = await self._clusterer.generate_embeddings(valid_terms, show_progress)
        terms_with_embeddings = sum(1 for t in valid_terms if t.embedding is not None)
        logger.info(f"[TAXONOMY] STEP 3 COMPLETE: {terms_with_embeddings}/{len(valid_terms)} terms have embeddings")

        if terms_with_embeddings == 0:
            logger.error("[TAXONOMY] No embeddings generated! Check embedding API.")
            raise ValueError("No embeddings generated")

        # Step 4: Cluster terms
        logger.info(f"[TAXONOMY] STEP 4: Clustering {len(valid_terms)} terms")
        self._clusters = await self._clusterer.cluster_terms(valid_terms, show_progress)
        logger.info(f"[TAXONOMY] STEP 4 COMPLETE: Created {len(self._clusters)} clusters")

        # Step 5: Name clusters
        logger.info(f"[TAXONOMY] STEP 5: Naming {len(self._clusters)} clusters")
        self._clusters = await self._clusterer.name_clusters(self._clusters, show_progress)
        logger.info(f"[TAXONOMY] STEP 5 COMPLETE: Named {len(self._clusters)} clusters")

        # Step 6: Build hierarchy
        logger.info(f"[TAXONOMY] STEP 6: Building topic hierarchy")
        self._taxonomy = await self._build_hierarchy(self._clusters, valid_terms, show_progress)
        logger.info(f"[TAXONOMY] STEP 6 COMPLETE: Built hierarchy with {len(self._taxonomy.topics)} topics")

        # Update stats
        stats = self._extractor.get_stats()
        if stats:
            stats.valid_entities = len(valid_terms)
            stats.topics_created = len(self._taxonomy.topics)
            self._taxonomy.source_stats = stats.model_dump()

        logger.info(f"{'='*60}")
        logger.info(
            f"[TAXONOMY] COMPLETE: {len(self._taxonomy.topics)} topics, "
            f"{len(self._taxonomy.term_to_topic)} terms"
        )
        logger.info(f"{'='*60}")

        return self._taxonomy

    async def build_from_texts(
        self,
        texts: list[tuple[str, str]],
        show_progress: bool = True,
    ) -> Taxonomy:
        """
        Build taxonomy from text content directly.

        Args:
            texts: List of (source_name, text_content) tuples
            show_progress: Show progress bars

        Returns:
            Complete Taxonomy object
        """
        logger.info(f"Building taxonomy from {len(texts)} texts...")

        # Step 1: Load and extract terms
        self._extractor.load_texts(texts)
        self._terms = self._extractor.extract_terms(self.top_n)

        # Continue with same pipeline
        self._terms = await self._clusterer.validate_entities(self._terms, show_progress)
        valid_terms = [t for t in self._terms if t.is_valid_entity]
        valid_terms = await self._clusterer.generate_embeddings(valid_terms, show_progress)
        self._clusters = await self._clusterer.cluster_terms(valid_terms, show_progress)
        self._clusters = await self._clusterer.name_clusters(self._clusters, show_progress)
        self._taxonomy = await self._build_hierarchy(self._clusters, valid_terms, show_progress)

        stats = self._extractor.get_stats()
        if stats:
            stats.valid_entities = len(valid_terms)
            stats.topics_created = len(self._taxonomy.topics)
            self._taxonomy.source_stats = stats.model_dump()

        return self._taxonomy

    async def _build_hierarchy(
        self,
        clusters: list[TopicCluster],
        terms: list[ExtractedTerm],
        show_progress: bool,
    ) -> Taxonomy:
        """Build hierarchical taxonomy from flat clusters."""
        logger.info(f"Building hierarchy from {len(clusters)} clusters...")

        term_map = {t.term: t for t in terms}

        # For small number of clusters, just create flat taxonomy
        if len(clusters) <= 5:
            return self._create_flat_taxonomy(clusters, term_map)

        # Use LLM to build hierarchy
        client = get_llm_client()

        # Format clusters for LLM
        clusters_text = "\n\n".join(
            f"Cluster: {c.suggested_name or 'Unnamed'}\n"
            f"Terms: {', '.join(c.terms[:10])}\n"
            f"Contexts: {'; '.join(c.sample_contexts[:2])}"
            for c in clusters[:50]  # Limit to avoid token limits
        )

        user_message = f"""Organize these {len(clusters)} topic clusters into a hierarchy:

{clusters_text}

Create a taxonomy with:
1. Parent topics that group related clusters
2. 2-3 levels maximum
3. Clear names and descriptions"""

        messages = [
            {"role": "system", "content": HIERARCHY_PROMPT},
            {"role": "user", "content": user_message},
        ]

        try:
            response = await chat_to_schema(
                messages=messages,
                schema=_HierarchyLLMResponse,
                model=self.llm_model,
                client=client,
            )

            return self._convert_hierarchy_to_taxonomy(response.topics, clusters, term_map)

        except Exception as e:
            logger.warning(f"Hierarchy building failed: {e}, using flat taxonomy")
            return self._create_flat_taxonomy(clusters, term_map)

    def _create_flat_taxonomy(
        self,
        clusters: list[TopicCluster],
        term_map: dict[str, ExtractedTerm],
    ) -> Taxonomy:
        """Create a flat taxonomy (no hierarchy) from clusters."""
        taxonomy = Taxonomy()

        for cluster in clusters:
            topic_id = f"topic_{uuid.uuid4().hex[:8]}"

            # Calculate average embedding
            embeddings = []
            for term in cluster.terms:
                t = term_map.get(term)
                if t and t.embedding:
                    embeddings.append(t.embedding)

            avg_embedding = []
            if embeddings:
                avg_embedding = np.mean(embeddings, axis=0).tolist()

            topic = Topic(
                id=topic_id,
                canonical_name=cluster.suggested_name or cluster.terms[0].title(),
                description=f"Topic containing: {', '.join(cluster.terms[:5])}",
                terms=cluster.terms,
                parent_topic_id=None,
                children=[],
                embedding=avg_embedding,
                sample_contexts=cluster.sample_contexts[:3],
            )

            taxonomy.topics[topic_id] = topic
            taxonomy.root_topics.append(topic_id)

            for term in cluster.terms:
                taxonomy.term_to_topic[term] = topic_id

        return taxonomy

    def _convert_hierarchy_to_taxonomy(
        self,
        hierarchy: list[TopicHierarchyItem],
        clusters: list[TopicCluster],
        term_map: dict[str, ExtractedTerm],
    ) -> Taxonomy:
        """Convert LLM hierarchy response to Taxonomy object."""
        taxonomy = Taxonomy()

        # Create mapping from cluster name to cluster
        cluster_map = {c.suggested_name: c for c in clusters if c.suggested_name}

        def process_item(
            item: TopicHierarchyItem,
            parent_id: str | None = None,
        ) -> str:
            topic_id = f"topic_{uuid.uuid4().hex[:8]}"

            # Find matching cluster for terms
            matching_cluster = cluster_map.get(item.name)
            terms = item.terms if item.terms else (matching_cluster.terms if matching_cluster else [])
            contexts = matching_cluster.sample_contexts if matching_cluster else []

            # Calculate embedding
            embeddings = []
            for term in terms:
                t = term_map.get(term)
                if t and t.embedding:
                    embeddings.append(t.embedding)

            avg_embedding = []
            if embeddings:
                avg_embedding = np.mean(embeddings, axis=0).tolist()

            # Process children first
            child_ids = []
            for child in item.children:
                child_id = process_item(child, topic_id)
                child_ids.append(child_id)

            topic = Topic(
                id=topic_id,
                canonical_name=item.name,
                description=item.description,
                terms=terms,
                parent_topic_id=parent_id,
                children=child_ids,
                embedding=avg_embedding,
                sample_contexts=contexts[:3],
            )

            taxonomy.topics[topic_id] = topic

            for term in terms:
                taxonomy.term_to_topic[term] = topic_id

            return topic_id

        # Process all root items
        for item in hierarchy:
            topic_id = process_item(item, None)
            taxonomy.root_topics.append(topic_id)

        return taxonomy

    def save_taxonomy(self, path: str) -> None:
        """
        Save taxonomy to JSON file.

        Args:
            path: Output file path
        """
        if self._taxonomy is None:
            raise ValueError("No taxonomy built yet. Call build_from_files() first.")

        output = self._taxonomy.model_dump()
        Path(path).write_text(json.dumps(output, indent=2))
        logger.info(f"Saved taxonomy to {path}")

    def load_taxonomy(self, path: str) -> Taxonomy:
        """
        Load taxonomy from JSON file.

        Args:
            path: Input file path

        Returns:
            Loaded Taxonomy object
        """
        data = json.loads(Path(path).read_text())
        self._taxonomy = Taxonomy.model_validate(data)
        logger.info(
            f"Loaded taxonomy with {len(self._taxonomy.topics)} topics "
            f"and {len(self._taxonomy.term_to_topic)} terms"
        )
        return self._taxonomy

    def get_taxonomy(self) -> Taxonomy | None:
        """Get the current taxonomy."""
        return self._taxonomy

    def get_terms(self) -> list[ExtractedTerm]:
        """Get extracted terms."""
        return self._terms

    def get_clusters(self) -> list[TopicCluster]:
        """Get topic clusters."""
        return self._clusters

    def print_taxonomy(self, max_terms: int = 5) -> None:
        """Print a summary of the taxonomy."""
        if self._taxonomy is None:
            print("No taxonomy built yet.")
            return

        print(f"\n{'='*60}")
        print(f"TAXONOMY SUMMARY")
        print(f"{'='*60}")
        print(f"Total topics: {len(self._taxonomy.topics)}")
        print(f"Total terms: {len(self._taxonomy.term_to_topic)}")
        print(f"Root topics: {len(self._taxonomy.root_topics)}")
        print()

        def print_topic(topic_id: str, indent: int = 0):
            topic = self._taxonomy.topics.get(topic_id)
            if not topic:
                return

            prefix = "  " * indent
            terms_preview = ", ".join(topic.terms[:max_terms])
            if len(topic.terms) > max_terms:
                terms_preview += f" ... (+{len(topic.terms) - max_terms} more)"

            print(f"{prefix}📁 {topic.canonical_name}")
            if topic.terms:
                print(f"{prefix}   Terms: {terms_preview}")
            if topic.description:
                print(f"{prefix}   Desc: {topic.description[:60]}...")

            for child_id in topic.children:
                print_topic(child_id, indent + 1)

        for root_id in self._taxonomy.root_topics:
            print_topic(root_id)
            print()
