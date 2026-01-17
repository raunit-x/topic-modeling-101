"""Context-aware topic clustering using embeddings and LLM."""

import asyncio
import logging
import uuid

from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm

from ..clustering.index import EmbeddingIndex
from ..embeddings import get_embeddings_batch
from ..openai_client import chat_to_schema, get_client as get_llm_client
from .prompts import (
    CLUSTER_NAMING_PROMPT,
    ENTITY_VALIDATION_PROMPT,
    SIMILARITY_CHECK_PROMPT,
)
from .schemas import (
    ExtractedTerm,
    TermSimilarity,
    TermSimilarityResponse,
    TermValidation,
    TermValidationResponse,
    TopicCluster,
)

logger = logging.getLogger(__name__)

# Default concurrency for LLM calls
DEFAULT_LLM_CONCURRENCY = 8


class EntityValidationRequest(BaseModel):
    """Request for entity validation."""

    term: str
    contexts: list[str]


class _ValidationLLMResponse(BaseModel):
    """LLM response for entity validation."""

    validations: list[TermValidation] = Field(..., description="Validation for each term")


class _SimilarityLLMResponse(BaseModel):
    """LLM response for similarity checking."""

    similarities: list[TermSimilarity] = Field(..., description="Similarity for each pair")


class _ClusterNameResponse(BaseModel):
    """LLM response for cluster naming."""

    name: str = Field(..., description="Canonical name for the cluster")
    description: str = Field(..., description="Brief description of the topic")


class TopicClusterer:
    """
    Cluster terms into topics using embeddings and context-aware LLM analysis.

    Pipeline:
    1. Validate terms as meaningful entities (with context)
    2. Generate embeddings for valid terms
    3. Use KNN to find candidate similar terms
    4. LLM verification of similarity (with context)
    5. Build clusters using Union-Find
    6. Name clusters using LLM
    """

    def __init__(
        self,
        knn_k: int = 10,
        similarity_threshold: float = 0.6,
        min_confidence: float = 0.7,
        batch_size: int = 10,
        llm_model: str = "gpt-4o-mini",
        max_concurrency: int = DEFAULT_LLM_CONCURRENCY,
    ):
        """
        Initialize the topic clusterer.

        Args:
            knn_k: Number of KNN candidates per term
            similarity_threshold: Minimum embedding similarity for candidates
            min_confidence: Minimum LLM confidence to accept similarity
            batch_size: Batch size for LLM calls
            llm_model: OpenAI model for LLM calls
            max_concurrency: Maximum concurrent LLM requests (default: 8)
        """
        self.knn_k = knn_k
        self.similarity_threshold = similarity_threshold
        self.min_confidence = min_confidence
        self.batch_size = batch_size
        self.llm_model = llm_model
        self.max_concurrency = max_concurrency

        # State
        self._valid_terms: list[ExtractedTerm] = []
        self._index: EmbeddingIndex | None = None

    async def validate_entities(
        self,
        terms: list[ExtractedTerm],
        show_progress: bool = True,
    ) -> list[ExtractedTerm]:
        """
        Validate which terms are meaningful entities using LLM with context.

        Args:
            terms: Terms to validate
            show_progress: Show progress bar

        Returns:
            Terms with is_valid_entity set
        """
        logger.info(f"[VALIDATE] Starting entity validation for {len(terms)} terms")
        
        if not terms:
            logger.warning("[VALIDATE] No terms to validate!")
            return []

        # Log sample of terms
        sample_terms = [t.term for t in terms[:5]]
        logger.info(f"[VALIDATE] Sample terms: {sample_terms}")
        
        # Count terms with contexts
        terms_with_contexts = sum(1 for t in terms if t.sample_contexts)
        logger.info(f"[VALIDATE] {terms_with_contexts}/{len(terms)} terms have sample contexts")

        client = get_llm_client()

        # Create batches
        batches = []
        for i in range(0, len(terms), self.batch_size):
            batches.append((i, terms[i : i + self.batch_size]))

        n_batches = len(batches)
        logger.info(f"[VALIDATE] Processing {n_batches} batches with concurrency={self.max_concurrency}")

        # Semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def process_batch(batch_idx: int, batch: list[ExtractedTerm]) -> tuple[int, list[ExtractedTerm]]:
            async with semaphore:
                # Build prompt with contexts
                terms_text = "\n\n".join(
                    f"Term {j+1}: \"{t.term}\"\n"
                    f"Type: {t.term_type}\n"
                    f"Sample contexts:\n"
                    + "\n".join(f"  - \"{c.sentence}\"" for c in t.sample_contexts[:3])
                    for j, t in enumerate(batch)
                )

                user_message = f"""Analyze these {len(batch)} terms and determine if each is a meaningful entity.

{terms_text}

For each term, provide:
1. Is it a valid/meaningful entity? (true/false)
2. Brief reasoning
3. Suggested canonical form (if applicable)"""

                messages = [
                    {"role": "system", "content": ENTITY_VALIDATION_PROMPT},
                    {"role": "user", "content": user_message},
                ]

                try:
                    response = await chat_to_schema(
                        messages=messages,
                        schema=_ValidationLLMResponse,
                        model=self.llm_model,
                        client=client,
                    )

                    # Update terms with validation results
                    for term, validation in zip(batch, response.validations):
                        term.is_valid_entity = validation.is_valid_entity

                except Exception as e:
                    logger.warning(f"Validation batch {batch_idx} failed: {e}")
                    # Mark as valid by default on error
                    for term in batch:
                        term.is_valid_entity = True

                return (batch_idx, batch)

        # Create all tasks
        tasks = [process_batch(idx, batch) for idx, batch in batches]

        # Execute with progress bar
        if show_progress:
            results = await tqdm.gather(*tasks, desc="Validating entities", unit="batch")
        else:
            results = await asyncio.gather(*tasks)

        # Sort by batch index and flatten
        results = sorted(results, key=lambda x: x[0])
        validated = []
        for _, batch in results:
            validated.extend(batch)

        self._valid_terms = [t for t in validated if t.is_valid_entity]
        logger.info(
            f"[VALIDATE] Validated {len(self._valid_terms)} terms as entities "
            f"(rejected {len(validated) - len(self._valid_terms)})"
        )

        return validated

    async def generate_embeddings(
        self,
        terms: list[ExtractedTerm] | None = None,
        show_progress: bool = True,
    ) -> list[ExtractedTerm]:
        """
        Generate embeddings for terms.

        Args:
            terms: Terms to embed (uses valid terms if None)
            show_progress: Show progress bar

        Returns:
            Terms with embeddings set
        """
        logger.info(f"[EMBED] Starting embedding generation")
        
        if terms is None:
            logger.info(f"[EMBED] Using internal valid_terms list ({len(self._valid_terms)} terms)")
            terms = self._valid_terms

        if not terms:
            logger.warning("[EMBED] No terms to embed!")
            return []

        logger.info(f"[EMBED] Received {len(terms)} terms for embedding")

        terms_needing_embeddings = [t for t in terms if t.embedding is None]
        if not terms_needing_embeddings:
            logger.info("[EMBED] All terms already have embeddings")
            return terms

        logger.info(f"[EMBED] Generating embeddings for {len(terms_needing_embeddings)} terms...")

        texts = [t.term for t in terms_needing_embeddings]
        logger.info(f"[EMBED] Sample terms to embed: {texts[:5]}")
        
        embeddings = await get_embeddings_batch(texts)
        logger.info(f"[EMBED] Received {len(embeddings)} embeddings")

        for term, embedding in zip(terms_needing_embeddings, embeddings):
            term.embedding = embedding

        terms_with_embeddings = sum(1 for t in terms if t.embedding is not None)
        logger.info(f"[EMBED] Complete: {terms_with_embeddings}/{len(terms)} terms now have embeddings")

        return terms

    def build_index(self, terms: list[ExtractedTerm] | None = None) -> None:
        """
        Build FAISS index for KNN search.

        Args:
            terms: Terms to index (uses valid terms if None)
        """
        logger.info("[INDEX] Starting index build")
        
        if terms is None:
            logger.info(f"[INDEX] Using internal valid_terms list ({len(self._valid_terms)} terms)")
            terms = self._valid_terms

        if not terms:
            logger.error("[INDEX] No terms provided!")
            raise ValueError("No terms to index")

        logger.info(f"[INDEX] Checking {len(terms)} terms for embeddings")
        terms_with_embeddings = [t for t in terms if t.embedding is not None]
        
        if not terms_with_embeddings:
            logger.error(f"[INDEX] None of the {len(terms)} terms have embeddings!")
            # Log some sample terms to debug
            for t in terms[:5]:
                logger.error(f"[INDEX]   Term: '{t.term}', embedding: {t.embedding is not None}")
            raise ValueError("No terms with embeddings to index")

        logger.info(f"[INDEX] Building index with {len(terms_with_embeddings)} terms (out of {len(terms)})")

        self._index = EmbeddingIndex()
        self._index.add_batch(
            [t.term for t in terms_with_embeddings],
            [t.embedding for t in terms_with_embeddings],
        )
        logger.info(f"[INDEX] Index built successfully with {self._index.size} terms")

    async def cluster_terms(
        self,
        terms: list[ExtractedTerm] | None = None,
        show_progress: bool = True,
    ) -> list[TopicCluster]:
        """
        Cluster terms into topics using KNN + LLM similarity checking.

        Args:
            terms: Terms to cluster (uses valid terms if None)
            show_progress: Show progress bar

        Returns:
            List of topic clusters
        """
        if terms is None:
            terms = self._valid_terms

        if self._index is None:
            self.build_index(terms)

        term_map = {t.term: t for t in terms}
        logger.info(f"Clustering {len(terms)} terms...")

        # Collect all similarity pairs to check
        pairs_to_check: list[tuple[str, str, list[str], list[str]]] = []
        seen_pairs: set[tuple[str, str]] = set()

        for term in terms:
            if term.embedding is None:
                continue

            # Get KNN candidates
            candidates = self._index.search(
                term.embedding,
                k=self.knn_k,
                threshold=self.similarity_threshold,
                exclude_ids={term.term},
            )

            for candidate_term, score in candidates:
                pair_key = tuple(sorted([term.term, candidate_term]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                candidate = term_map.get(candidate_term)
                if candidate is None:
                    continue

                # Collect contexts for both terms
                contexts_a = [c.sentence for c in term.sample_contexts[:3]]
                contexts_b = [c.sentence for c in candidate.sample_contexts[:3]]
                pairs_to_check.append((term.term, candidate_term, contexts_a, contexts_b))

        logger.info(f"[CLUSTER] Checking {len(pairs_to_check)} candidate pairs with concurrency={self.max_concurrency}")

        # Check similarity with LLM
        client = get_llm_client()
        semaphore = asyncio.Semaphore(self.max_concurrency)

        # Create batches
        batches = []
        for i in range(0, len(pairs_to_check), self.batch_size):
            batches.append((i, pairs_to_check[i : i + self.batch_size]))

        async def check_similarity_batch(
            batch_idx: int, 
            batch: list[tuple[str, str, list[str], list[str]]]
        ) -> list[tuple[str, str, bool]]:
            """Check similarity for a batch and return matches."""
            async with semaphore:
                # Build prompt
                pairs_text = "\n\n".join(
                    f"Pair {j+1}:\n"
                    f"Term A: \"{term_a}\"\n"
                    f"Contexts for A:\n" + "\n".join(f"  - \"{c}\"" for c in ctx_a[:2]) + "\n"
                    f"Term B: \"{term_b}\"\n"
                    f"Contexts for B:\n" + "\n".join(f"  - \"{c}\"" for c in ctx_b[:2])
                    for j, (term_a, term_b, ctx_a, ctx_b) in enumerate(batch)
                )

                user_message = f"""Analyze these {len(batch)} term pairs.

{pairs_text}

For each pair, determine:
1. Do they refer to the same concept?
2. Confidence (0.0-1.0)
3. Brief reasoning
4. If same concept, what's the best canonical name?"""

                messages = [
                    {"role": "system", "content": SIMILARITY_CHECK_PROMPT},
                    {"role": "user", "content": user_message},
                ]

                matches = []
                try:
                    response = await chat_to_schema(
                        messages=messages,
                        schema=_SimilarityLLMResponse,
                        model=self.llm_model,
                        client=client,
                    )

                    for (term_a, term_b, _, _), sim in zip(batch, response.similarities):
                        if sim.is_same_concept and sim.confidence >= self.min_confidence:
                            matches.append((term_a, term_b, True))

                except Exception as e:
                    logger.warning(f"Similarity batch {batch_idx} failed: {e}")

                return matches

        # Create all tasks
        tasks = [check_similarity_batch(idx, batch) for idx, batch in batches]

        # Execute with progress bar
        if show_progress:
            all_matches = await tqdm.gather(*tasks, desc="Checking similarity", unit="batch")
        else:
            all_matches = await asyncio.gather(*tasks)

        # Build union-find from all matches
        union_find = _UnionFind()
        for matches in all_matches:
            for term_a, term_b, _ in matches:
                union_find.union(term_a, term_b)

        # Build clusters from union-find
        raw_clusters = union_find.get_clusters()
        logger.info(f"Found {len(raw_clusters)} raw clusters")

        # Convert to TopicCluster objects
        clusters = []
        for root, members in raw_clusters.items():
            # Collect sample contexts from all members
            sample_contexts = []
            for member in members:
                term = term_map.get(member)
                if term:
                    sample_contexts.extend(c.sentence for c in term.sample_contexts[:2])

            clusters.append(
                TopicCluster(
                    cluster_id=f"cluster_{uuid.uuid4().hex[:8]}",
                    terms=members,
                    sample_contexts=sample_contexts[:5],
                )
            )

        return clusters

    async def name_clusters(
        self,
        clusters: list[TopicCluster],
        show_progress: bool = True,
    ) -> list[TopicCluster]:
        """
        Generate names for clusters using LLM.

        Args:
            clusters: Clusters to name
            show_progress: Show progress bar

        Returns:
            Clusters with suggested_name set
        """
        logger.info(f"[NAME] Naming {len(clusters)} clusters with concurrency={self.max_concurrency}")

        if not clusters:
            return clusters

        client = get_llm_client()
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def name_single_cluster(idx: int, cluster: TopicCluster) -> tuple[int, TopicCluster]:
            async with semaphore:
                # Build prompt
                terms_list = ", ".join(f'"{t}"' for t in cluster.terms[:10])
                contexts = "\n".join(f"  - \"{c}\"" for c in cluster.sample_contexts[:3])

                user_message = f"""Name this topic cluster:

Terms: {terms_list}

Sample contexts:
{contexts}

Provide a canonical name and brief description."""

                messages = [
                    {"role": "system", "content": CLUSTER_NAMING_PROMPT},
                    {"role": "user", "content": user_message},
                ]

                try:
                    response = await chat_to_schema(
                        messages=messages,
                        schema=_ClusterNameResponse,
                        model=self.llm_model,
                        client=client,
                    )
                    cluster.suggested_name = response.name

                except Exception as e:
                    logger.warning(f"Naming failed for cluster {idx}: {e}")
                    # Use first term as fallback
                    cluster.suggested_name = cluster.terms[0].title() if cluster.terms else "Unknown"

                return (idx, cluster)

        # Create all tasks
        tasks = [name_single_cluster(i, cluster) for i, cluster in enumerate(clusters)]

        # Execute with progress bar
        if show_progress:
            results = await tqdm.gather(*tasks, desc="Naming clusters", unit="cluster")
        else:
            results = await asyncio.gather(*tasks)

        # Sort by index and return
        results = sorted(results, key=lambda x: x[0])
        return [cluster for _, cluster in results]


class _UnionFind:
    """Union-Find data structure for clustering."""

    def __init__(self):
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: str, y: str) -> None:
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

    def get_clusters(self) -> dict[str, list[str]]:
        clusters: dict[str, list[str]] = {}
        for x in self.parent:
            root = self.find(x)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(x)
        return clusters
