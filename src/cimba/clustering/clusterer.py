"""Main semantic clustering module."""

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Literal

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

from ..embeddings import get_embeddings_batch, get_client as get_embedding_client
from ..openai_client import get_client as get_llm_client
from .index import EmbeddingIndex
from .llm_layers import check_similarity_batch, verify_clusters_batch
from .sanity_checker import SanityChecker, SanityMode
from .schemas import (
    ClusteringResult,
    ClusterVerificationRequest,
    Document,
    KnownTruthPair,
    SimilarityCheckRequest,
)

logger = logging.getLogger(__name__)


class UnionFind:
    """Union-Find data structure for efficient cluster merging."""

    def __init__(self):
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def find(self, x: str) -> str:
        """Find the root of the set containing x with path compression."""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: str, y: str) -> None:
        """Union the sets containing x and y by rank."""
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
        """Get all clusters as a dict mapping root -> members."""
        clusters: dict[str, list[str]] = {}
        for x in self.parent:
            root = self.find(x)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(x)
        return clusters


class SemanticClusterer:
    """
    Semantic clustering using KNN candidate selection and two-layer LLM verification.

    Pipeline:
    1. Generate embeddings for documents missing them
    2. Build FAISS index for KNN search
    3. For each document, find KNN candidates
    4. First-pass LLM: identify similar pairs (with canary validation)
    5. Build clusters using Union-Find
    6. Verification LLM: validate each cluster (with canary validation)
    7. Resolve conflicts and finalize clusters
    """

    def __init__(
        self,
        knn_k: int = 20,
        similarity_threshold: float = 0.7,
        sanity_mode: SanityMode = "strict",
        custom_canaries: list[KnownTruthPair] | None = None,
        batch_size: int = 10,
        min_confidence: float = 0.6,
        llm_model: str = "gpt-4o-mini",
    ):
        """
        Initialize the semantic clusterer.

        Args:
            knn_k: Number of KNN candidates to consider per document
            similarity_threshold: Minimum embedding similarity for KNN candidates
            sanity_mode: How to handle canary check failures ("strict", "warn", "sample")
            custom_canaries: Custom canary pairs for sanity checking
            batch_size: Number of pairs to process per LLM call
            min_confidence: Minimum LLM confidence to accept a similarity match
            llm_model: OpenAI model to use for LLM calls
        """
        self.knn_k = knn_k
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        self.min_confidence = min_confidence
        self.llm_model = llm_model

        # Initialize sanity checker
        self.sanity_checker = SanityChecker(
            mode=sanity_mode,
            canaries=custom_canaries,
        )

        # State
        self._documents: dict[str, Document] = {}
        self._index: EmbeddingIndex | None = None
        self._clusters: dict[str, str] = {}  # doc_id -> cluster_id

    async def cluster(
        self,
        documents: list[dict],
        show_progress: bool = True,
    ) -> ClusteringResult:
        """
        Cluster documents by semantic similarity.

        Args:
            documents: List of document dicts with 'id', 'text', optional 'metadata', 'embedding'
            show_progress: Whether to show progress bars

        Returns:
            ClusteringResult with clustered documents
        """
        # Convert to Document objects
        docs = [Document(**d) for d in documents]
        self._documents = {d.id: d for d in docs}

        logger.info(f"Clustering {len(docs)} documents")

        # Step 1: Generate embeddings for documents missing them
        docs = await self._ensure_embeddings(docs, show_progress)

        # Step 2: Build FAISS index
        logger.info("Building embedding index...")
        self._index = EmbeddingIndex.build_from_documents(docs)

        # Step 3-4: Find KNN candidates and run first-pass LLM
        logger.info("Running first-pass similarity detection...")
        union_find = await self._first_pass_similarity(docs, show_progress)

        # Step 5: Get preliminary clusters
        preliminary_clusters = union_find.get_clusters()
        logger.info(f"Found {len(preliminary_clusters)} preliminary clusters")

        # Step 6: Verification LLM
        logger.info("Running cluster verification...")
        verified_clusters = await self._verify_clusters(preliminary_clusters, show_progress)

        # Step 7: Assign final cluster IDs
        result_docs = self._finalize_clusters(docs, verified_clusters)

        # Build result
        cluster_ids = set(d.cluster_id for d in result_docs if d.cluster_id)
        result = ClusteringResult(
            documents=result_docs,
            cluster_count=len(cluster_ids),
            sanity_check_stats=self.sanity_checker.get_stats(),
        )

        logger.info(f"Clustering complete: {result.cluster_count} clusters from {len(docs)} documents")
        return result

    async def _ensure_embeddings(
        self,
        docs: list[Document],
        show_progress: bool,
    ) -> list[Document]:
        """Generate embeddings for documents that don't have them."""
        docs_needing_embeddings = [d for d in docs if d.embedding is None]

        if not docs_needing_embeddings:
            logger.info("All documents already have embeddings")
            return docs

        logger.info(f"Generating embeddings for {len(docs_needing_embeddings)} documents...")

        texts = [d.text for d in docs_needing_embeddings]
        embeddings = await get_embeddings_batch(texts)

        for doc, embedding in zip(docs_needing_embeddings, embeddings):
            doc.embedding = embedding
            self._documents[doc.id].embedding = embedding

        return docs

    async def _first_pass_similarity(
        self,
        docs: list[Document],
        show_progress: bool,
    ) -> UnionFind:
        """Run first-pass similarity detection using KNN + LLM."""
        union_find = UnionFind()

        # Collect all similarity check requests
        all_requests: list[SimilarityCheckRequest] = []
        seen_pairs: set[tuple[str, str]] = set()

        for doc in docs:
            # Get KNN candidates
            candidates = self._index.search_by_doc_id(
                doc.id,
                k=self.knn_k,
                threshold=self.similarity_threshold,
            )

            for candidate_id, score in candidates:
                # Avoid duplicate pairs (a,b) and (b,a)
                pair_key = tuple(sorted([doc.id, candidate_id]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                candidate_doc = self._documents.get(candidate_id)
                if candidate_doc is None:
                    continue

                all_requests.append(
                    SimilarityCheckRequest(
                        id_a=doc.id,
                        text_a=doc.text,
                        id_b=candidate_id,
                        text_b=candidate_doc.text,
                    )
                )

        logger.info(f"Processing {len(all_requests)} candidate pairs")

        if not all_requests:
            return union_find

        # Process in batches
        client = get_llm_client()

        iterator = range(0, len(all_requests), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="First-pass LLM", unit="batch")

        for i in iterator:
            batch = all_requests[i : i + self.batch_size]

            # Inject canaries
            augmented_batch, canary_positions = self.sanity_checker.inject_canaries(batch)

            # Call LLM
            results = await check_similarity_batch(
                augmented_batch,
                client=client,
                model=self.llm_model,
            )

            # Validate and remove canaries
            filtered_results, _ = self.sanity_checker.validate_canaries(results, canary_positions)

            # Process results
            for result in filtered_results:
                if result.is_similar and result.confidence >= self.min_confidence:
                    union_find.union(result.doc_id_a, result.doc_id_b)

        return union_find

    async def _verify_clusters(
        self,
        clusters: dict[str, list[str]],
        show_progress: bool,
    ) -> dict[str, list[str]]:
        """Verify clusters using second-pass LLM."""
        # Filter to clusters with more than 1 member
        multi_member_clusters = {k: v for k, v in clusters.items() if len(v) > 1}

        if not multi_member_clusters:
            return clusters

        # Build verification requests
        verification_requests: list[ClusterVerificationRequest] = []
        for cluster_id, member_ids in multi_member_clusters.items():
            member_texts = []
            for doc_id in member_ids:
                doc = self._documents.get(doc_id)
                if doc:
                    member_texts.append((doc_id, doc.text))

            verification_requests.append(
                ClusterVerificationRequest(
                    cluster_id=cluster_id,
                    member_texts=member_texts,
                )
            )

        logger.info(f"Verifying {len(verification_requests)} multi-member clusters")

        # Process in batches
        client = get_llm_client()
        verified_clusters = dict(clusters)  # Start with all clusters

        iterator = range(0, len(verification_requests), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Verification LLM", unit="batch")

        for i in iterator:
            batch = verification_requests[i : i + self.batch_size]

            # Call verification LLM
            results = await verify_clusters_batch(
                batch,
                client=client,
                model=self.llm_model,
            )

            # Process verification results
            for result in results:
                if not result.is_valid and result.members_to_remove:
                    # Remove flagged members from cluster
                    cluster_members = verified_clusters.get(result.cluster_id, [])
                    for member_id in result.members_to_remove:
                        if member_id in cluster_members:
                            cluster_members.remove(member_id)
                            # Create singleton cluster for removed member
                            verified_clusters[member_id] = [member_id]

                    if cluster_members:
                        verified_clusters[result.cluster_id] = cluster_members
                    else:
                        del verified_clusters[result.cluster_id]

        return verified_clusters

    def _finalize_clusters(
        self,
        docs: list[Document],
        clusters: dict[str, list[str]],
    ) -> list[Document]:
        """Assign final cluster IDs to documents."""
        # Create mapping from doc_id to cluster_id
        doc_to_cluster: dict[str, str] = {}

        for cluster_root, members in clusters.items():
            # Generate a stable cluster ID
            cluster_id = f"cluster_{uuid.uuid5(uuid.NAMESPACE_DNS, cluster_root)}"
            for member_id in members:
                doc_to_cluster[member_id] = cluster_id

        # Update documents
        result_docs = []
        for doc in docs:
            doc.cluster_id = doc_to_cluster.get(doc.id, f"cluster_{doc.id}")
            result_docs.append(doc)

        return result_docs

    def save_clusters(self, path: str) -> None:
        """
        Save clustering results to a JSON file.

        Args:
            path: Path to output JSON file
        """
        docs_data = [d.model_dump() for d in self._documents.values()]

        output = {
            "documents": docs_data,
            "sanity_check_stats": self.sanity_checker.get_stats(),
        }

        Path(path).write_text(json.dumps(output, indent=2))
        logger.info(f"Saved clustering results to {path}")

    def load_clusters(self, path: str) -> list[Document]:
        """
        Load clustering results from a JSON file.

        Args:
            path: Path to input JSON file

        Returns:
            List of Document objects with cluster assignments
        """
        data = json.loads(Path(path).read_text())

        docs = [Document(**d) for d in data["documents"]]
        self._documents = {d.id: d for d in docs}

        logger.info(f"Loaded {len(docs)} documents from {path}")
        return docs

    @classmethod
    def from_config(
        cls,
        config: dict,
    ) -> "SemanticClusterer":
        """
        Create a SemanticClusterer from a configuration dict.

        Args:
            config: Configuration dictionary

        Returns:
            Configured SemanticClusterer instance
        """
        return cls(
            knn_k=config.get("knn_k", 20),
            similarity_threshold=config.get("similarity_threshold", 0.7),
            sanity_mode=config.get("sanity_mode", "strict"),
            batch_size=config.get("batch_size", 10),
            min_confidence=config.get("min_confidence", 0.6),
            llm_model=config.get("llm_model", "gpt-4o-mini"),
        )
