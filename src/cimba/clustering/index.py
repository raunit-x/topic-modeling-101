"""FAISS index wrapper for efficient KNN search."""

import numpy as np
import faiss

from .schemas import Document


class EmbeddingIndex:
    """Wrapper around FAISS for KNN similarity search on document embeddings."""

    def __init__(self, dimension: int | None = None):
        """
        Initialize the embedding index.

        Args:
            dimension: Embedding dimension. If None, will be set on first add.
        """
        self._dimension = dimension
        self._index: faiss.IndexFlatIP | None = None
        self._doc_ids: list[str] = []
        self._id_to_idx: dict[str, int] = {}

    @property
    def dimension(self) -> int | None:
        """Get the embedding dimension."""
        return self._dimension

    @property
    def size(self) -> int:
        """Get the number of documents in the index."""
        return len(self._doc_ids)

    def _ensure_index(self, dimension: int) -> None:
        """Create the FAISS index if it doesn't exist."""
        if self._index is None:
            self._dimension = dimension
            # Using IndexFlatIP for inner product (cosine similarity with normalized vectors)
            self._index = faiss.IndexFlatIP(dimension)

    def add(self, doc_id: str, embedding: list[float]) -> None:
        """
        Add a single document embedding to the index.

        Args:
            doc_id: Unique document identifier
            embedding: Document embedding vector
        """
        if doc_id in self._id_to_idx:
            raise ValueError(f"Document {doc_id} already exists in index")

        embedding_np = np.array([embedding], dtype=np.float32)
        # Normalize for cosine similarity
        faiss.normalize_L2(embedding_np)

        self._ensure_index(len(embedding))
        self._index.add(embedding_np)
        self._id_to_idx[doc_id] = len(self._doc_ids)
        self._doc_ids.append(doc_id)

    def add_batch(self, doc_ids: list[str], embeddings: list[list[float]]) -> None:
        """
        Add multiple document embeddings to the index.

        Args:
            doc_ids: List of unique document identifiers
            embeddings: List of embedding vectors
        """
        if not doc_ids or not embeddings:
            return

        if len(doc_ids) != len(embeddings):
            raise ValueError("doc_ids and embeddings must have the same length")

        # Check for duplicates
        for doc_id in doc_ids:
            if doc_id in self._id_to_idx:
                raise ValueError(f"Document {doc_id} already exists in index")

        embeddings_np = np.array(embeddings, dtype=np.float32)
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings_np)

        self._ensure_index(embeddings_np.shape[1])
        start_idx = len(self._doc_ids)
        self._index.add(embeddings_np)

        for i, doc_id in enumerate(doc_ids):
            self._id_to_idx[doc_id] = start_idx + i
            self._doc_ids.append(doc_id)

    def search(
        self,
        query_embedding: list[float],
        k: int = 10,
        threshold: float | None = None,
        exclude_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """
        Search for the k nearest neighbors of a query embedding.

        Args:
            query_embedding: Query embedding vector
            k: Number of neighbors to return
            threshold: Minimum similarity score (0-1 for normalized vectors)
            exclude_ids: Set of document IDs to exclude from results

        Returns:
            List of (doc_id, similarity_score) tuples, sorted by score descending
        """
        if self._index is None or self.size == 0:
            return []

        query_np = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_np)

        # Search for more than k if we need to filter
        search_k = min(k + len(exclude_ids or []) + 10, self.size)
        scores, indices = self._index.search(query_np, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for missing results
                continue

            doc_id = self._doc_ids[idx]

            if exclude_ids and doc_id in exclude_ids:
                continue

            if threshold is not None and score < threshold:
                continue

            results.append((doc_id, float(score)))

            if len(results) >= k:
                break

        return results

    def search_by_doc_id(
        self,
        doc_id: str,
        k: int = 10,
        threshold: float | None = None,
    ) -> list[tuple[str, float]]:
        """
        Search for neighbors of a document already in the index.

        Args:
            doc_id: ID of the document to find neighbors for
            k: Number of neighbors to return
            threshold: Minimum similarity score

        Returns:
            List of (doc_id, similarity_score) tuples, excluding the query doc
        """
        if doc_id not in self._id_to_idx:
            raise ValueError(f"Document {doc_id} not found in index")

        if self._index is None:
            return []

        idx = self._id_to_idx[doc_id]
        # Reconstruct the embedding from the index
        embedding = self._index.reconstruct(idx)

        return self.search(
            query_embedding=embedding.tolist(),
            k=k,
            threshold=threshold,
            exclude_ids={doc_id},
        )

    @classmethod
    def build_from_documents(cls, documents: list[Document]) -> "EmbeddingIndex":
        """
        Build an index from a list of documents with embeddings.

        Args:
            documents: List of Document objects with embeddings

        Returns:
            Populated EmbeddingIndex
        """
        index = cls()

        docs_with_embeddings = [d for d in documents if d.embedding is not None]
        if not docs_with_embeddings:
            return index

        doc_ids = [d.id for d in docs_with_embeddings]
        embeddings = [d.embedding for d in docs_with_embeddings]

        index.add_batch(doc_ids, embeddings)
        return index

    def get_embedding(self, doc_id: str) -> list[float] | None:
        """
        Retrieve the embedding for a document.

        Args:
            doc_id: Document ID

        Returns:
            Embedding vector or None if not found
        """
        if doc_id not in self._id_to_idx or self._index is None:
            return None

        idx = self._id_to_idx[doc_id]
        return self._index.reconstruct(idx).tolist()

    def clear(self) -> None:
        """Clear all documents from the index."""
        self._index = None
        self._doc_ids.clear()
        self._id_to_idx.clear()
