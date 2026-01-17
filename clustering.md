# Semantic Statement Clustering

A production-ready module for clustering statements by semantic similarity. Statements that mean the same thing (e.g., "how many customers" and "customer count") are grouped into the same cluster.

## Overview

This module uses a multi-stage pipeline to ensure high-quality clusters:

1. **Embedding Generation** - Convert text to vector representations using OpenAI embeddings
2. **KNN Candidate Selection** - Use FAISS to efficiently find similar statement candidates
3. **First-Pass LLM** - Verify semantic equivalence of candidate pairs
4. **Cluster Verification LLM** - Validate proposed clusters and remove outliers
5. **Sanity Checking** - Inject known truth pairs (canaries) to validate LLM behavior

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Documents     │────▶│   Embeddings    │────▶│   FAISS Index   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Final Clusters │◀────│  Verification   │◀────│  First-Pass LLM │
│     (JSON)      │     │      LLM        │     │  + Canaries     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Installation

Ensure you have the required dependencies:

```bash
uv sync --all-extras
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key"
```

Or add it to a `.env` file in the project root:

```
OPENAI_API_KEY=your-api-key
```

## Quick Start

### Basic Usage

```python
import asyncio
from src.cimba.clustering import SemanticClusterer

documents = [
    {"id": "1", "text": "how many customers do I have"},
    {"id": "2", "text": "customer count"},
    {"id": "3", "text": "total number of customers"},
    {"id": "4", "text": "product inventory details"},
]

async def main():
    clusterer = SemanticClusterer()
    result = await clusterer.cluster(documents)
    
    print(f"Found {result.cluster_count} clusters")
    for doc in result.documents:
        print(f"  [{doc.cluster_id}] {doc.text}")

asyncio.run(main())
```

### Run the Example Script

```bash
uv run python clustering.py
```

## Configuration

### SemanticClusterer Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `knn_k` | 20 | Number of KNN candidates to consider per document |
| `similarity_threshold` | 0.7 | Minimum embedding similarity for KNN candidates |
| `sanity_mode` | "strict" | How to handle canary failures: `"strict"`, `"warn"`, `"sample"` |
| `custom_canaries` | None | Custom known truth pairs for sanity checking |
| `batch_size` | 10 | Number of pairs to process per LLM call |
| `min_confidence` | 0.6 | Minimum LLM confidence to accept a similarity match |
| `llm_model` | "gpt-4o-mini" | OpenAI model for LLM calls |

### Example with Custom Configuration

```python
from src.cimba.clustering import SemanticClusterer, KnownTruthPair

clusterer = SemanticClusterer(
    knn_k=10,
    similarity_threshold=0.5,
    sanity_mode="warn",  # Log warnings instead of failing
    batch_size=5,
    min_confidence=0.7,
)

# Add domain-specific canary pairs
clusterer.sanity_checker.add_canaries([
    KnownTruthPair(
        text_a="order count",
        text_b="number of orders",
        expected_similar=True,
    ),
    KnownTruthPair(
        text_a="revenue",
        text_b="expenses",
        expected_similar=False,
    ),
])
```

## Sanity Checking with Canaries

Canaries are known truth pairs injected into LLM requests to validate model behavior. If the LLM incorrectly classifies a canary, it indicates potential issues.

### Default Canaries

The module includes default canaries:

| Text A | Text B | Expected |
|--------|--------|----------|
| "how many customers do I have" | "customer count" | Similar |
| "total revenue" | "sum of all sales" | Similar |
| "what is the average order value" | "mean order amount" | Similar |
| "customer count" | "product inventory" | NOT Similar |
| "total revenue" | "profit margin" | NOT Similar |

### Sanity Modes

| Mode | Behavior |
|------|----------|
| `strict` | Raises `SanityCheckError` on any canary failure (recommended for production) |
| `warn` | Logs a warning but continues processing (useful during development) |
| `sample` | Only checks canaries on a random subset of batches (cost optimization) |

## Input/Output Format

### Input Documents

Documents should be a list of dictionaries with the following fields:

```python
{
    "id": str,           # Required: unique identifier
    "text": str,         # Required: the statement text
    "metadata": str,     # Optional: additional context
    "embedding": list,   # Optional: pre-computed embedding vector
}
```

### Output

The `cluster()` method returns a `ClusteringResult`:

```python
result = await clusterer.cluster(documents)

# Access clustered documents
for doc in result.documents:
    print(f"{doc.id}: {doc.text} -> {doc.cluster_id}")

# Cluster count
print(f"Total clusters: {result.cluster_count}")

# Sanity check statistics
print(result.sanity_check_stats)
# {'total_checks': 5, 'passed': 5, 'failed': 0, 'skipped': 0, 'pass_rate': 1.0}
```

### Saving and Loading Results

```python
# Save to JSON
clusterer.save_clusters("clusters.json")

# Load from JSON
documents = clusterer.load_clusters("clusters.json")
```

## Running Tests

```bash
# Run unit tests (no API key required)
uv run pytest tests/test_clustering.py -v -m "not integration"

# Run all tests including integration (requires API key)
uv run pytest tests/test_clustering.py -v
```

## Architecture

### Module Structure

```
src/cimba/clustering/
├── __init__.py          # Public API exports
├── schemas.py           # Pydantic models (Document, SimilarityPair, etc.)
├── index.py             # EmbeddingIndex - FAISS wrapper for KNN
├── llm_layers.py        # LLM functions for similarity/verification
├── sanity_checker.py    # Canary injection and validation
└── clusterer.py         # Main SemanticClusterer class
```

### Key Components

**EmbeddingIndex** (`index.py`)
- Wraps FAISS for efficient similarity search
- Normalizes vectors for cosine similarity
- Supports add, search, and batch operations

**LLM Layers** (`llm_layers.py`)
- `check_similarity_batch()` - First-pass pairwise similarity detection
- `verify_clusters_batch()` - Second-pass cluster validation

**SanityChecker** (`sanity_checker.py`)
- Injects canary pairs into LLM requests
- Validates LLM responses against known truths
- Tracks pass/fail statistics

**SemanticClusterer** (`clusterer.py`)
- Orchestrates the full pipeline
- Uses Union-Find for efficient cluster merging
- Handles embedding generation, KNN, LLM calls, and persistence

## Performance Considerations

- **Batch Processing**: LLM calls are batched to reduce API calls
- **KNN Pre-filtering**: FAISS reduces the number of pairs to evaluate
- **Async Operations**: All I/O operations are async for concurrency
- **Embedding Caching**: Pre-computed embeddings are reused if provided

## Troubleshooting

### ModuleNotFoundError: No module named 'faiss'

```bash
uv sync --all-extras
```

### OpenAI API Key Error

Ensure your API key is set:

```bash
export OPENAI_API_KEY="sk-..."
```

### SanityCheckError

If you're getting canary failures in strict mode:

1. Check if the LLM model is returning expected results
2. Switch to `warn` mode during development
3. Review and adjust your custom canaries
