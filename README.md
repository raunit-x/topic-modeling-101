A production-ready toolkit for semantic analysis, including **statement clustering** and **topic taxonomy extraction**. Both modules leverage embeddings and LLMs to deliver high-quality results with built-in quality assurance.

## Features

| Module | Description |
|--------|-------------|
| **Semantic Clustering** | Groups statements with the same meaning (e.g., "how many customers" = "customer count") |
| **Topic Taxonomy** | Extracts and organizes entities from text into a hierarchical taxonomy |

---

## Installation

```bash
uv sync --all-extras
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key"
```

Or add it to a `.env` file:

```
OPENAI_API_KEY=your-api-key
```

---

# Semantic Statement Clustering

Groups statements by semantic similarity. Statements that mean the same thing are grouped into the same cluster.

## Pipeline Overview

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

1. **Embedding Generation** - Convert text to vectors using OpenAI embeddings
2. **KNN Candidate Selection** - Use FAISS to find similar statement candidates
3. **First-Pass LLM** - Verify semantic equivalence of candidate pairs
4. **Cluster Verification LLM** - Validate proposed clusters and remove outliers
5. **Sanity Checking** - Inject known truth pairs (canaries) to validate LLM behavior

## Quick Start

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

### Run the Example

```bash
uv run python clustering.py
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `knn_k` | 20 | Number of KNN candidates to consider per document |
| `similarity_threshold` | 0.7 | Minimum embedding similarity for KNN candidates |
| `sanity_mode` | "strict" | How to handle canary failures: `"strict"`, `"warn"`, `"sample"` |
| `custom_canaries` | None | Custom known truth pairs for sanity checking |
| `batch_size` | 10 | Number of pairs to process per LLM call |
| `min_confidence` | 0.6 | Minimum LLM confidence to accept a similarity match |
| `llm_model` | "gpt-4o-mini" | OpenAI model for LLM calls |

### Custom Configuration Example

```python
from src.cimba.clustering import SemanticClusterer, KnownTruthPair

clusterer = SemanticClusterer(
    knn_k=10,
    similarity_threshold=0.5,
    sanity_mode="warn",
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

Canaries are known truth pairs injected into LLM requests to validate model behavior.

### Default Canaries

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
| `strict` | Raises `SanityCheckError` on any canary failure |
| `warn` | Logs a warning but continues processing |
| `sample` | Only checks canaries on a random subset of batches |

## Module Structure

```
src/cimba/clustering/
├── __init__.py          # Public API exports
├── schemas.py           # Pydantic models
├── index.py             # EmbeddingIndex - FAISS wrapper
├── llm_layers.py        # LLM functions for similarity/verification
├── sanity_checker.py    # Canary injection and validation
└── clusterer.py         # Main SemanticClusterer class
```

---

# Topic Taxonomy Extraction

Extracts entities from text using TF-IDF and LLM analysis, then clusters them into a hierarchical taxonomy. This is useful for building NER-like systems, creating search taxonomies, or understanding the key concepts in a corpus.

## Pipeline Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Text Corpus    │────▶│    TF-IDF       │────▶│  Top N Terms    │
│   (files)       │     │   Extraction    │     │ (with contexts) │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Taxonomy      │◀────│    Hierarchy    │◀────│  LLM Clustering │
│   (JSON)        │     │    Building     │     │  + Validation   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

1. **TF-IDF Extraction** - Extract top keywords, bigrams, and trigrams with context samples
2. **Entity Validation** - LLM validates which terms are meaningful entities (using context)
3. **Embedding Generation** - Generate embeddings for validated terms
4. **KNN Clustering** - Find candidate term pairs using FAISS similarity search
5. **LLM Similarity Check** - Verify if terms refer to the same concept (using context)
6. **Cluster Naming** - LLM generates canonical names and descriptions for clusters
7. **Hierarchy Building** - Organize clusters into a hierarchical taxonomy

## Quick Start

```python
import asyncio
from src.cimba.ner import TaxonomyBuilder

async def main():
    builder = TaxonomyBuilder(
        top_n_terms=1000,
        min_contexts=2,
    )
    
    # Load corpus
    builder.load_files("./data/", pattern="*.txt")
    
    # Build taxonomy
    taxonomy = await builder.build()
    
    # Save results
    builder.save_taxonomy("taxonomy.json")
    
    print(f"Created {len(taxonomy.topics)} topics")
    for topic_id in taxonomy.root_topics:
        topic = taxonomy.topics[topic_id]
        print(f"  {topic.canonical_name}: {len(topic.terms)} terms")

asyncio.run(main())
```

### Run the Example

```bash
uv run python ner_taxonomy.py
```

## Configuration

### TaxonomyBuilder Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_n_terms` | 5000 | Maximum number of terms to extract from TF-IDF |
| `min_contexts` | 3 | Minimum context samples required per term |
| `max_contexts` | 5 | Maximum context samples to store per term |
| `similarity_threshold` | 0.7 | Minimum embedding similarity for clustering candidates |
| `knn_k` | 20 | Number of KNN neighbors to consider |
| `batch_size` | 10 | Number of items per LLM batch call |
| `llm_concurrency` | 8 | Maximum concurrent LLM API calls |
| `llm_model` | "gpt-4o-mini" | OpenAI model for LLM analysis |

### TFIDFExtractor Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_df` | 2 | Minimum document frequency for a term |
| `max_df` | 0.8 | Maximum document frequency (as ratio) |
| `min_term_length` | 2 | Minimum character length for unigrams |
| `sentence_level` | True | Treat each sentence as a document for TF-IDF |

### TopicClusterer Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `similarity_threshold` | 0.7 | Minimum embedding similarity for candidates |
| `knn_k` | 20 | Number of KNN neighbors |
| `batch_size` | 10 | LLM batch size |
| `llm_concurrency` | 8 | Maximum concurrent LLM calls |

## Context-Aware LLM Analysis

A key feature of the topic taxonomy module is **context-aware LLM analysis**. Instead of just passing terms to the LLM, we include sample sentences showing how each term is used:

```
Term: "dahi"
Contexts:
- "the dahi is fresh and creamy today"
- "we need to order more dahi for the store"
- "customers prefer packaged dahi over loose"

Term: "curd"
Contexts:
- "fresh curd available at dairy counter"
- "curd rice is a popular lunch option"
```

This helps the LLM:
1. Understand the actual meaning in your domain
2. Distinguish between homonyms (same word, different meanings)
3. Identify synonyms that may not be obvious without context

## LLM Concurrency

All LLM calls use controlled concurrency (default: 8 concurrent requests) to:
- Maximize throughput without overwhelming the API
- Prevent rate limiting
- Handle large corpora efficiently

```python
builder = TaxonomyBuilder(
    llm_concurrency=16,  # Increase for faster processing
)
```

## Output Format

The taxonomy is saved as JSON with the following structure:

```json
{
  "topics": {
    "topic_123": {
      "id": "topic_123",
      "canonical_name": "Dairy Products",
      "description": "Milk-based products including curd, yogurt, and cheese",
      "terms": ["dahi", "curd", "yogurt", "paneer"],
      "parent_topic_id": null,
      "children": ["topic_456", "topic_789"],
      "sample_contexts": ["..."]
    }
  },
  "term_to_topic": {
    "dahi": "topic_123",
    "curd": "topic_123"
  },
  "root_topics": ["topic_123", "topic_234"],
  "created_at": "2026-01-17T10:30:00Z",
  "source_stats": {
    "total_documents": 150,
    "total_sentences": 3420,
    "terms_extracted": 5000,
    "valid_entities": 1234,
    "clusters_created": 456
  }
}
```

## Module Structure

```
src/cimba/ner/
├── __init__.py          # Public API exports
├── schemas.py           # Pydantic models
├── prompts.py           # LLM system prompts
├── tfidf_extractor.py   # TF-IDF term extraction with context
├── topic_clusterer.py   # Context-aware LLM clustering
└── taxonomy.py          # TaxonomyBuilder orchestrator
```

### Key Components

**TFIDFExtractor** (`tfidf_extractor.py`)
- Extracts unigrams, bigrams, and trigrams using TF-IDF
- Collects sample context sentences for each term
- Supports sentence-level document splitting

**TopicClusterer** (`topic_clusterer.py`)
- Validates entities using LLM with context samples
- Generates embeddings for validated terms
- Clusters similar terms using KNN + LLM verification
- Names clusters with canonical names and descriptions

**TaxonomyBuilder** (`taxonomy.py`)
- Orchestrates the full pipeline
- Builds hierarchical taxonomy from clusters
- Manages persistence (save/load)

---

## Running Tests

```bash
# Run all unit tests (no API key required)
uv run pytest tests/ -v -m "not integration"

# Run clustering tests
uv run pytest tests/test_clustering.py -v -m "not integration"

# Run NER/taxonomy tests
uv run pytest tests/test_ner.py -v -m "not integration"

# Run integration tests (requires API key)
uv run pytest tests/ -v
```

---

## Performance Considerations

### Semantic Clustering
- **Batch Processing**: LLM calls are batched to reduce API calls
- **KNN Pre-filtering**: FAISS reduces the number of pairs to evaluate
- **Async Operations**: All I/O operations are async for concurrency
- **Embedding Caching**: Pre-computed embeddings are reused if provided

### Topic Taxonomy
- **Sentence-Level TF-IDF**: More accurate term frequency calculation
- **Context Sampling**: Limits stored contexts to reduce memory usage
- **Concurrent LLM Calls**: Configurable concurrency (default: 8)
- **Union-Find Clustering**: Efficient cluster merging algorithm

---

## Troubleshooting

### ModuleNotFoundError: No module named 'faiss'

```bash
uv sync --all-extras
```

### OpenAI API Key Error

```bash
export OPENAI_API_KEY="sk-..."
```

### SanityCheckError (Clustering)

If you're getting canary failures in strict mode:
1. Check if the LLM model is returning expected results
2. Switch to `warn` mode during development
3. Review and adjust your custom canaries

### No Terms Extracted (Taxonomy)

If TF-IDF returns zero terms:
1. Ensure `sentence_level=True` (default) for proper document splitting
2. Lower `min_df` if you have few documents
3. Check that files contain enough text content

### Rate Limiting

If you're hitting OpenAI rate limits:
1. Reduce `llm_concurrency` (e.g., to 4)
2. Increase `batch_size` to send more items per request
3. Add delays between major pipeline stages
