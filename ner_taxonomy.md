# Topic Taxonomy Builder

A module for building hierarchical topic taxonomies from text corpora. Extracts keywords/bigrams/trigrams using TF-IDF, clusters similar terms using embeddings and context-aware LLM analysis, and outputs a structured taxonomy that can be used for NER-like entity extraction.

## Overview

The taxonomy builder uses a multi-stage pipeline:

1. **TF-IDF Extraction** - Extract top N keywords, bigrams, trigrams with sample contexts
2. **Entity Validation** - LLM validates which terms are meaningful entities
3. **Term Clustering** - Embed terms, use KNN + LLM to cluster similar terms
4. **Hierarchy Building** - LLM organizes clusters into hierarchical taxonomy
5. **Output** - JSON taxonomy with topics, terms, and relationships

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Text Files    │────▶│   TF-IDF        │────▶│  Terms +        │
│   (.txt)        │     │   Extraction    │     │  Contexts       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Taxonomy      │◀────│   Hierarchy     │◀────│   Term          │
│   (JSON)        │     │   Building      │     │   Clustering    │
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

## Quick Start

### Run the Demo

```bash
uv run python ner_taxonomy.py --demo
```

This runs on sample Indian cooking texts and produces a taxonomy of ingredients.

### Build from Your Own Files

```bash
uv run python ner_taxonomy.py /path/to/your/text/files
```

### Programmatic Usage

```python
import asyncio
from src.cimba.ner import TaxonomyBuilder

async def main():
    builder = TaxonomyBuilder(
        top_n=5000,           # Extract top 5000 terms
        contexts_per_term=5,  # Collect 5 sample sentences per term
        min_df=2,             # Term must appear in at least 2 docs
        knn_k=10,             # Consider 10 nearest neighbors
        similarity_threshold=0.6,
        min_confidence=0.7,
    )
    
    # Build from directory
    taxonomy = await builder.build_from_files("./data/corpus/")
    
    # Or from text list
    texts = [
        ("doc1.txt", "Your text content here..."),
        ("doc2.txt", "More text content..."),
    ]
    taxonomy = await builder.build_from_texts(texts)
    
    # Save results
    builder.save_taxonomy("taxonomy.json")
    builder.print_taxonomy()

asyncio.run(main())
```

## Configuration

### TaxonomyBuilder Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_n` | 5000 | Number of top terms to extract via TF-IDF |
| `contexts_per_term` | 5 | Sample sentences to collect per term |
| `min_df` | 2 | Minimum document frequency for a term |
| `knn_k` | 10 | Number of KNN candidates for clustering |
| `similarity_threshold` | 0.6 | Minimum embedding similarity for candidates |
| `min_confidence` | 0.7 | Minimum LLM confidence to accept similarity |
| `batch_size` | 10 | Batch size for LLM calls |
| `llm_model` | "gpt-4o-mini" | OpenAI model for LLM calls |

## How Context-Aware LLM Works

When the LLM evaluates terms, it receives **sample sentences** showing how the term is used:

```
Term: "dahi"
Sample contexts:
  - "Add fresh dahi to the curry for a creamy texture"
  - "The dahi was too sour today"
  - "Mix dahi with cucumber for raita"

Term: "curd"
Sample contexts:
  - "Set the curd overnight for best results"
  - "The curd has thickened nicely"
  - "Add curd to cool down the spice"

Questions:
1. Is "dahi" a meaningful entity?
2. Is "curd" a meaningful entity?
3. Do they refer to the same concept?
4. If yes, what should be the canonical name?
```

This context helps the LLM:
- Verify if the term is a true entity worth extracting
- Understand the semantic domain
- Make better clustering decisions

## Output Format

The taxonomy is saved as JSON:

```json
{
  "topics": {
    "topic_abc123": {
      "id": "topic_abc123",
      "canonical_name": "Fermented Dairy",
      "description": "Cultured dairy products like yogurt and curd",
      "terms": ["dahi", "curd", "yogurt"],
      "parent_topic_id": "topic_dairy",
      "children": [],
      "sample_contexts": [
        "Add fresh dahi to the curry",
        "Set the curd overnight"
      ]
    }
  },
  "term_to_topic": {
    "dahi": "topic_abc123",
    "curd": "topic_abc123",
    "yogurt": "topic_abc123"
  },
  "root_topics": ["topic_dairy", "topic_spices"],
  "source_stats": {
    "total_documents": 100,
    "total_tokens": 50000,
    "terms_extracted": 5000,
    "valid_entities": 3500,
    "topics_created": 150
  }
}
```

## Using the Taxonomy

### Look Up Terms

```python
# Get topic for a term
topic = taxonomy.get_topic_by_term("dahi")
print(topic.canonical_name)  # "Fermented Dairy"

# Get all terms
all_terms = taxonomy.get_all_terms()

# Get topic hierarchy path
path = taxonomy.get_topic_path(topic.id)
print(" > ".join(path))  # "Dairy > Fermented Dairy"
```

### Load Saved Taxonomy

```python
builder = TaxonomyBuilder()
taxonomy = builder.load_taxonomy("taxonomy.json")
```

## Pipeline Components

### TFIDFExtractor

Extracts terms with TF-IDF scoring and collects sample contexts:

```python
from src.cimba.ner import TFIDFExtractor

extractor = TFIDFExtractor(min_df=2, contexts_per_term=5)
extractor.load_files("./data/", pattern="*.txt")
terms = extractor.extract_terms(top_n=5000)

for term in terms[:10]:
    print(f"{term.term} ({term.term_type}): {term.tfidf_score:.3f}")
    for ctx in term.sample_contexts:
        print(f"  - {ctx.sentence}")
```

### TopicClusterer

Clusters terms using embeddings and context-aware LLM:

```python
from src.cimba.ner import TopicClusterer

clusterer = TopicClusterer(knn_k=10, similarity_threshold=0.6)

# Validate which terms are entities
terms = await clusterer.validate_entities(terms)
valid_terms = [t for t in terms if t.is_valid_entity]

# Generate embeddings
valid_terms = await clusterer.generate_embeddings(valid_terms)

# Cluster similar terms
clusters = await clusterer.cluster_terms(valid_terms)

# Name clusters
clusters = await clusterer.name_clusters(clusters)
```

## Running Tests

```bash
# Run unit tests (no API key required)
uv run pytest tests/test_ner.py -v -m "not integration"

# Run all tests including integration (requires API key)
uv run pytest tests/test_ner.py -v
```

## File Structure

```
src/cimba/ner/
├── __init__.py          # Public API exports
├── schemas.py           # Pydantic models (ExtractedTerm, Topic, Taxonomy, etc.)
├── tfidf_extractor.py   # TF-IDF extraction with context collection
├── topic_clusterer.py   # Embedding + LLM clustering
└── taxonomy.py          # TaxonomyBuilder and hierarchy management
```

## Tips for Best Results

1. **Corpus Size**: Works best with 50+ documents for meaningful patterns
2. **Domain Focus**: Better results with domain-specific corpora
3. **min_df**: Increase for larger corpora to filter noise
4. **top_n**: Start with 1000-2000, increase if you need more coverage
5. **contexts_per_term**: 3-5 contexts usually sufficient for LLM understanding

## Troubleshooting

### No terms extracted

- Check that files match the glob pattern
- Lower `min_df` for small corpora
- Ensure files are UTF-8 encoded text

### Too many generic terms

- Increase `min_df` to filter rare terms
- Lower `max_df_ratio` in TFIDFExtractor to filter very common terms
- The entity validation step should filter non-entities

### LLM errors

- Check API key is set correctly
- Lower `batch_size` if hitting rate limits
- Switch to a different `llm_model` if needed
