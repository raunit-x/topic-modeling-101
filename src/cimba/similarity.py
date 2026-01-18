"""
Hybrid semantic similarity module combining semantic, syntactic, and topic signals.

This module provides a `semantic_similarity` function that produces polarized,
human-like similarity scores without requiring model training.

TRADE-OFFS vs PRODUCTION-GRADE SOLUTION:
-----------------------------------------
| This Implementation              | Production-Grade Alternative            |
|----------------------------------|----------------------------------------|
| Heuristic weights (0.6/0.25/0.15)| Learned weights from labeled data      |
| Sigmoid calibration              | Fine-tuned classification head (MLP)   |
| Simple tokenization              | BM25 or learned sparse representations |
| ~50ms latency                    | ColBERT/cross-encoder for higher quality|

For production with labeled training data, consider:
1. Training a lightweight MLP reranker on (query_emb, doc_emb, features) -> score
2. Using contrastive or listwise loss for better ranking
3. Adding temperature scaling for polarized outputs

See: LightweightReranker pattern with learned feature weights.
"""

import math
import re
from typing import TYPE_CHECKING
from rapidfuzz import fuzz

from .embeddings import cosine_similarity, get_embedding, get_embeddings_batch

if TYPE_CHECKING:
    from .ner.schemas import Taxonomy


# =============================================================================
# Default Configuration
# =============================================================================

# Weights for blending signals (must sum to 1.0 when all signals present)
# Semantic is weighted heavily since it captures meaning
DEFAULT_WEIGHT_SEMANTIC = 0.85
DEFAULT_WEIGHT_SYNTACTIC = 0.10
DEFAULT_WEIGHT_TOPIC = 0.05

# Calibration parameters for polarizing scores
# midpoint: raw scores below this map to <0.5, above map to >0.5
# steepness: higher = more polarized (sharper sigmoid)
# Note: OpenAI embeddings for similar texts typically score 0.65-0.85 cosine
# We use midpoint=0.65 to ensure semantically similar texts score > 0.5
DEFAULT_CALIBRATION_MIDPOINT = 0.65
DEFAULT_CALIBRATION_STEEPNESS = 15.0

# Stopwords to exclude from syntactic similarity
STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
    "from", "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "and", "but", "if", "or",
    "because", "until", "while", "although", "though", "over", "up", "down",
    "out", "off", "away", "back", "around", "across", "along", "upon",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
    "that", "these", "those", "am", "about", "against", "any", "both",
    "many", "much", "also", "now", "even", "still", "already", "yet",
})


# =============================================================================
# Core Functions
# =============================================================================


def calibrate_score(
    raw_score: float,
    midpoint: float = DEFAULT_CALIBRATION_MIDPOINT,
    steepness: float = DEFAULT_CALIBRATION_STEEPNESS,
) -> float:
    """
    Apply sigmoid calibration to polarize a raw score.

    Transforms scores using: 1 / (1 + exp(-steepness * (raw - midpoint)))
    
    This creates "human-like" polarized outputs:
    - Scores above midpoint → closer to 1.0
    - Scores below midpoint → closer to 0.0
    - Higher steepness = more extreme polarization

    Args:
        raw_score: Input score (typically 0-1 range)
        midpoint: Center point of the sigmoid (default 0.5)
        steepness: Controls polarization intensity (default 10.0)
            - steepness=5: gentle curve
            - steepness=10: moderate polarization
            - steepness=20: aggressive polarization

    Returns:
        Calibrated score between 0 and 1

    Example:
        >>> calibrate_score(0.7, midpoint=0.5, steepness=10)
        0.88...  # Above midpoint → pushed toward 1
        >>> calibrate_score(0.3, midpoint=0.5, steepness=10)
        0.11...  # Below midpoint → pushed toward 0
    """
    return 1.0 / (1.0 + math.exp(-steepness * (raw_score - midpoint)))


def normalize_tokens(text: str) -> list[str]:
    """
    Tokenize and normalize text for syntactic comparison.

    Performs:
    - Lowercase conversion
    - Punctuation removal
    - Stopword filtering
    - Whitespace normalization

    Args:
        text: Input text to tokenize

    Returns:
        List of normalized tokens (content words only)
    """
    # Lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    
    # Split and filter
    tokens = text.split()
    tokens = [t for t in tokens if t and t not in STOPWORDS and len(t) > 1]
    
    return tokens


def syntactic_similarity(text_a: str, text_b: str) -> float:
    """
    Compute lexical overlap using token partial ratio.
    Useful for catching exact or near-exact matches that embeddings might miss.

    Args:
        text_a: First text
        text_b: Second text

    Returns:
        Jaccard similarity (0-1) of content word sets

    Example:
        >>> syntactic_similarity("How many customers", "customer count")
        0.25  # "customer/customers" overlap
    """
    tokens_a = set(normalize_tokens(text_a))
    tokens_b = set(normalize_tokens(text_b))
    
    return fuzz.partial_ratio(" ".join(tokens_a), " ".join(tokens_b)) / 100


def extract_topics_from_text(text: str, taxonomy: "Taxonomy") -> set[str]:
    """
    Extract topic IDs from text using a taxonomy's term-to-topic mapping.

    Scans text for known terms and returns their associated topic IDs.

    Args:
        text: Text to analyze
        taxonomy: Taxonomy with term_to_topic mapping

    Returns:
        Set of topic IDs found in the text
    """
    text_lower = text.lower()
    found_topics: set[str] = set()
    
    # Check each term in the taxonomy
    for term, topic_id in taxonomy.term_to_topic.items():
        # Simple substring match (could be enhanced with word boundaries)
        if term.lower() in text_lower:
            found_topics.add(topic_id)
    
    return found_topics


def topic_similarity(text_a: str, text_b: str, taxonomy: "Taxonomy") -> float:
    """
    Compute topic overlap using a pre-built taxonomy.

    Extracts topics from both texts and computes Jaccard similarity
    on the topic ID sets.

    Args:
        text_a: First text
        text_b: Second text
        taxonomy: Taxonomy with term_to_topic mapping

    Returns:
        Jaccard similarity (0-1) of topic sets, or 0 if no topics found
    """
    topics_a = extract_topics_from_text(text_a, taxonomy)
    topics_b = extract_topics_from_text(text_b, taxonomy)
    
    if not topics_a or not topics_b:
        return 0.0
    
    intersection = topics_a & topics_b
    union = topics_a | topics_b
    
    return len(intersection) / len(union)


def semantic_similarity_from_embeddings(
    emb_a: list[float],
    emb_b: list[float],
    midpoint: float = 0.82,
    steepness: float = 15.0,
) -> float:
    """
    Compute calibrated semantic similarity from pre-computed embeddings.

    Uses cosine similarity with sigmoid calibration optimized for
    embedding score distributions (which typically cluster in 0.7-0.95 range).

    Args:
        emb_a: First embedding vector
        emb_b: Second embedding vector
        midpoint: Calibration midpoint (default 0.82 for embeddings)
        steepness: Calibration steepness (default 15.0)

    Returns:
        Calibrated similarity score (0-1)
    """
    raw_cosine = cosine_similarity(emb_a, emb_b)
    return calibrate_score(raw_cosine, midpoint=midpoint, steepness=steepness)


async def semantic_similarity(
    text_a: str,
    text_b: str,
    taxonomy: "Taxonomy | None" = None,
    weight_semantic: float = DEFAULT_WEIGHT_SEMANTIC,
    weight_syntactic: float = DEFAULT_WEIGHT_SYNTACTIC,
    weight_topic: float = DEFAULT_WEIGHT_TOPIC,
    calibration_midpoint: float = DEFAULT_CALIBRATION_MIDPOINT,
    calibration_steepness: float = DEFAULT_CALIBRATION_STEEPNESS,
) -> float:
    """
    Compute hybrid semantic similarity between two texts.

    Combines three signals for robust similarity scoring:
    1. **Semantic**: Embedding cosine similarity (captures meaning)
    2. **Syntactic**: Token Jaccard overlap (catches exact matches)
    3. **Topic**: Taxonomy-based topic overlap (domain matching, optional)

    The final score is calibrated with sigmoid to produce polarized outputs
    that feel more "human-like" (strong matches → ~1.0, weak → ~0.0).

    Args:
        text_a: First text to compare
        text_b: Second text to compare
        taxonomy: Optional Taxonomy for topic-based scoring
        weight_semantic: Weight for semantic signal (default 0.6)
        weight_syntactic: Weight for syntactic signal (default 0.25)
        weight_topic: Weight for topic signal (default 0.15)
        calibration_midpoint: Sigmoid midpoint (default 0.5)
        calibration_steepness: Sigmoid steepness (default 10.0)

    Returns:
        Similarity score between 0 and 1

    Example:
        >>> await semantic_similarity(
        ...     "How many customers do I have",
        ...     "My customer count"
        ... )
        0.92  # High score: same meaning, shared "customer" token
        
        >>> await semantic_similarity(
        ...     "How many customers do I have",
        ...     "How many products do I have"
        ... )
        0.35  # Low score: different meaning despite similar syntax

    Note:
        If taxonomy is not provided, its weight is redistributed to
        semantic and syntactic signals proportionally.
    """
    # Compute semantic similarity (embedding-based)
    embeddings = await get_embeddings_batch([text_a, text_b])
    emb_a, emb_b = embeddings[0], embeddings[1]
    raw_semantic = cosine_similarity(emb_a, emb_b)
    if raw_semantic <= 0.65: # openai embeddings typically score 0.7+ for very similar texts
        raw_semantic *= 0.8
    else:
        raw_semantic *= 1.2
    # Compute syntactic similarity (token overlap)
    raw_syntactic = syntactic_similarity(text_a, text_b)
    
    # Compute topic similarity if taxonomy available
    if taxonomy is not None:
        raw_topic = topic_similarity(text_a, text_b, taxonomy)
        # Use provided weights
        w_sem = weight_semantic
        w_syn = weight_syntactic
        w_top = weight_topic
    else:
        raw_topic = 0.0
        # Redistribute topic weight proportionally
        total_non_topic = weight_semantic + weight_syntactic
        if total_non_topic > 0:
            w_sem = weight_semantic / total_non_topic
            w_syn = weight_syntactic / total_non_topic
        else:
            w_sem = 0.5
            w_syn = 0.5
        w_top = 0.0
    
    # Blend signals
    raw_blended = (
        w_sem * raw_semantic +
        w_syn * raw_syntactic +
        w_top * raw_topic
    )
    
    # Apply calibration for polarized output
    return calibrate_score(
        raw_blended,
        midpoint=calibration_midpoint,
        steepness=calibration_steepness,
    )


async def semantic_similarity_batch(
    pairs: list[tuple[str, str]],
    taxonomy: "Taxonomy | None" = None,
    weight_semantic: float = DEFAULT_WEIGHT_SEMANTIC,
    weight_syntactic: float = DEFAULT_WEIGHT_SYNTACTIC,
    weight_topic: float = DEFAULT_WEIGHT_TOPIC,
    calibration_midpoint: float = DEFAULT_CALIBRATION_MIDPOINT,
    calibration_steepness: float = DEFAULT_CALIBRATION_STEEPNESS,
) -> list[float]:
    """
    Compute semantic similarity for multiple text pairs efficiently.

    Batches embedding requests for better performance when comparing
    many pairs.

    Args:
        pairs: List of (text_a, text_b) tuples to compare
        taxonomy: Optional Taxonomy for topic-based scoring
        weight_semantic: Weight for semantic signal
        weight_syntactic: Weight for syntactic signal
        weight_topic: Weight for topic signal
        calibration_midpoint: Sigmoid midpoint
        calibration_steepness: Sigmoid steepness

    Returns:
        List of similarity scores, one per pair
    """
    if not pairs:
        return []
    
    # Collect all unique texts
    all_texts = []
    text_to_idx: dict[str, int] = {}
    for text_a, text_b in pairs:
        for text in (text_a, text_b):
            if text not in text_to_idx:
                text_to_idx[text] = len(all_texts)
                all_texts.append(text)
    
    # Batch embed all texts
    embeddings = await get_embeddings_batch(all_texts)
    
    # Compute similarity for each pair
    results = []
    for text_a, text_b in pairs:
        emb_a = embeddings[text_to_idx[text_a]]
        emb_b = embeddings[text_to_idx[text_b]]
        
        # Semantic
        raw_semantic = cosine_similarity(emb_a, emb_b)
        
        # Syntactic
        raw_syntactic = syntactic_similarity(text_a, text_b)
        
        # Topic
        if taxonomy is not None:
            raw_topic = topic_similarity(text_a, text_b, taxonomy)
            w_sem = weight_semantic
            w_syn = weight_syntactic
            w_top = weight_topic
        else:
            raw_topic = 0.0
            total_non_topic = weight_semantic + weight_syntactic
            if total_non_topic > 0:
                w_sem = weight_semantic / total_non_topic
                w_syn = weight_syntactic / total_non_topic
            else:
                w_sem = 0.5
                w_syn = 0.5
            w_top = 0.0
        
        # Blend and calibrate
        raw_blended = w_sem * raw_semantic + w_syn * raw_syntactic + w_top * raw_topic
        score = calibrate_score(raw_blended, calibration_midpoint, calibration_steepness)
        results.append(score)
    
    return results
