"""TF-IDF extraction with context collection for term discovery."""

import logging
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from .schemas import CorpusStats, ExtractedTerm, TermContext

logger = logging.getLogger(__name__)

# Configure logging format for this module
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


# Common stopwords to filter out
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "as", "is", "was", "are", "were", "been", "be", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might", "must",
    "shall", "can", "need", "dare", "ought", "used", "it", "its", "this", "that",
    "these", "those", "i", "you", "he", "she", "we", "they", "what", "which", "who",
    "whom", "whose", "where", "when", "why", "how", "all", "each", "every", "both",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "also", "now", "here", "there",
}


def tokenize(text: str) -> list[str]:
    """
    Tokenize text into lowercase words.

    Args:
        text: Input text

    Returns:
        List of lowercase tokens
    """
    # Simple tokenization - split on non-alphanumeric, keep only words
    tokens = re.findall(r"\b[a-zA-Z][a-zA-Z0-9]*\b", text.lower())
    return tokens


def extract_ngrams(tokens: list[str], n: int) -> list[str]:
    """
    Extract n-grams from a list of tokens.

    Args:
        tokens: List of tokens
        n: N-gram size (1, 2, or 3)

    Returns:
        List of n-gram strings
    """
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def split_into_sentences(text: str) -> list[tuple[str, int, int]]:
    """
    Split text into sentences with their positions.

    Args:
        text: Input text

    Returns:
        List of (sentence, start_pos, end_pos) tuples
    """
    # Simple sentence splitting on . ! ? followed by space or end
    sentences = [
        x.strip().lower() for x in text.split("\n") if len(x.strip()) > 10
    ]
    return [
        (sentence, 0, len(sentence)) for sentence in sentences
    ]

class TFIDFExtractor:
    """
    Extract keywords, bigrams, and trigrams using TF-IDF with context collection.

    Processes a corpus of text files and extracts the most interesting terms
    along with sample sentences showing how each term is used.
    """

    def __init__(
        self,
        min_df: int = 2,
        max_df_ratio: float = 0.8,
        contexts_per_term: int = 5,
    ):
        """
        Initialize the TF-IDF extractor.

        Args:
            min_df: Minimum document frequency for a term
            max_df_ratio: Maximum document frequency ratio (filter very common terms)
            contexts_per_term: Number of sample contexts to collect per term
        """
        self.min_df = min_df
        self.max_df_ratio = max_df_ratio
        self.contexts_per_term = contexts_per_term

        # Internal state
        self._documents: list[dict] = []  # [{path, text, tokens, sentences}]
        self._term_doc_freq: Counter = Counter()
        self._term_contexts: defaultdict = defaultdict(list)  # term -> [TermContext]
        self._corpus_stats: CorpusStats | None = None

    def load_files(
        self, 
        directory: str, 
        pattern: str = "*.txt",
        sentence_level: bool = True,
    ) -> int:
        """
        Load text files from a directory.

        Args:
            directory: Path to directory containing text files
            pattern: Glob pattern for matching files
            sentence_level: If True, treat each sentence/line as a separate document
                           (recommended for TF-IDF to work properly)

        Returns:
            Number of documents loaded
        """
        logger.info(f"[LOAD] Starting to load files from: {directory}")
        logger.info(f"[LOAD] Mode: {'sentence-level' if sentence_level else 'file-level'} documents")
        
        dir_path = Path(directory)
        if not dir_path.exists():
            raise ValueError(f"Directory not found: {directory}")

        files = list(dir_path.glob(pattern))
        logger.info(f"[LOAD] Direct glob found {len(files)} files")
        if not files:
            # Try recursive search
            files = list(dir_path.rglob(pattern))
            logger.info(f"[LOAD] Recursive glob found {len(files)} files")

        logger.info(f"[LOAD] Processing {len(files)} files matching '{pattern}'")

        total_sentences = 0
        for i, file_path in enumerate(files):
            try:
                text = file_path.read_text(encoding="utf-8")
                logger.debug(f"[LOAD] [{i+1}/{len(files)}] Processing {file_path.name} ({len(text)} chars)")
                
                if sentence_level:
                    # Split into sentences/lines and treat each as a document
                    sentences = self._split_into_sentences(text)
                    for j, sentence in enumerate(sentences):
                        if len(sentence.strip()) > 10:  # Skip very short sentences
                            source = f"{file_path.name}:line{j+1}"
                            self._process_document(source, sentence)
                            total_sentences += 1
                else:
                    # Treat entire file as one document
                    self._process_document(str(file_path), text)
                    
            except Exception as e:
                logger.warning(f"[LOAD] Failed to process {file_path}: {e}")

        if sentence_level:
            logger.info(f"[LOAD] Split {len(files)} files into {total_sentences} sentence-documents")
        
        logger.info(f"[LOAD] Successfully loaded {len(self._documents)} documents")
        return len(self._documents)
    
    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences for document-level processing."""
        # Split by newlines first (common in structured docs)
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        
        # If we get very few lines, try splitting by periods
        if len(lines) < 5:
            import re
            # Split on sentence boundaries
            sentences = re.split(r'(?<=[.!?])\s+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            if len(sentences) > len(lines):
                return sentences
        
        return lines

    def load_texts(
        self, 
        texts: list[tuple[str, str]],
        sentence_level: bool = True,
    ) -> int:
        """
        Load texts directly (without files).

        Args:
            texts: List of (source_name, text_content) tuples
            sentence_level: If True, treat each sentence/line as a separate document

        Returns:
            Number of documents loaded
        """
        logger.info(f"[LOAD] Loading {len(texts)} text sources")
        logger.info(f"[LOAD] Mode: {'sentence-level' if sentence_level else 'text-level'} documents")
        
        total_sentences = 0
        for source_name, text in texts:
            if sentence_level:
                sentences = self._split_into_sentences(text)
                for j, sentence in enumerate(sentences):
                    if len(sentence.strip()) > 10:
                        source = f"{source_name}:line{j+1}"
                        self._process_document(source, sentence)
                        total_sentences += 1
            else:
                self._process_document(source_name, text)

        if sentence_level:
            logger.info(f"[LOAD] Split {len(texts)} texts into {total_sentences} sentence-documents")
        
        logger.info(f"[LOAD] Successfully loaded {len(self._documents)} documents")
        return len(self._documents)

    def _process_document(self, source: str, text: str) -> None:
        """Process a single document and update term frequencies."""
        tokens = tokenize(text)
        sentences = split_into_sentences(text)

        logger.debug(f"[PROCESS] {Path(source).name}: {len(tokens)} tokens, {len(sentences)} sentences")

        # Extract all n-grams
        unigrams = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
        bigrams = extract_ngrams(tokens, 2)
        trigrams = extract_ngrams(tokens, 3)

        # Filter bigrams/trigrams containing only stopwords
        bigrams = [b for b in bigrams if not all(w in STOPWORDS for w in b.split())]
        trigrams = [t for t in trigrams if not all(w in STOPWORDS for w in t.split())]

        logger.debug(f"[PROCESS] {Path(source).name}: {len(unigrams)} unigrams, {len(bigrams)} bigrams, {len(trigrams)} trigrams")

        # Track unique terms in this document
        doc_terms = set(unigrams + bigrams + trigrams)

        # Update document frequency
        for term in doc_terms:
            self._term_doc_freq[term] += 1

        # Collect contexts for terms
        self._collect_contexts(source, text, sentences, doc_terms)

        # Store document info
        self._documents.append(
            {
                "source": source,
                "text": text,
                "tokens": tokens,
                "sentences": sentences,
                "unigrams": unigrams,
                "bigrams": bigrams,
                "trigrams": trigrams,
            }
        )

    def _collect_contexts(
        self,
        source: str,
        text: str,
        sentences: list[tuple[str, int, int]],
        terms: set[str],
    ) -> None:
        """Collect sample contexts for terms."""
        for sentence, start, end in sentences:
            sentence_lower = sentence.lower()
            for term in terms:
                if term in sentence_lower:
                    # Find position in sentence
                    match = re.search(re.escape(term), sentence_lower)
                    if match:
                        context = TermContext(
                            sentence=sentence,
                            source_file=source,
                            char_start=match.start(),
                            char_end=match.end(),
                        )
                        # Only keep up to contexts_per_term contexts
                        if len(self._term_contexts[term]) < self.contexts_per_term * 2:
                            self._term_contexts[term].append(context)

    def _calculate_tfidf(self, term: str, doc_idx: int) -> float:
        """Calculate TF-IDF score for a term in a document."""
        doc = self._documents[doc_idx]
        all_terms = doc["unigrams"] + doc["bigrams"] + doc["trigrams"]

        # Term frequency in document
        tf = all_terms.count(term) / max(len(all_terms), 1)

        # Inverse document frequency
        df = self._term_doc_freq.get(term, 1)
        n_docs = len(self._documents)
        idf = math.log(n_docs / df) if df > 0 else 0

        return tf * idf

    def extract_terms(self, top_n: int = 5000) -> list[ExtractedTerm]:
        """
        Extract top terms by TF-IDF score.

        Args:
            top_n: Number of top terms to extract

        Returns:
            List of ExtractedTerm objects sorted by TF-IDF score
        """
        logger.info(f"[EXTRACT] Starting term extraction (top_n={top_n})")
        
        if not self._documents:
            raise ValueError("No documents loaded. Call load_files() first.")

        n_docs = len(self._documents)
        max_df = int(n_docs * self.max_df_ratio)
        logger.info(f"[EXTRACT] Processing {n_docs} documents (min_df={self.min_df}, max_df={max_df})")

        # Calculate aggregate TF-IDF for each term
        term_scores: dict[str, float] = defaultdict(float)
        term_types: dict[str, str] = {}
        filtered_by_df = 0

        for doc_idx, doc in enumerate(self._documents):
            for term_type, terms in [
                ("unigram", doc["unigrams"]),
                ("bigram", doc["bigrams"]),
                ("trigram", doc["trigrams"]),
            ]:
                for term in set(terms):
                    df = self._term_doc_freq.get(term, 0)

                    # Filter by document frequency
                    if df < self.min_df or df > max_df:
                        filtered_by_df += 1
                        continue

                    score = self._calculate_tfidf(term, doc_idx)
                    term_scores[term] = max(term_scores[term], score)
                    term_types[term] = term_type

        logger.info(f"[EXTRACT] Scored {len(term_scores)} unique terms (filtered {filtered_by_df} by df)")

        # Sort by score and take top N
        sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        logger.info(f"[EXTRACT] Selected top {len(sorted_terms)} terms by TF-IDF score")

        # Build ExtractedTerm objects
        extracted = []
        terms_with_contexts = 0
        for term, score in sorted_terms:
            contexts = self._term_contexts.get(term, [])[:self.contexts_per_term]
            if contexts:
                terms_with_contexts += 1

            extracted.append(
                ExtractedTerm(
                    term=term,
                    term_type=term_types.get(term, "unigram"),
                    tfidf_score=score,
                    document_frequency=self._term_doc_freq.get(term, 0),
                    sample_contexts=contexts,
                )
            )

        # Count by type
        type_counts = {}
        for t in extracted:
            type_counts[t.term_type] = type_counts.get(t.term_type, 0) + 1

        # Update stats
        self._corpus_stats = CorpusStats(
            total_documents=len(self._documents),
            total_tokens=sum(len(d["tokens"]) for d in self._documents),
            unique_unigrams=len(
                set(t for d in self._documents for t in d["unigrams"])
            ),
            unique_bigrams=len(set(t for d in self._documents for t in d["bigrams"])),
            unique_trigrams=len(
                set(t for d in self._documents for t in d["trigrams"])
            ),
            terms_extracted=len(extracted),
            valid_entities=0,
            topics_created=0,
        )

        logger.info(
            f"[EXTRACT] Extracted {len(extracted)} terms: "
            f"{type_counts.get('unigram', 0)} unigrams, "
            f"{type_counts.get('bigram', 0)} bigrams, "
            f"{type_counts.get('trigram', 0)} trigrams"
        )
        logger.info(f"[EXTRACT] {terms_with_contexts}/{len(extracted)} terms have sample contexts")
        logger.info(f"[EXTRACT] Corpus stats: {self._corpus_stats.total_tokens} tokens across {n_docs} docs")

        return extracted

    def get_stats(self) -> CorpusStats | None:
        """Get corpus statistics."""
        return self._corpus_stats

    def get_contexts_for_term(self, term: str) -> list[TermContext]:
        """Get all collected contexts for a term."""
        return self._term_contexts.get(term, [])[:self.contexts_per_term]
