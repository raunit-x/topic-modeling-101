"""Sanity checking for LLM responses using known truth pairs (canaries)."""

import logging
import random
from typing import Literal

from .schemas import KnownTruthPair, SimilarityCheckRequest, SimilarityPair

logger = logging.getLogger(__name__)


class SanityCheckError(Exception):
    """Raised when a sanity check fails in strict mode."""

    def __init__(self, message: str, failed_canaries: list[tuple[KnownTruthPair, SimilarityPair]]):
        super().__init__(message)
        self.failed_canaries = failed_canaries


# Default canary pairs for sanity checking
DEFAULT_CANARIES: list[KnownTruthPair] = [
    # Positive pairs (should be similar)
    KnownTruthPair(
        text_a="how many customers do I have",
        text_b="customer count",
        expected_similar=True,
    ),
    KnownTruthPair(
        text_a="total revenue",
        text_b="sum of all sales",
        expected_similar=True,
    ),
    KnownTruthPair(
        text_a="what is the average order value",
        text_b="mean order amount",
        expected_similar=True,
    ),
    # Negative pairs (should NOT be similar)
    KnownTruthPair(
        text_a="customer count",
        text_b="product inventory",
        expected_similar=False,
    ),
    KnownTruthPair(
        text_a="total revenue",
        text_b="profit margin",
        expected_similar=False,
    ),
]


SanityMode = Literal["strict", "warn", "sample"]


class SanityChecker:
    """
    Injects known truth pairs (canaries) into LLM requests to validate model behavior.

    Usage modes:
    - strict: Fail immediately if any canary fails (recommended for production)
    - warn: Log warning but continue (useful during development/EDA)
    - sample: Only check canaries on a percentage of batches (for cost optimization)
    """

    def __init__(
        self,
        mode: SanityMode = "strict",
        canaries: list[KnownTruthPair] | None = None,
        sample_rate: float = 0.1,
        canaries_per_batch: int = 1,
    ):
        """
        Initialize the sanity checker.

        Args:
            mode: How to handle failed checks ("strict", "warn", or "sample")
            canaries: Custom canary pairs (uses defaults if None)
            sample_rate: For "sample" mode, probability of checking each batch (0-1)
            canaries_per_batch: Number of canary pairs to inject per batch
        """
        self.mode = mode
        self.canaries = canaries if canaries is not None else DEFAULT_CANARIES.copy()
        self.sample_rate = sample_rate
        self.canaries_per_batch = canaries_per_batch

        # Statistics tracking
        self.stats = {
            "total_checks": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
        }

    def should_check_batch(self) -> bool:
        """Determine if this batch should include canary checks."""
        if self.mode == "sample":
            return random.random() < self.sample_rate
        return True

    def select_canaries(self, count: int | None = None) -> list[KnownTruthPair]:
        """
        Select canary pairs for injection.

        Args:
            count: Number of canaries to select (defaults to canaries_per_batch)

        Returns:
            List of selected canary pairs
        """
        count = count or self.canaries_per_batch
        count = min(count, len(self.canaries))

        if count >= len(self.canaries):
            return self.canaries.copy()

        return random.sample(self.canaries, count)

    def create_canary_requests(
        self,
        canaries: list[KnownTruthPair] | None = None,
    ) -> list[tuple[SimilarityCheckRequest, KnownTruthPair]]:
        """
        Create similarity check requests from canary pairs.

        Args:
            canaries: Canary pairs to convert (selects automatically if None)

        Returns:
            List of (request, original_canary) tuples
        """
        if canaries is None:
            canaries = self.select_canaries()

        requests = []
        for i, canary in enumerate(canaries):
            request = SimilarityCheckRequest(
                id_a=f"__canary_{i}_a__",
                text_a=canary.text_a,
                id_b=f"__canary_{i}_b__",
                text_b=canary.text_b,
            )
            requests.append((request, canary))

        return requests

    def inject_canaries(
        self,
        pairs: list[SimilarityCheckRequest],
    ) -> tuple[list[SimilarityCheckRequest], list[tuple[int, KnownTruthPair]]]:
        """
        Inject canary pairs into a batch of similarity check requests.

        Args:
            pairs: Original similarity check requests

        Returns:
            Tuple of:
            - Augmented list with canaries injected at random positions
            - List of (index, canary) tuples for later validation
        """
        if not self.should_check_batch():
            self.stats["skipped"] += 1
            return pairs, []

        canary_requests = self.create_canary_requests()
        if not canary_requests:
            return pairs, []

        # Create a copy and insert canaries at random positions
        augmented = pairs.copy()
        canary_positions = []

        for request, canary in canary_requests:
            # Insert at a random position
            pos = random.randint(0, len(augmented))
            augmented.insert(pos, request)
            canary_positions.append((pos, canary))

        # Update positions after all insertions (positions shift as we insert)
        # Re-calculate actual positions in the final list
        final_positions = []
        for request, canary in canary_requests:
            for i, req in enumerate(augmented):
                if req.id_a == request.id_a and req.id_b == request.id_b:
                    final_positions.append((i, canary))
                    break

        return augmented, final_positions

    def validate_canaries(
        self,
        results: list[SimilarityPair],
        canary_positions: list[tuple[int, KnownTruthPair]],
    ) -> tuple[list[SimilarityPair], bool]:
        """
        Validate canary results and remove them from the response.

        Args:
            results: Full results including canaries
            canary_positions: List of (index, canary) tuples from inject_canaries

        Returns:
            Tuple of:
            - Filtered results with canaries removed
            - True if all canaries passed, False otherwise

        Raises:
            SanityCheckError: If mode is "strict" and any canary fails
        """
        if not canary_positions:
            return results, True

        self.stats["total_checks"] += len(canary_positions)

        # Identify canary result indices
        canary_indices = {pos for pos, _ in canary_positions}

        # Validate each canary
        failed_canaries = []
        for pos, canary in canary_positions:
            if pos >= len(results):
                logger.warning(f"Canary at position {pos} not found in results (len={len(results)})")
                continue

            result = results[pos]
            expected = canary.expected_similar

            if result.is_similar != expected:
                failed_canaries.append((canary, result))
                self.stats["failed"] += 1
                logger.warning(
                    f"Canary check FAILED: '{canary.text_a}' vs '{canary.text_b}' - "
                    f"expected is_similar={expected}, got {result.is_similar}"
                )
            else:
                self.stats["passed"] += 1
                logger.debug(
                    f"Canary check passed: '{canary.text_a}' vs '{canary.text_b}'"
                )

        # Filter out canary results
        filtered_results = [r for i, r in enumerate(results) if i not in canary_indices]

        # Handle failures based on mode
        all_passed = len(failed_canaries) == 0

        if not all_passed:
            if self.mode == "strict":
                raise SanityCheckError(
                    f"{len(failed_canaries)} canary check(s) failed",
                    failed_canaries=failed_canaries,
                )
            elif self.mode == "warn":
                logger.warning(
                    f"{len(failed_canaries)} canary check(s) failed - continuing in warn mode"
                )

        return filtered_results, all_passed

    def get_stats(self) -> dict:
        """Get sanity check statistics."""
        stats = self.stats.copy()
        total = stats["passed"] + stats["failed"]
        stats["pass_rate"] = stats["passed"] / total if total > 0 else 1.0
        return stats

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self.stats = {
            "total_checks": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
        }

    def add_canary(self, canary: KnownTruthPair) -> None:
        """Add a custom canary pair."""
        self.canaries.append(canary)

    def add_canaries(self, canaries: list[KnownTruthPair]) -> None:
        """Add multiple custom canary pairs."""
        self.canaries.extend(canaries)
