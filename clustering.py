"""Example usage of the semantic clustering module.

This script demonstrates how to cluster statements by semantic similarity.
Statements that mean the same thing (e.g., "how many customers" and "customer count")
will be grouped into the same cluster.
"""

import asyncio

from src.cimba.clustering import (
    DEFAULT_CANARIES,
    KnownTruthPair,
    SemanticClusterer,
)


# Example documents to cluster
EXAMPLE_DOCS = [
    {"id": "1", "text": "how many customers do I have", "metadata": "customer query"},
    {"id": "2", "text": "customer count", "metadata": "customer metric"},
    {"id": "3", "text": "total number of customers", "metadata": "customer calculation"},
    {"id": "4", "text": "product inventory details", "metadata": "inventory info"},
    {"id": "5", "text": "number of products in stock", "metadata": "inventory count"},
    {"id": "6", "text": "what is the total revenue", "metadata": "revenue query"},
    {"id": "7", "text": "sum of all sales", "metadata": "sales calculation"},
    {"id": "8", "text": "average order value", "metadata": "order metric"},
    {"id": "9", "text": "mean order amount", "metadata": "order calculation"},
]


async def main():
    """Run semantic clustering on example documents."""

    # Create clusterer with default settings
    clusterer = SemanticClusterer(
        knn_k=5,  # Consider top 5 nearest neighbors
        similarity_threshold=0.5,  # Minimum embedding similarity
        sanity_mode="warn",  # Log warnings for failed canaries (use "strict" in production)
        batch_size=5,  # Process 5 pairs per LLM call
        min_confidence=0.7,  # Minimum LLM confidence to accept similarity
    )

    # Optionally add custom canary pairs for your domain
    custom_canaries = [
        KnownTruthPair(
            text_a="number of orders",
            text_b="order count",
            expected_similar=True,
        ),
    ]
    clusterer.sanity_checker.add_canaries(custom_canaries)

    # Run clustering
    print(f"Clustering {len(EXAMPLE_DOCS)} documents...")
    result = await clusterer.cluster(EXAMPLE_DOCS)

    # Display results
    print(f"\nFound {result.cluster_count} clusters:\n")

    # Group by cluster
    clusters: dict[str, list] = {}
    for doc in result.documents:
        cluster_id = doc.cluster_id
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(doc)

    for i, (cluster_id, docs) in enumerate(clusters.items(), 1):
        print(f"Cluster {i}:")
        for doc in docs:
            print(f"  - [{doc.id}] {doc.text}")
        print()

    # Show sanity check stats
    print("Sanity check statistics:")
    for key, value in result.sanity_check_stats.items():
        print(f"  {key}: {value}")

    # Save results
    clusterer.save_clusters("clusters.json")
    print("\nResults saved to clusters.json")


if __name__ == "__main__":
    asyncio.run(main())
