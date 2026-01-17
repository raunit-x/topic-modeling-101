"""Example usage of the topic taxonomy builder.

This script demonstrates how to build a topic taxonomy from a corpus of text files.
The taxonomy groups semantically similar terms into topics, creating a hierarchy
that can be used for NER-like entity extraction.
"""

import asyncio
from pathlib import Path

from src.cimba.ner import TaxonomyBuilder


# Example: Create sample corpus for demonstration
SAMPLE_TEXTS = [
    (
        "recipe_1.txt",
        """
        Traditional Indian Curry Recipe
        
        For this delicious curry, you'll need fresh dahi (yogurt) to marinate the chicken.
        The dahi helps tenderize the meat and adds a creamy texture. If you don't have dahi,
        you can use curd or plain yogurt as a substitute.
        
        Add paneer cubes for a vegetarian version. The paneer should be fresh and firm.
        Some people prefer cottage cheese, but authentic paneer works best.
        
        Heat ghee in a pan. Ghee adds a rich flavor that regular butter cannot match.
        Add cumin seeds, coriander powder, and turmeric. These spices are essential
        for the authentic taste.
        
        Simmer with tomatoes until the oil separates. Add garam masala at the end.
        Serve with basmati rice or naan bread.
        """,
    ),
    (
        "recipe_2.txt",
        """
        South Indian Sambar
        
        Sambar is a lentil-based vegetable stew popular in South India.
        Start by cooking toor dal until soft. Toor dal, also known as pigeon peas,
        gives the sambar its distinctive taste.
        
        For the sambar powder, dry roast coriander seeds, cumin seeds, fenugreek,
        and dried red chilies. Grind to a fine powder.
        
        Add vegetables like drumstick, eggplant, and okra. The vegetables should be
        cut into bite-sized pieces.
        
        Tamarind paste gives the sambar its tangy flavor. Adjust the amount based
        on your preference. Finish with a tempering of mustard seeds, curry leaves,
        and dried red chili in coconut oil.
        
        Serve hot with steamed rice or idli. Sambar is also great with dosa.
        """,
    ),
    (
        "recipe_3.txt",
        """
        North Indian Butter Chicken
        
        Butter chicken, or murgh makhani, is a rich and creamy dish.
        
        Marinate chicken pieces in a mixture of yogurt, ginger-garlic paste,
        and kashmiri red chili powder. The yogurt tenderizes the meat.
        
        Grill or tandoor the marinated chicken until charred.
        
        For the sauce, blend tomatoes, cashews, and cream. The cashews add
        body and richness. Cook with butter, cream, and kasuri methi.
        
        The key to authentic butter chicken is using enough butter and cream.
        Some recipes also add honey for a touch of sweetness.
        
        Garnish with fresh cream and serve with naan or jeera rice.
        """,
    ),
    (
        "ingredients_guide.txt",
        """
        Essential Indian Cooking Ingredients
        
        DAIRY PRODUCTS:
        - Ghee (clarified butter): Used for cooking and flavoring
        - Paneer (Indian cottage cheese): Fresh, unaged cheese
        - Dahi/Curd/Yogurt: Fermented dairy, essential for marinades
        - Malai (cream): Used in rich gravies
        - Khoya (reduced milk solite): For sweets
        
        LENTILS AND LEGUMES:
        - Toor dal (pigeon peas): For sambar
        - Chana dal (split chickpeas): For snacks and curries
        - Moong dal: For light dishes
        - Urad dal (black gram): For idli batter
        - Masoor dal (red lentils): Quick-cooking
        
        SPICES:
        - Cumin (jeera): Earthy, warm flavor
        - Coriander (dhania): Citrusy, mild
        - Turmeric (haldi): Yellow color, earthy
        - Garam masala: Warming spice blend
        - Kasuri methi (dried fenugreek leaves): Bitter, aromatic
        
        GRAINS:
        - Basmati rice: Long-grain, aromatic
        - Atta (whole wheat flour): For rotis
        - Besan (chickpea flour): For pakoras
        - Semolina (sooji/rava): For upma, halwa
        """,
    ),
]


async def main():
    """Build taxonomy from sample texts."""

    print("=" * 60)
    print("TOPIC TAXONOMY BUILDER")
    print("=" * 60)

    # Create taxonomy builder with configuration
    builder = TaxonomyBuilder(
        top_n=100,  # Extract top 100 terms (small for demo)
        contexts_per_term=3,
        min_df=1,  # Allow terms appearing in just 1 doc (for small corpus)
        knn_k=5,
        similarity_threshold=0.5,
        min_confidence=0.6,
        batch_size=5,
    )

    # Build taxonomy from sample texts
    print(f"\nProcessing {len(SAMPLE_TEXTS)} sample texts...")
    taxonomy = await builder.build_from_texts(SAMPLE_TEXTS)

    # Print summary
    builder.print_taxonomy(max_terms=5)

    # Save to file
    builder.save_taxonomy("taxonomy.json")
    print(f"\nTaxonomy saved to taxonomy.json")

    # Show some examples
    print("\n" + "=" * 60)
    print("EXAMPLE LOOKUPS")
    print("=" * 60)

    for term in ["dahi", "paneer", "cumin", "basmati rice"]:
        topic = taxonomy.get_topic_by_term(term)
        if topic:
            path = taxonomy.get_topic_path(topic.id)
            print(f"'{term}' -> {' > '.join(path)}")
        else:
            print(f"'{term}' -> (not found)")


async def build_from_directory():
    """Build taxonomy from a directory of text files."""

    import sys

    if len(sys.argv) < 2:
        print("Usage: python ner_taxonomy.py <directory>")
        print("       python ner_taxonomy.py --demo")
        return

    if sys.argv[1] == "--demo":
        await main()
        return

    directory = sys.argv[1]
    if not Path(directory).exists():
        print(f"Directory not found: {directory}")
        return

    builder = TaxonomyBuilder(
        top_n=5000,
        contexts_per_term=5,
        min_df=2,
        knn_k=10,
        similarity_threshold=0.6,
        min_confidence=0.7,
    )

    print(f"Building taxonomy from {directory}...")
    taxonomy = await builder.build_from_files(directory)
    builder.print_taxonomy()
    builder.save_taxonomy("taxonomy.json")
    print("\nTaxonomy saved to taxonomy.json")


if __name__ == "__main__":
    asyncio.run(build_from_directory())
