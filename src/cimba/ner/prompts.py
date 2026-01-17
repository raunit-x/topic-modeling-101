"""LLM prompts for topic clustering and taxonomy building."""

# Entity validation prompt - determines if a term is a meaningful entity
ENTITY_VALIDATION_PROMPT = """You are an expert at identifying meaningful entities in text.

A meaningful entity is a specific concept, object, action, or attribute that:
1. Refers to something concrete and identifiable
2. Would be useful for categorization or search
3. Is not too generic (like "thing", "stuff", "good")
4. Is not purely functional (like "click", "submit", "next")

For each term, review the sample contexts to understand how it's used, then decide if it's a meaningful entity.

Be strict: only mark as valid entities that would be genuinely useful for taxonomy/NER purposes."""


# Similarity check prompt - determines if two terms refer to the same concept
SIMILARITY_CHECK_PROMPT = """You are an expert at semantic analysis.

Your task is to determine if pairs of terms refer to the SAME underlying concept.
Terms are the same concept if:
1. They are synonyms (dahi = curd = yogurt in Indian context)
2. One is a variant/abbreviation of the other
3. They refer to the exact same thing in different words

They are NOT the same if:
1. They are related but distinct concepts (milk vs cream)
2. One is a category of the other (dairy vs milk)
3. They overlap but aren't identical

Use the sample contexts to understand how each term is actually used.
Provide a canonical name that best represents the shared concept."""


# Cluster naming prompt - generates canonical names for topic clusters
CLUSTER_NAMING_PROMPT = """You are an expert at organizing concepts into taxonomies.

Given a cluster of related terms, provide:
1. A clear canonical name for this topic
2. A brief description of what this topic covers

The name should be:
- Specific enough to distinguish from other topics
- General enough to encompass all terms in the cluster
- Professional and consistent in style"""


# Hierarchy building prompt - organizes clusters into a hierarchical taxonomy
HIERARCHY_PROMPT = """You are an expert at organizing concepts into taxonomies.

Given these topic clusters, create a hierarchical taxonomy where:
1. Related topics are grouped under parent topics
2. Parent topics have clear, general names
3. The hierarchy has 2-3 levels maximum
4. Each leaf topic contains the original terms

Guidelines:
- Create parent topics only when there are 2+ related child topics
- Don't create deep hierarchies (max 3 levels)
- Use consistent naming conventions
- Include brief descriptions for each topic"""
