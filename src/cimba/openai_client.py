"""OpenAI client with async native structured output support."""

import os
import json
from typing import TypeVar, Any

from openai import AsyncOpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "gpt-4o-mini"


def get_client() -> AsyncOpenAI:
    """Get or create async OpenAI client."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    return AsyncOpenAI(api_key=api_key)


def pydantic_to_json_schema(model: type[BaseModel]) -> dict:
    """
    Convert a Pydantic model to a JSON schema compatible with OpenAI's structured outputs.

    Handles:
    - Converting anyOf nullable types to proper format
    - Resolving $ref references
    - Adding additionalProperties: false
    - Making all properties required (OpenAI strict mode requirement)

    Args:
        model: Pydantic model class

    Returns:
        JSON schema dict compatible with OpenAI strict mode
    """
    schema = model.model_json_schema()

    # Extract $defs for reference resolution
    defs = schema.pop("$defs", {})

    def resolve_refs(obj: Any) -> Any:
        """Recursively resolve $ref references."""
        if isinstance(obj, dict):
            # Handle $ref
            if "$ref" in obj:
                ref_path = obj["$ref"].split("/")[-1]
                if ref_path in defs:
                    resolved = resolve_refs(defs[ref_path].copy())
                    # Merge any additional properties (like default)
                    for key, value in obj.items():
                        if key != "$ref":
                            resolved[key] = value
                    return resolved
                return obj

            # Handle anyOf for nullable types (convert to type array)
            if "anyOf" in obj:
                any_of = obj["anyOf"]
                # Check if it's a nullable pattern: [{"type": "X"}, {"type": "null"}]
                types = []
                non_null_schema = None
                for item in any_of:
                    if isinstance(item, dict):
                        if item.get("type") == "null":
                            types.append("null")
                        elif "type" in item:
                            types.append(item["type"])
                            non_null_schema = item
                        elif "$ref" in item:
                            # Resolve the reference
                            resolved = resolve_refs(item)
                            non_null_schema = resolved

                if "null" in types and non_null_schema:
                    # Convert to nullable type format
                    result = non_null_schema.copy() if isinstance(non_null_schema, dict) else {"type": non_null_schema}
                    if "type" in result and isinstance(result["type"], str):
                        result["type"] = [result["type"], "null"]
                    # Copy over description, title, and default if present
                    if "description" in obj:
                        result["description"] = obj["description"]
                    if "title" in obj:
                        result["title"] = obj["title"]
                    if "default" in obj:
                        result["default"] = obj["default"]
                    return resolve_refs(result)

            # Recursively process all values
            return {k: resolve_refs(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve_refs(item) for item in obj]
        return obj

    def add_strict_properties(obj: Any) -> Any:
        """Add strict mode properties to object schemas."""
        if isinstance(obj, dict):
            if obj.get("type") == "object" and "properties" in obj:
                obj["additionalProperties"] = False
                # Make all properties required for strict mode
                obj["required"] = list(obj.get("properties", {}).keys())

            # Recurse into nested structures
            for key, value in obj.items():
                if isinstance(value, dict):
                    add_strict_properties(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            add_strict_properties(item)
        return obj

    # First resolve all refs, then add strict properties
    resolved = resolve_refs(schema)
    return add_strict_properties(resolved)


async def chat_to_schema(
    messages: list[dict],
    schema: type[T],
    model: str = DEFAULT_MODEL,
    client: AsyncOpenAI | None = None,
    temperature: float = 0.0,
) -> T:
    """
    Send a chat completion request and parse the response into a Pydantic model.

    Uses OpenAI's native structured output feature (response_format with json_schema).

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        schema: Pydantic model class to parse the response into
        model: The model to use (default: gpt-4o-mini)
        client: Optional AsyncOpenAI client (creates one if not provided)
        temperature: Sampling temperature (default: 0.0 for deterministic output)

    Returns:
        Parsed Pydantic model instance

    Example:
        >>> from cimba.schemas import ClassificationResult
        >>> result = await chat_to_schema(
        ...     messages=[{"role": "user", "content": "Classify: I love this product!"}],
        ...     schema=ClassificationResult,
        ... )
        >>> print(result.label, result.confidence)
    """
    if client is None:
        client = get_client()

    json_schema = pydantic_to_json_schema(schema)

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "strict": True,
                "schema": json_schema,
            },
        },
    )

    content = response.choices[0].message.content

    if content is None:
        raise ValueError("No content in response")

    # Parse JSON and validate with Pydantic
    data = json.loads(content)
    return schema.model_validate(data)


async def chat_to_schema_with_system(
    system_prompt: str,
    user_message: str,
    schema: type[T],
    model: str = DEFAULT_MODEL,
    client: AsyncOpenAI | None = None,
    temperature: float = 0.0,
) -> T:
    """
    Convenience function that takes system prompt and user message separately.

    Args:
        system_prompt: The system prompt to set context
        user_message: The user's message/query
        schema: Pydantic model class to parse the response into
        model: The model to use (default: gpt-4o-mini)
        client: Optional AsyncOpenAI client (creates one if not provided)
        temperature: Sampling temperature

    Returns:
        Parsed Pydantic model instance
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    return await chat_to_schema(
        messages=messages,
        schema=schema,
        model=model,
        client=client,
        temperature=temperature,
    )
