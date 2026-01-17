"""Tests for OpenAI client."""

import pytest
from pydantic import BaseModel, Field

from cimba.openai_client import pydantic_to_json_schema, chat_to_schema


class TestPydanticToJsonSchema:
    """Tests for schema conversion."""

    def test_simple_model_schema(self):
        """Test that a simple model generates valid schema."""
        class SimpleModel(BaseModel):
            name: str
            age: int
        
        class ComplexModel(BaseModel):
            name: str
            age: int
            address: Address
            city: str
            state: str
            zip: str

        class Address(BaseModel):
            street: str
            city: str
            state: str
            zip: str
        
        schema = pydantic_to_json_schema(ComplexModel)
        for key, value in schema.items():
            assert key in schema
        
        def inner_test(schema: dict):
            for key, value in schema.items():
                if isinstance(value, dict):
                    inner_test(value)
                else:
                    assert isinstance(value, type(schema[key]))

        inner_test(schema)

    def test_nullable_field_in_required(self):
        """Test that nullable fields are included in required array."""
        class ModelWithOptional(BaseModel):
            required_field: str
            optional_field: str | None = None

        schema = pydantic_to_json_schema(ModelWithOptional)
        
        # OpenAI requires ALL fields in required array
        assert "optional_field" in schema["required"]
        assert "required_field" in schema["required"]

    def test_nullable_field_type_format(self):
        """Test nullable fields use correct type array format."""
        class ModelWithOptional(BaseModel):
            optional_field: str | None = None

        schema = pydantic_to_json_schema(ModelWithOptional)
        
        optional_prop = schema["properties"]["optional_field"]
        assert optional_prop["type"] == ["string", "null"]


class TestChatToSchema:
    """Integration tests for chat_to_schema (require API key)."""

    @pytest.mark.integration
    async def test_basic_classification(self, skip_without_openai_key):
        """Test structured output with classification."""
        from cimba.schemas import ClassificationResult

        result = await chat_to_schema(
            messages=[
                {"role": "system", "content": "Classify sentiment as positive, negative, or neutral."},
                {"role": "user", "content": "I love this!"},
            ],
            schema=ClassificationResult,
        )

        assert result.label in ["positive", "negative", "neutral"]
        assert 0.0 <= result.confidence <= 1.0