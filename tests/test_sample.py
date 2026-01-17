"""Sample tests to verify pytest setup is working."""

import pytest


def test_simple_assertion():
    """Basic test to verify pytest works."""
    assert 1 + 1 == 2


def test_string_operations():
    """Test string operations."""
    text = "hello world"
    assert text.upper() == "HELLO WORLD"
    assert text.split() == ["hello", "world"]


def test_list_operations():
    """Test list operations."""
    numbers = [1, 2, 3, 4, 5]
    assert sum(numbers) == 15
    assert len(numbers) == 5
    assert max(numbers) == 5


class TestMathOperations:
    """Group of math-related tests."""

    def test_addition(self):
        assert 2 + 2 == 4

    def test_subtraction(self):
        assert 10 - 5 == 5

    def test_multiplication(self):
        assert 3 * 4 == 12

    def test_division(self):
        assert 20 / 4 == 5.0


@pytest.mark.parametrize(
    "input_val,expected",
    [
        (1, 1),
        (2, 4),
        (3, 9),
        (4, 16),
        (5, 25),
    ],
)
def test_squares(input_val, expected):
    """Parametrized test for squares."""
    assert input_val**2 == expected


def test_fixture_usage(sample_texts):
    """Test that fixtures work correctly."""
    assert len(sample_texts) == 3
    assert all(isinstance(t, str) for t in sample_texts)
