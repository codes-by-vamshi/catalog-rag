"""Tests for normalizer.py"""
import pytest
from src.normalizer import (
    normalise_text,
    normalise_code,
    extract_codes,
    split_code,
    derive_base_codes,
    generate_aliases,
)


class TestNormaliseText:
    def test_lowercases(self):
        assert normalise_text("Hello World") == "hello world"

    def test_collapses_whitespace(self):
        assert normalise_text("hello   world") == "hello world"

    def test_strips(self):
        assert normalise_text("  hi  ") == "hi"

    def test_unicode_normalisation(self):
        # Non-breaking space
        result = normalise_text("hello world")
        assert result == "hello world"


class TestNormaliseCode:
    def test_uppercases(self):
        assert normalise_code("36-5150/01") == "36-5150/01"

    def test_strips_spaces(self):
        assert normalise_code("36 - 5150") == "36-5150"


class TestExtractCodes:
    def test_single_code(self):
        assert extract_codes("Product code: 36-5150/01") == ["36-5150/01"]

    def test_multiple_codes(self):
        codes = extract_codes("Codes: 36-5150/01 and 24-9186")
        assert "36-5150/01" in codes
        assert "24-9186" in codes

    def test_no_codes(self):
        assert extract_codes("No codes here just text") == []

    def test_code_in_sentence(self):
        codes = extract_codes("The model 25-3518/01 is available.")
        assert "25-3518/01" in codes

    def test_base_code_only(self):
        codes = extract_codes("See 36-5150 for details.")
        assert "36-5150" in codes


class TestSplitCode:
    def test_with_variant(self):
        base, suffix = split_code("36-5150/01")
        assert base == "36-5150"
        assert suffix == "01"

    def test_without_variant(self):
        base, suffix = split_code("36-5150")
        assert base == "36-5150"
        assert suffix is None

    def test_variant_06(self):
        base, suffix = split_code("36-5150/06")
        assert base == "36-5150"
        assert suffix == "06"

    def test_strips_whitespace(self):
        base, suffix = split_code("  36-5150/01  ")
        assert base == "36-5150"
        assert suffix == "01"


class TestDeriveBaseCodes:
    def test_deduplicates(self):
        codes = ["36-5150/01", "36-5150/06"]
        bases = derive_base_codes(codes)
        assert bases == ["36-5150"]

    def test_multiple_families(self):
        codes = ["36-5150/01", "24-9186"]
        bases = derive_base_codes(codes)
        assert "36-5150" in bases
        assert "24-9186" in bases

    def test_empty(self):
        assert derive_base_codes([]) == []


class TestGenerateAliases:
    def test_includes_code(self):
        aliases = generate_aliases(None, ["36-5150/01"])
        assert "36-5150/01" in aliases

    def test_includes_base_code(self):
        aliases = generate_aliases(None, ["36-5150/01"])
        assert "36-5150" in aliases

    def test_includes_product_name(self):
        aliases = generate_aliases("Compression Machine", ["36-5150"])
        assert "Compression Machine" in aliases

    def test_includes_lowercase_name(self):
        aliases = generate_aliases("Compression Machine", [])
        assert "compression machine" in aliases

    def test_no_duplicates(self):
        aliases = generate_aliases("Test Product", ["36-5150"])
        assert len(aliases) == len(set(aliases))

    def test_strips_standards(self):
        aliases = generate_aliases("ADR Auto Compression BS EN Machine", ["36-5150"])
        # Should include a version without the standards markers
        assert any("bs en" not in a.lower() for a in aliases if "adr" in a.lower())

    def test_no_very_short_aliases(self):
        aliases = generate_aliases("A B", [])
        assert all(len(a) > 2 for a in aliases)
