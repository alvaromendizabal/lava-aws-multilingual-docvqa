"""Tests for multilingual normalization and source parsers."""

from decimal import Decimal

import pytest

from lava.evaluation.normalization import (
    normalize_text,
    parse_decimal,
    parse_evidence_pages,
    parse_list_answer,
)


def test_unicode_and_whitespace_normalization() -> None:
    """NFKC and whitespace normalization should be conservative and deterministic."""
    assert normalize_text("  ＡＢＣ\u00a0  日本語。 ") == "abc 日本語"


def test_grouped_number_and_percentage_parsing() -> None:
    """Grouped digits and percentages should map to exact Decimal values."""
    assert parse_decimal("1,000") == Decimal("1E+3")
    assert parse_decimal("25%") == Decimal("0.25")


def test_evidence_page_parser_sorts_and_deduplicates() -> None:
    """Evidence pages should be canonicalized as positive sorted unique integers."""
    assert parse_evidence_pages("[3, 1, 3]") == (1, 3)


def test_evidence_page_parser_rejects_zero() -> None:
    """PDF page numbers are one-indexed."""
    with pytest.raises(ValueError, match="positive"):
        parse_evidence_pages("[0]")


def test_list_parser_supports_json_and_lines() -> None:
    """List parsing should preserve item text while supporting common encodings."""
    assert parse_list_answer('["東京", "大阪"]') == ("東京", "大阪")
    assert parse_list_answer("東京\n大阪") == ("東京", "大阪")
