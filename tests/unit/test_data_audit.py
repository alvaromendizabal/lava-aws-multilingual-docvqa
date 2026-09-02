"""Unit tests for the LAVA data-audit helpers."""

from lava.data.audit import (
    classify_answer_type,
    classify_language,
    parse_page_references,
    split_from_key,
)


def test_language_classification() -> None:
    """Japanese, Vietnamese, and fallback language signals are recognized."""
    assert classify_language("これは質問です") == "ja"
    assert classify_language("Đây là một câu hỏi") == "vi"
    assert classify_language("plain text", "vi") == "vi"


def test_answer_type_classification() -> None:
    """Numeric, list, and list-like answers are separated."""
    assert classify_answer_type("12.5") == "number"
    assert classify_answer_type('["a", "b"]') == "list"
    assert classify_answer_type("alpha; beta") == "list_like"


def test_page_reference_parsing() -> None:
    """Page lists are deduplicated while retaining order."""
    assert parse_page_references("[1, 3, 3]") == [1, 3]
    assert parse_page_references("pages 2 and 7") == [2, 7]


def test_split_inference() -> None:
    """Raw S3 keys map to the intended corpus split."""
    assert split_from_key("raw/kaggle/train_pdfs/train_pdfs/j_1.pdf") == "train"
    assert split_from_key("raw/kaggle/test_pdfs/test_pdfs/v_1.pdf") == "test"
