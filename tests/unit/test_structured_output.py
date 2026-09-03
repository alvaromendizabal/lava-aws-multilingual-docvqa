from __future__ import annotations

import json

import pytest

from lava.readers.structured_output import (
    append_strict_json_instruction,
    extract_balanced_json_object,
    parse_structured_output,
    strip_reasoning_blocks,
)


def test_balanced_json_handles_braces_inside_strings() -> None:
    text = (
        'prefix {"answer":"a } brace","evidence_pages":[1],"confidence":0.8,"abstain":false} suffix'
    )
    candidate = extract_balanced_json_object(text)
    assert candidate is not None
    assert json.loads(candidate)["answer"] == "a } brace"


def test_reasoning_block_is_removed_before_parsing() -> None:
    raw = '<think>private reasoning {not json}</think> {"answer":"yes","evidence_pages":[2],"confidence":1,"abstain":false}'
    result = parse_structured_output(raw, valid_page_numbers=[2])
    assert result.valid
    assert result.parsed is not None
    assert result.parsed.answer == "yes"


def test_missing_object_fails_closed() -> None:
    result = parse_structured_output("I cannot answer")
    assert not result.valid
    assert result.error == "missing_json_object"


def test_extra_keys_are_rejected() -> None:
    raw = '{"answer":"yes","evidence_pages":[1],"confidence":0.8,"abstain":false,"extra":1}'
    result = parse_structured_output(raw)
    assert not result.valid
    assert result.error is not None and result.error.startswith("schema_error:")


def test_abstention_semantics_are_strict() -> None:
    invalid = '{"answer":"guess","evidence_pages":[],"confidence":0.1,"abstain":true}'
    assert not parse_structured_output(invalid).valid
    valid = '{"answer":"","evidence_pages":[],"confidence":0.1,"abstain":true}'
    assert parse_structured_output(valid).valid


def test_evidence_pages_must_be_allowed() -> None:
    raw = '{"answer":"yes","evidence_pages":[3],"confidence":0.8,"abstain":false}'
    result = parse_structured_output(raw, valid_page_numbers=[1, 2])
    assert not result.valid


def test_append_contract_preserves_multimodal_message() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "page.png"},
                {"type": "text", "text": "What is shown?"},
            ],
        }
    ]
    updated = append_strict_json_instruction(messages)
    assert updated is not messages
    assert updated[0]["content"][0]["image"] == "page.png"
    assert "OUTPUT CONTRACT" in updated[0]["content"][1]["text"]
    assert "OUTPUT CONTRACT" not in messages[0]["content"][1]["text"]


def test_strip_unclosed_thinking_block_does_not_publish_reasoning() -> None:
    assert strip_reasoning_blocks("<think>unfinished secret reasoning") == ""


def test_non_string_raw_text_is_rejected() -> None:
    with pytest.raises(TypeError):
        parse_structured_output(123)  # type: ignore[arg-type]
