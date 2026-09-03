"""Strict, reasoning-safe structured-output handling for oracle reader generations."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

PROMPT_VERSION = "oracle-reader-json-v3"
_REQUIRED_KEYS = frozenset({"answer", "evidence_pages", "confidence", "abstain"})
_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class StructuredReaderOutput:
    """Validated model response contract used by the oracle-reader benchmark."""

    answer: str
    evidence_pages: tuple[int, ...]
    confidence: float
    abstain: bool

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        payload["evidence_pages"] = list(self.evidence_pages)
        return payload


@dataclass(frozen=True, slots=True)
class StructuredParseResult:
    """Structured parse result with an explicit error instead of an ambiguous fallback."""

    raw_text: str
    candidate_json: str | None
    parsed: StructuredReaderOutput | None
    error: str | None

    @property
    def valid(self) -> bool:
        """Whether a complete response passed the strict schema."""
        return self.parsed is not None and self.error is None

    def as_dict(self) -> dict[str, object]:
        """Return metadata safe for logs; raw model text is intentionally omitted."""
        return {
            "valid": self.valid,
            "candidate_json_present": self.candidate_json is not None,
            "error": self.error,
            "parsed": self.parsed.as_dict() if self.parsed else None,
        }


def strict_output_instruction(*, evidence_pages: Sequence[int] | None = None) -> str:
    """Build the strict JSON-only contract appended to the final user turn."""
    page_rule = "Use only page numbers visible in the supplied evidence."
    if evidence_pages:
        allowed = ", ".join(str(int(page)) for page in evidence_pages)
        page_rule = f"evidence_pages may contain only these page numbers: [{allowed}]."
    return (
        "\n\nOUTPUT CONTRACT — FOLLOW EXACTLY:\n"
        "Return exactly one JSON object and nothing else. Do not use Markdown fences, "
        "XML, commentary, or a preamble. The object must have exactly these four keys:\n"
        '{"answer":"...","evidence_pages":[1],"confidence":0.8,"abstain":false}\n'
        "answer must be a string. evidence_pages must be a JSON array of integer page "
        "numbers. confidence must be a number from 0.0 through 1.0. abstain must be a "
        "JSON boolean. If the evidence is insufficient, set abstain=true, answer to an "
        "empty string, and evidence_pages to an empty array. "
        f"{page_rule}"
    )


def append_strict_json_instruction(
    messages: Sequence[Mapping[str, Any]],
    *,
    evidence_pages: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Copy chat messages and append the contract to the last user text item."""
    copied: list[dict[str, Any]] = []
    for message in messages:
        item = dict(message)
        content = item.get("content")
        if isinstance(content, list):
            item["content"] = [
                dict(part) if isinstance(part, Mapping) else part for part in content
            ]
        copied.append(item)

    instruction = strict_output_instruction(evidence_pages=evidence_pages)
    for message in reversed(copied):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = content.rstrip() + instruction
            return copied
        if isinstance(content, list):
            for part in reversed(content):
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text")
                    if isinstance(text, str):
                        part["text"] = text.rstrip() + instruction
                        return copied
            content.append({"type": "text", "text": instruction.lstrip()})
            return copied
        raise TypeError("The final user message has unsupported content.")
    raise ValueError("No user message exists to receive the structured-output contract.")


def strip_reasoning_blocks(text: str) -> str:
    """Remove explicit Qwen-style reasoning blocks before public response parsing."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    cleaned = _THINK_BLOCK.sub("", text).strip()
    lowered = cleaned.lower()
    if lowered.startswith("<think>") and "</think>" not in lowered:
        return ""
    return cleaned


def extract_balanced_json_object(text: str) -> str | None:
    """Return the first complete JSON object while respecting strings and escapes."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    start: int | None = None
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if start is None:
            if char == "{":
                start = index
                depth = 1
                in_string = False
                escaped = False
            continue

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
            if depth < 0:
                return None
    return None


def extract_json_candidate(text: str) -> str:
    """Extract a balanced response object, or preserve text for legacy error handling."""
    cleaned = strip_reasoning_blocks(text)
    candidate = extract_balanced_json_object(cleaned)
    return candidate if candidate is not None else cleaned


def _validate_pages(value: object, valid_page_numbers: set[int] | None) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise TypeError("evidence_pages must be a JSON array")
    pages: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError("evidence_pages must contain integers only")
        if item <= 0:
            raise ValueError("evidence_pages must contain positive page numbers")
        if valid_page_numbers is not None and item not in valid_page_numbers:
            raise ValueError(f"evidence page {item} is not in the supplied evidence")
        if item not in pages:
            pages.append(item)
    return tuple(pages)


def validate_structured_payload(
    payload: Mapping[str, object],
    *,
    valid_page_numbers: Iterable[int] | None = None,
) -> StructuredReaderOutput:
    """Validate exact keys, primitive types, page references, and abstention semantics."""
    keys = frozenset(payload)
    if keys != _REQUIRED_KEYS:
        missing = sorted(_REQUIRED_KEYS - keys)
        extra = sorted(keys - _REQUIRED_KEYS)
        raise ValueError(f"response keys must be exact; missing={missing}, extra={extra}")

    answer = payload["answer"]
    if not isinstance(answer, str):
        raise TypeError("answer must be a string")

    abstain = payload["abstain"]
    if not isinstance(abstain, bool):
        raise TypeError("abstain must be a boolean")

    confidence = payload["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise TypeError("confidence must be numeric")
    confidence_value = float(confidence)
    if not 0.0 <= confidence_value <= 1.0:
        raise ValueError("confidence must be between 0.0 and 1.0")

    allowed = None if valid_page_numbers is None else {int(page) for page in valid_page_numbers}
    pages = _validate_pages(payload["evidence_pages"], allowed)

    if abstain:
        if answer.strip() or pages:
            raise ValueError("abstaining responses must have an empty answer and no evidence pages")
    elif not answer.strip():
        raise ValueError("non-abstaining responses must contain a non-empty answer")

    return StructuredReaderOutput(
        answer=answer.strip(),
        evidence_pages=pages,
        confidence=confidence_value,
        abstain=abstain,
    )


def parse_structured_output(
    raw_text: str,
    *,
    valid_page_numbers: Iterable[int] | None = None,
) -> StructuredParseResult:
    """Parse a generation with balanced-object extraction and strict schema validation."""
    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string")
    cleaned = strip_reasoning_blocks(raw_text)
    candidate = extract_balanced_json_object(cleaned)
    if candidate is None:
        return StructuredParseResult(raw_text, None, None, "missing_json_object")
    try:
        loaded = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return StructuredParseResult(raw_text, candidate, None, f"invalid_json:{exc.msg}")
    if not isinstance(loaded, dict):
        return StructuredParseResult(raw_text, candidate, None, "json_root_not_object")
    try:
        parsed = validate_structured_payload(loaded, valid_page_numbers=valid_page_numbers)
    except (TypeError, ValueError) as exc:
        return StructuredParseResult(raw_text, candidate, None, f"schema_error:{exc}")
    return StructuredParseResult(raw_text, candidate, parsed, None)
