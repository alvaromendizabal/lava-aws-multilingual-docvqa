"""Unicode-safe normalization and source-value parsers for LAVA evaluation."""

from __future__ import annotations

import ast
import json
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Any

_WHITESPACE_RE = re.compile(r"\s+")
_NUMERIC_RE = re.compile(
    r"^[\s\u00a0]*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"[\s\u00a0]*(%|％)?[\s\u00a0]*$",
)
_BULLET_RE = re.compile(r"(?:^|\n)\s*(?:[-*•·]|\d+[.)])\s+")


def normalize_text(value: str) -> str:
    """Apply conservative Unicode normalization for deterministic comparisons."""
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.replace("\u00a0", " ").strip().casefold()
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    return normalized.strip(" \t\r\n\"'“”‘’.,;:。、「」『』()[]{}")


def parse_decimal(value: str) -> Decimal | None:
    """Parse a standalone number, including grouped digits and percentages."""
    normalized = unicodedata.normalize("NFKC", value).replace(",", "")
    match = _NUMERIC_RE.fullmatch(normalized)
    if match is None:
        return None
    try:
        number = Decimal(match.group(1))
    except InvalidOperation:
        return None
    if match.group(2) is not None:
        number /= Decimal(100)
    return number.normalize()


def parse_evidence_pages(value: str | int | list[Any] | tuple[Any, ...]) -> tuple[int, ...]:
    """Parse evidence pages from JSON, Python literals, scalars, or delimited text."""
    parsed: Any = value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ()
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(stripped)
                break
            except (ValueError, SyntaxError, json.JSONDecodeError):
                parsed = stripped
        if isinstance(parsed, str):
            parsed = re.findall(r"\d+", parsed)

    if isinstance(parsed, int):
        candidates = [parsed]
    elif isinstance(parsed, (list, tuple, set)):
        candidates = list(parsed)
    else:
        message = f"Unable to parse evidence pages from {value!r}"
        raise TypeError(message)

    pages: list[int] = []
    for candidate in candidates:
        try:
            page = int(candidate)
        except (TypeError, ValueError) as error:
            message = f"Invalid evidence page {candidate!r}"
            raise ValueError(message) from error
        if page < 1:
            message = f"Evidence page must be positive, received {page}"
            raise ValueError(message)
        pages.append(page)
    return tuple(sorted(set(pages)))


def parse_list_answer(value: str) -> tuple[str, ...]:
    """Parse JSON/Python lists and conservative line- or delimiter-separated answers."""
    stripped = value.strip()
    if not stripped:
        return ()

    parsed: Any = stripped
    if stripped[0] in "[(" and stripped[-1] in ")]":
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(stripped)
                break
            except (ValueError, SyntaxError, json.JSONDecodeError):
                parsed = stripped

    if isinstance(parsed, (list, tuple)):
        return tuple(str(item).strip() for item in parsed if str(item).strip())

    bullet_parts = [part.strip() for part in _BULLET_RE.split(stripped) if part.strip()]
    if len(bullet_parts) > 1:
        return tuple(bullet_parts)

    for separator in ("\n", ";", "；", "|"):
        parts = tuple(part.strip() for part in stripped.split(separator) if part.strip())
        if len(parts) > 1:
            return parts

    return (stripped,)
