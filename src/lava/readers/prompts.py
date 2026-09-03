"""Versioned answer-grounding prompts for multilingual document reading."""

from __future__ import annotations

from collections.abc import Sequence

from lava.evaluation.schemas import AnswerFormat

PROMPT_VERSION = "oracle-reader-json-v3"
SYSTEM_INSTRUCTION = (
    "You are an evidence-grounded multilingual document reader. Use only the supplied "
    "pages. Never use outside knowledge. Return only the requested JSON object. Do not "
    "reveal hidden reasoning or add markdown."
)

_FORMAT_GUIDANCE = {
    AnswerFormat.STRING: (
        "Return a concise string in the question language. Preserve source terminology."
    ),
    AnswerFormat.NUMBER: (
        "Return a number string. Preserve a unit only when the question or source requires it."
    ),
    AnswerFormat.UNORDERED_LIST: (
        "Return a JSON array of concise strings. Include each distinct item once; order is irrelevant."
    ),
    AnswerFormat.ORDERED_LIST: (
        "Return a JSON array of concise strings in the order required by the question."
    ),
}


def build_reader_instruction(
    *,
    question: str,
    language: str,
    answer_format: AnswerFormat,
    available_pages: Sequence[int],
) -> str:
    """Build a strict, answer-blind instruction for one oracle-evidence example."""
    if not question.strip():
        raise ValueError("Question cannot be blank")
    pages = tuple(sorted({int(page) for page in available_pages}))
    if not pages or any(page < 1 for page in pages):
        raise ValueError("Available pages must be positive one-indexed integers")
    page_text = ", ".join(str(page) for page in pages)
    return (
        "Task: answer the question using only the supplied evidence pages.\n"
        f"Question language: {language}\n"
        f"Required answer format: {answer_format.value}\n"
        f"Available one-indexed evidence pages: [{page_text}]\n"
        f"Answer rule: {_FORMAT_GUIDANCE[answer_format]}\n\n"
        "Evidence rule: cite only the minimum supplied page set that directly supports the answer. "
        "If the evidence does not support an answer, set abstain=true, answer to an empty string "
        "or empty array, confidence=0, and evidence_pages=[]. Confidence is the probability that "
        "the answer and cited pages are both correct.\n\n"
        "Return exactly this JSON schema and no other keys:\n"
        '{"answer": <string or array>, "evidence_pages": [<positive integers>], '
        '"confidence": <number from 0 to 1>, "abstain": <boolean>}\n\n'
        f"Question:\n{question.strip()}"
    )
