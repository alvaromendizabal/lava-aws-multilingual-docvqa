"""Independent validation of retained reader predictions and diagnostic scores."""

from __future__ import annotations

import hashlib

from lava.evaluation.judges import NormalizedExactJudge
from lava.evaluation.metric import score_question, set_f1
from lava.evaluation.schemas import PredictionRecord, ReferenceRecord
from lava.readers.parsing import ReaderOutputError, parse_reader_response
from lava.readers.schemas import BenchmarkRecord, OracleExample, ReaderPrediction


def verify_benchmark_record(record: BenchmarkRecord, example: OracleExample, raw_text: str) -> None:
    """Reparse and rescore an exact generation against its frozen reference."""
    if record.question_id != example.question_id:
        raise RuntimeError("Benchmark question identity mismatch")
    for key in ("document_alias", "language", "answer_format"):
        if getattr(record, key) != getattr(example, key):
            raise RuntimeError(f"Benchmark question alignment mismatch: {key}")
    if record.gold_evidence_pages != example.evidence_pages:
        raise RuntimeError("Benchmark evidence alignment mismatch")
    try:
        prediction = parse_reader_response(
            question_id=example.question_id,
            answer_format=example.answer_format,
            raw_response=raw_text,
            allowed_pages=example.evidence_pages,
        )
    except ReaderOutputError as error:
        prediction = ReaderPrediction(
            question_id=example.question_id,
            answer_format=example.answer_format,
            answer="",
            evidence_pages=(),
            confidence=0.0,
            abstain=True,
            schema_valid=False,
            parser_error_code=error.code,
            raw_response_sha256=hashlib.sha256(raw_text.encode()).hexdigest(),
        )
    if record.prediction != prediction:
        raise RuntimeError("Benchmark parsed prediction differs from raw generation")
    reference = ReferenceRecord(
        question_id=example.question_id,
        document_id=example.document_id,
        question=example.question,
        answer_format=example.answer_format,
        answer=example.answer,
        evidence_pages=example.evidence_pages,
        language=example.language,
    )
    score = score_question(
        reference,
        PredictionRecord(
            question_id=example.question_id,
            answer=prediction.answer,
            evidence_pages=example.evidence_pages,
        ),
        judge=NormalizedExactJudge(),
    )
    if (
        record.normalized_exact_answer_score != score.answer_score
        or record.self_grounding_f1 != set_f1(example.evidence_pages, prediction.evidence_pages)
        or record.oracle_fixed_overall_diagnostic != (score.answer_score + 1.0) / 2.0
    ):
        raise RuntimeError("Benchmark score differs from independently recomputed score")
