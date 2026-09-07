"""Score real retrieved-page predictions without changing the established judge contract."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import fmean, quantiles
from typing import Any

from lava.evaluation.metric import score_dataset
from lava.evaluation.schemas import PredictionRecord, ReferenceRecord
from lava.evaluation.semantic import (
    DurableSemanticJudge,
    GemmaDecision,
    ImmutableS3Objects,
    digest,
    encode,
    judge_contract,
)
from lava.evaluation.statistics import compare_document_scores
from lava.evaluation.submission_store import atomic_write, cached_source
from lava.readers.oracle_assets import document_aliases, parse_training_csv
from lava.readers.runtime_logging import RuntimeEventLogger
from lava.readers.schemas import ReaderTelemetry
from lava.readers.system import store_for, system_contract, validate_manifest, validate_prediction
from lava.retrieval.evaluation import pilot_sources


def score_predictions(
    manifest: dict[str, Any],
    inference: dict[str, Any],
    references: tuple[ReferenceRecord, ...],
    judge: Any,
) -> dict[str, Any]:
    """Evaluate all questions, including malformed responses and retrieval misses.

    References enter only here, after inference. Public output contains aliases,
    scores and counts, never source questions, answers, PDF IDs or page contents.
    """
    contract = manifest["contract"]
    requests = validate_manifest(manifest, contract)
    if (
        inference.get("schema_version") != 1
        or inference.get("contract_id") != contract["contract_id"]
        or inference.get("inputs_sha256") != manifest["inputs_sha256"]
        or len(inference.get("records", [])) != len(requests)
    ):
        raise ValueError("Complete matching inference is required before scoring")
    by_id = {reference.question_id: reference for reference in references}
    if len(by_id) != len(references) or set(by_id) != {r.question_id for r in requests}:
        raise ValueError("Reference and prediction question coverage differ")
    aliases = document_aliases(references)
    public_references: list[ReferenceRecord] = []
    predictions: list[PredictionRecord] = []
    telemetry: list[ReaderTelemetry] = []
    details: list[dict[str, Any]] = []
    for index, (request, record) in enumerate(zip(requests, inference["records"], strict=True), 1):
        if record.get("contract_id") != contract["contract_id"]:
            raise ValueError("Per-question checkpoint contract mismatch")
        prediction, measured = validate_prediction(request, record)
        reference = by_id[request.question_id]
        if (
            reference.document_id != request.document_id
            or reference.question != request.question
            or reference.language != request.language
            or reference.answer_format != request.answer_format
            or aliases[request.document_id] != request.document_alias
        ):
            raise ValueError("Reader request differs from its frozen reference")
        public_id = f"q-{index:02d}"
        public_references.append(
            reference.model_copy(
                update={"question_id": public_id, "document_id": request.document_alias}
            )
        )
        predictions.append(
            PredictionRecord(
                question_id=public_id,
                answer=prediction.answer,
                evidence_pages=prediction.evidence_pages,
            )
        )
        gold, predicted, available = (
            set(reference.evidence_pages),
            set(prediction.evidence_pages),
            set(request.available_pages),
        )
        details.append(
            {
                "question": public_id,
                "document": request.document_alias,
                "language": request.language,
                "answer_format": request.answer_format.value,
                "retrieval_recall": len(gold & available) / len(gold),
                "all_evidence_retrieved": gold.issubset(available),
                "evidence_precision": len(gold & predicted) / len(predicted) if predicted else 0.0,
                "evidence_recall": len(gold & predicted) / len(gold),
                "schema_valid": prediction.schema_valid,
                "abstain": prediction.abstain,
                "parser_error": prediction.parser_error_code,
                "input_pages": len(available),
            }
        )
        telemetry.append(measured)
    # Actual predicted pages are scored; gold pages are NEVER substituted.
    scores, metrics = score_dataset(tuple(public_references), tuple(predictions), judge=judge)
    for detail, score in zip(details, scores, strict=True):
        detail.update(
            answer_score=score.answer_score,
            evidence_f1=score.grounding_score,
            local_lava=score.overall_score,
        )
        detail["failure_category"] = (
            "invalid_response"
            if not detail["schema_valid"]
            else "missing_evidence"
            if not detail["all_evidence_retrieved"]
            else "answer_incomplete_or_incorrect"
            if score.answer_score < 1
            else "evidence_citation_incomplete"
            if score.grounding_score < 1
            else "full_credit"
        )
    generation = [row.generation_seconds for row in telemetry]
    total = [row.total_seconds for row in telemetry]
    return {
        "metrics": metrics,
        "per_question": details,
        "diagnostics": {
            "schema_valid_rate": fmean(float(row["schema_valid"]) for row in details),
            "abstention_rate": fmean(float(row["abstain"]) for row in details),
            "evidence_precision": fmean(row["evidence_precision"] for row in details),
            "evidence_recall": fmean(row["evidence_recall"] for row in details),
            "retrieval_recall": fmean(row["retrieval_recall"] for row in details),
            "all_evidence_retrieved": fmean(
                float(row["all_evidence_retrieved"]) for row in details
            ),
            "failure_counts": dict(Counter(row["failure_category"] for row in details)),
        },
        "runtime": {
            "generation_mean_seconds": fmean(generation),
            "generation_p50_seconds": quantiles(generation, n=100, method="inclusive")[49],
            "generation_p95_seconds": quantiles(generation, n=100, method="inclusive")[94],
            "reader_total_mean_seconds": fmean(total),
            "reader_total_p95_seconds": quantiles(total, n=100, method="inclusive")[94],
            "sum_reader_total_seconds": sum(total),
            "peak_allocated_gib": max(row.peak_cuda_memory_allocated_mib for row in telemetry)
            / 1024,
            "peak_reserved_gib": max(row.peak_cuda_memory_reserved_mib for row in telemetry) / 1024,
            "scope": "Per-question reader telemetry; excludes retrieval, provisioning and checkpoint I/O. First uncached question includes model load. Resume may span jobs.",
        },
    }


def publish_summary(root: Path, summary: dict[str, Any]) -> None:
    """Materialize verified public output; interruption is recovered by the same command."""
    path = root / "reports/system/summary.json"
    payload = encode(summary)
    atomic_write(path, payload)
    atomic_write(path.with_suffix(".sha256"), (digest(payload) + "\n").encode())


def load_summary(root: Path) -> dict[str, Any] | None:
    """Absent is pending, never a fabricated zero; stale or corrupt results are rejected."""
    path = root / "reports/system/summary.json"
    if not path.exists():
        return None
    payload = path.read_bytes()
    if digest(payload) != path.with_suffix(".sha256").read_text().strip():
        raise ValueError("System report checksum mismatch")
    summary = json.loads(payload)
    if (
        summary.get("contract") != system_contract(root)
        or summary.get("judge_contract") != judge_contract(root)
        or summary.get("scoring_source_sha256") != digest(Path(__file__).read_bytes())
    ):
        raise ValueError("System report is stale for current inference or scoring code")
    return summary


def evaluate_system(root: Path, s3: Any, bucket: str, logger: RuntimeEventLogger) -> dict[str, Any]:
    """Reuse accepted decisions and inference; no GPU resource can be created here."""
    contract = system_contract(root)
    store = store_for(s3, bucket, contract["contract_id"])
    manifest, inference = store.read("inputs.json"), store.read("inference.json")
    if manifest is None or inference is None:
        raise ValueError(
            "Complete retrieved-page inference is required; no answer scores published"
        )
    sources = pilot_sources(root, contract["retrieval"]["config"])
    references = parse_training_csv(
        cached_source(
            s3, bucket, root / "artifacts/retrieval/inputs/train.csv", sources["train.csv"], logger
        )
    )
    judging = judge_contract(root)
    protocol = json.loads((root / "configs/evaluation_protocol.lock.json").read_bytes())[
        "protocol_lock_id"
    ]
    judge = DurableSemanticJudge(
        judging,
        ImmutableS3Objects(
            s3, bucket, f"experiments/oracle-reader/evaluation/{protocol}/{judging['contract_id']}"
        ),
        GemmaDecision(judging["config"], logger, root / "artifacts/semantic_judge/model_cache"),
        logger,
    )
    # The exact same controls and prompt used for oracle scores remain mandatory.
    judge.validate()
    with logger.stage("system.semantic.scoring", heartbeat_seconds=15):
        result = score_predictions(manifest, inference, references, judge)
    baseline_path = (
        root
        / "reports/oracle_reader/runs"
        / contract["config"]["oracle_job"]
        / "semantic_summary.json"
    )
    baseline_bytes = baseline_path.read_bytes()
    if digest(baseline_bytes) != baseline_path.with_suffix(".sha256").read_text().strip():
        raise ValueError("Oracle comparison checksum mismatch")
    baseline = json.loads(baseline_bytes)
    if baseline["judge_contract"] != judging:
        raise ValueError("Oracle and retrieved results require the same semantic judge contract")
    paired = compare_document_scores(
        {name: values["overall"] for name, values in baseline["metrics"]["by_document"].items()},
        {name: values["overall"] for name, values in result["metrics"]["by_document"].items()},
        seed=20260902,
    )
    summary = {
        "schema_version": 1,
        "status": "Retrieved-evidence pilot scored",
        "split": "training_diagnostic",
        "official_server_parity": False,
        "official_server_score": None,
        "contract": contract,
        "judge_contract": judging,
        "scoring_source_sha256": digest(Path(__file__).read_bytes()),
        "input_sha256": manifest["inputs_sha256"],
        "inference_sha256": digest(encode(inference)),
        "oracle_summary_sha256": digest(baseline_bytes),
        "oracle_question_micro": baseline["metrics"]["question_micro"],
        "paired_document_comparison": paired,
        "question_mean_delta": result["metrics"]["question_micro"]["overall"]
        - baseline["metrics"]["question_micro"]["overall"],
        "limitations": "All 16 previously examined training questions from five PDFs; not held-out performance, a leaderboard result, or evidence of model superiority. Confidence intervals are exploratory. No fine-tuning was performed.",
        **result,
    }
    key = f"evaluation/{digest(encode(summary))}.json"
    store.write(key, summary)
    if store.read(key) != summary:
        raise ValueError("System scoring report failed durable read-back")
    publish_summary(root, summary)
    logger.emit(
        "system.evaluation.completed",
        question_count=16,
        document_count=5,
        new_decisions=judge.new_decisions,
        reused_decisions=judge.reused_decisions,
    )
    return summary
