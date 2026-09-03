"""Run, score, and persist oracle-evidence reader experiments."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

import boto3

from lava.evaluation.judges import NormalizedExactJudge
from lava.evaluation.metric import score_question, set_f1
from lava.evaluation.schemas import PredictionRecord, ReferenceRecord
from lava.readers.oracle_assets import load_oracle_examples
from lava.readers.prompts import PROMPT_VERSION
from lava.readers.reader_factory import build_reader
from lava.readers.schemas import BenchmarkRecord, ResolvedModel


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _checksum_sha256(payload: bytes) -> str:
    return base64.b64encode(hashlib.sha256(payload).digest()).decode()


def _safe_mean(values: list[float]) -> float:
    return fmean(values) if values else 0.0


def _summary(records: tuple[BenchmarkRecord, ...]) -> dict[str, Any]:
    by_document: dict[str, list[float]] = defaultdict(list)
    for record in records:
        by_document[record.document_alias].append(record.normalized_exact_answer_score)
    return {
        "record_count": len(records),
        "normalized_exact_answer_micro": _safe_mean(
            [record.normalized_exact_answer_score for record in records]
        ),
        "normalized_exact_answer_document_macro": _safe_mean(
            [_safe_mean(values) for values in by_document.values()]
        ),
        "self_grounding_f1_micro": _safe_mean([record.self_grounding_f1 for record in records]),
        "oracle_fixed_overall_diagnostic": _safe_mean(
            [record.oracle_fixed_overall_diagnostic for record in records]
        ),
        "schema_valid_rate": _safe_mean(
            [float(record.prediction.schema_valid) for record in records]
        ),
        "abstention_rate": _safe_mean([float(record.prediction.abstain) for record in records]),
        "mean_model_load_seconds": _safe_mean(
            [record.telemetry.model_load_seconds for record in records]
        ),
        "mean_preprocessing_seconds": _safe_mean(
            [record.telemetry.preprocessing_seconds for record in records]
        ),
        "mean_generation_seconds": _safe_mean(
            [record.telemetry.generation_seconds for record in records]
        ),
        "mean_total_seconds": _safe_mean([record.telemetry.total_seconds for record in records]),
        "mean_prompt_tokens": _safe_mean(
            [float(record.telemetry.prompt_tokens) for record in records]
        ),
        "mean_generated_tokens": _safe_mean(
            [float(record.telemetry.generated_tokens) for record in records]
        ),
        "max_peak_cuda_memory_allocated_mib": max(
            (record.telemetry.peak_cuda_memory_allocated_mib for record in records),
            default=0.0,
        ),
        "max_peak_cuda_memory_reserved_mib": max(
            (record.telemetry.peak_cuda_memory_reserved_mib for record in records),
            default=0.0,
        ),
        "language_counts": dict(sorted(Counter(record.language for record in records).items())),
        "answer_format_counts": dict(
            sorted(Counter(record.answer_format.value for record in records).items())
        ),
        "parser_error_counts": dict(
            sorted(
                Counter(record.prediction.parser_error_code or "none" for record in records).items()
            )
        ),
        "document_scores": {
            alias: _safe_mean(values) for alias, values in sorted(by_document.items())
        },
        "metric_boundary": (
            "Normalized-exact diagnostics only; not the organizer-server semantic score."
        ),
    }


def _put_json(s3: Any, *, bucket: str, key: str, payload: bytes, content_type: str) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=payload,
        ContentType=content_type,
        ChecksumSHA256=_checksum_sha256(payload),
        Metadata={"sha256": _sha256(payload)},
    )


def run_oracle_benchmark(
    *,
    bucket: str,
    region: str,
    manifest_s3_uri: str,
    output_s3_prefix: str,
    protocol_lock_id: str,
    model_spec: ResolvedModel,
    experiment_id: str,
    limit: int,
) -> dict[str, Any]:
    """Run a bounded oracle benchmark and upload private and public artifacts."""
    s3 = boto3.client("s3", region_name=region)
    bucket_prefix = f"s3://{bucket}/"
    if not manifest_s3_uri.startswith(bucket_prefix):
        raise ValueError("Oracle manifest must be in the configured private project bucket")
    manifest_key = manifest_s3_uri.removeprefix(bucket_prefix)
    manifest_response = s3.get_object(Bucket=bucket, Key=manifest_key)
    manifest_payload = manifest_response["Body"].read()
    manifest_sha = _sha256(manifest_payload)
    metadata_sha = manifest_response.get("Metadata", {}).get("sha256")
    if metadata_sha and metadata_sha != manifest_sha:
        raise ValueError("Oracle manifest SHA-256 metadata does not match its bytes")
    examples = load_oracle_examples(manifest_payload)[:limit]
    if not examples:
        raise ValueError("Oracle manifest contained no examples")
    if any(example.protocol_lock_id != protocol_lock_id for example in examples):
        raise ValueError("Oracle examples do not match the frozen evaluation protocol")
    reader = build_reader(model_spec, region=region)
    judge = NormalizedExactJudge()
    git_sha = os.environ.get("LAVA_GIT_COMMIT_SHA") or _git_sha()
    records: list[BenchmarkRecord] = []
    for example in examples:
        prediction, telemetry = reader.predict(example)
        reference = ReferenceRecord(
            question_id=example.question_id,
            document_id=example.document_id,
            question=example.question,
            answer_format=example.answer_format,
            answer=example.answer,
            evidence_pages=example.evidence_pages,
            language=example.language,
        )
        oracle_prediction = PredictionRecord(
            question_id=example.question_id,
            answer=prediction.answer,
            evidence_pages=example.evidence_pages,
        )
        score = score_question(reference, oracle_prediction, judge=judge)
        records.append(
            BenchmarkRecord(
                experiment_id=experiment_id,
                protocol_lock_id=protocol_lock_id,
                asset_manifest_sha256=manifest_sha,
                git_commit_sha=git_sha,
                prompt_version=PROMPT_VERSION,
                model_key=model_spec.model_key,
                model_id=model_spec.model_id,
                model_revision=model_spec.revision,
                question_id=example.question_id,
                document_alias=example.document_alias,
                language=example.language,
                answer_format=example.answer_format,
                gold_evidence_pages=example.evidence_pages,
                prediction=prediction,
                telemetry=telemetry,
                normalized_exact_answer_score=score.answer_score,
                self_grounding_f1=set_f1(
                    example.evidence_pages,
                    prediction.evidence_pages,
                ),
                oracle_fixed_overall_diagnostic=(score.answer_score + 1.0) / 2.0,
            )
        )
    frozen_records = tuple(records)
    private_payload = (
        "\n".join(
            json.dumps(record.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
            for record in frozen_records
        )
        + "\n"
    ).encode()
    public_summary = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "experiment_id": experiment_id,
        "protocol_lock_id": protocol_lock_id,
        "asset_manifest_sha256": manifest_sha,
        "git_commit_sha": git_sha,
        "prompt_version": PROMPT_VERSION,
        "model_key": model_spec.model_key,
        "model_id": model_spec.model_id,
        "model_revision": model_spec.revision,
        "input_mode": model_spec.input_mode.value,
        "generation": model_spec.generation.model_dump(mode="json"),
        **_summary(frozen_records),
    }
    output_prefix = output_s3_prefix.removeprefix(bucket_prefix).rstrip("/")
    _put_json(
        s3,
        bucket=bucket,
        key=f"{output_prefix}/private_records.jsonl",
        payload=private_payload,
        content_type="application/x-ndjson",
    )
    summary_payload = (json.dumps(public_summary, indent=2, sort_keys=True) + "\n").encode()
    _put_json(
        s3,
        bucket=bucket,
        key=f"{output_prefix}/public_summary.json",
        payload=summary_payload,
        content_type="application/json",
    )
    model_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "public_summary.json").write_bytes(summary_payload)
    return public_summary
