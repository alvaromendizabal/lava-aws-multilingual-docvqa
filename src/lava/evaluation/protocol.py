"""Freeze a private, nested document-isolated LAVA evaluation protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

from lava.evaluation.normalization import parse_evidence_pages
from lava.evaluation.schemas import AnswerFormat, ReferenceRecord
from lava.evaluation.splits import build_nested_leave_one_document_out_folds

_REQUIRED_COLUMNS = {
    "id",
    "file_id",
    "question",
    "answer_format",
    "answer",
    "evidence_page_number",
    "language",
}
_CODE_PATHS = (
    "src/lava/evaluation/schemas.py",
    "src/lava/evaluation/normalization.py",
    "src/lava/evaluation/judges.py",
    "src/lava/evaluation/matching.py",
    "src/lava/evaluation/metric.py",
    "src/lava/evaluation/retrieval.py",
    "src/lava/evaluation/splits.py",
    "src/lava/evaluation/statistics.py",
    "src/lava/evaluation/protocol.py",
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_records(payload: bytes) -> tuple[ReferenceRecord, ...]:
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    columns = set(reader.fieldnames or ())
    missing = sorted(_REQUIRED_COLUMNS - columns)
    if missing:
        message = f"train.csv is missing required columns: {missing}"
        raise ValueError(message)
    records: list[ReferenceRecord] = []
    for row_number, row in enumerate(reader, start=2):
        try:
            record = ReferenceRecord(
                question_id=row["id"],
                document_id=row["file_id"],
                question=row["question"],
                answer_format=AnswerFormat.from_raw(row["answer_format"]),
                answer=row["answer"],
                evidence_pages=parse_evidence_pages(row["evidence_page_number"]),
                language=row["language"],
            )
        except (KeyError, TypeError, ValueError) as error:
            message = f"Invalid train.csv row {row_number}: {error}"
            raise ValueError(message) from error
        records.append(record)
    if len({record.question_id for record in records}) != len(records):
        message = "train.csv question IDs are not unique"
        raise ValueError(message)
    return tuple(records)


def _private_reference_jsonl(records: tuple[ReferenceRecord, ...]) -> bytes:
    lines = [
        json.dumps(record.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        for record in sorted(records, key=lambda item: item.question_id)
    ]
    return ("\n".join(lines) + "\n").encode()


def _private_fold_payload(
    records: tuple[ReferenceRecord, ...],
) -> tuple[bytes, list[dict[str, Any]]]:
    folds = build_nested_leave_one_document_out_folds(records)
    document_aliases = {
        document_id: f"doc-{index:02d}"
        for index, document_id in enumerate(
            sorted({record.document_id for record in records}),
            start=1,
        )
    }
    private_payload = {
        "strategy": "nested-leave-one-document-out",
        "outer_folds": [fold.model_dump(mode="json") for fold in folds],
    }
    public_folds: list[dict[str, Any]] = []
    record_by_id = {record.question_id: record for record in records}
    for fold in folds:
        validation_records = [
            record_by_id[question_id] for question_id in fold.validation_question_ids
        ]
        public_folds.append(
            {
                "fold_id": fold.fold_id,
                "validation_document_alias": document_aliases[fold.validation_document_id],
                "training_document_count": len(fold.training_document_ids),
                "training_question_count": len(fold.training_question_ids),
                "validation_question_count": len(fold.validation_question_ids),
                "inner_fold_count": len(fold.inner_folds),
                "validation_language_counts": dict(
                    sorted(Counter(record.language for record in validation_records).items())
                ),
                "validation_answer_format_counts": dict(
                    sorted(
                        Counter(record.answer_format.value for record in validation_records).items()
                    )
                ),
                "validation_evidence_cardinality_counts": dict(
                    sorted(
                        Counter(
                            str(len(record.evidence_pages)) for record in validation_records
                        ).items()
                    )
                ),
            }
        )
    return _canonical_json_bytes(private_payload), public_folds


def _verify_audit(audit_path: Path) -> tuple[dict[str, Any], str]:
    summary = json.loads(audit_path.read_text(encoding="utf-8"))
    profiles = {Path(profile["s3_key"]).name: profile for profile in summary["csv_profiles"]}
    if summary["pdf_success_count"] != 205 or summary["pdf_error_count"] != 0:
        message = "The complete 205-PDF audit has not passed"
        raise ValueError(message)
    if profiles["train.csv"]["selected_columns"]["answer"] != "answer":
        message = "The corrected training answer-column audit is missing"
        raise ValueError(message)
    if profiles["test.csv"]["selected_columns"]["answer"] is not None:
        message = "The test set must remain unlabeled"
        raise ValueError(message)
    return summary, _sha256_file(audit_path)


def _verify_training_pdfs(
    s3_client: Any,
    bucket: str,
    records: tuple[ReferenceRecord, ...],
) -> None:
    for document_id in sorted({record.document_id for record in records}):
        key = f"raw/kaggle/train_pdfs/train_pdfs/{document_id}.pdf"
        try:
            response = s3_client.head_object(Bucket=bucket, Key=key)
        except ClientError as error:
            message = f"Training PDF is unavailable: s3://{bucket}/{key}"
            raise RuntimeError(message) from error
        if int(response["ContentLength"]) < 1:
            message = f"Training PDF is empty: s3://{bucket}/{key}"
            raise ValueError(message)


def _upload_bytes(
    s3_client: Any,
    *,
    bucket: str,
    key: str,
    payload: bytes,
    content_type: str,
) -> dict[str, str]:
    digest = _sha256_bytes(payload)
    response = s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=payload,
        ContentType=content_type,
        ServerSideEncryption="AES256",
        Metadata={"sha256": digest, "project": "lava-docvqa"},
    )
    head = s3_client.head_object(Bucket=bucket, Key=key)
    if int(head["ContentLength"]) != len(payload):
        message = f"S3 length verification failed for s3://{bucket}/{key}"
        raise RuntimeError(message)
    if head["Metadata"].get("sha256") != digest:
        message = f"S3 SHA-256 verification failed for s3://{bucket}/{key}"
        raise RuntimeError(message)
    return {
        "key": key,
        "sha256": digest,
        "version_id": str(response.get("VersionId", "")),
    }


def freeze_protocol(
    *,
    bucket: str,
    region: str,
    config_path: Path,
    audit_path: Path,
) -> dict[str, Any]:
    """Create nested folds, immutable provenance locks, and public reports."""
    audit_summary, audit_file_sha256 = _verify_audit(audit_path)
    s3_client = boto3.client("s3", region_name=region)
    try:
        train_payload = s3_client.get_object(
            Bucket=bucket,
            Key="raw/kaggle/train.csv",
        )["Body"].read()
    except ClientError as error:
        message = f"Unable to read s3://{bucket}/raw/kaggle/train.csv"
        raise RuntimeError(message) from error
    records = _load_records(train_payload)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_questions = int(config["expected_question_count"])
    expected_documents = int(config["expected_document_count"])
    if len(records) != expected_questions:
        message = f"Expected {expected_questions} labeled questions, found {len(records)}"
        raise ValueError(message)
    document_count = len({record.document_id for record in records})
    if document_count != expected_documents:
        message = f"Expected {expected_documents} labeled documents, found {document_count}"
        raise ValueError(message)
    _verify_training_pdfs(s3_client, bucket, records)

    private_references = _private_reference_jsonl(records)
    private_folds, public_folds = _private_fold_payload(records)
    code_hashes = {path: _sha256_file(Path(path)) for path in _CODE_PATHS}
    source_hashes: dict[str, str] = {
        "train_csv_sha256": _sha256_bytes(train_payload),
        "data_audit_file_sha256": audit_file_sha256,
        "data_audit_source_sha256": str(audit_summary["audit_source_sha256"]),
    }
    raw_manifest = Path("reports/raw_data_manifest_summary.json")
    if raw_manifest.exists():
        source_hashes[str(raw_manifest)] = _sha256_file(raw_manifest)

    git_head = _git_head()
    lock_core = {
        "schema_version": 1,
        "protocol_name": config["protocol_name"],
        "competition": "lava-challenge-2026",
        "source_git_commit_sha": git_head,
        "split_strategy": "nested-leave-one-document-out",
        "expected_question_count": expected_questions,
        "expected_document_count": expected_documents,
        "outer_fold_count": expected_documents,
        "inner_fold_count_per_outer_fold": expected_documents - 1,
        "random_seed": int(config["random_seed"]),
        "official_metric_spec": config["official_metric_spec"],
        "retrieval_metric_spec": config["retrieval_metric_spec"],
        "selection_policy": config["selection_policy"],
        "competition_constraints": config["competition_constraints"],
        "judge_specification_boundary": config["judge_specification_boundary"],
        "source_hashes": source_hashes,
        "code_hashes": code_hashes,
        "config_sha256": _sha256_file(config_path),
        "private_reference_sha256": _sha256_bytes(private_references),
        "private_fold_manifest_sha256": _sha256_bytes(private_folds),
    }
    lock_id = _sha256_bytes(_canonical_json_bytes(lock_core))
    lock_payload = {**lock_core, "protocol_lock_id": lock_id}
    lock_bytes = json.dumps(lock_payload, indent=2, sort_keys=True).encode()

    public_summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_git_commit_sha": git_head,
        "protocol_lock_id": lock_id,
        "question_count": len(records),
        "document_count": document_count,
        "outer_fold_count": expected_documents,
        "inner_fold_count_total": expected_documents * (expected_documents - 1),
        "language_counts": dict(sorted(Counter(record.language for record in records).items())),
        "answer_format_counts": dict(
            sorted(Counter(record.answer_format.value for record in records).items())
        ),
        "evidence_cardinality_counts": dict(
            sorted(Counter(str(len(record.evidence_pages)) for record in records).items())
        ),
        "folds": public_folds,
        "official_metric_spec": config["official_metric_spec"],
        "retrieval_metric_spec": config["retrieval_metric_spec"],
        "judge_specification_boundary": config["judge_specification_boundary"],
        "privacy_boundary": (
            "Questions, answers, actual document IDs, and per-question fold assignments "
            "are stored only under ignored local artifacts and private S3 paths."
        ),
    }
    summary_bytes = json.dumps(public_summary, indent=2, sort_keys=True).encode()
    report_lines = [
        "# LAVA Evaluation Protocol",
        "",
        f"Protocol lock: `{lock_id}`",
        "",
        "## Design",
        "",
        "- Five outer leave-one-document-out folds estimate generalization.",
        "- Four inner document folds inside each outer training set govern any label-based tuning.",
        "- No document or question crosses an outer or inner partition.",
        "- Question-micro and document-macro scores are both mandatory.",
        "- All five outer document scores are displayed.",
        "- Architecture development uses external public data first.",
        "- LAVA test or leaderboard feedback cannot become a tuning loop.",
        "",
        "## Published challenge metric",
        "",
        "Answer correctness and exact evidence-page grounding receive equal weight. String and ",
        "number answers use a Gemma-3 1B semantic judge; unordered lists use optimal one-to-one ",
        "semantic matching followed by F1; ordered lists use semantic LCS; evidence pages use ",
        "exact set F1.",
        "",
        "## Retrieval diagnostics",
        "",
        "Recall@k, all-evidence success@k, MRR@k, MAP@k, and nDCG@k are frozen at page ",
        "budgets 1, 2, 3, 5, and 10. These diagnose retrieval independently of reader quality.",
        "",
        "## Public outer-fold summary",
        "",
    ]
    for fold in public_folds:
        report_lines.append(
            f"- `{fold['fold_id']}`: validate `{fold['validation_document_alias']}` "
            f"({fold['validation_question_count']} questions); train on "
            f"{fold['training_document_count']} documents / "
            f"{fold['training_question_count']} questions; "
            f"{fold['inner_fold_count']} inner folds."
        )
    report_lines.extend(
        [
            "",
            "## Evaluator boundary",
            "",
            "The organizers publish Gemma-3 1B and the metric formulas but not the exact ",
            "judge prompt, decoding configuration, or checkpoint revision. This local evaluator ",
            "is therefore official-structure-compatible, not claimed server-identical. The ",
            "oracle-reader phase will pin a Gemma runtime and quantify judge sensitivity.",
            "",
        ]
    )
    report_bytes = "\n".join(report_lines).encode()

    private_directory = Path("artifacts/evaluation_protocol")
    _atomic_write(private_directory / "reference_records.jsonl", private_references)
    _atomic_write(private_directory / "nested_fold_manifest.json", private_folds)
    _atomic_write(Path("configs/evaluation_protocol.lock.json"), lock_bytes)
    _atomic_write(
        Path("reports/evaluation/evaluation_protocol_summary.json"),
        summary_bytes,
    )
    _atomic_write(Path("reports/evaluation/EVALUATION_PROTOCOL.md"), report_bytes)

    immutable_prefix = f"splits/evaluation-protocol/v1/{lock_id}"
    uploads = (
        (
            f"{immutable_prefix}/reference_records.jsonl",
            private_references,
            "application/x-ndjson",
        ),
        (
            f"{immutable_prefix}/nested_fold_manifest.json",
            private_folds,
            "application/json",
        ),
        (
            f"{immutable_prefix}/protocol_lock.json",
            lock_bytes,
            "application/json",
        ),
        (
            f"{immutable_prefix}/evaluation_protocol_summary.json",
            summary_bytes,
            "application/json",
        ),
        (
            "splits/evaluation-protocol/latest/protocol_lock.json",
            lock_bytes,
            "application/json",
        ),
        (
            "reports/evaluation-protocol/latest/evaluation_protocol_summary.json",
            summary_bytes,
            "application/json",
        ),
    )
    receipts = [
        _upload_bytes(
            s3_client,
            bucket=bucket,
            key=key,
            payload=payload,
            content_type=content_type,
        )
        for key, payload, content_type in uploads
    ]
    _atomic_write(
        private_directory / "s3_upload_receipts.json",
        json.dumps(receipts, indent=2, sort_keys=True).encode(),
    )
    return public_summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/evaluation_protocol.json"),
    )
    parser.add_argument(
        "--audit-summary",
        type=Path,
        default=Path("reports/data_audit/data_audit_summary_full.json"),
    )
    return parser.parse_args()


def main() -> None:
    """Freeze the protocol and print only the sanitized public summary."""
    arguments = _parse_args()
    summary = freeze_protocol(
        bucket=arguments.bucket,
        region=arguments.region,
        config_path=arguments.config,
        audit_path=arguments.audit_summary,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("EVALUATION_PROTOCOL_FROZEN")


if __name__ == "__main__":
    main()
