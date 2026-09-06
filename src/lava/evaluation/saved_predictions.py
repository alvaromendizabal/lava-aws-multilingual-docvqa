"""Evaluate verified saved reader outputs without creating reader compute."""

from __future__ import annotations

import io
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, quantiles
from typing import Any

from lava.evaluation.metric import score_dataset
from lava.evaluation.schemas import PredictionRecord, ReferenceRecord
from lava.evaluation.semantic import DurableSemanticJudge, digest, encode
from lava.readers.artifact_gate import verify_training_model_artifact
from lava.readers.evaluation_contract import load_evaluation_contract, validate_evaluation_manifest
from lava.readers.sagemaker_artifacts import canonical_output_s3_prefix, split_s3_uri
from lava.readers.schemas import BenchmarkRecord


class ArtifactSnapshot:
    """Reuse the exact bytes read by the artifact gate for downstream evaluation."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self.cache: dict[str, tuple[dict[str, Any], bytes]] = {}

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        key = json.dumps(kwargs, sort_keys=True)
        if key not in self.cache:
            response = self.client.get_object(**kwargs)
            with response.pop("Body") as body:
                payload = body.read()
            self.cache[key] = response, payload
        metadata, payload = self.cache[key]
        return {**metadata, "Body": io.BytesIO(payload)}

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        return dict(self.client.list_objects_v2(**kwargs))

    def get_paginator(self, name: str) -> Any:
        return self.client.get_paginator(name)


@dataclass(frozen=True)
class SavedRun:
    """Private scored inputs and public source identities from one accepted job."""

    job_name: str
    bucket: str
    records: tuple[BenchmarkRecord, ...]
    references: tuple[ReferenceRecord, ...]
    predictions: tuple[PredictionRecord, ...]
    source: dict[str, str]


def load_saved_run(root: Path, sagemaker: Any, s3: Any, job_name: str) -> SavedRun:
    """Audit once, then derive metrics from that same in-memory S3 snapshot."""
    description = sagemaker.describe_training_job(TrainingJobName=job_name)
    if description.get("HyperParameters", {}).get("mode") != "benchmark":
        raise ValueError("Performance evaluation requires a complete benchmark, not a smoke run")
    snapshot = ArtifactSnapshot(s3)
    verify_training_model_artifact(
        sagemaker_client=sagemaker, s3_client=snapshot, job_name=job_name
    )
    params = description["HyperParameters"]
    canonical = canonical_output_s3_prefix(description, job_name=job_name)
    if canonical is None or params.get("output_s3_prefix", canonical) != canonical:
        raise ValueError("Reader output location differs from the verified canonical prefix")
    bucket, prefix = split_s3_uri(canonical)
    contract = load_evaluation_contract(root)
    manifest_bucket, manifest_key = split_s3_uri(params["manifest_s3_uri"])
    with snapshot.get_object(
        Bucket=manifest_bucket, Key=manifest_key, VersionId=contract["private_manifest_version_id"]
    )["Body"] as body:
        examples = validate_evaluation_manifest(body.read(), contract)
    with snapshot.get_object(Bucket=bucket, Key=prefix + "/private_records.jsonl")["Body"] as body:
        payload = body.read()
    with snapshot.get_object(Bucket=bucket, Key=prefix + "/public_summary.json")["Body"] as body:
        summary_payload = body.read()
    records = tuple(BenchmarkRecord.model_validate_json(line) for line in payload.splitlines())
    aliases = {record.question_id: record.document_alias for record in records}
    references = tuple(
        ReferenceRecord(
            question_id=e.question_id,
            document_id=aliases[e.question_id],
            question=e.question,
            answer_format=e.answer_format,
            answer=e.answer,
            evidence_pages=e.evidence_pages,
            language=e.language,
        )
        for e in examples
    )
    predictions = tuple(
        PredictionRecord(
            question_id=r.question_id,
            answer=r.prediction.answer,
            evidence_pages=r.prediction.evidence_pages,
        )
        for r in records
    )
    return SavedRun(
        job_name,
        bucket,
        records,
        references,
        predictions,
        {
            "job_name": job_name,
            "model_key": params["model_key"],
            "source_records_sha256": digest(payload),
            "source_summary_sha256": digest(summary_payload),
            "asset_manifest_sha256": contract["private_manifest_sha256"],
            "protocol_lock_id": contract["protocol_lock_id"],
        },
    )


def supporting_metrics(run: SavedRun) -> dict[str, Any]:
    """Measure evidence, output reliability, and observed inference latency separately."""
    records = run.records
    latency = [r.telemetry.generation_seconds for r in records]
    if not records:
        raise ValueError("Cannot evaluate an empty prediction set")
    precision, recall, exact_pages = [], [], []
    by_document: dict[str, list[float]] = defaultdict(list)
    for r in records:
        gold, predicted = set(r.gold_evidence_pages), set(r.prediction.evidence_pages)
        overlap = len(gold & predicted)
        precision.append(overlap / len(predicted) if predicted else 0.0)
        recall.append(overlap / len(gold))
        exact_pages.append(float(gold == predicted))
        by_document[r.document_alias].append(r.self_grounding_f1)
    return {
        "schema_version": 1,
        "source": run.source,
        "question_count": len(records),
        "grounding": {
            "question_average_f1": fmean(r.self_grounding_f1 for r in records),
            "document_average_f1": fmean(fmean(values) for values in by_document.values()),
            "question_average_precision": fmean(precision),
            "question_average_recall": fmean(recall),
            "exact_page_set_rate": fmean(exact_pages),
            "evidence_scope": "oracle_supplied_pages",
        },
        "answer": {
            "normalized_exact_question_average": fmean(
                r.normalized_exact_answer_score for r in records
            ),
            "full_credit_rate": fmean(float(r.normalized_exact_answer_score == 1) for r in records),
            "zero_credit_rate": fmean(float(r.normalized_exact_answer_score == 0) for r in records),
            "schema_valid_rate": fmean(float(r.prediction.schema_valid) for r in records),
            "abstention_rate": fmean(float(r.prediction.abstain) for r in records),
        },
        "latency": {
            "generation_p50_seconds": quantiles(latency, n=100, method="inclusive")[49]
            if len(latency) > 1
            else latency[0],
            "generation_p95_seconds": quantiles(latency, n=100, method="inclusive")[94]
            if len(latency) > 1
            else latency[0],
            "generation_tokens_per_second": sum(r.telemetry.generated_tokens for r in records)
            / sum(latency)
            if sum(latency)
            else None,
            "scope": "generation_only; excludes provisioning, model loading, preprocessing, and storage",
        },
        "semantic_vqa_score": None,
        "official_server_score": None,
    }


def evaluate_semantic(run: SavedRun, judge: DurableSemanticJudge) -> dict[str, Any]:
    """Apply published formulas to predicted pages, never substituting gold evidence."""
    judge.validate()
    with judge.logger.stage("semantic.scoring", heartbeat_seconds=15):
        scores, metrics = score_dataset(run.references, run.predictions, judge=judge)
    artifact = {
        "schema_version": 1,
        "source": run.source,
        "judge_contract": judge.contract,
        "judge_identity": judge.identity,
        "status": "local_published_formula_score",
        "official_server_score": None,
        "evidence_scope": "oracle_supplied_pages",
        "metrics": metrics,
    }
    prefix = f"runs/{run.job_name}/{run.source['source_records_sha256']}"
    judge.objects.write(
        prefix + "/private_scores.json",
        {
            "source": run.source,
            "judge_identity": judge.identity,
            "scores": [score.model_dump(mode="json") for score in scores],
        },
    )
    judge.objects.write(prefix + "/public_summary.json", artifact)
    judge.logger.emit(
        "semantic.evaluation.completed",
        job_name=run.job_name,
        question_count=len(scores),
        new_decisions=judge.new_decisions,
        reused_decisions=judge.reused_decisions,
    )
    return artifact


def save_public_evaluation(root: Path, job_name: str, name: str, payload: dict[str, Any]) -> None:
    """Publish aggregate artifacts alongside a checksum, retaining the original reader outputs."""
    if name not in {"supporting_metrics", "semantic_summary"}:
        raise ValueError("Unknown evaluation artifact name")
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", job_name):
        raise ValueError("Invalid SageMaker job name")
    directory = root / "reports/oracle_reader/runs" / job_name
    directory.mkdir(parents=True, exist_ok=True)
    data = encode(payload)
    for filename, content in (
        (f"{name}.json", data),
        (f"{name}.sha256", (digest(data) + "\n").encode()),
    ):
        path = directory / filename
        temporary = path.with_name(f".{filename}.tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
