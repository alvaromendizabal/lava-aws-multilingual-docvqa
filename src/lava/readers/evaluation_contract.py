"""Frozen coverage and resource contract for the descriptive reader pilot."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from lava.readers.oracle_assets import load_oracle_examples
from lava.readers.schemas import OracleExample, SageMakerJobPlan


def load_evaluation_contract(root: Path) -> dict[str, Any]:
    """Load the pilot contract and verify its link to the immutable protocol."""
    contract = json.loads((root / "configs/oracle_reader_evaluation.json").read_text())
    lock = json.loads((root / "configs/evaluation_protocol.lock.json").read_text())
    if contract["protocol_lock_id"] != lock["protocol_lock_id"]:
        raise ValueError("Evaluation contract does not match the frozen protocol")
    if contract["question_count"] != 16 or contract["document_count"] != 5:
        raise ValueError("This pilot requires the frozen 16 questions and five documents")
    return contract


def validate_evaluation_plan(plan: SageMakerJobPlan, root: Path) -> None:
    """Require complete coverage and an already verified model/hardware path."""
    contract = load_evaluation_contract(root)
    if plan.mode != "benchmark" or plan.limit != contract["question_count"]:
        raise ValueError("Benchmark mode requires exactly the complete frozen question set")
    if plan.protocol_lock_id != contract["protocol_lock_id"]:
        raise ValueError("Benchmark plan protocol mismatch")
    if contract["models"].get(plan.model_key) != plan.instance_type:
        raise ValueError("Benchmark requires a verified model and its configured GPU path")
    if plan.managed_spot or plan.creates_endpoint or plan.instance_count != 1:
        raise ValueError("Benchmark requires one on-demand instance without an endpoint")


def validate_evaluation_manifest(
    payload: bytes, contract: dict[str, Any]
) -> tuple[OracleExample, ...]:
    """Verify exact bytes, unique IDs, and all frozen coverage before loading weights."""
    if hashlib.sha256(payload).hexdigest() != contract["private_manifest_sha256"]:
        raise ValueError("Oracle manifest differs from the frozen SHA-256")
    examples = load_oracle_examples(payload)
    if len(examples) != contract["question_count"]:
        raise ValueError("Oracle manifest question count mismatch")
    if len({item.question_id for item in examples}) != len(examples):
        raise ValueError("Oracle manifest contains duplicate question IDs")
    if any(item.protocol_lock_id != contract["protocol_lock_id"] for item in examples):
        raise ValueError("Oracle manifest protocol mismatch")
    counts = {
        "language_counts": Counter(item.language for item in examples),
        "answer_format_counts": Counter(item.answer_format.value for item in examples),
        "document_question_counts": Counter(item.document_alias for item in examples),
    }
    for key, observed in counts.items():
        if dict(observed) != contract[key]:
            raise ValueError(f"Oracle manifest {key} mismatch")
    return tuple(examples)
