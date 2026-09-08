"""A fixed second read of first-pass citations, without reference-label access."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

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
from lava.evaluation.system import load_summary, score_predictions
from lava.readers.model_registry import load_resolved_model
from lava.readers.oracle_assets import parse_training_csv
from lava.readers.runtime_logging import RuntimeEventLogger
from lava.readers.schemas import ReaderInput
from lava.readers.system import (
    infer_requests,
    store_for,
    system_contract,
    validate_manifest,
    validate_prediction,
)
from lava.readers.system_execution import training_request
from lava.retrieval.evaluation import pilot_sources


def refinement_contract(root: Path) -> dict[str, Any]:
    """Bind the measured first pass, exact policy, reader and implementation."""
    base = system_contract(root)
    config = json.loads((root / "configs/system_refinement.json").read_bytes())
    if (
        config["base_contract_id"] != base["contract_id"]
        or config["policy"] != "reread_self_cited_pages"
    ):
        raise ValueError("Refinement requires the frozen first pass and citation-only policy")
    base.pop("contract_id")
    base["refinement"] = config
    for name in (
        "src/lava/readers/refinement.py",
        "scripts/refine_system.py",
        "pipelines/refinement/run.sh",
    ):
        base["source_sha256"][name] = digest((root / name).read_bytes())
    return {**base, "contract_id": digest(encode(base))}


def focus_inputs(manifest: dict[str, Any], inference: dict[str, Any]) -> list[dict[str, Any]]:
    """Use only validated first-pass citations; empty/invalid outputs keep their input pages."""
    requests = validate_manifest(manifest, manifest["contract"])
    if (
        inference["contract_id"] != manifest["contract"]["contract_id"]
        or inference["inputs_sha256"] != manifest["inputs_sha256"]
        or len(inference["records"]) != len(requests)
    ):
        raise ValueError("First-pass inference coverage differs")
    focused = []
    for request, record in zip(requests, inference["records"], strict=True):
        if record["contract_id"] != inference["contract_id"]:
            raise ValueError("Mixed first-pass checkpoints")
        prediction, _ = validate_prediction(request, record)
        chosen = set(prediction.evidence_pages) if prediction.schema_valid else set()
        pages = (
            tuple(page for page in request.pages if page.page_number in chosen)
            if chosen
            else request.pages
        )
        value = request.model_copy(update={"pages": pages}).model_dump(mode="json")
        ReaderInput.model_validate(value)
        focused.append(value)
    return focused


def prepare_refinement(
    root: Path, s3: Any, bucket: str, logger: RuntimeEventLogger
) -> dict[str, Any]:
    contract = refinement_contract(root)
    store = store_for(s3, bucket, contract["contract_id"])
    saved = store.read("inputs.json")
    if saved is not None:
        validate_manifest(saved, contract)
        logger.emit("refinement.inputs.reused", questions=16)
        return saved
    base = store_for(s3, bucket, contract["refinement"]["base_contract_id"])
    manifest, inference = base.read("inputs.json"), base.read("inference.json")
    if manifest is None or inference is None:
        raise ValueError("Complete first-pass inference is required")
    if digest(encode(inference)) != contract["refinement"]["base_inference_sha256"]:
        raise ValueError("First-pass inference checksum differs from the frozen experiment")
    inputs = focus_inputs(manifest, inference)
    result = {
        "schema_version": 1,
        "contract": contract,
        "inputs": inputs,
        "inputs_sha256": digest(encode(inputs)),
    }
    validate_manifest(result, contract)
    store.write("inputs.json", result)
    if store.read("inputs.json") != result:
        raise ValueError("Focused inputs failed durable read-back")
    logger.emit(
        "refinement.inputs.verified",
        questions=len(inputs),
        first_pass_pages=sum(len(row["pages"]) for row in manifest["inputs"]),
        second_pass_pages=sum(len(row["pages"]) for row in inputs),
        uses_gold_labels=False,
    )
    return result


def refinement_training_request(*args: Any, **kwargs: Any) -> dict[str, Any]:
    request = training_request(*args, **kwargs)
    request["TrainingJobName"] = request["TrainingJobName"].replace(
        "lava-system-", "lava-refine-", 1
    )
    request["OutputDataConfig"]["S3OutputPath"] = request["OutputDataConfig"][
        "S3OutputPath"
    ].replace("/system-runs/", "/refinement-runs/")
    request["AlgorithmSpecification"]["ContainerArguments"][1] = request["AlgorithmSpecification"][
        "ContainerArguments"
    ][1].replace("/oracle_reader/run.sh", "/refinement/run.sh")
    request["Environment"]["PYTHONPATH"] = "/opt/ml/code/src"
    request["Tags"][1]["Value"] = "self-citation-refinement"
    return request


def evaluate_refinement(
    root: Path, s3: Any, bucket: str, logger: RuntimeEventLogger
) -> dict[str, Any]:
    contract = refinement_contract(root)
    base = load_summary(root)
    if base is None or base["inference_sha256"] != contract["refinement"]["base_inference_sha256"]:
        raise ValueError("The verified first-pass evaluation is required")
    store = store_for(s3, bucket, contract["contract_id"])
    manifest, inference = store.read("inputs.json"), store.read("inference.json")
    if manifest is None or inference is None:
        raise ValueError("Complete refinement inference is required")
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
    judge.validate()
    with logger.stage("refinement.evaluate", heartbeat_seconds=15):
        result = score_predictions(manifest, inference, references, judge)
    summary = {
        "schema_version": 1,
        "status": "Self-citation refinement scored",
        "split": "training_diagnostic",
        "official_server_parity": False,
        "official_server_score": None,
        "contract": contract,
        "judge_contract": judging,
        "scoring_source_sha256": digest((root / "src/lava/evaluation/system.py").read_bytes()),
        "input_sha256": manifest["inputs_sha256"],
        "inference_sha256": digest(encode(inference)),
        "oracle_summary_sha256": base["oracle_summary_sha256"],
        "oracle_question_micro": base["oracle_question_micro"],
        "first_pass_question_micro": base["metrics"]["question_micro"],
        "first_pass_summary_sha256": digest((root / "reports/system/summary.json").read_bytes()),
        "question_mean_delta": result["metrics"]["question_micro"]["overall"]
        - base["oracle_question_micro"]["overall"],
        "first_pass_delta": result["metrics"]["question_micro"]["overall"]
        - base["metrics"]["question_micro"]["overall"],
        "paired_document_comparison": compare_document_scores(
            {d: m["overall"] for d, m in base["metrics"]["by_document"].items()},
            {d: m["overall"] for d, m in result["metrics"]["by_document"].items()},
            seed=20260902,
        ),
        "limitations": "Post-hoc development hypothesis on the same 16 previously examined questions. Second-pass telemetry excludes the first pass; both passes are required. No held-out or official leaderboard claim.",
        **result,
    }
    store.write("evaluation.json", summary)
    if store.read("evaluation.json") != summary:
        raise ValueError("Refinement evaluation failed durable read-back")
    path = root / "reports/system/refinement.json"
    payload = encode(summary)
    atomic_write(path, payload)
    atomic_write(path.with_suffix(".sha256"), (digest(payload) + "\n").encode())
    logger.emit("refinement.scored", **result["metrics"]["question_micro"])
    return summary


def load_refinement(root: Path) -> dict[str, Any] | None:
    path = root / "reports/system/refinement.json"
    if not path.exists():
        return None
    payload = path.read_bytes()
    if digest(payload) != path.with_suffix(".sha256").read_text().strip():
        raise ValueError("Refinement summary checksum mismatch")
    summary = json.loads(payload)
    if (
        summary["contract"] != refinement_contract(root)
        or summary["judge_contract"] != judge_contract(root)
        or summary["scoring_source_sha256"]
        != digest((root / "src/lava/evaluation/system.py").read_bytes())
        or summary["first_pass_summary_sha256"]
        != digest((root / "reports/system/summary.json").read_bytes())
    ):
        raise ValueError("Refinement report lineage is stale")
    return summary


if __name__ == "__main__":
    import boto3

    from lava.readers.reader_factory import build_reader

    root = Path("/opt/ml/code")
    contract = refinement_contract(root)
    if os.environ["LAVA_SYSTEM_CONTRACT"] != contract["contract_id"]:
        raise ValueError("Remote refinement contract mismatch")
    logger = RuntimeEventLogger(
        "refinement.job", jsonl_path=Path("/opt/ml/model/private/events.jsonl")
    )
    s3 = boto3.client("s3", region_name=os.environ["AWS_DEFAULT_REGION"])
    store = store_for(s3, os.environ["LAVA_BUCKET"], contract["contract_id"])
    manifest = store.read("inputs.json")
    if manifest is None:
        raise ValueError("Prepared refinement inputs are required")
    reader = build_reader(
        load_resolved_model(
            root / "configs/oracle_reader_models.lock.json", contract["config"]["model_key"]
        ),
        region=os.environ["AWS_DEFAULT_REGION"],
    )
    with logger.stage("refinement.job", heartbeat_seconds=15):
        infer_requests(manifest, contract, store, reader.predict, logger)
