"""Bounded, operator-routed evidence review after an exhausted OCR search.

Routes contain page numbers and optional geometric crops, never candidate answers.
This stage preserves existing predictions and cannot export or upload a partial CSV.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pymupdf
import yaml

from lava.evaluation.semantic import digest, encode
from lava.evaluation.submission_store import cached_source
from lava.readers.model_registry import load_resolved_model
from lava.readers.private_artifacts import read_raw_response
from lava.readers.qwen35 import Qwen35Reader
from lava.readers.runtime_logging import RuntimeEventLogger
from lava.readers.system import put_blob, store_for, validate_prediction
from lava.readers.test_inference import (
    TestPageAsset,
    TestReaderInput,
    render_test_asset,
    test_contract,
)
from pipelines.submission.inference import configure_runtime
from pipelines.submission.recovery import (
    check_recovery_record,
    complete_answer,
    inspect_base,
    recovery_contract,
)

INSTRUCTION = (
    "The question may require connecting information across the supplied pages. "
    "Compare the chart, table, photograph and relevant text together before deciding whether "
    "the evidence is sufficient. Cite every supplied physical page needed for that connection. "
    "Some images are geometric detail crops of their labelled page, not additional pages. "
    "Use no external knowledge. If the supplied evidence is insufficient, still abstain."
)
POLICY = {
    "version": "operator-routed-cross-page-v1",
    "maximum_questions": 5,
    "maximum_candidates_per_question": 2,
    "maximum_pages_per_candidate": 4,
    "maximum_model_calls": 10,
    "maximum_work_seconds": 1200,
    "maximum_crop_dpi": 600,
    "instruction": INSTRUCTION,
    "selection": "first structurally complete prediction in the frozen route order",
    "routing": "operator review of test documents, without reference answers",
    "csv_export": False,
}


def read_pinned(s3: Any, bucket: str, source: dict) -> dict:
    """Read a precise immutable envelope and close its body even on validation failure."""
    response = s3.get_object(Bucket=bucket, Key=source["key"], VersionId=source["version_id"])
    with response["Body"] as body:
        payload = body.read()
    if response.get("VersionId") != source["version_id"]:
        raise ValueError("Pinned source version mismatch")
    if digest(payload) != source["sha256"]:
        raise ValueError("Pinned source checksum mismatch")
    if response.get("Metadata", {}).get("sha256") != source["sha256"]:
        raise ValueError("Pinned source metadata checksum mismatch")
    envelope = json.loads(payload)
    if envelope.get("state") != "complete" or not isinstance(envelope.get("value"), dict):
        raise ValueError("Pinned source must be a complete envelope")
    return envelope["value"]


def contract_for(root: Path, plan: dict) -> dict:
    value = {
        "schema_version": 1,
        "plan_sha256": digest(encode(plan)),
        "policy": POLICY,
        "source_sha256": {
            name: digest((root / name).read_bytes())
            for name in ("pipelines/submission/targeted.py", "pipelines/submission/run_targeted.sh")
        },
    }
    return {**value, "contract_id": digest(encode(value))}


def validate_routes(plan: dict, requests: tuple, failed: set[str], page_counts: dict) -> None:
    """Reject changed coverage, repeated questions, unsafe crops and unbounded routing."""
    routes, blocked = plan["routes"], plan["blocked"]
    if set(routes) & set(blocked) or set(routes) | set(blocked) != failed:
        raise ValueError("Routes and explicit blockers must partition the unresolved questions")
    if len(failed) > POLICY["maximum_questions"]:
        raise ValueError("Targeted question allowance exceeded")
    originals = {r.question_id: r for r in requests}
    for qid, candidates in routes.items():
        if not 1 <= len(candidates) <= POLICY["maximum_candidates_per_question"]:
            raise ValueError("Invalid candidate count")
        seen = set()
        for candidate in candidates:
            if set(candidate) != {"pages", "crops"}:
                raise ValueError("Routes may contain only page numbers and crops")
            pages = candidate["pages"]
            if (
                not 1 <= len(pages) <= POLICY["maximum_pages_per_candidate"]
                or any(
                    type(n) is not int or not 1 <= n <= page_counts[originals[qid].document_id]
                    for n in pages
                )
                or pages != sorted(set(pages))
                or not set(candidate["crops"]) <= {str(n) for n in pages}
            ):
                raise ValueError("Invalid physical page route")
            for crop in candidate["crops"].values():
                if (
                    not isinstance(crop, list)
                    or len(crop) != 4
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in crop)
                    or not 0 <= crop[0] < crop[2] <= 1
                    or not 0 <= crop[1] < crop[3] <= 1
                ):
                    raise ValueError("Crop must be a finite normalized rectangle")
            identity = digest(encode(candidate))
            if identity in seen:
                raise ValueError("Duplicate targeted candidate")
            seen.add(identity)
    if any(not isinstance(reason, str) or not reason.strip() for reason in blocked.values()):
        raise ValueError("Every blocked question needs an explicit reason")


def check_targeted(saved: dict, original: Any, identity: str, route: dict) -> Any:
    """Reparse the actual generation and bind its question, source and selected geometry."""
    if saved.get("contract_id") != identity or saved.get("route") != route:
        raise ValueError("Targeted answer contract or route mismatch")
    request = TestReaderInput.model_validate(saved["request"])
    for field in (
        "question_id",
        "document_id",
        "document_alias",
        "question",
        "language",
        "answer_format",
    ):
        if getattr(request, field) != getattr(original, field):
            raise ValueError("Targeted recovery changed the frozen question")
    if list(request.available_pages) != route["pages"]:
        raise ValueError("Targeted request differs from its page route")
    for page in request.pages:
        for field in ("source_pdf_s3_uri", "source_pdf_sha256", "source_pdf_version_id"):
            if getattr(page, field) != getattr(original.pages[0], field):
                raise ValueError("Targeted recovery changed the frozen PDF")
    if not re.fullmatch(r"[0-9a-f]{40}", saved.get("inference_code_commit", "")):
        raise ValueError("Targeted generation requires a committed source revision")
    prediction, _ = validate_prediction(request, saved)
    return prediction


class CrossPageReader(Qwen35Reader):
    """Keep native text and supply one image per page, including labelled detail views."""

    route: dict

    def _messages(self, example):
        messages, images, pixels = super()._messages(example)
        details = ", ".join(f"page {n}: {rect}" for n, rect in self.route["crops"].items())
        messages[1]["content"].insert(
            0, {"type": "text", "text": INSTRUCTION + "\nDetail crops: " + (details or "none")}
        )
        return messages, images, pixels


def prepare_asset(
    s3: Any,
    bucket: str,
    payload: bytes,
    source: dict,
    original: Any,
    number: int,
    crop: list | None,
    render: dict,
    identity: str,
) -> dict:
    asset_identity = {
        "targeted_contract": identity,
        "source": source,
        "page": number,
        "alias": original.document_alias,
        "render": render,
        "crop": crop,
    }
    asset = render_test_asset(
        s3,
        bucket,
        payload,
        source,
        original.document_id,
        original.document_alias,
        number,
        render,
        asset_identity,
    )
    if crop is not None:
        with pymupdf.open(stream=payload, filetype="pdf") as pdf:
            page = pdf[number - 1]
            w, h = page.rect.width, page.rect.height
            clip = pymupdf.Rect(crop[0] * w, crop[1] * h, crop[2] * w, crop[3] * h)
            pix = page.get_pixmap(dpi=POLICY["maximum_crop_dpi"], clip=clip, alpha=False)
            data = pix.tobytes("png")
        key = f"experiments/submissions/test-assets/{digest(encode(asset_identity))}/detail.png"
        version = put_blob(s3, bucket, key, data, "image/png")
        asset.update(
            asset_version="targeted-page-detail-v1",
            image_s3_uri=f"s3://{bucket}/{key}",
            image_sha256=digest(data),
            image_version_id=version,
            width_pixels=pix.width,
            height_pixels=pix.height,
            dpi=POLICY["maximum_crop_dpi"],
        )
    return TestPageAsset.model_validate(asset).model_dump(mode="json")


def verify_parent(root: Path, s3: Any, bucket: str, plan: dict) -> tuple:
    """Verify all inherited outputs before allocating the reader or generating anything."""
    base = test_contract(root)
    if base["contract_id"] != plan["base_contract_id"]:
        raise ValueError("Frozen base implementation changed")
    manifest = read_pinned(s3, bucket, plan["inputs"])
    inference = read_pinned(s3, bucket, plan["base_inference"])
    requests, failed = inspect_base(manifest, inference, base)
    if digest(encode(inference)) != plan["base_inference_value_sha256"]:
        raise ValueError("Base inference value changed")
    parent = read_pinned(s3, bucket, plan["parent_contract"])
    report = read_pinned(s3, bucket, plan["parent_report"])
    if parent != recovery_contract(root, base, digest(encode(inference))):
        raise ValueError("Frozen parent implementation changed")
    if (
        parent["contract_id"] != plan["parent_contract_id"]
        or report["contract_id"] != parent["contract_id"]
    ):
        raise ValueError("Parent recovery contract mismatch")
    if parent["base_inference_sha256"] != digest(encode(inference)):
        raise ValueError("Parent recovery has a different base")
    refs = plan["parent_answers"]
    if len({r["question_id"] for r in refs}) != len(refs):
        raise ValueError("Duplicate inherited answer")
    originals = {r.question_id: r for r in requests}

    def verify(ref):
        qid = ref["question_id"]
        if qid not in failed:
            raise ValueError("Inherited recovery overwrites a complete base answer")
        saved = read_pinned(s3, bucket, ref)
        request = TestReaderInput.model_validate(saved["request"])
        context = {}
        for number in request.available_pages:
            suffix = f"ocr/{request.document_id}/{number:04d}.json"
            obj = s3.get_object(
                Bucket=bucket,
                Key=f"experiments/submissions/system/{parent['contract_id']}/{suffix}",
            )
            with obj["Body"] as stream:
                payload = stream.read()
            if digest(payload) != obj.get("Metadata", {}).get("sha256"):
                raise ValueError("Inherited OCR checksum mismatch")
            envelope = json.loads(payload)
            if envelope.get("state") != "complete":
                raise ValueError("Incomplete inherited OCR")
            context[number] = envelope["value"]
        prediction = check_recovery_record(saved, originals[qid], context, parent["contract_id"])
        if (
            not complete_answer(prediction)
            or saved["inference_code_commit"] != plan["parent_inference_commit"]
        ):
            raise ValueError("Inherited recovery is incomplete or uses a different revision")
        return qid

    with ThreadPoolExecutor(max_workers=8) as pool:
        inherited = set(pool.map(verify, refs))
    remaining = set(failed) - inherited
    preserved = len(requests) - len(remaining)
    if report["recovered"] != len(inherited) or set(report["unresolved"]) != remaining:
        raise ValueError("Inherited coverage differs from the final parent report")
    if preserved != plan["expected_preserved_count"]:
        raise ValueError("Preserved answer count mismatch")
    validate_routes(plan, requests, remaining, manifest["page_counts"])
    return base, manifest, requests, preserved


def run(root: Path, s3: Any, bucket: str, plan: dict, logger: Any) -> dict:
    started = time.monotonic()
    contract = contract_for(root, plan)
    if contract["contract_id"] != os.environ["LAVA_TARGETED_CONTRACT"]:
        raise ValueError("Targeted launch identity mismatch")
    base, manifest, requests, preserved = verify_parent(root, s3, bucket, plan)
    store = store_for(s3, bucket, contract["contract_id"])
    store.write("contract.json", contract)
    existing = store.read("report.json")
    if existing is not None:
        raise ValueError("Targeted review already finished; inspect its saved report, do not rerun")
    renderer = yaml.safe_load((root / "configs/oracle_reader_benchmark.yaml").read_bytes())[
        "asset_builder"
    ]
    render = renderer["render_profiles"][renderer["active_render_profile"]]
    reader = CrossPageReader(
        load_resolved_model(
            root / "configs/oracle_reader_models.lock.json", base["config"]["model_key"]
        ),
        region=os.environ["AWS_DEFAULT_REGION"],
    )
    originals = {r.question_id: r for r in requests}
    recovered, calls = [], 0
    for qid, candidates in sorted(plan["routes"].items()):
        original = originals[qid]
        page = original.pages[0]
        source = {
            "key": page.source_pdf_s3_uri.split(f"s3://{bucket}/", 1)[1],
            "sha256": page.source_pdf_sha256,
            "version_id": page.source_pdf_version_id,
        }
        payload = cached_source(
            s3,
            bucket,
            root / "artifacts/submission/inputs" / f"{original.document_id}.pdf",
            source,
            logger,
        )
        with pymupdf.open(stream=payload, filetype="pdf") as pdf:
            if len(pdf) != manifest["page_counts"][original.document_id]:
                raise ValueError("Targeted PDF page count mismatch")
        for route in candidates:
            if (
                calls >= POLICY["maximum_model_calls"]
                or time.monotonic() - started > POLICY["maximum_work_seconds"]
            ):
                raise RuntimeError("Targeted allowance exhausted; saved answers preserved")
            candidate = TestReaderInput.model_validate(
                {
                    **original.model_dump(mode="json"),
                    "pages": [
                        prepare_asset(
                            s3,
                            bucket,
                            payload,
                            source,
                            original,
                            n,
                            route["crops"].get(str(n)),
                            render,
                            contract["contract_id"],
                        )
                        for n in route["pages"]
                    ],
                }
            )
            key = "attempts/" + digest(encode(candidate.model_dump(mode="json"))) + ".json"
            saved = store.read(key)
            if saved is None:
                reader.route = route
                with logger.stage("targeted.question.inference", heartbeat_seconds=15):
                    prediction, telemetry = reader.predict(candidate)
                calls += 1
                saved = {
                    "contract_id": contract["contract_id"],
                    "route": route,
                    "request": candidate.model_dump(mode="json"),
                    "input_sha256": digest(encode(candidate.model_dump(mode="json"))),
                    "prediction": prediction.model_dump(mode="json"),
                    "telemetry": telemetry.model_dump(mode="json"),
                    "raw_response": read_raw_response(qid),
                    "inference_code_commit": os.environ["LAVA_GIT_COMMIT_SHA"],
                }
                check_targeted(saved, original, contract["contract_id"], route)
                store.write(key, saved)
                if store.read(key) != saved:
                    raise ValueError("Targeted attempt read-back failed")
            prediction = check_targeted(saved, original, contract["contract_id"], route)
            if complete_answer(prediction):
                answer_key = f"answers/{digest(qid.encode())}.json"
                store.write(answer_key, saved)
                if store.read(answer_key) != saved:
                    raise ValueError("Targeted answer read-back failed")
                recovered.append(qid)
                break
        logger.emit(
            "targeted.question.completed", recovered=len(recovered), total=len(plan["routes"])
        )
    unresolved = sorted((set(plan["routes"]) | set(plan["blocked"])) - set(recovered))
    report = {
        "schema_version": 1,
        "contract_id": contract["contract_id"],
        "parent_contract_id": plan["parent_contract_id"],
        "preserved_count": preserved,
        "additional_recovered_count": len(recovered),
        "additional_recovered_ids": recovered,
        "total_complete_count": preserved + len(recovered),
        "unresolved": unresolved,
        "blocked": plan["blocked"],
        "new_model_calls": calls,
        "elapsed_seconds": time.monotonic() - started,
        "csv_exported": False,
        "uploaded_to_kaggle": False,
    }
    store.write("report.json", report)
    if store.read("report.json") != report:
        raise ValueError("Targeted report read-back failed")
    return report


def main() -> int:
    import boto3
    import torch

    print(json.dumps(configure_runtime(torch), sort_keys=True), flush=True)
    root = Path(__file__).resolve().parents[2]
    s3 = boto3.client("s3", region_name=os.environ["AWS_DEFAULT_REGION"])
    bucket = os.environ["LAVA_BUCKET"]
    plan = read_pinned(s3, bucket, json.loads(os.environ["LAVA_TARGETED_PLAN"]))
    logger = RuntimeEventLogger(
        "submission.targeted", jsonl_path=Path("/opt/ml/model/private/events.jsonl")
    )
    result = run(root, s3, bucket, plan, logger)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
