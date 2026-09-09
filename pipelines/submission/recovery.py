"""Recover unanswered test questions using OCR retrieval and the frozen 9B reader.

Base checkpoints are never modified. Every additional read, OCR page and final
choice has a separate immutable identity. No reference answers are loaded.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import re
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pymupdf
import yaml

from lava.evaluation.semantic import ImmutableS3Objects, digest, encode
from lava.evaluation.submission import build_submission, validate_submission_csv
from lava.evaluation.submission_store import atomic_write, cached_source, persist_submission
from lava.readers.model_registry import load_resolved_model
from lava.readers.private_artifacts import read_raw_response
from lava.readers.qwen35 import Qwen35Reader
from lava.readers.runtime_logging import RuntimeEventLogger
from lava.readers.system import store_for, validate_prediction
from lava.readers.test_inference import (
    TestPageAsset,
    TestReaderInput,
    render_test_asset,
    submission_inputs,
    test_contract,
    test_sources,
    validate_test_manifest,
)
from lava.retrieval.lexical import BM25Index, PageText
from pipelines.submission.inference import configure_runtime

OCR_REVISION = "87416418657359cb625c412a48b6e1d6d41c29bd"
OCR_HASHES = {
    "eng": "7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2",
    "jpn": "1f5de9236d2e85f5fdf4b3c500f2d4926f8d9449f28f5394472d9e8d83b91b4d",
    "vie": "79df64caf7bcfb2a27df5042ecb6121e196eada34da774956995747636d5bfa1",
}
POLICY = {
    "version": "ocr-bm25-visual-recovery-v1",
    "page_batch": 4,
    "ocr_dpi": 200,
    "ocr_psm": 11,
    "ocr_workers": 8,
    "tesserocr": "2.9.2",
    "tesseract": "tesseract 5.5.1",
    "native_characters_per_page": 2200,
    "ocr_characters_per_page": 2200,
    "maximum_model_calls": 700,
    "maximum_work_seconds": 6600,
    "selection": "first complete non-abstaining answer in the fixed page-search order",
    "unresolved": "save all attempts and refuse CSV export",
}


def complete_answer(prediction: Any) -> bool:
    """Structural completeness is distinct from answer correctness."""
    return bool(
        prediction.schema_valid
        and not prediction.abstain
        and prediction.answer.strip() not in {"", "[]"}
        and prediction.evidence_pages
    )


def inspect_base(manifest: dict, inference: dict, contract: dict) -> tuple:
    requests = validate_test_manifest(manifest, contract)
    if (
        inference.get("contract_id") != contract["contract_id"]
        or inference.get("inputs_sha256") != manifest["inputs_sha256"]
        or len(inference.get("records", [])) != len(requests)
    ):
        raise ValueError("Recovery requires complete, compatible base inference")
    failed = []
    for request, saved in zip(requests, inference["records"], strict=True):
        if saved.get("contract_id") != contract["contract_id"]:
            raise ValueError("Mixed base contracts")
        if not re.fullmatch(r"[0-9a-f]{40}", saved.get("inference_code_commit", "")):
            raise ValueError("Base prediction lacks a committed inference revision")
        prediction, _ = validate_prediction(request, saved)
        if not complete_answer(prediction):
            failed.append(request.question_id)
    return requests, failed


def recovery_contract(root: Path, base: dict, inference_sha256: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{64}", inference_sha256):
        raise ValueError("A complete base inference checksum is required")
    value = {
        "schema_version": 1,
        "base_contract_id": base["contract_id"],
        "base_inference_sha256": inference_sha256,
        "model": base["model"],
        "policy": POLICY,
        "ocr_revision": OCR_REVISION,
        "ocr_sha256": OCR_HASHES,
        "source_sha256": {
            name: digest((root / name).read_bytes())
            for name in (
                "pipelines/submission/recovery.py",
                "pipelines/submission/run_recovery.sh",
                "pipelines/submission/inference.py",
            )
        },
    }
    return {**value, "contract_id": digest(encode(value))}


def page_search_order(ranking: tuple, page_count: int) -> tuple[tuple[int, ...], ...]:
    """Try strong joint context, neighbours, focused pages, then remaining PDF coverage."""
    ranked = [int(number) for number, _ in ranking]
    if set(ranked) != set(range(1, page_count + 1)) or len(ranked) != page_count:
        raise ValueError("Recovery ranking must cover each physical PDF page exactly once")
    candidates = [ranked[:4]]
    for number in ranked[:2]:
        candidates.append(list(range(max(1, number - 1), min(page_count, number + 1) + 1)))
    candidates.extend([[number] for number in ranked[:4]])
    candidates.extend(ranked[start : start + 4] for start in range(4, page_count, 4))
    result = []
    for candidate in candidates:
        pages = tuple(sorted(set(candidate)))
        if pages and pages not in result:
            result.append(pages)
    return tuple(result)


def normalize_ocr(text: str) -> str:
    """Remove OCR-inserted spacing between adjacent Japanese characters for retrieval."""
    return re.sub(r"(?<=[\u3040-\u30ff\u3400-\u9fff])\s+(?=[\u3040-\u30ff\u3400-\u9fff])", "", text)


def prepare_ocr_models(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for language, expected in OCR_HASHES.items():
        target = destination / f"{language}.traineddata"
        if target.exists() and digest(target.read_bytes()) == expected:
            continue
        url = (
            "https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/"
            f"{OCR_REVISION}/{language}.traineddata"
        )
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
        if digest(payload) != expected:
            raise ValueError("OCR language model checksum mismatch")
        atomic_write(target, payload)


def ocr_page(pdf_path: str, number: int, language: str, tessdata: str) -> dict:
    """Each process owns its PDF and Tesseract handles; no shared MuPDF threads."""
    from PIL import Image
    from tesserocr import OEM, PSM, PyTessBaseAPI

    with pymupdf.open(pdf_path) as pdf:
        page = pdf[number - 1]
        native = page.get_text("text", sort=True)
        pixmap = page.get_pixmap(dpi=POLICY["ocr_dpi"], colorspace=pymupdf.csGRAY, alpha=False)
        image = Image.frombytes("L", (pixmap.width, pixmap.height), pixmap.samples)
    languages = {"ja": "jpn+eng", "vi": "vie+eng", "en": "eng"}
    with PyTessBaseAPI(
        path=tessdata, lang=languages[language], psm=PSM.SPARSE_TEXT, oem=OEM.LSTM_ONLY
    ) as api:
        api.SetImage(image)
        text = api.GetUTF8Text()
        confidence = api.MeanTextConf()
    image.close()
    return {"number": number, "native": native, "ocr": text, "ocr_confidence": confidence}


def document_text(
    path: Path,
    count: int,
    language: str,
    tessdata: Path,
    store: Any,
    document: str,
    pool: Any,
    logger: Any,
) -> list[dict]:
    pages = {}
    pending = {}
    for number in range(1, count + 1):
        key = f"ocr/{document}/{number:04d}.json"
        saved = store.read(key)
        if saved is not None:
            if saved["number"] != number:
                raise ValueError("OCR page checkpoint mismatch")
            pages[number] = saved
        else:
            pending[pool.submit(ocr_page, str(path), number, language, str(tessdata))] = key
    with logger.stage("recovery.ocr", heartbeat_seconds=15):
        for future in as_completed(pending):
            page = future.result()
            key = pending[future]
            store.write(key, page)
            if store.read(key) != page:
                raise ValueError("OCR checkpoint read-back failed")
            pages[page["number"]] = page
            logger.emit("recovery.ocr.page", completed=len(pages), total=count)
    return [pages[number] for number in range(1, count + 1)]


class RecoveryReader(Qwen35Reader):
    """Use the same image reader with clearly labelled, bounded OCR context."""

    context: dict[int, dict]

    def _messages(self, example):
        messages, images, pixels = super()._messages(example)
        for item in messages[1]["content"]:
            if item.get("type") != "text":
                continue
            match = re.match(r"Native PDF text for page (\d+):", item["text"])
            if match:
                number = int(match.group(1))
                page = self.context[number]
                item["text"] = (
                    f"Native PDF text for page {number}:\n"
                    f"{page['native'][: POLICY['native_characters_per_page']]}\n"
                    f"OCR transcription for page {number} (may contain errors; use the image "
                    f"as the authority):\n{page['ocr'][: POLICY['ocr_characters_per_page']]}"
                )
        return messages, images, pixels


def check_recovery_record(saved: dict, original: Any, context: dict, identity: str) -> Any:
    if saved.get("contract_id") != identity:
        raise ValueError("Recovery answer contract mismatch")
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
            raise ValueError("Recovery changed the original question")
    original_source = original.pages[0]
    for page in request.pages:
        for field in ("source_pdf_s3_uri", "source_pdf_sha256", "source_pdf_version_id"):
            if getattr(page, field) != getattr(original_source, field):
                raise ValueError("Recovery evidence changed the frozen PDF source")
    expected = {str(number): context[number] for number in request.available_pages}
    if saved.get("context_sha256") != digest(encode(expected)):
        raise ValueError("Recovery OCR context checksum mismatch")
    prediction, _ = validate_prediction(request, saved)
    if not re.fullmatch(r"[0-9a-f]{40}", saved.get("inference_code_commit", "")):
        raise ValueError("Recovery requires a committed inference revision")
    return prediction


def export_recovered(
    root: Path,
    s3: Any,
    bucket: str,
    manifest: dict,
    inference: dict,
    contract: dict,
    recovered: dict,
    contexts: dict,
    logger: Any,
) -> dict:
    """Reparse all chosen raw responses and preserve every complete base prediction."""
    base = test_contract(root)
    requests, failed = inspect_base(manifest, inference, base)
    if contract != recovery_contract(root, base, digest(encode(inference))):
        raise ValueError("Recovery export contract mismatch")
    if set(recovered) != set(failed):
        raise ValueError("Every unresolved question requires one complete recovery decision")
    for document, context in contexts.items():
        if set(context) != set(range(1, manifest["page_counts"][document] + 1)) or any(
            page["number"] != number for number, page in context.items()
        ):
            raise ValueError("Recovery context does not cover the frozen PDF")
    inputs = submission_inputs(root, s3, bucket, logger)
    rows, commits, unresolved = [], set(), []
    for request, saved in zip(requests, inference["records"], strict=True):
        original = inputs.questions[request.question_id]
        if any(
            getattr(request, field) != original[column]
            for field, column in (
                ("document_id", "file_id"),
                ("question", "question"),
                ("answer_format", "answer_format"),
                ("language", "language"),
            )
        ):
            raise ValueError("Recovery inputs differ from the pinned competition question")
        prediction, _ = validate_prediction(request, saved)
        if request.question_id in recovered:
            saved = recovered[request.question_id]
            prediction = check_recovery_record(
                saved, request, contexts[request.document_id], contract["contract_id"]
            )
        if not complete_answer(prediction):
            unresolved.append(request.question_id)
        commits.add(saved["inference_code_commit"])
        rows.append(
            {
                "question_id": request.question_id,
                "answer": prediction.answer,
                "evidence_pages": list(prediction.evidence_pages),
            }
        )
    if unresolved:
        raise ValueError(f"{len(unresolved)} questions remain unanswered; no CSV exported")
    payload = b"".join(encode(row) for row in rows)
    provenance = {
        "split": "test",
        "evidence_scope": "retrieved_full_document_pages",
        "test_csv_sha256": inputs.source_sha256["test.csv"],
        "question_count": len(rows),
        "model_id": base["model"]["model_id"],
        "model_revision": base["model"]["revision"],
        "code_commit": os.environ["LAVA_GIT_COMMIT_SHA"],
        "inference_code_commits": sorted(commits),
        "predictions_sha256": digest(payload),
        "page_counts_sha256": manifest["page_counts_sha256"],
        "inference_contract": contract["contract_id"],
        "base_inference_contract": base["contract_id"],
        "base_inference_sha256": digest(encode(inference)),
        "recovery_answers_sha256": digest(encode(recovered)),
        "recovery_contract": contract,
        "preserved_answer_count": len(rows) - len(failed),
        "recovered_answer_count": len(failed),
    }
    csv_payload = build_submission(inputs, payload, manifest["page_counts"], provenance)
    validation = validate_submission_csv(csv_payload, inputs, manifest["page_counts"])
    output_manifest = {
        "schema_version": 1,
        "competition": base["submission"]["competition"],
        "source_sha256": inputs.source_sha256,
        "prediction_sha256": digest(payload),
        "provenance": provenance,
        "submission_sha256": digest(csv_payload),
        "validation": validation,
    }
    target = persist_submission(s3, bucket, root, csv_payload, output_manifest, logger)
    atomic_write(root / "artifacts/submission/submission.csv", csv_payload)
    atomic_write(root / "artifacts/submission/manifest.json", encode(output_manifest))
    result = {
        "bundle_id": target.parent.name,
        "submission_sha256": digest(csv_payload),
        "csv_key": f"experiments/submissions/lava-challenge-2026/{target.parent.name}/submission.csv",
        "validation": validation,
        "uploaded_to_kaggle": False,
        "recovery_contract_id": contract["contract_id"],
    }
    store = store_for(s3, bucket, contract["contract_id"])
    store.write("export.json", result)
    if store.read("export.json") != result:
        raise ValueError("Recovery export receipt read-back failed")
    return result


def run(root: Path, s3: Any, bucket: str, logger: Any) -> dict:
    started = time.monotonic()
    base = test_contract(root)
    if base["contract_id"] != os.environ["LAVA_BASE_CONTRACT"]:
        raise ValueError("Frozen base implementation changed")
    base_store = store_for(s3, bucket, base["contract_id"])
    manifest, inference = base_store.read("inputs.json"), base_store.read("inference.json")
    if manifest is None or inference is None:
        raise ValueError("Complete saved base inference is required")
    inference_sha = digest(encode(inference))
    if inference_sha != os.environ["LAVA_BASE_INFERENCE_SHA256"]:
        raise ValueError("Base inference differs from the reviewed snapshot")
    requests, failed = inspect_base(manifest, inference, base)
    contract = recovery_contract(root, base, inference_sha)
    if contract["contract_id"] != os.environ["LAVA_RECOVERY_CONTRACT"]:
        raise ValueError("Recovery contract differs from the reviewed launch")
    store = store_for(s3, bucket, contract["contract_id"])
    store.write("contract.json", contract)
    inputs = submission_inputs(root, s3, bucket, logger)
    existing = store.read("export.json")
    if existing is not None:
        prefix = f"experiments/submissions/lava-challenge-2026/{existing['bundle_id']}"
        if existing["csv_key"] != prefix + "/submission.csv":
            raise ValueError("Recovered CSV receipt points outside its immutable bundle")
        saved_manifest = ImmutableS3Objects(s3, bucket, prefix).read("manifest.json")
        response = s3.get_object(Bucket=bucket, Key=existing["csv_key"])
        with response["Body"] as stream:
            payload = stream.read()
        if (
            saved_manifest is None
            or digest(encode(saved_manifest)) != existing["bundle_id"]
            or saved_manifest["provenance"]["recovery_contract"] != contract
            or saved_manifest["submission_sha256"] != digest(payload)
            or existing["submission_sha256"] != digest(payload)
            or validate_submission_csv(payload, inputs, manifest["page_counts"])
            != existing["validation"]
        ):
            raise ValueError("Previously completed recovery export failed verification")
        atomic_write(root / "artifacts/submission/submission.csv", payload)
        atomic_write(root / "artifacts/submission/manifest.json", encode(saved_manifest))
        logger.emit("recovery.export.reused", rows=len(inputs.order))
        return existing
    sources = test_sources(root, inputs, base)
    tessdata = root / "artifacts/submission/tessdata"
    prepare_ocr_models(tessdata)
    renderer = yaml.safe_load((root / "configs/oracle_reader_benchmark.yaml").read_bytes())[
        "asset_builder"
    ]
    render = renderer["render_profiles"][renderer["active_render_profile"]]
    grouped = defaultdict(list)
    for request in requests:
        if request.question_id in failed:
            grouped[request.document_id].append(request)
    reader = RecoveryReader(
        load_resolved_model(
            root / "configs/oracle_reader_models.lock.json", base["config"]["model_key"]
        ),
        region=os.environ["AWS_DEFAULT_REGION"],
    )
    recovered, contexts = {}, {}
    model_calls = 0
    with ProcessPoolExecutor(
        max_workers=POLICY["ocr_workers"], mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for document, questions in sorted(grouped.items()):
            path = root / "artifacts/submission/inputs" / f"{document}.pdf"
            source = sources[document]
            pdf_bytes = cached_source(s3, bucket, path, source, logger)
            with pymupdf.open(stream=pdf_bytes, filetype="pdf") as pdf:
                if len(pdf) != manifest["page_counts"][document]:
                    raise ValueError("Recovery PDF page count mismatch")
            pages = document_text(
                path,
                manifest["page_counts"][document],
                questions[0].language,
                tessdata,
                store,
                document,
                pool,
                logger,
            )
            context = {page["number"]: page for page in pages}
            contexts[document] = context
            reader.context = context
            index = BM25Index(
                tuple(
                    PageText(p["number"], normalize_ocr(p["native"] + "\n" + p["ocr"]))
                    for p in pages
                )
            )
            assets = {
                p.page_number: p for q in requests if q.document_id == document for p in q.pages
            }
            for request in questions:
                answer_key = f"answers/{digest(request.question_id.encode())}.json"
                saved = store.read(answer_key)
                if saved is not None:
                    prediction = check_recovery_record(
                        saved, request, context, contract["contract_id"]
                    )
                    if not complete_answer(prediction):
                        raise ValueError(
                            "Recovery decision checkpoint contains an incomplete answer"
                        )
                    recovered[request.question_id] = saved
                    continue
                for selected in page_search_order(index.rank(request.question), len(pages)):
                    if (
                        time.monotonic() - started > POLICY["maximum_work_seconds"]
                        or model_calls >= POLICY["maximum_model_calls"]
                    ):
                        raise RuntimeError(
                            "Bounded recovery allowance reached; checkpoints preserved"
                        )
                    for number in selected:
                        if number not in assets:
                            identity = {
                                "source": source,
                                "render": render,
                                "page": number,
                                "alias": request.document_alias,
                                "implementation": base["source_sha256"][
                                    "src/lava/readers/test_inference.py"
                                ],
                                "pymupdf": base["retrieval"]["pymupdf_version"],
                            }
                            key = f"pages/{digest(encode(identity))}.json"
                            asset = store.read(key)
                            if asset is None:
                                asset = render_test_asset(
                                    s3,
                                    bucket,
                                    pdf_bytes,
                                    source,
                                    document,
                                    request.document_alias,
                                    number,
                                    render,
                                    identity,
                                )
                                store.write(key, asset)
                            assets[number] = TestPageAsset.model_validate(asset)
                    candidate = TestReaderInput.model_validate(
                        {
                            **request.model_dump(mode="json"),
                            "pages": [assets[n].model_dump(mode="json") for n in selected],
                        }
                    )
                    context_sha = digest(encode({str(n): context[n] for n in selected}))
                    attempt_id = digest(
                        encode(
                            {
                                "request": candidate.model_dump(mode="json"),
                                "context_sha256": context_sha,
                            }
                        )
                    )
                    attempt_key = f"attempts/{attempt_id}.json"
                    saved = store.read(attempt_key)
                    if saved is None:
                        with logger.stage("recovery.question.inference", heartbeat_seconds=15):
                            prediction, telemetry = reader.predict(candidate)
                        model_calls += 1
                        saved = {
                            "contract_id": contract["contract_id"],
                            "request": candidate.model_dump(mode="json"),
                            "input_sha256": digest(encode(candidate.model_dump(mode="json"))),
                            "context_sha256": context_sha,
                            "prediction": prediction.model_dump(mode="json"),
                            "telemetry": telemetry.model_dump(mode="json"),
                            "raw_response": read_raw_response(candidate.question_id),
                            "inference_code_commit": os.environ["LAVA_GIT_COMMIT_SHA"],
                        }
                        check_recovery_record(saved, request, context, contract["contract_id"])
                        store.write(attempt_key, saved)
                        if store.read(attempt_key) != saved:
                            raise ValueError("Recovery attempt read-back failed")
                    prediction = check_recovery_record(
                        saved, request, context, contract["contract_id"]
                    )
                    logger.emit(
                        "recovery.attempt.completed",
                        model_calls=model_calls,
                        schema_valid=prediction.schema_valid,
                        abstain=prediction.abstain,
                    )
                    if complete_answer(prediction):
                        store.write(answer_key, saved)
                        if store.read(answer_key) != saved:
                            raise ValueError("Recovery decision read-back failed")
                        recovered[request.question_id] = saved
                        break
                logger.emit(
                    "recovery.question.completed", recovered=len(recovered), total=len(failed)
                )
    report = {
        "schema_version": 1,
        "contract_id": contract["contract_id"],
        "base_question_count": len(requests),
        "base_preserved": len(requests) - len(failed),
        "recovered": len(recovered),
        "unresolved": sorted(set(failed) - set(recovered)),
        "new_model_calls": model_calls,
        "elapsed_seconds": time.monotonic() - started,
    }
    previous_report = store.read("report.json")
    if previous_report is None:
        store.write("report.json", report)
    elif any(
        previous_report[key] != report[key]
        for key in (
            "contract_id",
            "base_question_count",
            "base_preserved",
            "recovered",
            "unresolved",
        )
    ):
        raise ValueError("Recovery decisions disagree with the completed report")
    logger.emit("recovery.completed", **report)
    return export_recovered(
        root, s3, bucket, manifest, inference, contract, recovered, contexts, logger
    )


def main() -> int:
    import boto3
    import tesserocr
    import torch

    if tesserocr.tesseract_version().splitlines()[0] != POLICY["tesseract"]:
        raise RuntimeError("Unexpected OCR runtime version")
    print(json.dumps(configure_runtime(torch), sort_keys=True), flush=True)
    logger = RuntimeEventLogger(
        "submission.recovery", jsonl_path=Path("/opt/ml/model/private/events.jsonl")
    )
    with logger.stage("submission.recovery", heartbeat_seconds=15):
        result = run(
            Path(__file__).resolve().parents[2],
            boto3.client("s3", region_name=os.environ["AWS_DEFAULT_REGION"]),
            os.environ["LAVA_BUCKET"],
            logger,
        )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
