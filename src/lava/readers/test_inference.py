"""User-operated full-test inference and CSV export; never call the Kaggle API.

The training diagnostic and test export have separate contracts. Test requests
contain no labels. Artifacts are private, immutable, and checkpointed per answer.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
from collections import defaultdict
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import pymupdf
import yaml
from pydantic import Field

from lava.evaluation.schemas import AnswerFormat
from lava.evaluation.semantic import digest, encode
from lava.evaluation.submission import (
    SubmissionInputs,
    build_submission,
    load_submission_inputs,
    validate_submission_csv,
)
from lava.evaluation.submission_store import atomic_write, cached_source, persist_submission
from lava.readers.model_registry import load_resolved_model
from lava.readers.oracle_assets import _page_layout, _render_page
from lava.readers.private_artifacts import read_raw_response
from lava.readers.runtime_logging import RuntimeEventLogger
from lava.readers.schemas import OraclePageAsset, ReaderInput
from lava.readers.system import (
    put_blob,
    select_pages,
    store_for,
    system_contract,
    validate_prediction,
)
from lava.readers.system_execution import execute_prepared, training_request
from lava.retrieval.lexical import PageText, RetrievalQuery
from lava.retrieval.pipeline import extract_pages, rank_query, stage
from lava.retrieval.storage import CheckpointStore


class TestPageAsset(OraclePageAsset):
    """Extend presentation aliases for the 200-document test set, without changing the pilot."""

    __test__ = False
    document_alias: str = Field(pattern=r"^doc-[0-9]{3}$")


class TestReaderInput(ReaderInput):
    """Same label-free input semantics, with distinct aliases for all test documents."""

    __test__ = False
    document_alias: str = Field(pattern=r"^doc-[0-9]{3}$")
    pages: tuple[TestPageAsset, ...] = Field(min_length=1, max_length=10)


def test_contract(root: Path) -> dict[str, Any]:
    """Pin test inputs and implementation independently of the existing pilot contract."""
    contract = system_contract(root)
    contract.pop("contract_id")
    submission = json.loads((root / "configs/submission.json").read_bytes())
    contract["config"] = {
        **contract["config"],
        "split": "test",
        "question_count": submission["expected_question_count"],
        "document_count": submission["expected_document_count"],
        "max_runtime_seconds": submission["maximum_inference_seconds"],
    }
    contract["submission"] = submission
    for name in ("src/lava/readers/test_inference.py", "pipelines/submission/run.sh"):
        contract["source_sha256"][name] = digest((root / name).read_bytes())
    return {**contract, "contract_id": digest(encode(contract))}


def submission_inputs(
    root: Path, s3: Any, bucket: str, logger: RuntimeEventLogger
) -> SubmissionInputs:
    """Read only the checksum-pinned test/template files, never training labels."""
    config = json.loads((root / "configs/submission.json").read_bytes())
    data = {
        name: cached_source(s3, bucket, root / "artifacts/submission/inputs" / name, source, logger)
        for name, source in config["source_files"].items()
    }
    return load_submission_inputs(data["test.csv"], data["sample_submission.csv"], config)


def test_sources(
    root: Path, inputs: SubmissionInputs, contract: dict[str, Any]
) -> dict[str, dict[str, str]]:
    """Bind every test PDF to the original versioned data audit, not directory guesses."""
    payload = (root / "reports/raw_data_manifest.csv").read_bytes()
    if digest(payload) != contract["retrieval"]["config"]["source_manifest_sha256"]:
        raise ValueError("Raw PDF inventory checksum mismatch")
    sources = {}
    for row in csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))):
        if not row["name"].startswith("test_pdfs/"):
            continue
        document = Path(row["name"]).stem
        if document in sources or not re.fullmatch(r"[A-Za-z0-9_-]+", document):
            raise ValueError("Duplicate or unsafe test document identity")
        if not row["version_id"] or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError("Test PDF requires an immutable version and checksum")
        sources[document] = {
            "key": row["s3_key"],
            "sha256": row["sha256"],
            "version_id": row["version_id"],
        }
    if set(sources) != {row["file_id"] for row in inputs.questions.values()}:
        raise ValueError("Test questions and audited test PDFs have different coverage")
    return sources


def validate_test_manifest(
    manifest: dict[str, Any], contract: dict[str, Any]
) -> tuple[TestReaderInput, ...]:
    """Reject label-bearing, partial, cross-contract, or out-of-range test requests."""
    if manifest.get("contract") != contract or contract["config"]["split"] != "test":
        raise ValueError("Test inference contract mismatch")
    requests = tuple(TestReaderInput.model_validate(row) for row in manifest["inputs"])
    counts = manifest["page_counts"]
    if (
        manifest.get("schema_version") != 1
        or len(requests) != contract["config"]["question_count"]
        or len({row.question_id for row in requests}) != len(requests)
        or len(counts) != contract["config"]["document_count"]
        or set(counts) != {row.document_id for row in requests}
        or any(type(n) is not int or n < 1 for n in counts.values())
        or any(
            len(row.pages) != min(counts[row.document_id], contract["config"]["page_budget"])
            for row in requests
        )
        or any(page.page_number > counts[row.document_id] for row in requests for page in row.pages)
        or digest(encode(manifest["inputs"])) != manifest.get("inputs_sha256")
        or digest(encode(counts)) != manifest.get("page_counts_sha256")
    ):
        raise ValueError("Test input coverage, page range or checksum mismatch")
    return requests


def render_test_asset(
    s3: Any,
    bucket: str,
    payload: bytes,
    source: dict[str, str],
    document: str,
    alias: str,
    number: int,
    render: dict[str, Any],
    identity: dict[str, Any],
) -> dict[str, Any]:
    """Render one selected page; commit exact bytes before its reusable completion record."""
    with pymupdf.open(stream=payload, filetype="pdf") as pdf:
        page = pdf[number - 1]
        image, width, height = _render_page(page, **render)
        text = page.get_text("text", sort=True).encode()
        layout, words, blocks, images = _page_layout(page)
    fields: dict[str, Any] = {}
    for kind, suffix, data, mime in (
        ("image", "png", image, "image/png"),
        ("text", "txt", text, "text/plain"),
        ("layout", "json", layout, "application/json"),
    ):
        key = f"experiments/submissions/test-assets/{digest(encode(identity))}/{kind}.{suffix}"
        put_blob(s3, bucket, key, data, mime)
        fields.update({f"{kind}_s3_uri": f"s3://{bucket}/{key}", f"{kind}_sha256": digest(data)})
    return TestPageAsset(
        asset_version="test-retrieved-pages-v1",
        document_id=document,
        document_alias=alias,
        page_number=number,
        source_pdf_s3_uri=f"s3://{bucket}/{source['key']}",
        source_pdf_sha256=source["sha256"],
        source_pdf_version_id=source["version_id"],
        width_pixels=width,
        height_pixels=height,
        dpi=render["dpi"],
        native_text_characters=len(text.decode()),
        word_count=words,
        text_block_count=blocks,
        embedded_image_count=images,
        **fields,
    ).model_dump(mode="json")


def prepare_test_inputs(
    root: Path, s3: Any, bucket: str, logger: RuntimeEventLogger
) -> dict[str, Any]:
    """Process one PDF at a time; independently resume extraction, ranking and page rendering."""
    contract = test_contract(root)
    store = store_for(s3, bucket, contract["contract_id"])
    saved = store.read("inputs.json")
    if saved is not None:
        validate_test_manifest(saved, contract)
        logger.emit("export.inputs.reused", question_count=len(saved["inputs"]))
        return saved
    inputs = submission_inputs(root, s3, bucket, logger)
    sources = test_sources(root, inputs, contract)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for question in inputs.questions.values():
        grouped[question["file_id"]].append(question)
    renderer = yaml.safe_load((root / "configs/oracle_reader_benchmark.yaml").read_bytes())[
        "asset_builder"
    ]
    render = renderer["render_profiles"][renderer["active_render_profile"]]
    retrieval_id = digest(encode(contract["retrieval"]))
    retrieval_store = CheckpointStore(
        s3, bucket, f"experiments/submissions/retrieval/{retrieval_id}"
    )
    requests: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for index, (document, questions) in enumerate(sorted(grouped.items()), 1):
        source, alias = sources[document], f"doc-{index:03d}"
        # One in-memory PDF, never the 200-document corpus. Original S3 bytes remain versioned.
        payload = cached_source(
            s3, bucket, root / "artifacts/submission/inputs" / f"{document}.pdf", source, logger
        )
        extraction, _ = stage(
            retrieval_store,
            {"implementation": retrieval_id, "source": source},
            partial(extract_pages, payload),
            logger=logger,
            name="documents",
        )
        if extraction["pdf_sha256"] != source["sha256"]:
            raise ValueError("Cached test extraction does not match the audited PDF")
        pages = tuple(PageText(**page) for page in extraction["pages"])
        counts[document] = len(pages)
        assets = {}
        for question in questions:
            query = RetrievalQuery(question["id"], document, question["question"])
            ranking, _ = stage(
                retrieval_store,
                {
                    "implementation": retrieval_id,
                    "query": asdict(query),
                    "pages_sha256": digest(encode([asdict(page) for page in pages])),
                },
                partial(rank_query, query, pages, contract["retrieval"]["config"]["bm25"]),
                logger=logger,
                name="queries",
            )
            if ranking["question_id"] != query.question_id or ranking["document_id"] != document:
                raise ValueError("Cached test ranking belongs to a different question")
            selected = select_pages(ranking, len(pages), contract["config"]["page_budget"])
            for number in selected:
                if number not in assets:
                    identity = {
                        "source": source,
                        "render": render,
                        "page": number,
                        "alias": alias,
                        "implementation": contract["source_sha256"][
                            "src/lava/readers/test_inference.py"
                        ],
                        "pymupdf": contract["retrieval"]["pymupdf_version"],
                    }
                    asset, _ = stage(
                        store,
                        identity,
                        partial(
                            render_test_asset,
                            s3,
                            bucket,
                            payload,
                            source,
                            document,
                            alias,
                            number,
                            render,
                            identity,
                        ),
                        logger=logger,
                        name="pages",
                    )
                    assets[number] = TestPageAsset.model_validate(asset)
            request = TestReaderInput(
                question_id=query.question_id,
                document_id=document,
                document_alias=alias,
                question=query.question,
                answer_format=AnswerFormat(question["answer_format"]),
                language=question["language"],
                pages=tuple(assets[n] for n in selected),
            )
            requests.append(request.model_dump(mode="json"))
            logger.emit(
                "export.input.prepared",
                completed=len(requests),
                total=len(inputs.order),
                documents_completed=index - 1,
            )
    requests.sort(key=lambda row: row["question_id"])
    manifest = {
        "schema_version": 1,
        "contract": contract,
        "inputs": requests,
        "inputs_sha256": digest(encode(requests)),
        "page_counts": counts,
        "page_counts_sha256": digest(encode(counts)),
    }
    validate_test_manifest(manifest, contract)
    store.write("inputs.json", manifest)
    if store.read("inputs.json") != manifest:
        raise ValueError("Test inputs failed durable read-back")
    logger.emit("export.inputs.completed", questions=len(requests), documents=len(counts))
    return manifest


def test_training_request(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Reuse the guarded job machinery with a distinct name, entrypoint and output prefix."""
    request = training_request(*args, manifest_validator=validate_test_manifest, **kwargs)
    request["TrainingJobName"] = request["TrainingJobName"].replace(
        "lava-system-", "lava-export-", 1
    )
    request["OutputDataConfig"]["S3OutputPath"] = request["OutputDataConfig"][
        "S3OutputPath"
    ].replace("/system-runs/", "/test-runs/")
    request["AlgorithmSpecification"]["ContainerArguments"][1] = request["AlgorithmSpecification"][
        "ContainerArguments"
    ][1].replace("/oracle_reader/run.sh", "/submission/run.sh")
    request["Environment"]["PYTHONPATH"] = "/opt/ml/code/src"
    request["Tags"][1]["Value"] = "user-test-export"
    return request


def execute_test(
    root: Path,
    session: Any,
    bucket: str,
    region: str,
    logger: RuntimeEventLogger,
    *,
    acknowledge_charges: str,
    attempt: int = 1,
    allow_retry: bool = False,
    maximum_training_usd: float = 15.0,
) -> dict[str, Any]:
    """User-authorized single GPU attempt; $15 is an estimate guard, not an account cap."""
    return execute_prepared(
        root,
        session,
        bucket,
        region,
        logger,
        contract=test_contract(root),
        request_factory=test_training_request,
        manifest_validator=validate_test_manifest,
        receipt_directory="artifacts/submission/runtime",
        acknowledge_charges=acknowledge_charges,
        attempt=attempt,
        allow_retry=allow_retry,
        maximum_training_usd=maximum_training_usd,
    )


def infer_test(root: Path, s3: Any, bucket: str, identity: str, logger: RuntimeEventLogger) -> None:
    """Remote inference reads only verified test requests and saves every model response."""
    from lava.readers.reader_factory import build_reader

    contract = test_contract(root)
    if contract["contract_id"] != identity:
        raise ValueError("GPU test contract differs from prepared inputs")
    store = store_for(s3, bucket, identity)
    manifest = store.read("inputs.json")
    if manifest is None:
        raise ValueError("Test inputs have not been prepared")
    requests = validate_test_manifest(manifest, contract)
    reader = build_reader(
        load_resolved_model(
            root / "configs/oracle_reader_models.lock.json", contract["config"]["model_key"]
        ),
        region=os.environ.get("AWS_DEFAULT_REGION", "us-west-2"),
    )
    records, reused = [], 0
    for number, request in enumerate(requests, 1):
        key = f"answers/{digest(request.question_id.encode())}.json"
        saved = store.read(key)
        if saved is None:
            with logger.stage("system.question.inference", heartbeat_seconds=15):
                prediction, telemetry = reader.predict(request)
                saved = {
                    "contract_id": identity,
                    "input_sha256": digest(encode(request.model_dump(mode="json"))),
                    "prediction": prediction.model_dump(mode="json"),
                    "telemetry": telemetry.model_dump(mode="json"),
                    "raw_response": read_raw_response(request.question_id),
                    "inference_code_commit": os.environ.get("LAVA_GIT_COMMIT_SHA"),
                }
                validate_prediction(request, saved)
                store.write(key, saved)
                if store.read(key) != saved:
                    raise ValueError("Test answer failed durable read-back")
        else:
            reused += 1
        if saved.get("contract_id") != identity:
            raise ValueError("Test answer checkpoint contract mismatch")
        prediction, _ = validate_prediction(request, saved)
        records.append(saved)
        logger.emit(
            "system.question.completed",
            completed=number,
            total=len(requests),
            reused_count=reused,
            schema_valid=prediction.schema_valid,
        )
    result = {
        "schema_version": 1,
        "contract_id": identity,
        "inputs_sha256": manifest["inputs_sha256"],
        "records": records,
    }
    store.write("inference.json", result)
    if store.read("inference.json") != result:
        raise ValueError("Test inference failed durable read-back")
    logger.emit(
        "system.job.completed",
        completed=len(records),
        reused_count=reused,
        uploaded_to_kaggle=False,
    )


def export_test(root: Path, s3: Any, bucket: str, logger: RuntimeEventLogger) -> Path:
    """Validate complete real predictions, then materialize a user-downloadable CSV."""
    contract = test_contract(root)
    store = store_for(s3, bucket, contract["contract_id"])
    manifest, inference = store.read("inputs.json"), store.read("inference.json")
    if manifest is None or inference is None:
        raise ValueError("Full test inference is not complete; no submission.csv was created")
    requests = validate_test_manifest(manifest, contract)
    if (
        inference.get("contract_id") != contract["contract_id"]
        or inference.get("inputs_sha256") != manifest["inputs_sha256"]
        or len(inference["records"]) != len(requests)
    ):
        raise ValueError("Incomplete or incompatible test inference")
    inputs = submission_inputs(root, s3, bucket, logger)
    for request in requests:
        original = inputs.questions.get(request.question_id)
        if original is None or any(
            getattr(request, field) != original[column]
            for field, column in (
                ("document_id", "file_id"),
                ("question", "question"),
                ("answer_format", "answer_format"),
                ("language", "language"),
            )
        ):
            raise ValueError("Saved reader request differs from the pinned test question")
    predictions: list[dict[str, Any]] = []
    commits: set[str] = set()
    invalid: list[str] = []
    for request, saved in zip(requests, inference["records"], strict=True):
        if saved.get("contract_id") != contract["contract_id"]:
            raise ValueError("Per-answer test contract mismatch")
        prediction, _ = validate_prediction(request, saved)
        if not prediction.schema_valid or prediction.abstain:
            invalid.append(request.question_id)
        predictions.append(
            {
                "question_id": request.question_id,
                "answer": prediction.answer,
                "evidence_pages": list(prediction.evidence_pages),
            }
        )
        commits.add(saved["inference_code_commit"])
    if invalid:
        # Never invent fallback answers or drop a failed row to get an export through validation.
        atomic_write(
            root / "artifacts/submission/invalid_questions.json", encode({"question_ids": invalid})
        )
        raise ValueError(
            f"{len(invalid)} invalid/abstained answers require review; no export created"
        )
    if not commits or any(
        not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value) for value in commits
    ):
        raise ValueError("Every answer requires an immutable inference code revision")
    payload = b"".join(encode(row) for row in predictions)
    provenance = {
        "split": "test",
        "evidence_scope": "retrieved_full_document_pages",
        "test_csv_sha256": inputs.source_sha256["test.csv"],
        "question_count": len(inputs.order),
        "model_id": contract["model"]["model_id"],
        "model_revision": contract["model"]["revision"],
        "code_commit": min(commits),
        "inference_code_commits": sorted(commits),
        "code_commit_semantics": "representative revision; all compatible revisions are recorded",
        "predictions_sha256": digest(payload),
        "page_counts_sha256": manifest["page_counts_sha256"],
        "inference_contract": contract["contract_id"],
    }
    csv_payload = build_submission(inputs, payload, manifest["page_counts"], provenance)
    validation = validate_submission_csv(csv_payload, inputs, manifest["page_counts"])
    output_manifest = {
        "schema_version": 1,
        "competition": contract["submission"]["competition"],
        "source_sha256": inputs.source_sha256,
        "prediction_sha256": digest(payload),
        "provenance": provenance,
        "submission_sha256": digest(csv_payload),
        "validation": validation,
    }
    target = persist_submission(s3, bucket, root, csv_payload, output_manifest, logger)
    for name, data in (
        ("predictions.jsonl", payload),
        ("page_counts.json", encode(manifest["page_counts"])),
        ("provenance.json", encode(provenance)),
    ):
        atomic_write(target.parent / name, data)
    # Current is a verified local convenience pointer; immutable version remains in S3.
    atomic_write(root / "artifacts/submission/submission.csv", csv_payload)
    atomic_write(root / "artifacts/submission/manifest.json", encode(output_manifest))
    logger.emit(
        "export.download.ready",
        filename="artifacts/submission/submission.csv",
        rows=len(inputs.order),
        sha256=digest(csv_payload),
        uploaded_to_kaggle=False,
    )
    return target


if __name__ == "__main__":
    import boto3

    root = Path(__file__).resolve().parents[3]
    log = RuntimeEventLogger("system.job", jsonl_path=Path("/opt/ml/model/private/events.jsonl"))
    with log.stage("system.job", heartbeat_seconds=15):
        infer_test(
            root,
            boto3.client("s3", region_name=os.environ["AWS_DEFAULT_REGION"]),
            os.environ["LAVA_BUCKET"],
            os.environ["LAVA_SYSTEM_CONTRACT"],
            log,
        )
