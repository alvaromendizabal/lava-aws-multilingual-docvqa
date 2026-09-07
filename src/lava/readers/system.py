"""Label-blind retrieved-page inputs and durable end-to-end reader inference.

Keep the established retriever and semantic judge contracts unchanged. Every
completed stage is acknowledged only after checksum-verified S3 read-back.
"""

from __future__ import annotations

import base64
import json
import math
import os
import time
from collections.abc import Callable
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import pymupdf
import yaml
from botocore.exceptions import ClientError

from lava.evaluation.semantic import digest, encode
from lava.evaluation.submission_store import cached_source
from lava.readers.model_registry import load_resolved_model
from lava.readers.oracle_assets import (
    _page_layout,
    _render_page,
    document_aliases,
    parse_training_csv,
)
from lava.readers.parsing import ReaderOutputError, parse_reader_response
from lava.readers.private_artifacts import read_raw_response
from lava.readers.runtime_logging import RuntimeEventLogger
from lava.readers.schemas import OraclePageAsset, ReaderInput, ReaderPrediction, ReaderTelemetry
from lava.retrieval.lexical import PageText, RetrievalQuery
from lava.retrieval.pipeline import (
    ObjectStore,
    extract_pages,
    implementation_contract,
    rank_query,
    stage,
)
from lava.retrieval.storage import CheckpointStore

SYSTEM_PREFIX = "experiments/submissions/system"


def system_config(root: Path) -> dict[str, Any]:
    """Validate the deliberately bounded portfolio experiment, not a search space."""
    config = json.loads((root / "configs/system_evaluation.json").read_bytes())
    expected = {
        "schema_version": 1,
        "model_key": "qwen35_9b_fused_direct",
        "page_budget": 5,
        "retrieval_method": "bm25",
        "question_count": 16,
        "document_count": 5,
        "split": "training_diagnostic",
        "official_server_parity": False,
        "max_runtime_seconds": 1800,
        "max_pending_seconds": 86400,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("System configuration differs from the frozen final pilot")
    return config


def system_contract(root: Path) -> dict[str, Any]:
    """Stable across report-only commits and merges; invalidate changed inference inputs."""
    config = system_config(root)
    model = load_resolved_model(
        root / "configs/oracle_reader_models.lock.json", config["model_key"]
    )
    retrieval_config = json.loads((root / "configs/retrieval.json").read_bytes())
    paths = (
        "src/lava/readers/system.py",
        "src/lava/readers/qwen35.py",
        "src/lava/readers/schemas.py",
        "src/lava/readers/parsing.py",
        "src/lava/readers/prompts.py",
        "src/lava/readers/structured_output.py",
        "src/lava/readers/reader_factory.py",
        "src/lava/readers/private_artifacts.py",
        "pipelines/oracle_reader/requirements-gpu.txt",
        "configs/oracle_reader_benchmark.yaml",
        "src/lava/readers/hardware.py",
        "pipelines/oracle_reader/job_entry.py",
        "pipelines/oracle_reader/run.sh",
    )
    contract = {
        "schema_version": 1,
        "config": config,
        "model": model.model_dump(mode="json"),
        "retrieval": implementation_contract(root, retrieval_config),
        "source_sha256": {name: digest((root / name).read_bytes()) for name in paths},
    }
    return {**contract, "contract_id": digest(encode(contract))}


def store_for(s3: Any, bucket: str, contract_id: str) -> CheckpointStore:
    """All new private work remains within the previously granted project prefix."""
    if len(contract_id) != 64 or any(c not in "0123456789abcdef" for c in contract_id):
        raise ValueError("Invalid system contract identity")
    return CheckpointStore(s3, bucket, f"{SYSTEM_PREFIX}/{contract_id}")


def put_blob(s3: Any, bucket: str, key: str, payload: bytes, content_type: str) -> str | None:
    """Install immutable content; a retry verifies existing bytes rather than overwriting."""
    checksum = digest(payload)
    try:
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=payload,
            ContentType=content_type,
            IfNoneMatch="*",
            Metadata={"sha256": checksum},
            ChecksumSHA256=base64.b64encode(bytes.fromhex(checksum)).decode(),
        )
    except ClientError as error:
        if not CheckpointStore._precondition(error):
            raise
    response = s3.get_object(Bucket=bucket, Key=key)
    with response["Body"] as stream:
        observed = stream.read()
    if observed != payload or response.get("Metadata", {}).get("sha256") != checksum:
        raise ValueError("System artifact failed immutable read-back verification")
    return response.get("VersionId")


def select_pages(ranking: dict[str, Any], page_count: int, budget: int) -> tuple[int, ...]:
    """Select by label-blind rank; present selected physical pages in document order."""
    order = ranking["bm25"]
    if (
        not isinstance(order, list)
        or any(type(number) is not int for number in order)
        or len(order) != page_count
        or set(order) != set(range(1, page_count + 1))
        or type(budget) is not int
        or page_count < 1
        or not 1 <= budget <= 10
    ):
        raise ValueError("Ranking must cover every physical page exactly once")
    return tuple(sorted(order[:budget]))


def prepare_inputs(root: Path, s3: Any, bucket: str, logger: RuntimeEventLogger) -> dict[str, Any]:
    """Reuse the existing PDF/query checkpoints and render only the selected unique pages."""
    from lava.retrieval.evaluation import pilot_sources

    contract = system_contract(root)
    config = contract["config"]
    store = store_for(s3, bucket, contract["contract_id"])
    saved = store.read("inputs.json")
    if saved is not None:
        validate_manifest(saved, contract)
        logger.emit("system.inputs.reused", question_count=len(saved["inputs"]))
        return saved
    retrieval_config = contract["retrieval"]["config"]
    sources = pilot_sources(root, retrieval_config)
    cache = root / "artifacts/retrieval/inputs"
    references = parse_training_csv(
        cached_source(s3, bucket, cache / "train.csv", sources["train.csv"], logger)
    )
    aliases = document_aliases(references)
    if len(references) != config["question_count"] or len(aliases) != config["document_count"]:
        raise ValueError("Incomplete final-pilot source coverage")
    retrieval_id = digest(encode(contract["retrieval"]))
    retrieval_store = CheckpointStore(
        s3, bucket, f"experiments/submissions/retrieval/{retrieval_id}"
    )
    renderer = yaml.safe_load((root / "configs/oracle_reader_benchmark.yaml").read_bytes())[
        "asset_builder"
    ]
    render = renderer["render_profiles"][renderer["active_render_profile"]]
    inputs = []
    for document_id, alias in aliases.items():
        source = sources[f"{document_id}.pdf"]

        def load_pdf(source=source, document_id=document_id):
            return cached_source(s3, bucket, cache / f"{document_id}.pdf", source, logger)

        extraction, _ = stage(
            retrieval_store,
            {"implementation": retrieval_id, "source": source},
            lambda: extract_pages(load_pdf()),
            logger=logger,
            name="documents",
        )
        if extraction["pdf_sha256"] != source["sha256"]:
            raise ValueError("Retrieval extraction differs from frozen PDF")
        pages = tuple(PageText(**page) for page in extraction["pages"])
        page_assets: dict[int, OraclePageAsset] = {}
        # This projection deliberately excludes the answer and gold evidence fields.
        queries = [
            (RetrievalQuery(r.question_id, r.document_id, r.question), r.answer_format, r.language)
            for r in references
            if r.document_id == document_id
        ]
        for query, answer_format, language in queries:
            ranking, _ = stage(
                retrieval_store,
                {
                    "implementation": retrieval_id,
                    "query": asdict(query),
                    "pages_sha256": digest(encode([asdict(page) for page in pages])),
                },
                partial(rank_query, query, pages, retrieval_config["bm25"]),
                logger=logger,
                name="queries",
            )
            if ranking["question_id"] != query.question_id or ranking["document_id"] != document_id:
                raise ValueError("Ranking belongs to a different request")
            selected = select_pages(ranking, len(pages), config["page_budget"])
            for number in selected:
                if number in page_assets:
                    continue
                identity = {
                    "source": source,
                    "render": render,
                    "page": number,
                    "renderer": contract["retrieval"]["source_sha256"],
                    "pymupdf": contract["retrieval"]["pymupdf_version"],
                }

                def render_asset(
                    number: int = number,
                    identity: dict[str, Any] = identity,
                    document_id: str = document_id,
                    alias: str = alias,
                    source: dict[str, str] = source,
                    load_pdf: Callable[[], bytes] = load_pdf,
                ) -> dict[str, Any]:
                    with pymupdf.open(stream=load_pdf(), filetype="pdf") as document:
                        page = document[number - 1]
                        image, width, height = _render_page(page, **render)
                        text = page.get_text("text", sort=True).encode()
                        layout, words, blocks, images = _page_layout(page)
                    base = f"{SYSTEM_PREFIX}/assets/{digest(encode(identity))}"
                    values: dict[str, Any] = {}
                    for kind, extension, payload, mime in (
                        ("image", "png", image, "image/png"),
                        ("text", "txt", text, "text/plain; charset=utf-8"),
                        ("layout", "json", layout, "application/json"),
                    ):
                        key = f"{base}/{kind}.{extension}"
                        put_blob(s3, bucket, key, payload, mime)
                        values.update(
                            {
                                f"{kind}_s3_uri": f"s3://{bucket}/{key}",
                                f"{kind}_sha256": digest(payload),
                                f"{kind}_version_id": None,
                            }
                        )
                    asset = OraclePageAsset(
                        asset_version="retrieved-pages-v1",
                        document_id=document_id,
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
                        **values,
                    )
                    return asset.model_dump(mode="json")

                asset, _ = stage(store, identity, render_asset, logger=logger, name="pages")
                page_assets[number] = OraclePageAsset.model_validate(asset)
            request = ReaderInput(
                question_id=query.question_id,
                document_id=document_id,
                document_alias=alias,
                question=query.question,
                answer_format=answer_format,
                language=language,
                pages=tuple(page_assets[number] for number in selected),
            )
            inputs.append(request.model_dump(mode="json"))
            logger.emit("system.input.prepared", completed=len(inputs), total=len(references))
    inputs.sort(key=lambda row: row["question_id"])
    manifest = {"schema_version": 1, "contract": contract, "inputs": inputs}
    manifest["inputs_sha256"] = digest(encode(inputs))
    validate_manifest(manifest, contract)
    store.write("inputs.json", manifest)
    if store.read("inputs.json") != manifest:
        raise ValueError("System input manifest failed durable read-back")
    logger.emit("system.inputs.completed", question_count=len(inputs), document_count=len(aliases))
    return manifest


def validate_manifest(
    manifest: dict[str, Any], contract: dict[str, Any]
) -> tuple[ReaderInput, ...]:
    """Reject mixed contracts, label-bearing requests and incomplete manifests."""
    if manifest.get("schema_version") != 1 or manifest.get("contract") != contract:
        raise ValueError("System input contract mismatch")
    inputs = tuple(ReaderInput.model_validate(row) for row in manifest["inputs"])
    config = contract["config"]
    if (
        len(inputs) != config["question_count"]
        or len({row.question_id for row in inputs}) != len(inputs)
        or len({row.document_id for row in inputs}) != config["document_count"]
        or any(len(row.pages) > config["page_budget"] for row in inputs)
        or digest(encode(manifest["inputs"])) != manifest.get("inputs_sha256")
    ):
        raise ValueError("System input coverage or checksum mismatch")
    return inputs


def validate_prediction(
    request: ReaderInput, saved: dict[str, Any]
) -> tuple[ReaderPrediction, ReaderTelemetry]:
    """Independently parse the exact saved generation; failed outputs remain counted failures."""
    if saved.get("input_sha256") != digest(encode(request.model_dump(mode="json"))):
        raise ValueError("Saved answer belongs to different reader inputs")
    raw = saved["raw_response"]
    try:
        parsed = parse_reader_response(
            question_id=request.question_id,
            answer_format=request.answer_format,
            raw_response=raw,
            allowed_pages=request.available_pages,
        )
    except ReaderOutputError as error:
        parsed = ReaderPrediction(
            question_id=request.question_id,
            answer_format=request.answer_format,
            answer="",
            evidence_pages=(),
            confidence=0.0,
            abstain=True,
            schema_valid=False,
            parser_error_code=error.code,
            raw_response_sha256=digest(raw.encode()),
        )
    prediction = ReaderPrediction.model_validate(saved["prediction"])
    if prediction != parsed:
        raise ValueError("Saved prediction disagrees with the independently parsed generation")
    telemetry = ReaderTelemetry.model_validate(saved["telemetry"])
    numeric = [value for value in telemetry.model_dump().values() if isinstance(value, float)]
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("Nonfinite reader telemetry")
    if telemetry.image_count != len(request.pages):
        raise ValueError("Telemetry image coverage differs from reader inputs")
    return prediction, telemetry


def infer_requests(
    manifest: dict[str, Any],
    contract: dict[str, Any],
    store: ObjectStore,
    predict: Callable[[ReaderInput], tuple[ReaderPrediction, ReaderTelemetry]],
    logger: RuntimeEventLogger,
    *,
    read_raw: Callable[[str], str] = read_raw_response,
) -> dict[str, Any]:
    """Resume each question independently; never instantiate weights on a fully cached run."""
    requests = validate_manifest(manifest, contract)
    started = time.perf_counter()
    records = []
    reused = 0
    for number, request in enumerate(requests, 1):
        key = f"answers/{digest(request.question_id.encode())}.json"
        saved = store.read(key)
        if saved is None:
            with logger.stage("system.question.inference", heartbeat_seconds=15):
                prediction, telemetry = predict(request)
                saved = {
                    "contract_id": contract["contract_id"],
                    "input_sha256": digest(encode(request.model_dump(mode="json"))),
                    "prediction": prediction.model_dump(mode="json"),
                    "telemetry": telemetry.model_dump(mode="json"),
                    "raw_response": read_raw(request.question_id),
                    "inference_code_commit": os.environ.get("LAVA_GIT_COMMIT_SHA"),
                }
                validate_prediction(request, saved)
                store.write(key, saved)
                if store.read(key) != saved:
                    raise ValueError("Completed answer failed durable read-back")
        else:
            reused += 1
        if saved.get("contract_id") != contract["contract_id"]:
            raise ValueError("Answer checkpoint uses a different inference contract")
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
        "contract_id": contract["contract_id"],
        "inputs_sha256": manifest["inputs_sha256"],
        "records": records,
    }
    store.write("inference.json", result)
    if store.read("inference.json") != result:
        raise ValueError("Inference completion failed durable read-back")
    logger.emit(
        "system.inference.completed",
        completed=len(records),
        reused_count=reused,
        new_count=len(records) - reused,
        total_elapsed_seconds=round(time.perf_counter() - started, 3),
    )
    return result


def run_inference(
    root: Path, s3: Any, bucket: str, identity: str, logger: RuntimeEventLogger
) -> dict:
    """GPU entry point. This function never reads labels or evaluation answers."""
    from lava.readers.reader_factory import build_reader

    contract = system_contract(root)
    if contract["contract_id"] != identity:
        raise ValueError("Remote code/model/retrieval contract differs from the prepared inputs")
    store = store_for(s3, bucket, identity)
    manifest = store.read("inputs.json")
    if manifest is None:
        raise ValueError("Prepare and verify reader inputs before allocating GPU work")
    # Reader construction is lazy; model loading occurs only for an uncached question.
    model = load_resolved_model(
        root / "configs/oracle_reader_models.lock.json", contract["config"]["model_key"]
    )
    reader = build_reader(model, region=os.environ.get("AWS_DEFAULT_REGION", "us-west-2"))
    return infer_requests(manifest, contract, store, reader.predict, logger)
