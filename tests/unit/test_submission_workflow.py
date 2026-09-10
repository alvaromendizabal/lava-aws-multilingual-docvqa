"""Real synthetic-PDF workflow regressions; not model-performance measurements."""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import runpy
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import boto3
import pymupdf
import pytest
import yaml
from botocore.exceptions import ClientError, EndpointConnectionError
from botocore.stub import Stubber
from botocore.validate import validate_parameters

from lava.evaluation.semantic import digest, encode
from lava.evaluation.submission import SUBMISSION_COLUMNS, TEST_COLUMNS
from lava.notebook_operations import download_link, run_operation
from lava.readers import reader_factory
from lava.readers import test_inference as workflow
from lava.readers.parsing import parse_reader_response
from lava.readers.runtime_logging import RuntimeEventLogger
from lava.readers.schemas import OraclePageAsset, ReaderTelemetry
from lava.readers.system import store_for
from lava.readers.system_execution import describe_or_none, ensure_job, verify_committed_source

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def recovery_module(monkeypatch):
    # Register a real module, matching multiprocessing/import behavior in the container.
    # The managed entrypoint adds the repository root to its module search path.
    monkeypatch.syspath_prepend(str(ROOT))
    spec = importlib.util.spec_from_file_location(
        "lava_submission_recovery", ROOT / "pipelines/submission/recovery.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_recovery_page_search_keeps_full_pdf_coverage_and_bounded_context(recovery_module):
    ranking = tuple((n, 10 - i) for i, n in enumerate([3, 8, 2, 1, 4, 5, 6, 7]))
    sets = recovery_module.page_search_order(ranking, 8)
    assert sets[0] == (1, 2, 3, 8)
    assert (2, 3, 4) in sets and (7, 8) in sets
    assert {page for pages in sets for page in pages} == set(range(1, 9))
    assert all(1 <= len(pages) <= 4 for pages in sets)
    assert len(sets) == len(set(sets))
    with pytest.raises(ValueError, match="each physical"):
        recovery_module.page_search_order(((1, 1.0), (1, 0.0)), 2)
    assert recovery_module.normalize_ocr("防 災 対 策 water supply") == "防災対策 water supply"


@pytest.fixture
def recovery_case(prepared, recovery_module, monkeypatch):
    p, module = prepared, recovery_module
    workflow.infer_test(
        p.root, p.s3, "bucket", p.contract["contract_id"], RuntimeEventLogger("test")
    )
    inference = deepcopy(
        store_for(p.s3, "bucket", p.contract["contract_id"]).read("inference.json")
    )
    requests = workflow.validate_test_manifest(p.manifest, p.contract)
    original = requests[0]
    raw = '{"answer":"","evidence_pages":[],"confidence":0.0,"abstain":true}'
    inference["records"][0].update(
        raw_response=raw,
        prediction=parse_reader_response(
            question_id=original.question_id,
            answer_format=original.answer_format,
            raw_response=raw,
            allowed_pages=original.available_pages,
        ).model_dump(mode="json"),
    )
    monkeypatch.setattr(module, "test_contract", lambda _: deepcopy(p.contract))
    (p.root / "pipelines/submission").mkdir(parents=True)
    for name in ("recovery.py", "run_recovery.sh", "inference.py"):
        shutil.copy(ROOT / "pipelines/submission" / name, p.root / "pipelines/submission" / name)
    contract = module.recovery_contract(p.root, p.contract, digest(encode(inference)))
    context = {
        n: {"number": n, "native": f"native page {n}", "ocr": f"OCR page {n}"} for n in (1, 2, 3)
    }
    candidate = workflow.TestReaderInput.model_validate(
        {
            **original.model_dump(mode="json"),
            "pages": [original.pages[1].model_dump(mode="json")],
        }
    )
    raw = '{"answer":"RECOVERED_VALUE","evidence_pages":[2],"confidence":0.8,"abstain":false}'
    saved = deepcopy(inference["records"][0])
    saved.update(
        contract_id=contract["contract_id"],
        request=candidate.model_dump(mode="json"),
        input_sha256=digest(encode(candidate.model_dump(mode="json"))),
        context_sha256=digest(encode({"2": context[2]})),
        raw_response=raw,
        inference_code_commit="c" * 40,
        prediction=parse_reader_response(
            question_id=candidate.question_id,
            answer_format=candidate.answer_format,
            raw_response=raw,
            allowed_pages=candidate.available_pages,
        ).model_dump(mode="json"),
    )
    saved["telemetry"]["image_count"] = 1
    saved["telemetry"]["raw_response_characters"] = len(raw)
    monkeypatch.setenv("LAVA_GIT_COMMIT_SHA", "c" * 40)
    return SimpleNamespace(
        p=p,
        module=module,
        inference=inference,
        contract=contract,
        original=original,
        context=context,
        saved=saved,
    )


def test_recovery_export_preserves_complete_base_answers_and_exact_template_order(recovery_case):
    c = recovery_case
    before = deepcopy(c.p.s3.values)
    result = c.module.export_recovered(
        c.p.root,
        c.p.s3,
        "bucket",
        c.p.manifest,
        c.inference,
        c.contract,
        {c.original.question_id: c.saved},
        {c.original.document_id: c.context},
        RuntimeEventLogger("test"),
    )
    for key, value in before.items():
        assert c.p.s3.values[key] == value
    payload = (c.p.root / "artifacts/submission/submission.csv").read_bytes()
    rows = list(csv.DictReader(io.StringIO(payload.decode())))
    assert [row["id"] for row in rows] == ["q3", "q2", "q1", "q0"]
    assert [row["answer"] for row in rows] == ["PRIVATE_ANSWER"] * 3 + ["RECOVERED_VALUE"]
    assert rows[-1]["evidence_page_number"] == "[2]"
    assert result["validation"]["row_count"] == 4
    assert result["submission_sha256"] == digest(payload)
    manifest = json.loads((c.p.root / "artifacts/submission/manifest.json").read_bytes())
    assert manifest["provenance"]["preserved_answer_count"] == 3
    assert manifest["provenance"]["recovered_answer_count"] == 1
    assert manifest["provenance"]["inference_code_commits"] == ["b" * 40, "c" * 40]
    assert store_for(c.p.s3, "bucket", c.contract["contract_id"]).read("export.json") == result


@pytest.mark.parametrize(
    "mutation", ["missing", "question", "source", "context", "prediction", "base"]
)
def test_recovery_rejects_missing_or_tampered_answers_before_csv_creation(recovery_case, mutation):
    c = recovery_case
    recovered = {c.original.question_id: deepcopy(c.saved)}
    if mutation == "missing":
        recovered.clear()
    elif mutation == "question":
        recovered[c.original.question_id]["request"]["question"] = "different question"
    elif mutation == "source":
        recovered[c.original.question_id]["request"]["pages"][0]["source_pdf_sha256"] = "f" * 64
    elif mutation == "context":
        recovered[c.original.question_id]["context_sha256"] = "f" * 64
    elif mutation == "prediction":
        recovered[c.original.question_id]["prediction"]["answer"] = "fabricated"
    else:
        c.inference["records"].pop()
    with pytest.raises(ValueError):
        c.module.export_recovered(
            c.p.root,
            c.p.s3,
            "bucket",
            c.p.manifest,
            c.inference,
            c.contract,
            recovered,
            {c.original.document_id: c.context},
            RuntimeEventLogger("test"),
        )
    assert not (c.p.root / "artifacts/submission/submission.csv").exists()


def test_recovery_never_converts_abstention_to_a_complete_answer(recovery_case):
    c = recovery_case
    _, failed = c.module.inspect_base(c.p.manifest, c.inference, c.p.contract)
    assert failed == [c.original.question_id]
    prediction, _ = workflow.validate_prediction(c.original, c.inference["records"][0])
    assert not c.module.complete_answer(prediction)


@pytest.fixture
def targeted_module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from pipelines.submission import targeted

    return targeted


def test_targeted_routes_partition_remaining_ids_and_reject_answer_hints(targeted_module):
    module = targeted_module
    requests = (SimpleNamespace(question_id="q0", document_id="d0"),)
    good = {
        "routes": {"q0": [{"pages": [1, 3], "crops": {"3": [0.1, 0.2, 0.8, 0.9]}}]},
        "blocked": {},
    }
    module.validate_routes(good, requests, {"q0"}, {"d0": 3})
    for mutation in ("overlap", "missing", "page", "duplicate", "crop", "answer", "bool", "count"):
        plan = deepcopy(good)
        if mutation == "overlap":
            plan["blocked"]["q0"] = "Unrelated source document"
        elif mutation == "missing":
            plan["routes"].clear()
        elif mutation == "page":
            plan["routes"]["q0"][0]["pages"] = [1, 4]
        elif mutation == "duplicate":
            plan["routes"]["q0"] *= 2
        elif mutation == "crop":
            plan["routes"]["q0"][0]["crops"]["3"][0] = float("nan")
        elif mutation == "answer":
            plan["routes"]["q0"][0]["answer"] = "Not allowed"
        elif mutation == "bool":
            plan["routes"]["q0"][0]["pages"] = [True, 3]
        else:
            plan["routes"]["q0"] *= 3
        with pytest.raises(ValueError):
            module.validate_routes(plan, requests, {"q0"}, {"d0": 3})
    module.validate_routes(
        {"routes": {}, "blocked": {"q0": "No matching evidence"}}, requests, {"q0"}, {"d0": 3}
    )


def test_targeted_reparses_generation_without_changing_question_or_source(
    recovery_case, targeted_module
):
    c, module = recovery_case, targeted_module
    route = {"pages": [2], "crops": {}}
    saved = {**deepcopy(c.saved), "route": route}
    assert (
        module.check_targeted(saved, c.original, c.contract["contract_id"], route).answer
        == "RECOVERED_VALUE"
    )
    for mutation in ("question", "source", "raw", "route", "commit"):
        changed = deepcopy(saved)
        if mutation == "question":
            changed["request"]["question"] = "different question"
        elif mutation == "source":
            changed["request"]["pages"][0]["source_pdf_sha256"] = "f" * 64
        elif mutation == "raw":
            changed["raw_response"] = (
                '{"answer":"","evidence_pages":[],"confidence":0,"abstain":true}'
            )
        elif mutation == "route":
            changed["route"] = {"pages": [1], "crops": {}}
        else:
            changed["inference_code_commit"] = "uncommitted"
        with pytest.raises(ValueError):
            module.check_targeted(changed, c.original, c.contract["contract_id"], route)


def test_targeted_pinned_read_closes_body_and_rejects_incomplete_or_changed_source(targeted_module):
    module = targeted_module
    for change in (None, "checksum", "version", "pending", "metadata"):
        value = (
            {"state": "pending"}
            if change == "pending"
            else {"state": "complete", "value": {"x": 1}}
        )
        payload = encode(value)
        body = io.BytesIO(payload)
        source = {"key": "source", "version_id": "v1", "sha256": digest(payload)}
        response = {"Body": body, "VersionId": "v1", "Metadata": {"sha256": digest(payload)}}
        if change == "checksum":
            source["sha256"] = "f" * 64
        elif change == "version":
            response["VersionId"] = "v2"
        elif change == "metadata":
            response["Metadata"] = {}
        client = Mock()
        client.get_object.return_value = response
        if change is None:
            assert module.read_pinned(client, "bucket", source) == {"x": 1}
        else:
            with pytest.raises(ValueError):
                module.read_pinned(client, "bucket", source)
        assert body.closed


def test_targeted_geometric_crop_is_bound_to_the_same_pdf(prepared, targeted_module):
    p, module = prepared, targeted_module
    original = workflow.validate_test_manifest(p.manifest, p.contract)[0]
    page = original.pages[0]
    source = {
        "key": page.source_pdf_s3_uri.split("s3://bucket/")[1],
        "sha256": page.source_pdf_sha256,
        "version_id": page.source_pdf_version_id,
    }
    payload = p.s3.values[source["key"]][0]
    renderer = yaml.safe_load((ROOT / "configs/oracle_reader_benchmark.yaml").read_bytes())[
        "asset_builder"
    ]
    render = renderer["render_profiles"][renderer["active_render_profile"]]
    crop = [0.1, 0.1, 0.6, 0.4]
    result = module.prepare_asset(
        p.s3, "bucket", payload, source, original, 1, crop, render, "e" * 64
    )
    assert result["source_pdf_sha256"] == page.source_pdf_sha256
    assert result["source_pdf_version_id"] == page.source_pdf_version_id
    assert result["page_number"] == 1 and result["asset_version"] == "targeted-page-detail-v1"
    with pymupdf.open(stream=payload, filetype="pdf") as pdf:
        rect = pdf[0].rect
        clip = pymupdf.Rect(
            crop[0] * rect.width, crop[1] * rect.height, crop[2] * rect.width, crop[3] * rect.height
        )
        expected = pdf[0].get_pixmap(dpi=600, clip=clip, alpha=False).tobytes("png")
    assert result["image_sha256"] == digest(expected)


def test_targeted_parent_verifies_all_inherited_records_read_only(
    recovery_case, targeted_module, monkeypatch
):
    c, module = recovery_case, targeted_module
    monkeypatch.setattr(module, "test_contract", lambda _: deepcopy(c.p.contract))

    def pinned(key, value):
        payload = encode({"state": "complete", "value": value})
        c.p.s3.put_object(Key=key, Body=payload, Metadata={"sha256": digest(payload)})
        return {"key": key, "sha256": digest(payload), "version_id": "v1"}

    plan = {
        "base_contract_id": c.p.contract["contract_id"],
        "base_inference_value_sha256": digest(encode(c.inference)),
        "parent_contract_id": c.contract["contract_id"],
        "parent_inference_commit": "c" * 40,
        "expected_preserved_count": 4,
        "inputs": pinned("parent-inputs", c.p.manifest),
        "base_inference": pinned("parent-base", c.inference),
        "parent_contract": pinned("parent-contract", c.contract),
        "parent_report": pinned(
            "parent-report",
            {"contract_id": c.contract["contract_id"], "recovered": 1, "unresolved": []},
        ),
        "parent_answers": [
            {"question_id": c.original.question_id, **pinned("parent-answer", c.saved)}
        ],
        "routes": {},
        "blocked": {},
    }
    for number, context in c.context.items():
        pinned(
            f"experiments/submissions/system/{c.contract['contract_id']}/ocr/{c.original.document_id}/{number:04d}.json",
            context,
        )
    before = deepcopy(c.p.s3.values)
    _, _, requests, preserved = module.verify_parent(c.p.root, c.p.s3, "bucket", plan)
    assert len(requests) == preserved == 4
    assert c.p.s3.values == before
    for mutation in ("missing", "overwrite", "count", "duplicate", "checksum"):
        changed = deepcopy(plan)
        if mutation == "missing":
            changed["parent_answers"] = []
        elif mutation == "overwrite":
            changed["parent_answers"][0]["question_id"] = requests[1].question_id
        elif mutation == "count":
            changed["expected_preserved_count"] = 3
        elif mutation == "duplicate":
            changed["parent_answers"] *= 2
        else:
            changed["parent_answers"][0]["sha256"] = "f" * 64
        with pytest.raises(ValueError):
            module.verify_parent(c.p.root, c.p.s3, "bucket", changed)
    assert c.p.s3.values == before


@pytest.mark.parametrize("abstain", [False, True])
def test_targeted_run_preserves_old_objects_and_never_exports_partial_csv(
    recovery_case, targeted_module, monkeypatch, abstain
):
    c, module = recovery_case, targeted_module
    for name in ("targeted.py", "run_targeted.sh"):
        shutil.copy(ROOT / "pipelines/submission" / name, c.p.root / "pipelines/submission" / name)
    requests = workflow.validate_test_manifest(c.p.manifest, c.p.contract)
    plan = {
        "routes": {c.original.question_id: [{"pages": [2], "crops": {}}]},
        "blocked": {},
        "parent_contract_id": c.contract["contract_id"],
    }
    monkeypatch.setattr(
        module, "verify_parent", lambda *args: (c.p.contract, c.p.manifest, requests, 3)
    )
    monkeypatch.setattr(module, "load_resolved_model", lambda *args: None)
    contract = module.contract_for(c.p.root, plan)
    monkeypatch.setenv("LAVA_TARGETED_CONTRACT", contract["contract_id"])
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
    raw = (
        c.saved["raw_response"]
        if not abstain
        else '{"answer":"","evidence_pages":[],"confidence":0,"abstain":true}'
    )
    calls = []

    class Reader:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, request):
            calls.append(request)
            prediction = parse_reader_response(
                question_id=request.question_id,
                answer_format=request.answer_format,
                raw_response=raw,
                allowed_pages=request.available_pages,
            )
            return prediction, ReaderTelemetry.model_validate(c.saved["telemetry"])

    monkeypatch.setattr(module, "CrossPageReader", Reader)
    monkeypatch.setattr(module, "read_raw_response", lambda _: raw)
    before = deepcopy(c.p.s3.values)
    report = module.run(c.p.root, c.p.s3, "bucket", plan, RuntimeEventLogger("test"))
    assert report["preserved_count"] == 3
    assert report["additional_recovered_count"] == (0 if abstain else 1)
    assert report["total_complete_count"] == (3 if abstain else 4)
    assert report["unresolved"] == ([c.original.question_id] if abstain else [])
    assert report["csv_exported"] is report["uploaded_to_kaggle"] is False
    assert len(calls) == report["new_model_calls"] == 1
    assert not (c.p.root / "artifacts/submission/submission.csv").exists()
    assert all(c.p.s3.values[key] == value for key, value in before.items())
    with pytest.raises(ValueError, match="already finished"):
        module.run(c.p.root, c.p.s3, "bucket", plan, RuntimeEventLogger("test"))
    assert len(calls) == 1


def test_recovery_worker_only_reads_failed_questions_and_resumes_verified_export(
    recovery_case, monkeypatch
):
    from concurrent.futures import Future

    c = recovery_case
    real_store_for = c.module.store_for

    def stores(s3, bucket, identity):
        if identity == c.p.contract["contract_id"]:
            return SimpleNamespace(
                read=lambda name: {"inputs.json": c.p.manifest, "inference.json": c.inference}[name]
            )
        return real_store_for(s3, bucket, identity)

    class InlinePool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, function, *args):
            future = Future()
            future.set_result(function(*args))
            return future

    calls = []

    def predict(request):
        calls.append(request.question_id)
        return c.p.predict(request)

    monkeypatch.setattr(c.module, "store_for", stores)
    monkeypatch.setattr(c.module, "ProcessPoolExecutor", InlinePool)
    monkeypatch.setattr(c.module, "prepare_ocr_models", lambda _: None)
    monkeypatch.setattr(
        c.module,
        "ocr_page",
        lambda path, number, language, models: {
            "number": number,
            "native": f"synthetic target {number}",
            "ocr": f"synthetic target {number}",
            "ocr_confidence": 90,
        },
    )
    monkeypatch.setattr(c.module, "load_resolved_model", lambda *a: None)
    monkeypatch.setattr(
        c.module, "RecoveryReader", lambda *a, **k: SimpleNamespace(predict=predict)
    )
    monkeypatch.setattr(
        c.module,
        "read_raw_response",
        lambda _: (
            '{"answer":"PRIVATE_ANSWER","evidence_pages":[1],"confidence":0.5,"abstain":false}'
        ),
    )
    monkeypatch.setenv("LAVA_BASE_CONTRACT", c.p.contract["contract_id"])
    monkeypatch.setenv("LAVA_BASE_INFERENCE_SHA256", digest(encode(c.inference)))
    monkeypatch.setenv("LAVA_RECOVERY_CONTRACT", c.contract["contract_id"])
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
    before = deepcopy(c.p.s3.values)
    result = c.module.run(c.p.root, c.p.s3, "bucket", RuntimeEventLogger("test"))
    assert calls == [c.original.question_id]
    assert all(c.p.s3.values[key] == value for key, value in before.items())
    target = c.p.root / "artifacts/submission/submission.csv"
    target.unlink()
    fail = Mock(side_effect=AssertionError("Completed recovery must not recompute"))
    monkeypatch.setattr(c.module, "prepare_ocr_models", fail)
    monkeypatch.setattr(c.module, "RecoveryReader", fail)
    assert c.module.run(c.p.root, c.p.s3, "bucket", RuntimeEventLogger("test")) == result
    assert digest(target.read_bytes()) == result["submission_sha256"]
    c.p.s3.values[result["csv_key"]] = (b"corrupted", {})
    with pytest.raises(ValueError, match="failed verification"):
        c.module.run(c.p.root, c.p.s3, "bucket", RuntimeEventLogger("test"))


def test_recovery_real_ocr_finds_text_in_an_image_only_pdf(tmp_path, recovery_module, monkeypatch):
    import os

    pytest.importorskip("tesserocr", reason="Optional OCR integration runtime")
    tessdata = os.environ.get("LAVA_TEST_TESSDATA")
    if not tessdata:
        pytest.skip("Set LAVA_TEST_TESSDATA to checksum-pinned OCR models")
    from PIL import Image, ImageDraw, ImageFont

    from lava.retrieval.lexical import BM25Index, PageText

    monkeypatch.setenv("OMP_THREAD_LIMIT", "1")
    image = Image.new("RGB", (1600, 400), "white")
    ImageDraw.Draw(image).text(
        (80, 120),
        "Emergency water supply 31415",
        fill="black",
        font=ImageFont.truetype("DejaVuSans.ttf", 64),
    )
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    path = tmp_path / "scanned.pdf"
    with pymupdf.open() as pdf:
        for _ in range(2):
            pdf.new_page(width=800, height=200)
        pdf[1].insert_image(pdf[1].rect, stream=stream.getvalue())
        assert pdf[1].get_text() == ""
        pdf.save(path)
    page = recovery_module.ocr_page(str(path), 2, "en", tessdata)
    assert page["native"] == ""
    assert "31415" in page["ocr"] and "water" in page["ocr"].lower()
    index = BM25Index((PageText(1, ""), PageText(2, page["ocr"])))
    assert index.rank("Emergency water supply")[0][0] == 2


def test_bounded_launch_resolves_real_reader_forward_annotations(prepared):
    # A fresh interpreter reproduces the container without pytest's test-file import hook.
    code = """
import json
import runpy
import sys
entrypoint = runpy.run_path(sys.argv[1])
run_module = runpy.run_module
sample = json.loads(sys.stdin.read())
def load_without_starting_inference(module_name, *, run_name, **kwargs):
    assert run_name == "__main__"
    namespace = run_module(module_name, run_name="__lava_schema_check__", **kwargs)
    model = namespace["TestReaderInput"]
    assert model.__pydantic_complete__
    request = model.model_validate(sample)
    assert request.document_alias == "doc-001"
    assert request.pages[0].page_number >= 1
runpy.run_module = load_without_starting_inference
entrypoint["run_inference"]()
print("SUBMISSION_LAUNCH_SCHEMA_VERIFIED")
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(ROOT / "pipelines/submission/inference.py")],
        input=json.dumps(prepared.manifest["inputs"][0]),
        text=True,
        capture_output=True,
        check=True,
        cwd=ROOT,
    )
    assert result.stdout.strip() == "SUBMISSION_LAUNCH_SCHEMA_VERIFIED"


@pytest.mark.parametrize(
    "available,count,memory_gib,accepted",
    [
        (True, 1, 96, True),
        (True, 1, 48, True),
        (True, 2, 96, False),
        (False, 0, 96, False),
        (True, 1, 24, False),
    ],
)
def test_bounded_launch_checks_gpu_and_memory_before_model_loading(
    available, count, memory_gib, accepted
):
    entrypoint = runpy.run_path(str(ROOT / "pipelines/submission/inference.py"))
    cuda = Mock()
    cuda.is_available.return_value = available
    cuda.device_count.return_value = count
    cuda.get_device_properties.return_value = SimpleNamespace(
        total_memory=memory_gib * 1024**3, name="synthetic GPU"
    )
    if accepted:
        result = entrypoint["configure_runtime"](SimpleNamespace(cuda=cuda))
        cuda.set_per_process_memory_fraction.assert_called_once_with(36 / memory_gib, 0)
        assert result["visible_cuda_devices"] == 1
    else:
        with pytest.raises(RuntimeError):
            entrypoint["configure_runtime"](SimpleNamespace(cuda=cuda))
        cuda.set_per_process_memory_fraction.assert_not_called()


def test_bounded_job_exports_complete_predictions_and_a_durable_receipt(prepared):
    p = prepared
    entrypoint = runpy.run_path(str(ROOT / "pipelines/submission/inference.py"))
    identity = p.contract["contract_id"]
    logger = RuntimeEventLogger("test")
    with pytest.raises(ValueError, match="not complete"):
        entrypoint["export_outputs"](p.root, p.s3, "bucket", identity, logger)
    assert store_for(p.s3, "bucket", identity).read("export.json") is None

    workflow.infer_test(p.root, p.s3, "bucket", identity, logger)
    receipt = entrypoint["export_outputs"](p.root, p.s3, "bucket", identity, logger)
    assert receipt["validation"]["row_count"] == 4
    assert receipt["validation"]["schema_valid"] is True
    assert store_for(p.s3, "bucket", identity).read("export.json") == receipt
    csv_payload = (p.root / "artifacts/submission/submission.csv").read_bytes()
    assert digest(csv_payload) == receipt["submission_sha256"]
    assert p.s3.get_object(Bucket="bucket", Key=receipt["csv_key"])["Body"].read() == csv_payload


@pytest.mark.parametrize(
    "code,message",
    [
        ("ValidationException", "Requested resource not found."),
        ("ValidationException", "Requested resource not found"),
        ("ValidationException", "Could not find training job"),
        ("ValidationException", "Training job does not exist"),
        ("ResourceNotFound", "missing"),
        ("ResourceNotFoundException", "missing"),
    ],
)
def test_missing_job_service_responses_are_normal_cache_misses(code, message):
    client = boto3.client(
        "sagemaker", region_name="us-west-2", aws_access_key_id="test", aws_secret_access_key="test"
    )
    with Stubber(client) as stub:
        stub.add_client_error(
            "describe_training_job",
            service_error_code=code,
            service_message=message,
            expected_params={"TrainingJobName": "lava-system-test"},
        )
        assert describe_or_none(client, "lava-system-test") is None
        stub.assert_no_pending_responses()


@pytest.mark.parametrize(
    "code,message",
    [
        ("AccessDeniedException", "Requested resource not found."),
        ("ValidationException", "Invalid training job name"),
        ("ValidationException", "Role not found or inaccessible"),
        ("ThrottlingException", "Try later"),
        ("InternalFailure", "Requested resource not found."),
    ],
)
def test_nonabsence_service_errors_never_authorize_a_new_job(code, message):
    client = Mock()
    failure = ClientError({"Error": {"Code": code, "Message": message}}, "DescribeTrainingJob")
    client.describe_training_job.side_effect = failure
    with pytest.raises(ClientError) as caught:
        ensure_job(
            client,
            {"TrainingJobName": "lava-system-test"},
            acknowledge_charges="YES",
            allow_retry=False,
            attempt=1,
            logger=RuntimeEventLogger("test"),
        )
    assert caught.value is failure
    client.create_training_job.assert_not_called()


def test_network_failure_is_not_absence():
    client = Mock()
    client.describe_training_job.side_effect = EndpointConnectionError(
        endpoint_url="https://invalid.test"
    )
    with pytest.raises(EndpointConnectionError):
        describe_or_none(client, "name")


def test_exact_studio_error_reaches_create_once_with_explicit_ack():
    client = Mock()
    client.describe_training_job.side_effect = ClientError(
        {"Error": {"Code": "ValidationException", "Message": "Requested resource not found."}},
        "DescribeTrainingJob",
    )
    client.get_paginator.return_value.paginate.return_value = [{"TrainingJobSummaries": []}]
    request = {"TrainingJobName": "lava-system-test"}
    result = ensure_job(
        client,
        request,
        acknowledge_charges="YES",
        allow_retry=False,
        attempt=1,
        logger=RuntimeEventLogger("test"),
    )
    assert result["TrainingJobStatus"] == "InProgress"
    client.create_training_job.assert_called_once_with(**request)


def test_notebook_edits_are_allowed_but_uncommitted_inference_code_is_rejected(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"], cwd=tmp_path, check=True
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "src/model.py").write_text("x = 1\n")
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "notebooks/example.ipynb").write_text("{}")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "test"], cwd=tmp_path, check=True)
    (tmp_path / "notebooks/example.ipynb").write_text('{"run":true}')
    verify_committed_source(tmp_path)
    (tmp_path / "src/another.py").write_text("x = 2\n")
    with pytest.raises(ValueError, match="source"):
        verify_committed_source(tmp_path)


@pytest.mark.parametrize("operation", ["pilot", "test", "export"])
def test_publication_mode_never_runs_cloud_or_export_even_when_saved_controls_are_enabled(
    tmp_path, monkeypatch, operation
):
    monkeypatch.setenv("LAVA_NOTEBOOK_PUBLICATION", "1")
    launch = Mock(side_effect=AssertionError("No child may run"))
    monkeypatch.setattr(subprocess, "Popen", launch)
    assert (
        run_operation(tmp_path, operation, enabled=True, acknowledge_charges="YES")["status"]
        == "disabled"
    )
    launch.assert_not_called()


@pytest.mark.parametrize("operation", ["pilot", "test"])
def test_interactive_paid_controls_require_acknowledgement(tmp_path, monkeypatch, operation):
    monkeypatch.delenv("LAVA_NOTEBOOK_PUBLICATION", raising=False)
    with pytest.raises(ValueError, match="ACKNOWLEDGE"):
        run_operation(tmp_path, operation, enabled=True)


def test_interactive_operation_runs_a_real_local_child_and_streams_its_progress(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.delenv("LAVA_NOTEBOOK_PUBLICATION", raising=False)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/prepare_submission.py").write_text('print("SYNTHETIC_PROGRESS")\n')
    assert run_operation(tmp_path, "export", enabled=True)["status"] == "completed"
    assert "SYNTHETIC_PROGRESS" in capsys.readouterr().out


def csv_payload(columns, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


class MemoryS3:
    def __init__(self):
        self.values = {}
        self.exceptions = SimpleNamespace(NoSuchKey=KeyError)

    def put_object(self, *, Key, Body, Metadata=None, **kwargs):
        prior = self.values.get(Key)
        if (kwargs.get("IfNoneMatch") == "*" and prior is not None) or (
            "IfMatch" in kwargs and (prior is None or kwargs["IfMatch"] != digest(prior[0]))
        ):
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.values[Key] = (Body, Metadata or {})
        return {"VersionId": "v1"}

    def get_object(self, *, Key, **kwargs):
        data, metadata = self.values[Key]
        return {
            "Body": io.BytesIO(data),
            "Metadata": metadata,
            "ETag": digest(data),
            "VersionId": "v1",
        }


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    s3 = MemoryS3()
    config = deepcopy(json.loads((ROOT / "configs/submission.json").read_bytes()))
    rows = [
        {
            "id": f"q{index}",
            "file_id": f"d{index % 2}",
            "question": "synthetic target",
            "answer_format": "string",
            "language": "ja",
        }
        for index in range(4)
    ]
    test = csv_payload(TEST_COLUMNS, rows)
    template = csv_payload(
        SUBMISSION_COLUMNS,
        [
            {"id": row["id"], "answer": "placeholder", "evidence_page_number": "[1]"}
            for row in reversed(rows)
        ],
    )
    config.update(
        expected_question_count=4,
        expected_document_count=2,
        answer_format_counts={"string": 4},
        language_counts={"ja": 4},
    )
    for name, data in (("test.csv", test), ("sample_submission.csv", template)):
        config["source_files"][name] = {"key": name, "sha256": digest(data), "version_id": "v1"}
        s3.values[name] = (data, {"sha256": digest(data)})
    inventory = []
    for index in range(2):
        with pymupdf.open() as pdf:
            for page in range(3):
                pdf.new_page().insert_text((72, 72), f"synthetic target {page + 1}")
            payload = pdf.tobytes()
        name = f"test_pdfs/test_pdfs/d{index}.pdf"
        s3.values[name] = (payload, {"sha256": digest(payload)})
        inventory.append(
            {"name": name, "s3_key": name, "sha256": digest(payload), "version_id": "v1"}
        )
    raw_manifest = csv_payload(("name", "s3_key", "sha256", "version_id"), inventory)
    (tmp_path / "configs").mkdir()
    (tmp_path / "reports").mkdir()
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "configs/submission.json").write_bytes(encode(config))
    (tmp_path / "reports/raw_data_manifest.csv").write_bytes(raw_manifest)
    shutil.copy(ROOT / "configs/oracle_reader_benchmark.yaml", tmp_path / "configs")
    contract = workflow.test_contract(ROOT)
    contract["config"].update(question_count=4, document_count=2)
    contract["submission"] = config
    contract["retrieval"]["config"]["source_manifest_sha256"] = digest(raw_manifest)
    monkeypatch.setattr(workflow, "test_contract", lambda _: deepcopy(contract))
    manifest = workflow.prepare_test_inputs(tmp_path, s3, "bucket", RuntimeEventLogger("synthetic"))
    monkeypatch.setattr(workflow, "load_resolved_model", lambda *a: None)
    monkeypatch.setenv("LAVA_GIT_COMMIT_SHA", "b" * 40)
    raw = '{"answer":"PRIVATE_ANSWER","evidence_pages":[1],"confidence":0.5,"abstain":false}'
    monkeypatch.setattr(workflow, "read_raw_response", lambda _: raw)

    def predict(request):
        prediction = parse_reader_response(
            question_id=request.question_id,
            answer_format=request.answer_format,
            raw_response=raw,
            allowed_pages=request.available_pages,
        )
        telemetry = ReaderTelemetry(
            model_load_seconds=0,
            preprocessing_seconds=1,
            generation_seconds=2,
            total_seconds=3,
            prompt_tokens=10,
            generated_tokens=2,
            image_count=len(request.pages),
            total_image_pixels=30000,
            raw_response_characters=len(raw),
            peak_cuda_memory_allocated_mib=1,
            peak_cuda_memory_reserved_mib=2,
            gpu_name="synthetic",
            cuda_compute_capability="synthetic",
            torch_version="test",
            transformers_version="test",
            dtype="bfloat16",
            attention_implementation="sdpa",
            deterministic_algorithms_enabled=True,
            template_switch_supported=True,
        )
        return prediction, telemetry

    reader = SimpleNamespace(predict=predict)
    monkeypatch.setattr(reader_factory, "build_reader", lambda *a, **k: reader)
    return SimpleNamespace(
        root=tmp_path, s3=s3, contract=contract, manifest=manifest, reader=reader, predict=predict
    )


def test_test_preparation_reuses_real_pdf_assets_and_preserves_label_free_identity(
    prepared, monkeypatch
):
    p = prepared
    fail = Mock(side_effect=AssertionError("No repeat computation"))
    for name in ("extract_pages", "rank_query", "render_test_asset"):
        monkeypatch.setattr(workflow, name, fail)
    assert (
        workflow.prepare_test_inputs(p.root, p.s3, "bucket", RuntimeEventLogger("test"))
        == p.manifest
    )
    requests = workflow.validate_test_manifest(p.manifest, p.contract)
    assert {row.document_alias for row in requests} == {"doc-001", "doc-002"}
    assert all(row.available_pages == (1, 2, 3) for row in requests)
    assert all(not hasattr(row, "answer") for row in requests)
    data = requests[0].pages[0].model_dump()
    assert workflow.TestPageAsset.model_validate({**data, "document_alias": "doc-200"})
    with pytest.raises(ValueError):
        OraclePageAsset.model_validate(data)


@pytest.mark.parametrize("mutation", ["label", "partial", "range", "hash", "wrong_split"])
def test_test_manifest_rejects_invalid_inputs(prepared, mutation):
    manifest = deepcopy(prepared.manifest)
    if mutation == "label":
        manifest["inputs"][0]["answer"] = "leak"
    if mutation == "partial":
        manifest["inputs"].pop()
    if mutation == "range":
        manifest["inputs"][0]["pages"][-1]["page_number"] = 1000
    if mutation == "hash":
        manifest["inputs_sha256"] = "0" * 64
    if mutation == "wrong_split":
        manifest["contract"]["config"]["split"] = "train"
    with pytest.raises(ValueError):
        workflow.validate_test_manifest(manifest, prepared.contract)


def test_interrupted_test_inference_and_export_reuse_completed_work(prepared, monkeypatch, capsys):
    p = prepared
    calls = []

    def interrupted(request):
        calls.append(request.question_id)
        if len(calls) == 3:
            raise KeyboardInterrupt()
        return p.predict(request)

    p.reader.predict = interrupted
    with pytest.raises(KeyboardInterrupt):
        workflow.infer_test(
            p.root, p.s3, "bucket", p.contract["contract_id"], RuntimeEventLogger("test")
        )
    p.reader.predict = p.predict
    monkeypatch.setenv("LAVA_GIT_COMMIT_SHA", "c" * 40)
    workflow.infer_test(
        p.root, p.s3, "bucket", p.contract["contract_id"], RuntimeEventLogger("test")
    )
    p.reader.predict = Mock(side_effect=AssertionError("No repeat inference"))
    workflow.infer_test(
        p.root, p.s3, "bucket", p.contract["contract_id"], RuntimeEventLogger("test")
    )
    first = workflow.export_test(p.root, p.s3, "bucket", RuntimeEventLogger("test"))
    assert workflow.export_test(p.root, p.s3, "bucket", RuntimeEventLogger("test")) == first
    rows = list(csv.DictReader(io.StringIO(first.read_text())))
    assert [row["id"] for row in rows] == ["q3", "q2", "q1", "q0"]
    assert all(row["answer"] == "PRIVATE_ANSWER" for row in rows)
    provenance = json.loads(first.with_name("manifest.json").read_bytes())["provenance"]
    assert provenance["inference_code_commits"] == ["b" * 40, "c" * 40]
    output = capsys.readouterr().out
    assert "PRIVATE_ANSWER" not in output
    monkeypatch.delenv("LAVA_NOTEBOOK_PUBLICATION", raising=False)
    link = download_link(p.root)
    assert "../artifacts/submission/submission.csv" in link and "PRIVATE_ANSWER" not in link
    (p.root / "artifacts/submission/submission.csv").write_text("corrupt")
    with pytest.raises(ValueError, match="manifest"):
        download_link(p.root)


def test_incomplete_inference_does_not_create_fake_csv(prepared):
    with pytest.raises(ValueError, match="not complete"):
        workflow.export_test(prepared.root, prepared.s3, "bucket", RuntimeEventLogger("test"))
    assert not list(prepared.root.rglob("submission.csv"))


def test_test_job_has_distinct_identity_limits_and_valid_service_contract(prepared):
    p = prepared
    request = workflow.test_training_request(
        ROOT,
        p.manifest,
        "bucket",
        "us-west-2",
        "arn:aws:iam::123456789012:role/test",
        1,
        "source/source.tar.gz",
        "a" * 64,
        "b" * 40,
    )
    assert request["TrainingJobName"].startswith("lava-export-")
    assert request["StoppingCondition"]["MaxRuntimeInSeconds"] == 7200
    assert request["ResourceConfig"]["InstanceCount"] == 1
    assert "/submission/run.sh" in request["AlgorithmSpecification"]["ContainerArguments"][1]
    client = boto3.client(
        "sagemaker", region_name="us-west-2", aws_access_key_id="test", aws_secret_access_key="test"
    )
    validate_parameters(
        request, client.meta.service_model.operation_model("CreateTrainingJob").input_shape
    )


def test_test_cli_requires_opt_in_before_loading_private_inputs(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prepare_submission.py", "--mode", "test"])
    main = runpy.run_path(str(ROOT / "scripts/prepare_submission.py"))["main"]
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


def test_notebook_has_generation_download_controls_but_no_upload_or_embedded_csv():
    import nbformat

    notebook = nbformat.read(ROOT / "notebooks/05_end_to_end_system_evaluation.ipynb", 4)
    source = "\n".join(cell.source for cell in notebook.cells)
    assert "RUN_TEST_INFERENCE = False" in source
    assert "EXPORT_SAVED_TEST = False" in source
    assert "download_link(ROOT)" in source
    assert "Generate and download your own submission" in source
    for prohibited in (
        "competition_submit(",
        "kaggle competitions submit",
        "data:text/csv",
        "data:application/octet-stream",
    ):
        assert prohibited not in source


def test_corrupted_saved_generation_is_not_reused_or_exported(prepared):
    p = prepared
    workflow.infer_test(
        p.root, p.s3, "bucket", p.contract["contract_id"], RuntimeEventLogger("test")
    )
    store = store_for(p.s3, "bucket", p.contract["contract_id"])
    key = f"{store.prefix}/inference.json"
    payload, _ = p.s3.values[key]
    changed = json.loads(payload)
    changed["value"]["records"][0]["raw_response"] = "tampered"
    data = encode(changed)
    p.s3.values[key] = (data, {"sha256": digest(data)})
    with pytest.raises(ValueError, match="parsed"):
        workflow.export_test(p.root, p.s3, "bucket", RuntimeEventLogger("test"))
    assert not list(p.root.rglob("submission.csv"))


def test_abstained_answers_remain_checkpoints_but_cannot_fake_a_complete_export(
    prepared, monkeypatch
):
    p = prepared
    raw = '{"answer":"","evidence_pages":[],"confidence":0.0,"abstain":true}'
    monkeypatch.setattr(workflow, "read_raw_response", lambda _: raw)

    def abstain(request):
        prediction = parse_reader_response(
            question_id=request.question_id,
            answer_format=request.answer_format,
            raw_response=raw,
            allowed_pages=request.available_pages,
        )
        return prediction, p.predict(request)[1]

    p.reader.predict = abstain
    workflow.infer_test(
        p.root, p.s3, "bucket", p.contract["contract_id"], RuntimeEventLogger("test")
    )
    with pytest.raises(ValueError, match="invalid/abstained"):
        workflow.export_test(p.root, p.s3, "bucket", RuntimeEventLogger("test"))
    assert not list(p.root.rglob("submission.csv"))
    assert (
        len(
            json.loads((p.root / "artifacts/submission/invalid_questions.json").read_bytes())[
                "question_ids"
            ]
        )
        == 4
    )
    assert store_for(p.s3, "bucket", p.contract["contract_id"]).read("inference.json") is not None
