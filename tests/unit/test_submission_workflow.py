"""Real synthetic-PDF workflow regressions; not model-performance measurements."""

from __future__ import annotations

import csv
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
