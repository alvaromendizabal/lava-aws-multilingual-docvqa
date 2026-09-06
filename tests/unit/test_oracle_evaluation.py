"""End-to-end pilot tests with synthetic data and injected cloud/model adapters."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

from lava.evaluation.reporting import load_report, render_report
from lava.readers import benchmark, evaluation_contract
from lava.readers.artifact_gate import verify_training_model_artifact
from lava.readers.model_registry import load_resolved_model
from lava.readers.parsing import ReaderOutputError, parse_reader_response
from lava.readers.private_artifacts import persist_raw_response
from lava.readers.sagemaker import build_job_plan, validate_submission_guardrails
from lava.readers.schemas import OracleExample, OraclePageAsset, ReaderPrediction, ReaderTelemetry

ROOT = Path(__file__).resolve().parents[2]


def test_benchmark_preview_requires_all_questions_and_verified_hardware() -> None:
    kwargs = {
        "repo_root": ROOT,
        "config_path": ROOT / "configs/oracle_reader_benchmark.yaml",
        "model_lock_path": ROOT / "configs/oracle_reader_models.lock.json",
        "bucket": "test-bucket",
    }
    for key in evaluation_contract.load_evaluation_contract(ROOT)["models"]:
        plan = build_job_plan(**kwargs, model_key=key, limit=16, mode="benchmark")
        assert plan.limit == 16
        assert plan.output_s3_prefix.endswith("/benchmark")
    with pytest.raises(ValueError, match="complete"):
        build_job_plan(**kwargs, model_key="qwen35_4b_fused_direct", limit=1, mode="benchmark")
    with pytest.raises(ValueError, match="verified"):
        build_job_plan(
            **kwargs,
            model_key="qwen35_4b_fused_direct",
            limit=16,
            mode="benchmark",
            instance_type="ml.g6e.2xlarge",
        )


def test_smoke_guard_still_rejects_sixteen_questions(monkeypatch) -> None:
    monkeypatch.setattr("lava.readers.sagemaker._git_clean", lambda _: True)
    plan = build_job_plan(
        repo_root=ROOT,
        config_path=ROOT / "configs/oracle_reader_benchmark.yaml",
        model_lock_path=ROOT / "configs/oracle_reader_models.lock.json",
        bucket="test-bucket",
        model_key="qwen35_4b_fused_direct",
        limit=16,
    )
    with pytest.raises(ValueError, match="exactly one"):
        validate_submission_guardrails(plan, repo_root=ROOT)


class MemoryS3:
    def __init__(self, objects):
        self.objects = objects
        self.version_requests = []

    def get_object(self, **kwargs):
        self.version_requests.append(kwargs.get("VersionId"))
        return {"Body": io.BytesIO(self.objects[kwargs["Key"]])}

    def put_object(self, **kwargs):
        self.objects[kwargs["Key"]] = kwargs["Body"]

    def list_objects_v2(self, **kwargs):
        return {"Contents": [{"Key": k} for k in self.objects if k.startswith(kwargs["Prefix"])]}


@pytest.fixture
def pilot(tmp_path, monkeypatch, capsys, request):
    contract = evaluation_contract.load_evaluation_contract(ROOT)
    examples = []
    formats = ["string"] * 7 + ["number"] * 4 + ["ordered_list"] + ["unordered_list"] * 4
    for alias, count in contract["document_question_counts"].items():
        for _ in range(count):
            index = len(examples)
            page = OraclePageAsset(
                asset_version="synthetic",
                document_id=alias,
                document_alias=alias,
                page_number=1,
                source_pdf_s3_uri="s3://test-bucket/source",
                source_pdf_sha256="a" * 64,
                image_s3_uri="s3://test-bucket/image",
                image_sha256="a" * 64,
                text_s3_uri="s3://test-bucket/text",
                text_sha256="a" * 64,
                layout_s3_uri="s3://test-bucket/layout",
                layout_sha256="a" * 64,
                width_pixels=100,
                height_pixels=100,
                dpi=180,
                native_text_characters=0,
                word_count=0,
                text_block_count=0,
                embedded_image_count=0,
            )
            examples.append(
                OracleExample(
                    protocol_lock_id=contract["protocol_lock_id"],
                    question_id=f"q{index:02d}",
                    document_id=alias,
                    document_alias=alias,
                    question="SYNTHETIC_PRIVATE_QUESTION",
                    answer_format=formats[index],
                    answer='["10"]' if "list" in formats[index] else "10",
                    evidence_pages=(1,),
                    language="vi" if index == 15 else "ja",
                    pages=(page,),
                )
            )
    payload = ("\n".join(e.model_dump_json() for e in examples) + "\n").encode()
    contract = {**contract, "private_manifest_sha256": hashlib.sha256(payload).hexdigest()}
    monkeypatch.setattr(evaluation_contract, "load_evaluation_contract", lambda _: contract)
    monkeypatch.setattr(benchmark, "load_evaluation_contract", lambda _: contract)
    s3 = MemoryS3({"manifest.jsonl": payload})
    monkeypatch.setattr(benchmark.boto3, "client", lambda *a, **kw: s3)
    monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path / "model"))
    monkeypatch.setenv("LAVA_PRIVATE_MODEL_DIR", str(tmp_path / "model/private"))
    monkeypatch.setenv("LAVA_GIT_COMMIT_SHA", "c" * 40)

    class Reader:
        def predict(self, example):
            if example.question_id == "q00":
                raw = getattr(request, "param", "not json")
            else:
                raw = json.dumps(
                    {
                        "answer": ["10"] if "list" in example.answer_format.value else "10",
                        "evidence_pages": [1],
                        "confidence": 0.9,
                        "abstain": False,
                    }
                )
            persist_raw_response(raw, question_id=example.question_id)
            try:
                prediction = parse_reader_response(
                    question_id=example.question_id,
                    answer_format=example.answer_format,
                    raw_response=raw,
                    allowed_pages=(1,),
                )
            except ReaderOutputError as error:
                prediction = ReaderPrediction(
                    question_id=example.question_id,
                    answer_format=example.answer_format,
                    answer="",
                    evidence_pages=(),
                    confidence=0,
                    abstain=True,
                    schema_valid=False,
                    parser_error_code=error.code,
                    raw_response_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                )
            telemetry = ReaderTelemetry(
                model_load_seconds=10,
                preprocessing_seconds=1,
                generation_seconds=2,
                total_seconds=3,
                prompt_tokens=50,
                generated_tokens=20,
                image_count=1,
                total_image_pixels=10000,
                raw_response_characters=len(raw),
                peak_cuda_memory_allocated_mib=100,
                peak_cuda_memory_reserved_mib=128,
                gpu_name="synthetic",
                cuda_compute_capability="synthetic",
                torch_version="synthetic",
                transformers_version="synthetic",
                dtype="bfloat16",
                attention_implementation="sdpa",
                deterministic_algorithms_enabled=True,
                template_switch_supported=True,
            )
            return prediction, telemetry

    monkeypatch.setattr(benchmark, "build_reader", lambda *a, **kw: Reader())
    model = load_resolved_model(
        ROOT / "configs/oracle_reader_models.lock.json", "qwen35_4b_fused_direct"
    )
    summary = benchmark.run_oracle_benchmark(
        bucket="test-bucket",
        region="us-west-2",
        manifest_s3_uri="s3://test-bucket/manifest.jsonl",
        output_s3_prefix="s3://test-bucket/run",
        protocol_lock_id=contract["protocol_lock_id"],
        model_spec=model,
        experiment_id="pilot",
        limit=16,
        evaluation_root=ROOT,
    )
    events = capsys.readouterr().out
    assert "SYNTHETIC_PRIVATE_QUESTION" not in events
    assert "question.completed" in events and "elapsed_seconds" in events
    artifact = "run/sagemaker-output/job/output/model"
    for file in (tmp_path / "model").rglob("*"):
        if file.is_file():
            s3.objects[artifact + "/" + file.relative_to(tmp_path / "model").as_posix()] = (
                file.read_bytes()
            )
    description = {
        "TrainingJobStatus": "Completed",
        "ModelArtifacts": {"S3ModelArtifacts": "s3://test-bucket/" + artifact},
        "OutputDataConfig": {
            "CompressionType": "NONE",
            "S3OutputPath": "s3://test-bucket/run/sagemaker-output",
        },
        "HyperParameters": {
            "mode": "benchmark",
            "limit": "16",
            "model_key": model.model_key,
            "experiment_id": "pilot",
            "manifest_s3_uri": "s3://test-bucket/manifest.jsonl",
        },
        "Environment": {"LAVA_GIT_COMMIT_SHA": "c" * 40},
        "ResourceConfig": {"InstanceType": model.instance_type},
    }

    class SageMaker:
        def describe_training_job(self, **kwargs):
            return description

    return s3, SageMaker(), summary, payload, contract, tmp_path


def verify(pilot):
    s3, sm, *_ = pilot
    return verify_training_model_artifact(
        s3_client=s3,
        sagemaker_client=sm,
        job_name="job",
        expected_output_s3_prefix="s3://test-bucket/run",
    )


def test_full_pilot_preserves_failed_outputs_and_verifies_every_question(pilot):
    gate = verify(pilot)
    s3, _, summary, _, _, tmp_path = pilot
    assert gate.raw_response_count == 16
    assert gate.schema_valid_rate == 15 / 16
    assert gate.parser_error_counts == {"missing_json_object": 1}
    assert summary["document_count"] == 5
    assert summary["normalized_exact_answer_micro"] == 15 / 16
    assert summary["normalized_exact_answer_document_macro"] == 0.95
    assert len((tmp_path / "model/private/records.jsonl").read_text().splitlines()) == 16
    assert "ypRTl7C6zNtPwF9LS9bb4bfkz66qBn_b" in s3.version_requests


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_record",
        "duplicate_record",
        "score",
        "raw",
        "metadata",
        "summary",
        "identity",
        "lineage",
    ],
)
def test_artifact_tampering_is_rejected(pilot, mutation):
    s3 = pilot[0]
    key = "run/private_records.jsonl"
    records = [json.loads(line) for line in s3.objects[key].splitlines()]
    if mutation == "missing_record":
        records.pop()
    if mutation == "duplicate_record":
        records[-1] = records[0]
    if mutation == "score":
        records[1]["normalized_exact_answer_score"] = 0
    if mutation == "identity":
        records[1]["document_alias"] = "doc-05"
    if mutation == "lineage":
        records[1]["git_commit_sha"] = "d" * 40
    s3.objects[key] = ("\n".join(json.dumps(r) for r in records) + "\n").encode()
    if mutation == "raw":
        key = next(k for k in s3.objects if "/q01-" in k and k.endswith(".txt"))
        s3.objects[key] = b"tampered"
    if mutation == "metadata":
        key = next(k for k in s3.objects if "/q01-" in k and k.endswith(".json"))
        data = json.loads(s3.objects[key])
        data["question_id"] = "q00"
        s3.objects[key] = json.dumps(data).encode()
    if mutation == "summary":
        data = json.loads(s3.objects["run/public_summary.json"])
        data["record_count"] = 15
        s3.objects["run/public_summary.json"] = json.dumps(data).encode()
    with pytest.raises((RuntimeError, ValueError)):
        verify(pilot)


def test_manifest_drift_is_rejected_before_model_loading(pilot):
    _, _, _, payload, contract, _ = pilot
    with pytest.raises(ValueError, match="SHA-256"):
        evaluation_contract.validate_evaluation_manifest(payload + b" ", contract)
    lines = payload.splitlines()
    lines[-1] = lines[0]
    duplicate = b"\n".join(lines) + b"\n"
    with pytest.raises(ValueError, match="duplicate"):
        evaluation_contract.validate_evaluation_manifest(
            duplicate,
            {**contract, "private_manifest_sha256": hashlib.sha256(duplicate).hexdigest()},
        )


def test_real_smokes_never_enter_paired_comparisons(tmp_path):
    import shutil

    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    for path in (ROOT / "reports/oracle_reader/runs").glob("*/public_summary.json"):
        if json.loads(path.read_text())["record_count"] == 1:
            shutil.copytree(path.parent, tmp_path / "reports/oracle_reader/runs" / path.parent.name)
    report = load_report(tmp_path)
    assert len(report["runs"]) >= 3
    assert report["paired_document_comparisons"] == []
    assert not any(r["complete"] for r in report["runs"])
    rendered = render_report(report)
    assert "Smoke only" in rendered
    assert "s3://" not in rendered
    assert "560403859723" not in rendered
    assert "<script" not in rendered


def test_report_rejects_modified_summary(tmp_path):
    import shutil

    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    shutil.copytree(ROOT / "reports/oracle_reader/runs", tmp_path / "reports/oracle_reader/runs")
    path = next((tmp_path / "reports/oracle_reader/runs").glob("*/public_summary.json"))
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum"):
        load_report(tmp_path)


@pytest.mark.parametrize("pilot", [""], indirect=True)
def test_empty_generation_is_a_measured_failure_in_the_full_pilot(pilot):
    gate = verify(pilot)
    assert gate.raw_response_count == 16
    assert gate.schema_valid_rate == 15 / 16


def test_raw_filenames_cannot_collide_after_sanitization(tmp_path):
    first = persist_raw_response("first", question_id="same/id", root=tmp_path)
    second = persist_raw_response("second", question_id="same?id", root=tmp_path)
    assert first.response_path != second.response_path
    assert Path(first.response_path).read_text() == "first"


def test_report_complete_pairs_require_compatible_protocol_and_generation(pilot, monkeypatch):
    import shutil

    from lava.evaluation import reporting

    _, _, summary, _, contract, root = pilot
    monkeypatch.setattr(reporting, "load_evaluation_contract", lambda _: contract)
    shutil.copytree(ROOT / "configs", root / "configs")
    for index in (1, 2):
        directory = root / f"reports/oracle_reader/runs/job-{index}"
        directory.mkdir(parents=True)
        payload = json.dumps(summary).encode()
        (directory / "public_summary.json").write_bytes(payload)
        manifest = {
            key: summary[key]
            for key in [
                "model_key",
                "model_id",
                "model_revision",
                "git_commit_sha",
                "protocol_lock_id",
            ]
        }
        manifest.update(
            job_name=f"job-{index}",
            instance_type="ml.g5.2xlarge",
            billable_time_seconds=60,
            public_summary_sha256=hashlib.sha256(payload).hexdigest(),
            artifact_gate={"raw_response_count": 16, "schema_valid_rate": 15 / 16},
        )
        (directory / "sync_manifest.json").write_text(json.dumps(manifest))
    report = load_report(root)
    assert len(report["paired_document_comparisons"]) == 1
    pair = report["paired_document_comparisons"][0]
    assert pair["mean_delta"] == 0
    assert pair["exact_two_sided_sign_flip_p_value"] == 1
    assert pair["document_count"] == 5
    assert "Full evaluation comes next" not in render_report(report)
    path = root / "reports/oracle_reader/runs/job-2/public_summary.json"
    changed = json.loads(path.read_text())
    changed["generation"]["seed"] += 1
    payload = json.dumps(changed).encode()
    path.write_bytes(payload)
    path = path.parent / "sync_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["public_summary_sha256"] = hashlib.sha256(payload).hexdigest()
    path.write_text(json.dumps(manifest))
    assert load_report(root)["paired_document_comparisons"] == []
