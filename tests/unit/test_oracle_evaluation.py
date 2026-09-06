"""End-to-end pilot tests with synthetic data and injected cloud/model adapters."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

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
        self.metadata = {}

    def get_object(self, **kwargs):
        self.version_requests.append(kwargs.get("VersionId"))
        return {
            "Body": io.BytesIO(self.objects[kwargs["Key"]]),
            "Metadata": self.metadata.get(kwargs["Key"], {}),
        }

    def put_object(self, **kwargs):
        if kwargs.get("IfNoneMatch") == "*" and kwargs["Key"] in self.objects:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.objects[kwargs["Key"]] = kwargs["Body"]
        self.metadata[kwargs["Key"]] = kwargs.get("Metadata", {})

    def get_paginator(self, operation):
        assert operation == "list_objects_v2"
        outer = self

        class Paginator:
            def paginate(self, **kwargs):
                # Deliberately exercise multiple listing pages.
                items = outer.list_objects_v2(**kwargs)["Contents"]
                for start in range(0, len(items), 3):
                    yield {"Contents": items[start : start + 3]}

        return Paginator()

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
            "checkpoint_schema_version": "1",
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


def test_saved_prediction_metrics_preserve_failures_and_verified_coverage(pilot, monkeypatch):
    from lava.evaluation import saved_predictions

    s3, sm, summary, _, contract, _ = pilot
    monkeypatch.setattr(saved_predictions, "load_evaluation_contract", lambda _: contract)
    run = saved_predictions.load_saved_run(ROOT, sm, s3, "job")
    metrics = saved_predictions.supporting_metrics(run)
    assert len(run.references) == len(run.predictions) == metrics["question_count"] == 16
    assert metrics["answer"]["schema_valid_rate"] == 15 / 16
    assert metrics["answer"]["abstention_rate"] == 1 / 16
    assert metrics["answer"]["full_credit_rate"] == 15 / 16
    assert metrics["answer"]["zero_credit_rate"] == 1 / 16
    assert metrics["grounding"]["question_average_f1"] == summary["self_grounding_f1_micro"]
    assert metrics["grounding"]["question_average_precision"] == 15 / 16
    assert metrics["grounding"]["question_average_recall"] == 15 / 16
    assert metrics["latency"]["generation_p50_seconds"] == 2
    assert metrics["latency"]["generation_p95_seconds"] == 2
    assert metrics["latency"]["generation_tokens_per_second"] == 10
    assert metrics["semantic_vqa_score"] is None
    assert metrics["official_server_score"] is None


def test_saved_evaluation_rejects_smoke_before_loading_objects(pilot):
    from lava.evaluation.saved_predictions import load_saved_run

    s3, sm, *_ = pilot
    sm.describe_training_job()["HyperParameters"]["mode"] = "smoke"
    with pytest.raises(ValueError, match="complete benchmark"):
        load_saved_run(ROOT, sm, s3, "job")


def test_latency_percentiles_and_zero_duration_are_explicit(pilot, monkeypatch):
    from dataclasses import replace

    from lava.evaluation import saved_predictions

    s3, sm, _, _, contract, _ = pilot
    monkeypatch.setattr(saved_predictions, "load_evaluation_contract", lambda _: contract)
    run = saved_predictions.load_saved_run(ROOT, sm, s3, "job")
    record = run.records[1]
    records = tuple(
        record.model_copy(
            update={
                "telemetry": record.telemetry.model_copy(update={"generation_seconds": seconds})
            }
        )
        for seconds in (0.0, 10.0)
    )
    measured = saved_predictions.supporting_metrics(replace(run, records=records))
    assert measured["latency"]["generation_p50_seconds"] == 5
    assert measured["latency"]["generation_p95_seconds"] == 9.5
    zero = saved_predictions.supporting_metrics(replace(run, records=records[:1]))
    assert zero["latency"]["generation_tokens_per_second"] is None
    with pytest.raises(ValueError, match="empty"):
        saved_predictions.supporting_metrics(replace(run, records=()))


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
    # Real semantic results now bind their evaluator source and frozen dependencies.
    shutil.copytree(
        ROOT / "src/lava/evaluation",
        tmp_path / "src/lava/evaluation",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copyfile(ROOT / "uv.lock", tmp_path / "uv.lock")
    assert load_report(tmp_path)["runs"]  # The complete fixture must pass before tampering.
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


def test_verified_4b_9b_report_exposes_mixed_results_and_uncertainty():
    report = load_report(ROOT)
    pair = next(
        p
        for p in report["paired_document_comparisons"]
        if "4b-fused" in p["baseline_job"] and "9b-fused" in p["challenger_job"]
    )
    assert pair["mean_delta"] == pytest.approx(-0.04119047619047619)
    assert (pair["documents_improved"], pair["documents_tied"], pair["documents_regressed"]) == (
        2,
        2,
        1,
    )
    assert pair["exact_two_sided_sign_flip_p_value"] == 1.0
    rendered = render_report(report)
    for required in (
        "+7.65 pp",
        "-4.12 pp",
        "-80.00 pp",
        "opposite directions",
        "-43.17 to +25.98",
        "2 documents improved, 2 tied, and 1 regressed",
    ):
        assert required in rendered
    assert "Pending: at least two" not in rendered
    assert "560403859723" not in rendered
    assert "s3://" not in rendered


def test_comparison_chart_has_accessible_numeric_table_and_bounded_bars():
    import re
    import xml.etree.ElementTree as ET

    rendered = render_report(load_report(ROOT))
    charts = re.findall(r"<svg[^>]*>.*?</svg>", rendered, flags=re.DOTALL)
    chart = next(ET.fromstring(svg) for svg in charts if "challenger minus baseline" in svg)
    assert chart.find("title") is not None
    bars = chart.findall("rect")
    assert len(bars) == 5
    for bar in bars:
        x, width = float(bar.attrib["x"]), float(bar.attrib["width"])
        assert 140 <= x <= x + width <= 490
    assert '<table class="comparison-table">' in rendered
    assert '<th scope="col">Questions</th>' in rendered


def test_comparison_escapes_model_labels():
    from copy import deepcopy

    from lava.evaluation.reporting import _comparison_detail

    report = deepcopy(load_report(ROOT))
    pair = report["paired_document_comparisons"][0]
    runs = {run["job_name"]: run for run in report["runs"]}
    runs[pair["challenger_job"]]["label"] = '<script>alert("model")</script>'
    rendered = _comparison_detail(pair, runs)
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered


def run_again(pilot, *, prefix="retry", source="run"):
    _, _, summary, _, _, _ = pilot
    model = load_resolved_model(
        ROOT / "configs/oracle_reader_models.lock.json", summary["model_key"]
    )
    return benchmark.run_oracle_benchmark(
        bucket="test-bucket",
        region="us-west-2",
        manifest_s3_uri="s3://test-bucket/manifest.jsonl",
        output_s3_prefix="s3://test-bucket/" + prefix,
        protocol_lock_id=summary["protocol_lock_id"],
        model_spec=model,
        experiment_id="pilot",
        limit=16,
        evaluation_root=ROOT,
        resume_s3_prefix="s3://test-bucket/" + source,
    )


def test_resume_reuses_all_answers_without_constructing_model(pilot, monkeypatch, capsys):
    import shutil

    s3, sm, original, _, _, tmp_path = pilot
    # Simulate losing the entire instance disk. Only S3 objects survive.
    shutil.rmtree(tmp_path / "model")
    monkeypatch.setattr(benchmark, "build_reader", lambda *a, **kw: pytest.fail("Model was loaded"))
    resumed = run_again(pilot)
    assert resumed["resume"] == {"reused_question_count": 16, "new_question_count": 0}
    for key in benchmark._summary(()):
        assert resumed[key] == original[key]
    assert len(list((tmp_path / "model/private/raw_responses").glob("*.txt"))) == 16
    events = capsys.readouterr().out
    assert events.count('"event": "question.reused"') == 16
    assert '"timestamp_utc"' in events and '"elapsed_seconds"' in events
    # Include the malformed response; reuse must not cherry-pick valid answers.
    assert resumed["schema_valid_rate"] == 15 / 16
    description = sm.describe_training_job()
    description["HyperParameters"]["resume_s3_prefix"] = "s3://test-bucket/run"
    description["OutputDataConfig"]["S3OutputPath"] = "s3://test-bucket/retry/sagemaker-output"
    artifact = "retry/sagemaker-output/job/output/model"
    description["ModelArtifacts"]["S3ModelArtifacts"] = "s3://test-bucket/" + artifact
    for file in (tmp_path / "model").rglob("*"):
        if file.is_file():
            s3.objects[artifact + "/" + file.relative_to(tmp_path / "model").as_posix()] = (
                file.read_bytes()
            )
    gate = verify_training_model_artifact(
        s3_client=s3,
        sagemaker_client=sm,
        job_name="job",
        expected_output_s3_prefix="s3://test-bucket/retry",
    )
    assert gate.raw_response_count == 16


def test_multiple_interruptions_keep_completed_questions(pilot, monkeypatch):
    from lava.readers.private_artifacts import read_raw_response

    s3, _, _, _, _, _ = pilot
    # Preserve first three completed answers, including a parser failure.
    checkpoints = sorted(k for k in s3.objects if k.startswith("run/checkpoints/"))
    for key in checkpoints:
        if json.loads(s3.objects[key])["record"]["question_id"] not in {"q00", "q01", "q02"}:
            del s3.objects[key]
    real_factory = benchmark.build_reader
    calls = []

    class InterruptingReader:
        def predict(self, example):
            calls.append(example.question_id)
            if len(calls) == 3:
                raise TimeoutError("Synthetic interruption")
            return real_factory().predict(example)

    monkeypatch.setattr(benchmark, "build_reader", lambda *a, **kw: InterruptingReader())
    with pytest.raises(TimeoutError, match="Synthetic"):
        run_again(pilot, prefix="attempt1")
    assert calls == ["q03", "q04", "q05"]
    assert "attempt1/public_summary.json" not in s3.objects
    first_raw = read_raw_response("q00")
    completed = [k for k in s3.objects if k.startswith("attempt1/checkpoints/questions/")]
    assert len(completed) == 5
    calls.clear()

    class CompletingReader:
        def predict(self, example):
            calls.append(example.question_id)
            return real_factory().predict(example)

    monkeypatch.setattr(benchmark, "build_reader", lambda *a, **kw: CompletingReader())
    resumed = run_again(pilot, prefix="attempt2", source="attempt1")
    assert calls == [f"q{i:02d}" for i in range(5, 16)]
    assert resumed["resume"] == {"reused_question_count": 5, "new_question_count": 11}
    assert first_raw == read_raw_response("q00")
    records = [
        json.loads(line) for line in s3.objects["attempt2/private_records.jsonl"].splitlines()
    ]
    assert len({r["question_id"] for r in records}) == 16


@pytest.mark.parametrize(
    "mutation", ["checksum", "code", "model", "prompt", "manifest", "record", "raw", "extra"]
)
def test_resume_rejects_corruption_before_loading_model(pilot, monkeypatch, mutation):
    s3 = pilot[0]
    key = next(k for k in s3.objects if k.startswith("run/checkpoints/"))
    data = json.loads(s3.objects[key])
    if mutation == "code":
        data["contract"]["git_commit_sha"] = "d" * 40
    elif mutation == "model":
        data["contract"]["model"]["revision"] = "d" * 40
    elif mutation == "prompt":
        data["contract"]["prompt_version"] = "different"
    elif mutation == "manifest":
        data["contract"]["asset_manifest_sha256"] = "a" * 64
    elif mutation == "record":
        data["record"]["normalized_exact_answer_score"] = 1
    elif mutation == "raw":
        data["raw_response"] = "changed"
    elif mutation == "extra":
        data["record"]["question_id"] = "outside-pilot"
    s3.objects[key] = json.dumps(data).encode()
    if mutation != "checksum":
        s3.metadata[key]["sha256"] = hashlib.sha256(s3.objects[key]).hexdigest()
    monkeypatch.setattr(
        benchmark, "build_reader", lambda *a, **kw: pytest.fail("Loaded incompatible model")
    )
    with pytest.raises((RuntimeError, ValueError)):
        run_again(pilot)
    assert "retry/public_summary.json" not in s3.objects


def test_checkpoint_upload_failure_does_not_claim_completion(pilot, monkeypatch, capsys):
    s3 = pilot[0]
    original = s3.put_object

    def fail_upload(**kwargs):
        if kwargs["Key"].startswith("retry/checkpoints/"):
            raise ClientError({"Error": {"Code": "AccessDenied"}}, "PutObject")
        return original(**kwargs)

    monkeypatch.setattr(s3, "put_object", fail_upload)
    with pytest.raises(ClientError, match="AccessDenied"):
        run_again(pilot)
    assert '"event": "question.completed"' not in capsys.readouterr().out
    assert "retry/public_summary.json" not in s3.objects


def test_checkpoint_put_is_idempotent_and_refuses_overwrite(pilot):
    from lava.readers.checkpoints import CheckpointStore, QuestionCheckpoint
    from lava.readers.schemas import BenchmarkRecord

    s3 = pilot[0]
    key = next(k for k in s3.objects if k.startswith("run/checkpoints/"))
    payload = json.loads(s3.objects[key])
    store = CheckpointStore(s3, bucket="test-bucket", prefix="run", contract=payload["contract"])
    saved = QuestionCheckpoint(
        BenchmarkRecord.model_validate(payload["record"]), payload["raw_response"]
    )
    before = s3.objects[key]
    store.save(saved)
    assert s3.objects[key] == before
    with pytest.raises(RuntimeError, match="overwrite"):
        store.save(QuestionCheckpoint(saved.record, "different generation"))
    assert s3.objects[key] == before


def test_interruption_during_checkpoint_copy_retains_parent_work(pilot, monkeypatch):
    s3 = pilot[0]
    original = s3.put_object
    calls = 0

    def interrupt(**kwargs):
        nonlocal calls
        if kwargs["Key"].startswith("attempt1/checkpoints/questions/"):
            calls += 1
            if calls == 2:
                raise TimeoutError("Interrupted while restoring saved work")
        return original(**kwargs)

    monkeypatch.setattr(s3, "put_object", interrupt)
    with pytest.raises(TimeoutError):
        run_again(pilot, prefix="attempt1")
    assert len([k for k in s3.objects if k.startswith("attempt1/checkpoints/questions/")]) == 1
    monkeypatch.setattr(s3, "put_object", original)
    monkeypatch.setattr(
        benchmark, "build_reader", lambda *a, **kw: pytest.fail("Lost parent checkpoint")
    )
    resumed = run_again(pilot, prefix="attempt2", source="attempt1")
    assert resumed["resume"]["reused_question_count"] == 16


def test_checkpoint_ancestry_cycles_fail_before_inference(pilot, monkeypatch):
    s3 = pilot[0]
    key = next(k for k in s3.objects if k.startswith("run/checkpoints/questions/"))
    contract = json.loads(s3.objects[key])["contract"]
    from lava.readers.checkpoints import CheckpointStore

    CheckpointStore(s3, bucket="test-bucket", prefix="run", contract=contract).link_parent("other")
    CheckpointStore(s3, bucket="test-bucket", prefix="other", contract=contract).link_parent("run")
    monkeypatch.setattr(benchmark, "build_reader", lambda *a, **kw: pytest.fail("Loaded model"))
    with pytest.raises(RuntimeError, match="cyclic"):
        run_again(pilot)


@pytest.mark.parametrize("status", ["Failed", "Stopped", "Completed", "InProgress"])
def test_resume_preflight_checks_terminal_status_and_immutable_plan(pilot, monkeypatch, status):
    from lava.readers.checkpoints import prepare_resume_plan

    s3, _, _, payload, _contract, _ = pilot
    monkeypatch.setattr("lava.readers.sagemaker._git_sha", lambda _: "c" * 40)
    plan = build_job_plan(
        repo_root=ROOT,
        config_path=ROOT / "configs/oracle_reader_benchmark.yaml",
        model_lock_path=ROOT / "configs/oracle_reader_models.lock.json",
        bucket="test-bucket",
        model_key="qwen35_4b_fused_direct",
        limit=16,
        mode="benchmark",
    )
    manifest_key = plan.manifest_s3_uri.split("test-bucket/")[1]
    s3.objects[manifest_key] = payload
    description = {
        "TrainingJobName": "failed-job",
        "TrainingJobStatus": status,
        "ResourceConfig": {"InstanceType": plan.instance_type},
        "Environment": {"LAVA_GIT_COMMIT_SHA": plan.git_commit_sha},
        "HyperParameters": {
            "checkpoint_schema_version": "1",
            "mode": "benchmark",
            "limit": "16",
            "model_key": plan.model_key,
            "manifest_s3_uri": plan.manifest_s3_uri,
            "protocol_lock_id": plan.protocol_lock_id,
            "experiment_id": f"benchmark-{plan.model_key}-{plan.git_commit_sha[:8]}",
            "output_s3_prefix": plan.output_s3_prefix,
        },
        "OutputDataConfig": {"S3OutputPath": plan.output_s3_prefix + "/sagemaker-output"},
    }
    if status in {"Completed", "InProgress"}:
        with pytest.raises(ValueError, match="Failed or Stopped"):
            prepare_resume_plan(
                plan=plan, description=description, s3=s3, repo_root=ROOT, attempt_id="test"
            )
        return
    updated, count = prepare_resume_plan(
        plan=plan, description=description, s3=s3, repo_root=ROOT, attempt_id="test"
    )
    assert count == 0  # A failure before any answer can restart at the same code revision.
    assert updated.output_s3_prefix == plan.output_s3_prefix + "/attempts/test"
    assert updated.resume_s3_prefix == plan.output_s3_prefix
    assert plan.resume_s3_prefix is None
    for change, expected in [("code", "same Git"), ("legacy", "predates"), ("model", "mismatch")]:
        import copy

        changed = copy.deepcopy(description)
        if change == "code":
            changed["Environment"]["LAVA_GIT_COMMIT_SHA"] = "d" * 40
        elif change == "legacy":
            del changed["HyperParameters"]["checkpoint_schema_version"]
        else:
            changed["HyperParameters"]["model_key"] = "qwen35_9b_fused_direct"
        with pytest.raises(ValueError, match=expected):
            prepare_resume_plan(
                plan=plan, description=changed, s3=s3, repo_root=ROOT, attempt_id="test"
            )


def test_durable_checkpoint_request_matches_pinned_s3_sdk(pilot):
    import base64

    import boto3
    from botocore.stub import Stubber

    from lava.readers.checkpoints import CheckpointStore, QuestionCheckpoint
    from lava.readers.schemas import BenchmarkRecord

    memory = pilot[0]
    key = next(k for k in memory.objects if k.startswith("run/checkpoints/questions/"))
    payload = memory.objects[key]
    data = json.loads(payload)
    client = boto3.session.Session().client(
        "s3",
        region_name="us-west-2",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )
    with Stubber(client) as stub:
        stub.add_response(
            "put_object",
            {},
            {
                "Bucket": "test-bucket",
                "Key": key,
                "Body": payload,
                "ContentType": "application/json",
                "IfNoneMatch": "*",
                "Metadata": {"sha256": hashlib.sha256(payload).hexdigest()},
                "ChecksumSHA256": base64.b64encode(hashlib.sha256(payload).digest()).decode(),
            },
        )
        CheckpointStore(client, bucket="test-bucket", prefix="run", contract=data["contract"]).save(
            QuestionCheckpoint(BenchmarkRecord.model_validate(data["record"]), data["raw_response"])
        )
        stub.assert_no_pending_responses()


def test_artifact_gate_rejects_false_resume_accounting(pilot):
    s3 = pilot[0]
    for key in [
        "run/public_summary.json",
        "run/sagemaker-output/job/output/model/public_summary.json",
    ]:
        data = json.loads(s3.objects[key])
        data["resume"]["reused_question_count"] = 16
        data["resume"]["new_question_count"] = 0
        s3.objects[key] = json.dumps(data).encode()
    with pytest.raises(RuntimeError, match="resume accounting"):
        verify(pilot)
