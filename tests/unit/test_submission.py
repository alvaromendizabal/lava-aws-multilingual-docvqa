"""Submission coverage, serialization, provenance and interrupted-publication regressions."""

from __future__ import annotations

import csv
import io
import json
import runpy
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError

from lava.evaluation.semantic import encode
from lava.evaluation.submission import (
    SUBMISSION_COLUMNS,
    TEST_COLUMNS,
    build_submission,
    load_submission_inputs,
    sha256,
    validate_submission_csv,
)
from lava.evaluation.submission_store import cached_source, persist_submission
from lava.readers.runtime_logging import RuntimeEventLogger

ROOT = Path(__file__).resolve().parents[2]


def csv_bytes(columns, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


@pytest.fixture
def example():
    formats = ["string", "number", "ordered_list", "unordered_list"]
    rows = [
        {
            "id": f"q{i}",
            "file_id": f"d{i % 2}",
            "question": "公開の合成質問",
            "answer_format": fmt,
            "language": "ja" if i % 2 else "vi",
        }
        for i, fmt in enumerate(formats)
    ]
    test = csv_bytes(TEST_COLUMNS, rows)
    template = csv_bytes(
        SUBMISSION_COLUMNS,
        [
            {"id": row["id"], "answer": "placeholder", "evidence_page_number": "[1]"}
            for row in reversed(rows)
        ],
    )
    contract = {
        "competition": "lava-challenge-2026",
        "expected_question_count": 4,
        "expected_document_count": 2,
        "answer_format_counts": dict.fromkeys(formats, 1),
        "language_counts": {"ja": 2, "vi": 2},
        "source_files": {
            name: {"key": name, "version_id": "v1", "sha256": sha256(data)}
            for name, data in {"test.csv": test, "sample_submission.csv": template}.items()
        },
    }
    predictions = [
        {"question_id": "q0", "answer": '東京, "本社"\nHà Nội', "evidence_pages": [2, 1]},
        {"question_id": "q1", "answer": "1,000 kg", "evidence_pages": [1]},
        {"question_id": "q2", "answer": "['東京', 'Hà Nội', '東京']", "evidence_pages": [2]},
        {"question_id": "q3", "answer": '["red", "blue"]', "evidence_pages": [1]},
    ]
    inputs = load_submission_inputs(test, template, contract)
    return SimpleNamespace(
        test=test,
        template=template,
        contract=contract,
        inputs=inputs,
        predictions=predictions,
        counts={"d0": 2, "d1": 1},
    )


def packed(example, predictions=None):
    payload = b"".join(
        encode(row) for row in (example.predictions if predictions is None else predictions)
    )
    provenance = {
        "split": "test",
        "evidence_scope": "retrieved_full_document_pages",
        "test_csv_sha256": sha256(example.test),
        "question_count": 4,
        "model_id": "public/synthetic",
        "model_revision": "a" * 40,
        "code_commit": "b" * 40,
        "predictions_sha256": sha256(payload),
        "page_counts_sha256": sha256(encode(example.counts)),
    }
    return payload, provenance


def test_unicode_csv_quoting_template_order_and_repeated_ordered_items(example):
    payload, provenance = packed(example)
    result = build_submission(example.inputs, payload, example.counts, provenance)
    rows = list(csv.DictReader(io.StringIO(result.decode())))
    assert [row["id"] for row in rows] == ["q3", "q2", "q1", "q0"]
    assert rows[-1]["answer"] == example.predictions[0]["answer"]
    assert json.loads(rows[1]["answer"]) == ["東京", "Hà Nội", "東京"]
    assert json.loads(rows[-1]["evidence_page_number"]) == [1, 2]
    check = validate_submission_csv(result, example.inputs, example.counts)
    assert check["schema_valid"] and check["row_count"] == 4
    assert not check["uploaded_to_kaggle"] and not check["model_quality_verified"]
    assert build_submission(example.inputs, payload, example.counts, provenance) == result


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "integer_id"])
def test_incomplete_or_invalid_prediction_ids_cannot_create_submission(example, mutation):
    rows = deepcopy(example.predictions)
    if mutation == "missing":
        rows.pop()
    elif mutation == "extra":
        rows.append({**rows[0], "question_id": "extra"})
    elif mutation == "duplicate":
        rows.append(rows[0])
    else:
        rows[0]["question_id"] = 1
    payload, provenance = packed(example, rows)
    with pytest.raises(ValueError):
        build_submission(example.inputs, payload, example.counts, provenance)


@pytest.mark.parametrize("pages", [[], [0], [-1], [3], [True], [1.0], ["1"], [1, 1], "[1]"])
def test_invalid_predicted_pages_are_rejected(example, pages):
    example.predictions[0]["evidence_pages"] = pages
    payload, provenance = packed(example)
    with pytest.raises(ValueError):
        build_submission(example.inputs, payload, example.counts, provenance)


@pytest.mark.parametrize(
    "answer",
    ["", "[]", "[True]", "[{}]", "[[1]]", "[1e999]", "('x',)", "__import__('os').getcwd()"],
)
def test_invalid_list_answers_are_rejected_without_execution(example, answer):
    example.predictions[2]["answer"] = answer
    payload, provenance = packed(example)
    with pytest.raises((ValueError, TypeError)):
        build_submission(example.inputs, payload, example.counts, provenance)


@pytest.mark.parametrize(
    "field,value",
    [
        ("split", "train"),
        ("evidence_scope", "oracle_supplied_pages"),
        ("test_csv_sha256", "wrong"),
        ("question_count", 16),
        ("model_revision", "main"),
        ("code_commit", ""),
        ("predictions_sha256", "wrong"),
        ("page_counts_sha256", "wrong"),
    ],
)
def test_oracle_or_unbound_provenance_is_rejected(example, field, value):
    payload, provenance = packed(example)
    provenance[field] = value
    with pytest.raises(ValueError):
        build_submission(example.inputs, payload, example.counts, provenance)


@pytest.mark.parametrize("counts", [{"d0": 2}, {"d0": True, "d1": 1}, {"d0": 2, "d1": 0}])
def test_complete_verified_page_counts_are_required(example, counts):
    example.counts = counts
    payload, provenance = packed(example)
    with pytest.raises(ValueError):
        build_submission(example.inputs, payload, example.counts, provenance)


def test_input_checksum_mismatch_stops_before_using_template_placeholders(example):
    with pytest.raises(ValueError, match="checksum"):
        load_submission_inputs(example.test + b"\n", example.template, example.contract)


@pytest.mark.parametrize("change", ["columns", "duplicate", "coverage", "format", "language"])
def test_corrupt_source_schema_or_coverage_is_rejected_even_with_matching_hash(example, change):
    rows = list(csv.DictReader(io.StringIO(example.test.decode())))
    columns = TEST_COLUMNS
    if change == "columns":
        columns = tuple(reversed(columns))
    elif change == "duplicate":
        rows[0]["id"] = rows[1]["id"]
    elif change == "coverage":
        rows.pop()
    elif change == "format":
        rows[0]["answer_format"] = "unsupported"
    else:
        rows[0]["language"] = "xx"
    payload = csv_bytes(columns, rows)
    example.contract["source_files"]["test.csv"]["sha256"] = sha256(payload)
    with pytest.raises(ValueError):
        load_submission_inputs(payload, example.template, example.contract)


def test_external_csv_validation_rejects_bad_row_order_and_extra_columns(example):
    payload, provenance = packed(example)
    result = build_submission(example.inputs, payload, example.counts, provenance)
    rows = list(csv.DictReader(io.StringIO(result.decode())))
    with pytest.raises(ValueError, match="row order"):
        validate_submission_csv(
            csv_bytes(SUBMISSION_COLUMNS, reversed(rows)), example.inputs, example.counts
        )
    with pytest.raises(ValueError, match="columns"):
        validate_submission_csv(
            result.replace(b"id,answer,", b"extra,id,answer,"), example.inputs, example.counts
        )


class MissingObject(Exception):
    pass


class Store:
    exceptions = SimpleNamespace(NoSuchKey=MissingObject)

    def __init__(self):
        self.objects, self.metadata, self.reads = {}, {}, []
        self.fail_manifest = False

    def get_object(self, **kwargs):
        key = kwargs["Key"]
        if key not in self.objects:
            raise MissingObject(key)
        body = io.BytesIO(self.objects[key])
        self.reads.append(body)
        return {"Body": body, "Metadata": self.metadata.get(key, {}), "VersionId": "v1"}

    def put_object(self, **kwargs):
        key = kwargs["Key"]
        if self.fail_manifest and key.endswith("manifest.json"):
            raise RuntimeError("simulated interruption")
        if key in self.objects:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.objects[key] = kwargs["Body"]
        self.metadata[key] = kwargs.get("Metadata", {})


def test_interrupted_bundle_publication_resumes_after_csv_without_overwrite(
    example, tmp_path, capsys
):
    s3, logger = Store(), RuntimeEventLogger("test")
    payload, provenance = packed(example)
    result = build_submission(example.inputs, payload, example.counts, provenance)
    manifest = {"submission_sha256": sha256(result), "provenance": provenance}
    s3.fail_manifest = True
    with pytest.raises(RuntimeError, match="interruption"):
        persist_submission(s3, "bucket", tmp_path, result, manifest, logger)
    assert len(s3.objects) == 1
    assert not list(tmp_path.rglob("submission.csv"))
    s3.fail_manifest = False
    target = persist_submission(s3, "bucket", tmp_path, result, manifest, logger)
    assert target.read_bytes() == result
    assert len(s3.objects) == 2
    assert persist_submission(s3, "bucket", tmp_path, result, manifest, logger) == target
    assert all(body.closed for body in s3.reads)
    assert '"event": "submission.csv.reused"' in capsys.readouterr().out


def test_conflicting_immutable_csv_is_rejected(example, tmp_path):
    s3, logger = Store(), RuntimeEventLogger("test")
    payload, provenance = packed(example)
    result = build_submission(example.inputs, payload, example.counts, provenance)
    manifest = {"submission_sha256": sha256(result)}
    persist_submission(s3, "bucket", tmp_path, result, manifest, logger)
    key = next(key for key in s3.objects if key.endswith("submission.csv"))
    s3.objects[key] = b"corrupt"
    with pytest.raises(ValueError, match="corrupted"):
        persist_submission(s3, "bucket", tmp_path, result, manifest, logger)


def test_cache_reuse_and_corruption_recovery_close_s3_body(example, tmp_path):
    s3, logger = Store(), RuntimeEventLogger("test")
    source = example.contract["source_files"]["test.csv"]
    s3.objects[source["key"]] = example.test
    path = tmp_path / "test.csv"
    assert cached_source(s3, "bucket", path, source, logger) == example.test
    assert cached_source(s3, "bucket", path, source, logger) == example.test
    assert len(s3.reads) == 1
    path.write_bytes(b"corrupt")
    assert cached_source(s3, "bucket", path, source, logger) == example.test
    assert len(s3.reads) == 2 and all(body.closed for body in s3.reads)


@pytest.mark.parametrize("fail", [False, True])
def test_cli_persists_terminal_events_and_totals(example, tmp_path, monkeypatch, fail):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/submission.json").write_bytes(encode(example.contract))
    s3 = Store()
    s3.objects.update({"test.csv": example.test, "sample_submission.csv": example.template})
    if fail:
        s3.objects["test.csv"] = b"corrupt"
    module = runpy.run_path(str(ROOT / "scripts/prepare_submission.py"))
    main = module["main"]
    monkeypatch.setitem(main.__globals__, "find_repo_root", lambda _: tmp_path)
    monkeypatch.setattr(module["boto3"], "client", lambda *a, **k: s3)
    monkeypatch.setenv("S3_BUCKET", "bucket")
    monkeypatch.setattr("sys.argv", ["prepare_submission.py", "--mode", "check"])
    if fail:
        with pytest.raises(ValueError, match="checksum"):
            main()
    else:
        assert main() == 0
    logs = next(
        data for key, data in s3.objects.items() if key.startswith("experiments/submissions/logs/")
    )
    events = [json.loads(line) for line in logs.splitlines()]
    assert events[-1]["event"] == ("submission.failed" if fail else "submission.finished")
    assert all("timestamp_utc" in event and "elapsed_seconds" in event for event in events)
    if not fail:
        assert events[-1]["total_elapsed_seconds"] >= 0


@pytest.mark.parametrize("source_corrupt", [False, True])
def test_cli_records_log_upload_failure_without_masking_input_error(
    example, tmp_path, monkeypatch, source_corrupt
):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/submission.json").write_bytes(encode(example.contract))
    s3 = Store()
    s3.objects.update({"test.csv": example.test, "sample_submission.csv": example.template})
    if source_corrupt:
        s3.objects["test.csv"] = b"corrupt"

    def denied_upload(**kwargs):
        raise ClientError({"Error": {"Code": "AccessDenied"}}, "PutObject")

    monkeypatch.setattr(s3, "put_object", denied_upload)
    module = runpy.run_path(str(ROOT / "scripts/prepare_submission.py"))
    main = module["main"]
    monkeypatch.setitem(main.__globals__, "find_repo_root", lambda _: tmp_path)
    monkeypatch.setattr(module["boto3"], "client", lambda *a, **k: s3)
    monkeypatch.setenv("S3_BUCKET", "bucket")
    monkeypatch.setattr("sys.argv", ["prepare_submission.py", "--mode", "check"])
    with pytest.raises(
        ValueError if source_corrupt else ClientError,
        match="checksum" if source_corrupt else "AccessDenied",
    ):
        main()
    path = next((tmp_path / "artifacts/submission/logs").glob("*.jsonl"))
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert events[-1]["event"] == "submission.log.persistence_failed"
    assert events[-1]["level"] == "ERROR"
    assert events[-1]["timestamp_utc"] and events[-1]["elapsed_seconds"] >= 0
    assert any(event["event"] == "submission.failed" for event in events) == source_corrupt
