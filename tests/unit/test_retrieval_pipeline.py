"""Multilingual ranking, label separation, metric denominators and recovery regressions."""

from __future__ import annotations

import hashlib
import io
import json
import math
import runpy
from copy import deepcopy
from dataclasses import fields
from pathlib import Path
from unittest.mock import Mock

import pymupdf
import pytest
from botocore.exceptions import ClientError

from lava.evaluation.schemas import ReferenceRecord
from lava.evaluation.semantic import encode
from lava.readers.runtime_logging import RuntimeEventLogger
from lava.retrieval.evaluation import evaluate_rankings, pilot_sources
from lava.retrieval.lexical import BM25Index, PageText, RetrievalQuery, tokenize
from lava.retrieval.pipeline import extract_pages, rank_query, stage
from lava.retrieval.storage import CheckpointStore

ROOT = Path(__file__).resolve().parents[2]


class FakeS3:
    """Conditional object semantics, including missing-key denial without ListBucket."""

    def __init__(self):
        self.objects = {}
        self.streams = []
        self.fail_completion = False
        self.deny = False

    def put_object(self, *, Bucket, Key, Body, Metadata, **kwargs):
        if self.deny:
            raise ClientError({"Error": {"Code": "AccessDenied"}}, "PutObject")
        existing = self.objects.get(Key)
        if kwargs.get("IfNoneMatch") == "*" and existing is not None:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        if "IfMatch" in kwargs and (existing is None or kwargs["IfMatch"] != existing["etag"]):
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        if self.fail_completion and json.loads(Body).get("state") == "complete":
            raise TimeoutError("Interrupted before commit")
        self.objects[Key] = {
            "body": Body,
            "metadata": Metadata,
            "etag": hashlib.sha256(Body).hexdigest(),
        }
        return {}

    def get_object(self, *, Bucket, Key, VersionId=None):
        if self.deny or Key not in self.objects:
            raise ClientError({"Error": {"Code": "AccessDenied"}}, "GetObject")
        value = self.objects[Key]
        stream = io.BytesIO(value["body"])
        self.streams.append(stream)
        return {
            "Body": stream,
            "Metadata": value["metadata"],
            "ETag": value["etag"],
            "VersionId": value.get("version_id"),
        }


@pytest.mark.parametrize(
    "question,target,other",
    [
        ("売上高", "年間売上高報告", "所在地と経営者"),
        ("doanh thu", "doanh thu năm nay", "địa chỉ văn phòng"),
        ("REVENUE", "revenue amount", "office address"),
    ],
)
def test_multilingual_question_ranks_the_matching_page(question, target, other):
    index = BM25Index((PageText(1, other), PageText(2, "", "textless"), PageText(3, target)))
    ranked = index.rank(question)
    assert ranked[0][0] == 3
    assert {page for page, _ in ranked} == {1, 2, 3}


def test_tokenizer_unicode_normalization_and_marks():
    assert tokenize("ＡＢＣ") == tokenize("abc")
    assert tokenize("Hà Nội") == tokenize("Ha\u0300 No\u0323\u0302i")
    assert tokenize("ma") != tokenize("má")
    assert "c2:売上" in tokenize("年間売上高")
    assert tokenize(" \n!? ") == ()


def test_bm25_matches_independent_hand_calculation():
    # Single-letter words have no character ngrams: lengths are exactly 2 and 1.
    rank = BM25Index((PageText(1, "x x"), PageText(2, "y"))).rank("x")
    expected = math.log(2) * 2 * 2.2 / (2 + 1.2 * (0.25 + 0.75 * 2 / 1.5))
    assert rank[0] == pytest.approx((1, expected))
    assert rank[1] == (2, 0.0)


def test_repeated_query_terms_do_not_change_scoring_and_ties_are_stable():
    index = BM25Index((PageText(1, "same"), PageText(2, "same"), PageText(3, "")))
    assert index.rank("same same") == index.rank("same")
    assert [p for p, _ in index.rank("missing")] == [1, 2, 3]
    assert [p for p, _ in index.rank("same")] == [1, 2, 3]


@pytest.mark.parametrize("pages", [(), (PageText(2, "x"),), (PageText(1, "x"), PageText(1, "y"))])
def test_missing_duplicate_or_shifted_pages_fail(pages):
    with pytest.raises(ValueError, match="complete PDF"):
        BM25Index(pages)


@pytest.mark.parametrize(
    "k1,b", [(0, 0.75), (1.2, -0.1), (float("nan"), 0.75), (1.2, float("inf"))]
)
def test_invalid_bm25_parameters_fail(k1, b):
    with pytest.raises(ValueError):
        BM25Index((PageText(1, "x"),), k1=k1, b=b)


def test_real_pdf_extraction_retains_blank_pages():
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((72, 72), "Revenue 2026")
        pdf.new_page()
        pdf.new_page().insert_text((72, 72), "Office")
        payload = pdf.tobytes()
    output = extract_pages(payload)
    assert output["pdf_sha256"] == hashlib.sha256(payload).hexdigest()
    assert [p["number"] for p in output["pages"]] == [1, 2, 3]
    assert output["pages"][1] == {"number": 2, "text": "", "status": "textless"}


def reference(qid, document, page, question="target"):
    return ReferenceRecord(
        question_id=qid,
        document_id=document,
        question=question,
        answer_format="string",
        answer="private answer",
        evidence_pages=(page,),
        language="ja",
    )


def test_ranker_has_no_label_fields_and_reference_edits_cannot_change_ranking():
    assert {field.name for field in fields(RetrievalQuery)} == {
        "question_id",
        "document_id",
        "question",
    }
    pages = (PageText(1, "unrelated"), PageText(2, "target"))
    original = reference("a", "d", 1)
    changed = original.model_copy(update={"answer": "different", "evidence_pages": (2,)})
    results = [
        rank_query(
            RetrievalQuery(row.question_id, row.document_id, row.question),
            pages,
            {"k1": 1.2, "b": 0.75},
        )
        for row in (original, changed)
    ]
    assert results[0] == results[1]


def test_document_average_and_empty_page_failures_keep_all_questions():
    refs = (reference("a", "d1", 1), reference("b", "d1", 1), reference("c", "d2", 2))
    documents = {
        "d1": (PageText(1, "target"), PageText(2, "other")),
        "d2": (PageText(1, "target"), PageText(2, "", "extraction_error")),
    }
    rankings = {
        row.question_id: rank_query(
            RetrievalQuery(row.question_id, row.document_id, row.question),
            documents[row.document_id],
            {"k1": 1.2, "b": 0.75},
        )
        for row in refs
    }
    result = evaluate_rankings(refs, rankings, documents, budgets=(1, 2))
    bm25 = result["methods"]["bm25"]
    assert bm25["question_average"]["1"]["recall_at_k"] == pytest.approx(2 / 3)
    assert bm25["document_average"]["1"]["recall_at_k"] == 0.5
    assert bm25["question_average"]["2"]["recall_at_k"] == 1
    assert result["document_coverage"][1]["gold_pages_without_text"] == 1
    assert result["document_coverage"][1]["extraction_errors"] == 1
    for mutation in ("missing", "duplicate", "out_of_bounds", "wrong_document"):
        altered = deepcopy(rankings)
        if mutation == "missing":
            del altered["a"]
        if mutation == "duplicate":
            altered["a"]["bm25"] = [1, 1]
        if mutation == "out_of_bounds":
            altered["a"]["bm25"] = [1, 3]
        if mutation == "wrong_document":
            altered["a"]["document_id"] = "d2"
        with pytest.raises(ValueError):
            evaluate_rankings(refs, altered, documents, budgets=(1, 2))


def test_checkpoint_survives_process_loss_and_needs_no_list_permission(capsys):
    s3 = FakeS3()
    store = CheckpointStore(s3, "private", "experiments/submissions/retrieval/test")
    logger = RuntimeEventLogger("test")
    compute = Mock(return_value={"ranking": [2, 1]})
    identity = {"input_sha256": "a" * 64}
    first, reused = stage(store, identity, compute, logger=logger, name="query")
    assert not reused
    restarted = CheckpointStore(s3, "private", store.prefix)
    second, reused = stage(restarted, identity, compute, logger=logger, name="query")
    assert reused and first == second
    compute.assert_called_once()
    assert all(stream.closed for stream in s3.streams)
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert all("timestamp_utc" in event and "elapsed_seconds" in event for event in events)
    assert events[-1]["event"] == "retrieval.stage.reused"


def test_pending_checkpoint_resumes_after_interrupted_commit():
    s3 = FakeS3()
    store = CheckpointStore(s3, "private", "test")
    compute = Mock(return_value={"ranking": [1]})
    args = {"logger": RuntimeEventLogger("test"), "name": "query"}
    s3.fail_completion = True
    with pytest.raises(TimeoutError):
        stage(store, {"q": 1}, compute, **args)
    assert store.read(next(iter(s3.objects)).removeprefix("test/")) is None
    s3.fail_completion = False
    assert stage(store, {"q": 1}, compute, **args)[1] is False
    assert stage(store, {"q": 1}, compute, **args)[1] is True
    assert compute.call_count == 2


def test_failed_computation_does_not_commit_and_other_completed_work_survives():
    store = CheckpointStore(FakeS3(), "private", "test")
    args = {"logger": RuntimeEventLogger("test"), "name": "query"}
    good = Mock(return_value={"ranking": [1]})
    stage(store, {"q": "good"}, good, **args)
    bad = Mock(side_effect=RuntimeError("model interrupted"))
    with pytest.raises(RuntimeError):
        stage(store, {"q": "bad"}, bad, **args)
    stage(store, {"q": "good"}, good, **args)
    good.assert_called_once()


def test_corruption_and_access_denial_are_never_treated_as_cache_misses():
    s3 = FakeS3()
    store = CheckpointStore(s3, "private", "test")
    store.write("a", {"score": 1})
    with pytest.raises(ValueError, match="Conflicting"):
        store.write("a", {"score": 2})
    s3.objects["test/a"]["body"] = b"corrupt"
    with pytest.raises(ValueError, match="checksum"):
        store.read("a")
    s3.deny = True
    with pytest.raises(ClientError, match="AccessDenied"):
        store.read("unknown")


def test_stage_identity_mismatch_rejected_even_with_valid_outer_storage():
    s3 = FakeS3()
    store = CheckpointStore(s3, "private", "test")
    args = {"logger": RuntimeEventLogger("test"), "name": "query"}
    stage(store, {"q": 1}, lambda: {"x": 1}, **args)
    key = next(iter(s3.objects))
    envelope = json.loads(s3.objects[key]["body"])
    envelope["value"]["identity"]["q"] = 2
    body = encode(envelope)
    s3.objects[key]["body"] = body
    s3.objects[key]["metadata"]["sha256"] = hashlib.sha256(body).hexdigest()
    with pytest.raises(ValueError, match="identity"):
        stage(store, {"q": 1}, lambda: {"x": 1}, **args)


def test_preview_reads_no_cloud_data(monkeypatch, capsys):
    command = runpy.run_path(str(ROOT / "scripts/evaluate_retrieval.py"))["main"]
    monkeypatch.setattr("sys.argv", ["evaluate_retrieval.py", "--mode", "preview"])
    cloud = Mock(side_effect=AssertionError("Preview must not contact AWS"))
    monkeypatch.setattr(command.__globals__["boto3"], "client", cloud)
    assert command() == 0
    cloud.assert_not_called()
    assert '"reads_private_sources": false' in capsys.readouterr().out


def test_sources_use_frozen_training_partition_only():
    config = json.loads((ROOT / "configs/retrieval.json").read_text())
    sources = pilot_sources(ROOT, config)
    assert len(sources) == 6 and "train.csv" in sources
    assert not any("test" in row["key"] for row in sources.values())
    with pytest.raises(ValueError, match="checksum"):
        pilot_sources(ROOT, {**config, "source_manifest_sha256": "0" * 64})


def test_complete_pipeline_restores_results_without_extraction_or_ranking(tmp_path, monkeypatch):
    import csv
    import shutil

    from lava.retrieval import evaluation

    (tmp_path / "reports").mkdir()
    for relative in ["src/lava/retrieval", "src/lava/evaluation", "src/lava/readers"]:
        shutil.copytree(ROOT / relative, tmp_path / relative)
    (tmp_path / "scripts").mkdir()
    shutil.copy(ROOT / "scripts/evaluate_retrieval.py", tmp_path / "scripts/evaluate_retrieval.py")
    shutil.copy(ROOT / "uv.lock", tmp_path / "uv.lock")
    monkeypatch.setattr(evaluation, "git_snapshot", lambda _: {"git_commit_sha": "a" * 40})
    s3 = FakeS3()

    def csv_payload(rows):
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        return buffer.getvalue().encode()

    rows = [
        {
            "id": f"q{i}",
            "file_id": f"d{i}",
            "question": "target",
            "answer_format": "string",
            "answer": "private",
            "evidence_page_number": "[2]",
            "language": "ja",
        }
        for i in range(2)
    ]
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((72, 72), "other")
        pdf.new_page().insert_text((72, 72), "target")
        content = pdf.tobytes()
    sources = {
        "train.csv": csv_payload(rows),
        "train_pdfs/d0.pdf": content,
        "train_pdfs/d1.pdf": content,
    }
    manifest = []
    for name, payload in sources.items():
        sha = hashlib.sha256(payload).hexdigest()
        s3.objects[name] = {
            "body": payload,
            "metadata": {"sha256": sha},
            "etag": sha,
            "version_id": "v1",
        }
        manifest.append({"name": name, "s3_key": name, "sha256": sha, "version_id": "v1"})
    payload = csv_payload(manifest)
    (tmp_path / "reports/raw_data_manifest.csv").write_bytes(payload)
    config = json.loads((ROOT / "configs/retrieval.json").read_text())
    config.update(
        expected_question_count=2,
        expected_document_count=2,
        source_manifest_sha256=hashlib.sha256(payload).hexdigest(),
    )
    first = evaluation.run_evaluation(tmp_path, s3, "private", config, RuntimeEventLogger("test"))
    public = (tmp_path / "reports/retrieval/summary.json").read_bytes()
    assert first["documents_computed"] == 2 and first["queries_computed"] == 2
    assert json.loads(public)["methods"]["bm25"]["question_average"]["1"]["recall_at_k"] == 1
    shutil.rmtree(tmp_path / "artifacts/retrieval/inputs")
    monkeypatch.setattr(
        evaluation, "extract_pages", Mock(side_effect=AssertionError("must reuse extraction"))
    )
    monkeypatch.setattr(
        evaluation, "rank_query", Mock(side_effect=AssertionError("must reuse rankings"))
    )
    second = evaluation.run_evaluation(tmp_path, s3, "private", config, RuntimeEventLogger("test"))
    assert second["documents_reused"] == 2 and second["queries_reused"] == 2
    assert (tmp_path / "reports/retrieval/summary.json").read_bytes() == public
    assert all(stream.closed for stream in s3.streams)


def test_public_chart_and_table_share_exact_values_and_escape_document_names():
    from lava.evaluation.walkthrough import render_table
    from lava.retrieval.reporting import document_rows, metric_rows, recall_chart, render_report

    refs = (reference("q", "d", 2),)
    documents = {"d": (PageText(1, "other"), PageText(2, "target"))}
    rankings = {
        "q": rank_query(RetrievalQuery("q", "d", "target"), documents["d"], {"k1": 1.2, "b": 0.75})
    }
    summary = {
        **evaluate_rankings(refs, rankings, documents, budgets=(1, 2)),
        "implementation": {"config": {"budgets": [1, 2]}},
        "question_count": 1,
        "document_count": 1,
        "completed_at_utc": "2026-09-07T00:00:00Z",
        "first_run_seconds": 1.234,
        "contract_id": "a" * 64,
    }
    table = metric_rows(summary)
    assert table[0]["Evidence recall"] == "0.0%"
    assert table[2]["Evidence recall"] == "100.0%"
    assert table[2]["MRR"] == "1.000"
    svg = recall_chart(summary)
    assert 'role="img"' in svg and "viewBox=" in svg
    assert "Multilingual BM25, k=1: 100.0%" in svg
    summary["methods"]["bm25"]["by"]["document"]["<script>"] = summary["methods"]["bm25"]["by"][
        "document"
    ].pop("doc-01")
    rendered = render_table(document_rows(summary), caption="Synthetic coverage")
    assert "<script>" not in rendered and "&lt;script&gt;" in rendered
    assert "not held-out" in render_report(summary)
