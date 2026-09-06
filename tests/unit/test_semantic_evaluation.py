"""Explicit recovery, provenance, metric-boundary, and CPU runtime contracts."""

from __future__ import annotations

import io
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError
from pydantic import ValidationError

from lava.evaluation.reporting import (
    _load_evaluation,
    current_model_runs,
    load_report,
    render_report,
)
from lava.evaluation.saved_predictions import (
    ArtifactSnapshot,
    SavedRun,
    evaluate_semantic,
    save_public_evaluation,
)
from lava.evaluation.schemas import PredictionRecord, ReferenceRecord
from lava.evaluation.semantic import (
    PROBES,
    DurableSemanticJudge,
    GemmaDecision,
    ImmutableS3Objects,
    JudgeConfig,
    digest,
    encode,
    judge_contract,
    parse_decision,
)
from lava.readers.runtime_logging import RuntimeEventLogger

ROOT = Path(__file__).resolve().parents[2]


class MissingObject(Exception):
    pass


class MemoryObjects:
    exceptions = SimpleNamespace(NoSuchKey=MissingObject)

    def __init__(self):
        self.objects = {}
        self.metadata = {}
        self.read_bodies = []
        self.reject_writes = False

    def get_object(self, **kwargs):
        key = kwargs["Key"]
        if key not in self.objects:
            raise MissingObject(key)
        body = io.BytesIO(self.objects[key])
        self.read_bodies.append(body)
        return {"Body": body, "Metadata": self.metadata[key]}

    def put_object(self, **kwargs):
        if self.reject_writes:
            raise ClientError({"Error": {"Code": "AccessDenied"}}, "PutObject")
        assert kwargs["IfNoneMatch"] == "*"
        key, payload = kwargs["Key"], kwargs["Body"]
        if key in self.objects:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.objects[key], self.metadata[key] = payload, kwargs["Metadata"]


def objects(client):
    return ImmutableS3Objects(client, "test-bucket", "semantic")


def judge(client, infer, contract=None):
    return DurableSemanticJudge(
        contract or {"contract_id": "a" * 64},
        objects(client),
        infer,
        RuntimeEventLogger("test.semantic"),
    )


@pytest.mark.parametrize("raw, expected", [("YES", True), (" NO\n", False)])
def test_unambiguous_decisions(raw, expected):
    assert parse_decision(raw) is expected


@pytest.mark.parametrize("raw", ["", "yes", "YES because", "YES NO", "```YES```"])
def test_ambiguous_decisions_stop_without_fallback(raw):
    client = MemoryObjects()
    evaluator = judge(client, lambda *_: raw)
    with pytest.raises(ValueError, match="exactly YES or NO"):
        evaluator.equivalent("reference", "candidate", language="ja")
    assert client.objects == {}
    assert evaluator.new_decisions == 0


def test_restart_reuses_yes_and_no_and_only_evaluates_missing_pairs(capsys):
    client = MemoryObjects()
    first = judge(client, lambda a, b, _: "YES" if a == b else "NO")
    assert first.equivalent("a", "a", language="ja")
    assert not first.equivalent("a", "b", language="ja")
    calls = []
    second = judge(client, lambda *args: calls.append(args) or "YES")
    assert second.equivalent("a", "a", language="ja")
    assert not second.equivalent("a", "b", language="ja")
    assert second.equivalent("c", "c", language="ja")
    assert calls == [("c", "c", "ja")]
    assert (second.reused_decisions, second.new_decisions) == (2, 1)
    assert all(body.closed for body in client.read_bodies)
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert all("timestamp_utc" in event and "elapsed_seconds" in event for event in events)
    assert all("reference" not in event and "prediction" not in event for event in events)


def test_rejected_persistence_cannot_increment_progress():
    client = MemoryObjects()
    client.reject_writes = True
    evaluator = judge(client, lambda *_: "NO")
    with pytest.raises(ClientError, match="AccessDenied"):
        evaluator.equivalent("a", "b", language="vi")
    assert evaluator.new_decisions == 0
    assert not client.objects


def test_conditional_writes_are_idempotent_and_conflicts_fail():
    store = objects(MemoryObjects())
    store.write("one.json", {"score": False})
    store.write("one.json", {"score": False})
    with pytest.raises(ValueError, match="Conflicting"):
        store.write("one.json", {"score": True})


@pytest.mark.parametrize("mutation", ["checksum", "request", "raw", "boolean"])
def test_corrupt_decisions_cannot_be_reused(mutation):
    client = MemoryObjects()
    evaluator = judge(client, lambda *_: "YES")
    evaluator.equivalent("a", "a", language="ja")
    key = next(iter(client.objects))
    value = json.loads(client.objects[key])
    if mutation == "request":
        value["request"]["language"] = "vi"
    elif mutation == "raw":
        value["raw_decision"] = "NO"
    elif mutation == "boolean":
        value["equivalent"] = 1
    client.objects[key] = encode(value)
    client.metadata[key]["sha256"] = (
        "invalid" if mutation == "checksum" else digest(client.objects[key])
    )
    with pytest.raises(ValueError):
        evaluator.equivalent("a", "a", language="ja")


def test_changed_contract_or_language_never_reuses_old_decisions():
    client = MemoryObjects()
    evaluator = judge(client, lambda *_: "YES")
    evaluator.equivalent("a", "a", language="ja")
    changed = judge(client, lambda *_: "NO", {"contract_id": "b" * 64})
    assert not changed.equivalent("a", "a", language="ja")
    assert not changed.equivalent("a", "a", language="vi")
    assert changed.new_decisions == 2


def test_failed_acceptance_probe_prevents_public_scores():
    client = MemoryObjects()
    with pytest.raises(ValueError, match="acceptance probe"):
        judge(client, lambda *_: "YES").validate()
    assert all("/runs/" not in key for key in client.objects)


def test_semantic_score_uses_predicted_pages_and_persists_idempotently():
    client = MemoryObjects()
    cases = {(a, b, lang): "YES" if expected else "NO" for a, b, lang, expected in PROBES}
    evaluator = judge(client, lambda a, b, lang: cases.get((a, b, lang), "YES"))
    reference = ReferenceRecord(
        question_id="q",
        document_id="doc-01",
        question="synthetic",
        answer_format="string",
        answer="10",
        evidence_pages=(1,),
        language="ja",
    )
    prediction = PredictionRecord(question_id="q", answer="10", evidence_pages=(2,))
    run = SavedRun(
        "job", "test-bucket", (), (reference,), (prediction,), {"source_records_sha256": "d" * 64}
    )
    result = evaluate_semantic(run, evaluator)
    assert result["metrics"]["question_micro"] == {"answer": 1.0, "grounding": 0.0, "overall": 0.5}
    assert result["official_server_score"] is None
    assert result["status"] == "local_published_formula_score"
    assert evaluate_semantic(run, evaluator) == result
    assert sum("public_summary" in key for key in client.objects) == 1


def test_snapshot_reuses_verified_bytes_and_closes_remote_stream():
    client = MemoryObjects()
    client.objects["file"], client.metadata["file"] = b"original", {}
    snapshot = ArtifactSnapshot(client)
    with snapshot.get_object(Bucket="bucket", Key="file")["Body"] as body:
        assert body.read() == b"original"
    client.objects["file"] = b"changed"
    with snapshot.get_object(Bucket="bucket", Key="file")["Body"] as body:
        assert body.read() == b"original"
    assert len(client.read_bodies) == 1 and client.read_bodies[0].closed


@pytest.mark.parametrize(
    "field,value",
    [
        ("device", "cuda"),
        ("dtype", "bfloat16"),
        ("do_sample", True),
        ("official_server_parity", True),
        ("threads", 0),
        ("model_revision", "main"),
    ],
)
def test_config_rejects_unimplemented_or_unpinned_runtime(field, value):
    config = json.loads((ROOT / "configs/semantic_judge.json").read_text())
    with pytest.raises(ValidationError):
        JudgeConfig.model_validate({**config, field: value})


def test_current_coverage_prefers_full_pilot_not_recent_smoke_or_best_score():
    full = {
        "model_key": "model",
        "complete": True,
        "job_name": "full",
        "summary": {"generated_at_utc": "2026-01-01"},
    }
    smoke = {
        **full,
        "complete": False,
        "job_name": "smoke",
        "summary": {"generated_at_utc": "2026-12-31", "normalized_exact_answer_micro": 1},
    }
    assert current_model_runs([smoke, full]) == [full]
    report = load_report(ROOT)
    completed = {r["model_key"] for r in report["current_models"] if r["complete"]}
    assert {"qwen35_4b_fused_direct", "qwen35_9b_fused_direct"} <= completed
    # Future completed 27B and semantic results must not break this regression check.
    for run in report["runs"]:
        run["semantic_summary"] = None
    html = render_report(report)
    assert html.count("16 / 16") >= 2
    assert "Not evaluated" in html and "<details>" in html


def test_stale_semantic_contract_remains_inspectable_but_is_not_current(tmp_path, monkeypatch):
    payload = {
        "source": {"job_name": "job", "source_summary_sha256": "a" * 64},
        "judge_contract": {"contract_id": "old"},
        "official_server_score": None,
        "status": "local_published_formula_score",
    }
    save_public_evaluation(tmp_path, "job", "semantic_summary", payload)
    monkeypatch.setattr("lava.evaluation.semantic.judge_contract", lambda _: {"contract_id": "new"})
    path = tmp_path / "reports/oracle_reader/runs/job/sync_manifest.json"
    result = _load_evaluation(path, "semantic_summary", "a" * 64)
    assert result["contract_current"] is False
    assert result["judge_contract"]["contract_id"] == "old"
    with pytest.raises(ValueError, match="different reader"):
        _load_evaluation(path, "semantic_summary", "b" * 64)
    payload["official_server_score"] = 1
    save_public_evaluation(tmp_path, "job", "semantic_summary", payload)
    with pytest.raises(ValueError, match="organizer-server"):
        _load_evaluation(path, "semantic_summary", "a" * 64)


def test_public_artifact_rejects_path_traversal(tmp_path):
    with pytest.raises(ValueError, match="job name"):
        save_public_evaluation(tmp_path, "../escape", "supporting_metrics", {})


def test_derived_artifact_checksum_tampering_fails(tmp_path):
    save_public_evaluation(tmp_path, "job", "supporting_metrics", {})
    directory = tmp_path / "reports/oracle_reader/runs/job"
    (directory / "supporting_metrics.json").write_text('{"tampered": true}')
    with pytest.raises(ValueError, match="checksum"):
        _load_evaluation(directory / "sync_manifest.json", "supporting_metrics", "a" * 64)


def test_permission_errors_are_not_cache_misses(monkeypatch):
    client = MemoryObjects()

    def denied(**kwargs):
        raise ClientError({"Error": {"Code": "AccessDenied"}}, "GetObject")

    monkeypatch.setattr(client, "get_object", denied)
    with pytest.raises(ClientError, match="AccessDenied"):
        objects(client).read("one.json")


def test_execution_notebook_runs_from_notebook_directory(monkeypatch):
    tables = []
    monkeypatch.chdir(ROOT / "notebooks")
    monkeypatch.setattr("IPython.display.display", tables.append)
    runpy.run_path(str(ROOT / "notebooks/02_verified_gpu_execution.py"))
    table = tables[0].set_index("reader")
    assert table.loc["Qwen3.5 · 4B", "coverage"] == "16 / 16"
    assert table.loc["Qwen3.5 · 9B", "coverage"] == "16 / 16"
    assert not table["local_lava_score"].isna().any()


def test_pinned_cpu_gemma_runtime_can_generate_with_synthetic_weights(tmp_path):
    """Exercise the real library/model path; this does not test pretrained judge quality."""
    import torch
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    config = judge_contract(ROOT)["config"]
    runtime = GemmaDecision(config, RuntimeEventLogger("test.cpu"), tmp_path)
    torch.set_num_threads(1)
    tiny = Gemma3TextConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=128,
        sliding_window=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    runtime.model = Gemma3ForCausalLM(tiny).eval()

    class Tokenizer:
        eos_token_id = 2

        def apply_chat_template(self, *args, **kwargs):
            return {
                "input_ids": torch.tensor([[1, 4, 5]]),
                "attention_mask": torch.ones((1, 3), dtype=torch.long),
            }

        def decode(self, tokens, **kwargs):
            assert 1 <= len(tokens) <= config["max_new_tokens"]
            return "NO"

    runtime.tokenizer = Tokenizer()
    assert runtime("a", "b", "ja") == "NO"
