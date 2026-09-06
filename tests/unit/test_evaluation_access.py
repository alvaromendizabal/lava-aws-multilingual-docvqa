"""Login, gated access, secret-safe failures, and no-compute preflight contracts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, LocalTokenNotFoundError

from lava.evaluation.access import (
    EvaluationAccessError,
    available_memory_bytes,
    check_cpu_memory,
    check_judge_access,
    hub_access_errors,
)
from lava.evaluation.semantic import GemmaDecision, judge_contract
from lava.readers.runtime_logging import RuntimeEventLogger

ROOT = Path(__file__).resolve().parents[2]
SECRET = "synthetic-private-credential-must-not-appear"


def test_access_check_uses_authenticated_metadata_not_weight_downloads(capsys):
    config = judge_contract(ROOT)["config"]
    calls = []

    class Api:
        def whoami(self, **kwargs):
            assert kwargs == {"token": True}
            return {"name": "synthetic-user", "auth": {"accessToken": SECRET}}

    def metadata(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(commit_hash=config["model_revision"])

    result = check_judge_access(
        config, RuntimeEventLogger("test.access"), api=Api(), metadata=metadata
    )
    assert result["username"] == "synthetic-user"
    assert result["downloaded_weights"] is False and result["new_cloud_compute"] is False
    assert len(calls) == 1
    assert calls[0][0].endswith(f"/{config['model_revision']}/config.json")
    assert calls[0][1] == {"token": True, "timeout": 10, "retry_on_errors": False}
    output = capsys.readouterr().out
    assert SECRET not in output
    assert "judge.access.verified" in output


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("missing", "not signed in"),
        ("gated", "Google's terms"),
        ("401", "--force"),
        ("403", "denied permission"),
        ("404", "pinned"),
        ("503", "Retry"),
        ("network", "does not establish that your login is invalid"),
    ],
)
def test_access_failures_are_actionable_and_do_not_leak_http_details(kind, expected, capsys):
    if kind == "missing":
        error = LocalTokenNotFoundError(SECRET)
    elif kind == "network":
        error = httpx.ConnectTimeout(SECRET)
    else:
        response = httpx.Response(
            403 if kind == "gated" else int(kind),
            request=httpx.Request("GET", "https://huggingface.co/private"),
        )
        error_type = GatedRepoError if kind == "gated" else HfHubHTTPError
        error = error_type(SECRET, response=response)
    logger = RuntimeEventLogger("test.access")
    with (
        pytest.raises(EvaluationAccessError, match=expected),
        logger.stage("access"),
        hub_access_errors(),
    ):
        raise error
    assert SECRET not in capsys.readouterr().out


def test_access_check_rejects_revision_mismatch():
    api = SimpleNamespace(whoami=lambda **_: {"name": "synthetic"})
    with pytest.raises(EvaluationAccessError, match="pinned"):
        check_judge_access(
            judge_contract(ROOT)["config"],
            RuntimeEventLogger("test"),
            api=api,
            metadata=lambda *a, **kw: SimpleNamespace(commit_hash="wrong"),
        )


def test_access_errors_do_not_hide_programming_errors():
    with pytest.raises(ValueError, match="unexpected"), hub_access_errors():
        raise ValueError("unexpected")


def test_gemma_load_translates_gate_error_before_logging(tmp_path, monkeypatch, capsys):
    from transformers import AutoTokenizer

    def denied(*args, **kwargs):
        raise GatedRepoError(
            SECRET,
            response=httpx.Response(
                403, request=httpx.Request("GET", "https://huggingface.co/private")
            ),
        )

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", denied)
    monkeypatch.setattr("lava.evaluation.semantic.check_cpu_memory", lambda _: None)
    runtime = GemmaDecision(judge_contract(ROOT)["config"], RuntimeEventLogger("test"), tmp_path)
    with pytest.raises(EvaluationAccessError, match="Google's terms"):
        runtime("a", "b", "ja")
    assert SECRET not in capsys.readouterr().out


def load_command():
    spec = importlib.util.spec_from_file_location(
        "evaluate_oracle_reader_test", ROOT / "scripts/evaluate_oracle_reader.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_mode_never_reads_private_predictions_or_creates_cloud_clients(monkeypatch):
    module = load_command()
    checks = []
    monkeypatch.setattr(module, "check_judge_access", lambda *a: checks.append(a))
    monkeypatch.setattr(module, "check_cpu_memory", lambda _: None)

    def forbidden(*args, **kwargs):
        pytest.fail("Access check must not create cloud clients or load private predictions")

    monkeypatch.setattr(module.boto3, "Session", forbidden)
    monkeypatch.setattr(module, "load_saved_run", forbidden)
    monkeypatch.setattr(module, "load_report", forbidden)
    assert (
        module.run_evaluation(ROOT, SimpleNamespace(mode="check"), RuntimeEventLogger("test")) == 0
    )
    assert len(checks) == 1


def test_cli_access_failure_has_saved_log_and_nonzero_status_without_traceback(
    tmp_path, monkeypatch, capsys
):
    module = load_command()
    monkeypatch.setattr(module, "find_repo_root", lambda _: tmp_path)
    monkeypatch.setattr("sys.argv", ["evaluate_oracle_reader.py", "--mode", "check"])

    def denied(*args):
        raise EvaluationAccessError("Finish Hugging Face login")

    monkeypatch.setattr(module, "run_evaluation", denied)
    assert module.main() == 2
    output = capsys.readouterr().out
    assert "EVALUATION_ACTION_REQUIRED" in output
    assert "Traceback" not in output
    paths = list((tmp_path / "artifacts/semantic_judge/runtime").glob("*/events.jsonl"))
    assert len(paths) == 1
    events = [json.loads(line) for line in paths[0].read_text().splitlines()]
    assert events[-1]["event"] == "evaluation.action_required"
    assert all("timestamp_utc" in row and "elapsed_seconds" in row for row in events)


def test_cli_interrupt_preserves_a_distinct_attempt_log(tmp_path, monkeypatch):
    module = load_command()
    monkeypatch.setattr(module, "find_repo_root", lambda _: tmp_path)
    monkeypatch.setattr("sys.argv", ["evaluate_oracle_reader.py", "--mode", "check"])

    def interrupted(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(module, "run_evaluation", interrupted)
    assert module.main() == 130
    assert module.main() == 130
    assert len(list((tmp_path / "artifacts/semantic_judge/runtime").glob("*/events.jsonl"))) == 2


@pytest.mark.parametrize(
    "mode,limit,used,expected_gib",
    [
        ("v2", "max", "0", 12),
        ("v2", str(4 * 1024**3), str(1024**3), 3),
        ("v1", str(16 * 1024**3), str(6 * 1024**3), 10),
        ("v2", str(4 * 1024**3), str(5 * 1024**3), 0),
    ],
)
def test_memory_check_respects_host_and_container_limits(tmp_path, mode, limit, used, expected_gib):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(f"MemTotal: {16 * 1024**2} kB\nMemAvailable: {12 * 1024**2} kB\n")
    group = tmp_path / "cgroup"
    group.mkdir()
    if mode == "v2":
        (group / "memory.max").write_text(limit)
        (group / "memory.current").write_text(used)
    else:
        (group / "memory").mkdir()
        (group / "memory/memory.limit_in_bytes").write_text(limit)
        (group / "memory/memory.usage_in_bytes").write_text(used)
    assert available_memory_bytes(meminfo, group) == expected_gib * 1024**3


def test_missing_memory_measurement_cannot_pass_readiness(tmp_path):
    with pytest.raises(EvaluationAccessError, match="Cannot verify available memory"):
        available_memory_bytes(tmp_path / "absent", tmp_path)


def test_memory_check_accepts_exact_threshold():
    result = check_cpu_memory(RuntimeEventLogger("test"), available=lambda: 8 * 1024**3)
    assert result["available_memory_gib"] == 8


def test_small_host_stops_before_model_loading(tmp_path, monkeypatch):
    from transformers import AutoTokenizer

    def forbidden(*args, **kwargs):
        pytest.fail("Insufficient memory must be detected before model download/loading")

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", forbidden)
    monkeypatch.setattr(
        "lava.evaluation.semantic.check_cpu_memory",
        lambda logger: check_cpu_memory(logger, available=lambda: 3 * 1024**3),
    )
    runtime = GemmaDecision(judge_contract(ROOT)["config"], RuntimeEventLogger("test"), tmp_path)
    with pytest.raises(EvaluationAccessError, match="ml.m7i.2xlarge"):
        runtime("a", "b", "ja")
