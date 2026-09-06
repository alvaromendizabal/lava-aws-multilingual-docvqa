from __future__ import annotations

from pathlib import Path


def test_preflight_script_exists_and_is_generic() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "scripts/preflight.py").read_text(encoding="utf-8")
    assert "--model-key" in text
    assert "qwen35_9b_fused_direct" in text
    assert "ORACLE_READER_PREFLIGHT_VERIFIED" in text
    assert "NO_PAID_SAGEMAKER_RESOURCE_WAS_CREATED" in text
    assert "phase5b" not in text.lower()


def test_full_pilot_preflight_runs_read_only_checks(monkeypatch, capsys):
    """Exercise the real command: full coverage passes without any resource creation."""
    import runpy
    from unittest.mock import MagicMock

    root = Path(__file__).resolve().parents[2]
    module = runpy.run_path(str(root / "scripts/preflight.py"))
    command = module["main"]
    globals_ = command.__globals__
    monkeypatch.setenv("S3_BUCKET", "test-bucket")
    monkeypatch.setattr(
        "sys.argv",
        ["preflight.py", "--mode", "benchmark", "--model-key", "qwen38_27b_nf4_g5_fused_direct"],
    )
    monkeypatch.setitem(globals_, "validate_sagemaker_sdk_contract", lambda _: None)
    monkeypatch.setitem(globals_, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setitem(globals_, "list_training_job_names", lambda **k: [])
    monkeypatch.setitem(globals_, "verify_training_quota", lambda **k: {"value": 1})
    session = MagicMock()
    clients = {name: MagicMock() for name in ("s3", "sts", "sagemaker", "service-quotas")}
    clients["sts"].get_caller_identity.return_value = {"Arn": "test-role"}
    session.client.side_effect = clients.__getitem__
    monkeypatch.setattr(globals_["boto3"].session, "Session", lambda **k: session)
    assert command() == 0
    output = capsys.readouterr().out
    assert '"mode": "benchmark"' in output
    assert '"limit": 16' in output
    assert "ORACLE_READER_PREFLIGHT_VERIFIED" in output
    assert "NO_PAID_SAGEMAKER_RESOURCE_WAS_CREATED" in output
    clients["sagemaker"].create_training_job.assert_not_called()
    clients["s3"].put_object.assert_not_called()
    assert clients["s3"].head_object.call_count == 2
