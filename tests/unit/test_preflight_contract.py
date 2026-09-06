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
