from __future__ import annotations

import ast
from pathlib import Path


def test_qwen_source_has_direct_template_switch_and_no_legacy_kwarg() -> None:
    path = Path("src/lava/readers/qwen35.py")
    text = path.read_text(encoding="utf-8")
    assert "chat_template_kwargs" not in text
    assert "append_strict_json_instruction" in text
    assert "persist_raw_response" in text
    tree = ast.parse(text)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "apply_chat_template"
    ]
    assert calls
    assert all(any(item.arg == "enable_thinking" for item in call.keywords) for call in calls)


def test_prompt_version_is_three() -> None:
    text = Path("src/lava/readers/prompts.py").read_text(encoding="utf-8")
    assert "oracle-reader-json-v3" in text
    assert "oracle-reader-json-v2" not in text


def test_gpu_dependency_set_pins_compatible_cachetools() -> None:
    lines = (
        Path("pipelines/oracle_reader/requirements-gpu.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert "cachetools==6.2.4" in lines


def test_existing_smoke_script_uses_artifact_gate() -> None:
    text = Path("scripts/run_oracle_reader_smoke.py").read_text(encoding="utf-8")
    assert "verify_training_model_artifact" in text
    assert "ORACLE_READER_ONE_QUESTION_SMOKE_VERIFIED" in text


def test_existing_prompt_regression_tracks_prompt_v3() -> None:
    text = Path("tests/unit/test_reader_prompts.py").read_text(encoding="utf-8")
    assert 'PROMPT_VERSION == "oracle-reader-json-v3"' in text
    assert "oracle-reader-json-v2" not in text
