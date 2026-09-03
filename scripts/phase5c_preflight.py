"""No-cost code and configuration gate for the structured-output smoke."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from lava.notebook_support import find_repo_root

_EXPECTED_BRANCH = "feat/oracle-reader-benchmark"
_EXPECTED_MODEL_KEY = "qwen35_4b_fused_direct"


def _assert_all_python_calls_have_keyword(path: Path, attribute: str, keyword: str) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attribute
    ]
    if not calls:
        raise RuntimeError(f"{path} must call {attribute}(...)")
    missing = [
        node.lineno for node in calls if not any(item.arg == keyword for item in node.keywords)
    ]
    if missing:
        raise RuntimeError(
            f"Every {attribute}(...) call in {path} must include {keyword}=...; "
            f"missing at lines {missing}"
        )
    return len(calls)


def _require_text(path: Path, required: str, forbidden: str | None = None) -> None:
    text = path.read_text(encoding="utf-8")
    if required not in text:
        raise RuntimeError(f"{path} is missing required marker: {required}")
    if forbidden and forbidden in text:
        raise RuntimeError(f"{path} contains forbidden legacy marker: {forbidden}")


def validate_code(root: Path) -> dict[str, object]:
    """Validate source-level invariants without contacting AWS."""
    qwen = root / "src/lava/readers/qwen35.py"
    prompts = root / "src/lava/readers/prompts.py"
    runner = root / "scripts/run_oracle_reader_smoke.py"
    requirements = root / "pipelines/oracle_reader/requirements-gpu.txt"
    entry = root / "pipelines/oracle_reader/job_entry.py"

    template_call_count = _assert_all_python_calls_have_keyword(
        qwen, "apply_chat_template", "enable_thinking"
    )
    _require_text(qwen, "append_strict_json_instruction", forbidden="chat_template_kwargs")
    _require_text(qwen, "persist_raw_response")
    _require_text(prompts, "oracle-reader-json-v3", forbidden="oracle-reader-json-v2")
    _require_text(requirements, "cachetools==6.2.4")
    _require_text(runner, "verify_training_model_artifact")
    _require_text(runner, "ORACLE_READER_ONE_QUESTION_SMOKE_VERIFIED")
    _require_text(entry, "RuntimeEventLogger")
    _require_text(entry, "heartbeat_seconds=15.0")

    return {
        "branch_required": _EXPECTED_BRANCH,
        "model_key": _EXPECTED_MODEL_KEY,
        "prompt_version": "oracle-reader-json-v3",
        "invalid_chat_template_kwargs_absent": True,
        "direct_enable_thinking_keyword_present": True,
        "apply_chat_template_call_count": template_call_count,
        "private_raw_response_persistence_present": True,
        "artifact_success_gate_present": True,
        "gpu_cachetools_pin": "6.2.4",
        "paid_resource_created": False,
    }


def main() -> int:
    """Print a machine-readable code gate; this command never submits AWS work."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-only", action="store_true")
    parser.parse_args()
    root = find_repo_root(Path(__file__).resolve())
    result = validate_code(root)
    print(json.dumps(result, indent=2, sort_keys=True))
    print("PHASE_5C_CODE_GATE_VERIFIED")
    print("NO_PAID_SAGEMAKER_RESOURCE_WAS_CREATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
