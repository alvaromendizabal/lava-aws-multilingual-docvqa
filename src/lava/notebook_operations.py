"""Explicit notebook actions and verified local downloads; no competition upload."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from contextlib import chdir
from pathlib import Path
from typing import Any

from IPython.display import FileLink

from lava.evaluation.semantic import digest
from lava.readers.runtime_logging import RuntimeEventLogger


def run_operation(
    root: Path,
    operation: str,
    *,
    enabled: bool = False,
    acknowledge_charges: str = "NO",
    attempt: int = 1,
    retry: bool = False,
    maximum_training_usd: float = 5.0,
) -> dict[str, Any]:
    """Run user-selected operations; reporting kernels always stay offline.

    The pilot action deliberately does not invoke notebook publication from
    within its own running notebook. Cloud work is still checkpointed by the
    canonical commands, independent of the browser's lifetime.
    """
    if operation not in {"pilot", "test", "export"}:
        raise ValueError("Unknown notebook operation")
    if type(enabled) is not bool or type(retry) is not bool:
        raise ValueError("Notebook run controls must be booleans")
    if type(attempt) is not int or not 1 <= attempt <= 99:
        raise ValueError("Attempt must be an integer from 1 to 99")
    logger = RuntimeEventLogger("notebook.operation")
    if os.environ.get("LAVA_NOTEBOOK_PUBLICATION") == "1" or not enabled:
        logger.emit("notebook.operation.disabled", operation=operation, cloud_access=False)
        return {"operation": operation, "status": "disabled", "uploaded_to_kaggle": False}
    if operation in {"pilot", "test"} and acknowledge_charges != "YES":
        raise ValueError("Set ACKNOWLEDGE_AWS_CHARGES = 'YES' to authorize a new GPU attempt")
    if isinstance(maximum_training_usd, bool) or not math.isfinite(maximum_training_usd) or maximum_training_usd <= 0:
        raise ValueError("Training estimate limit must be finite and positive")
    command = [sys.executable, "-u"]
    if operation == "pilot":
        command += ["scripts/evaluate_system.py", "--mode", "notebook"]
    else:
        command += ["scripts/prepare_submission.py", "--mode", operation]
    if operation in {"pilot", "test"}:
        command += ["--acknowledge-charges", acknowledge_charges, "--attempt", str(attempt),
                    "--retry", "YES" if retry else "NO", "--maximum-training-usd", str(maximum_training_usd)]
    with logger.stage("notebook.operation", heartbeat_seconds=15):
        with subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as process:
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                status = process.wait()
            except BaseException:
                # Stop the local child monitor, not an accepted managed AWS job.
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        if status:
            raise subprocess.CalledProcessError(status, command)
    return {"operation": operation, "status": "completed", "uploaded_to_kaggle": False}


def download_link(root: Path) -> str:
    """Link exact verified bytes without embedding private answers in notebook outputs."""
    import json

    if os.environ.get("LAVA_NOTEBOOK_PUBLICATION") == "1":
        return "Downloads are available only in your interactive workspace, after export."
    target = root / "artifacts/submission/submission.csv"
    manifest_path = target.with_name("manifest.json")
    if not target.is_file() or not manifest_path.is_file():
        return "No verified submission.csv exists yet. Generate test predictions, then export."
    manifest = json.loads(manifest_path.read_bytes())
    expected_count = json.loads((root / "configs/submission.json").read_bytes())["expected_question_count"]
    validation = manifest.get("validation", {})
    if digest(target.read_bytes()) != manifest.get("submission_sha256") or (
        validation.get("schema_valid") is not True or validation.get("row_count") != expected_count
    ):
        raise ValueError("The local download does not match its validated submission manifest")
    # FileLink checks local existence at rendering time; links are relative to
    # notebooks/, not the kernel's arbitrary launch directory. No data URI.
    with chdir(root / "notebooks"):
        return str(FileLink("../artifacts/submission/submission.csv", result_html_prefix="Download your generated CSV: ")._repr_html_())
