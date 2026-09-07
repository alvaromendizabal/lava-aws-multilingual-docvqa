"""Published notebook outputs are traceable, complete, and safe to read on GitHub."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import nbformat
import pytest

from lava.notebook_execution import analysis_input_digest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("stem", ["02_verified_gpu_execution", "03_model_scaling_and_cost"])
def test_published_snapshot_matches_current_sources_and_verified_inputs(stem):
    path = ROOT / "reports/notebooks" / f"{stem}.ipynb"
    manifest = json.loads(path.with_suffix(".manifest.json").read_text())
    source = ROOT / "notebooks" / path.name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["output_sha256"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["source_sha256"]
    assert manifest["analysis_inputs_sha256"] == analysis_input_digest(ROOT)
    notebook = nbformat.read(path, 4)
    canonical = nbformat.read(source, 4)
    assert [cell.source for cell in notebook.cells] == [cell.source for cell in canonical.cells]
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert len(code) == manifest["executed_code_cells"]
    assert [cell.execution_count for cell in code] == list(range(1, len(code) + 1))
    for cell in code:
        assert not any(output.output_type == "error" for output in cell.outputs)
        assert not any(output.get("name") == "stderr" for output in cell.outputs)
    serialized = path.read_text()
    assert not re.search(r"arn:aws:|s3://|AKIA[A-Z0-9]{16}|hf_[A-Za-z0-9]{20,}", serialized)
    assert '"contract_current": false' not in serialized
    assert "Full pilot scored" in serialized
    attribute = subprocess.check_output(
        ["git", "check-attr", "filter", "--", str(path.relative_to(ROOT))], cwd=ROOT, text=True
    )
    assert attribute.strip().endswith(": unset")
