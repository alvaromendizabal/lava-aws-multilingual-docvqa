"""The dependency-free example exercises real public contracts on authored data."""

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples/run_demo.py"


def test_demo_runs_without_site_packages_and_rejects_bad_citation(tmp_path):
    result = subprocess.run(
        [sys.executable, "-S", "-B", str(SCRIPT)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    report = json.loads(result.stdout)
    assert [(row["answer"], row["evidence_pages"]) for row in report["accepted"]] == [
        ("1200", [2]),
        ("4 giờ", [1]),
    ]
    assert report["submission_contract"]["row_count"] == 2
    assert report["submission_contract"]["schema_valid"] is True
    assert report["submission_contract"]["model_quality_verified"] is False
    assert report["submission_contract"]["uploaded_to_kaggle"] is False
    assert report["rejected"]["page"] == 3
    assert "page range" in report["rejected"]["reason"]
    assert report["semantic_limit"]["schema_valid"] is True
    assert report["semantic_limit"]["matches_authored_fixture"] is False
    assert report["model_inference_performed"] is False
    assert report["confidence_is_calibrated"] is False
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("change", ["incorrect_answer", "missing_evidence", "absent_page"])
def test_demo_fails_when_fixed_response_is_corrupted(tmp_path, change):
    module = runpy.run_path(str(SCRIPT))
    fixture = json.loads((ROOT / "examples/fixture.json").read_text())
    response = fixture["questions"][0]["provided_response"]
    if change == "incorrect_answer":
        response["answer"] = "9000"
    elif change == "missing_evidence":
        response["evidence_pages"] = []
    else:
        response["evidence_pages"] = [3]
    path = tmp_path / "corrupt_fixture.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")
    with pytest.raises(ValueError):
        module["run_demo"](path)
