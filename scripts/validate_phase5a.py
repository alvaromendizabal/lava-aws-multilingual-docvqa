"""Strict local verification for the Phase 5A foundation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/oracle_reader_benchmark.yaml").read_text())
    protocol_lock = json.loads((root / "configs/evaluation_protocol.lock.json").read_text())
    assert config["protocol_lock_id"] == protocol_lock["protocol_lock_id"]
    assert config["training_runtime"]["instance_count"] == 1
    assert config["training_runtime"]["no_endpoint"] is True
    assert config["training_runtime"]["max_runtime_seconds"] <= 3600
    models = config["models"]
    assert models["qwen35_4b_fused_direct"]["instance_type"] == "ml.g5.2xlarge"
    assert all(
        model["instance_type"] == "ml.g6e.2xlarge"
        for model_key, model in models.items()
        if model_key != "qwen35_4b_fused_direct"
    )
    for notebook_path in sorted((root / "notebooks").glob("0[01]_*.ipynb")):
        notebook = nbformat.read(notebook_path, as_version=4)
        for cell in notebook.cells:
            assert not cell.get("outputs", [])
            assert cell.get("execution_count") is None
    if not args.code_only:
        model_lock = json.loads((root / "configs/oracle_reader_models.lock.json").read_text())
        assert model_lock["candidate_count"] == 5
        assert model_lock["unique_model_repository_count"] == 2
        locked_models = {row["model_key"]: row for row in model_lock["resolved_models"]}
        assert set(locked_models) == set(models)
        for model_key, model in models.items():
            assert locked_models[model_key]["model_id"] == model["model_id"]
            assert locked_models[model_key]["instance_type"] == model["instance_type"]
        summary = json.loads(
            (root / "reports/oracle_reader/oracle_assets_summary.json").read_text()
        )
        assert summary["protocol_lock_id"] == config["protocol_lock_id"]
        assert summary["question_count"] == 16
        assert summary["document_count"] == 5
        assert summary["unique_evidence_page_count"] >= 5
    print("PHASE_5A_CODE_VERIFIED" if args.code_only else "PHASE_5A_PREFLIGHT_VERIFIED")


if __name__ == "__main__":
    main()
