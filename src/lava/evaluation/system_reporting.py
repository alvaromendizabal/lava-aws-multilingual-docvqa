"""Verify public figure lineage before including it in canonical notebooks."""

import hashlib
import json
from pathlib import Path


def verified_system_figure(root: Path, name: str) -> str:
    if name not in {"quality.svg", "documents.svg", "questions.svg"}:
        raise ValueError("Unknown published system figure")
    folder = root / "reports/system"
    manifest = json.loads((folder / "figures.json").read_bytes())
    if manifest["summary_filename"] not in {"summary.json", "refinement.json"}:
        raise ValueError("Unknown figure summary source")
    for path, expected in (
        (folder / manifest["summary_filename"], manifest["summary_sha256"]),
        (folder / "summary.json", manifest["first_pass_summary_sha256"]),
        (root / "scripts/report_system.py", manifest["source_sha256"]),
        (folder / name, manifest["files"][name]),
    ):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("System figure is stale or corrupt")
    return (folder / name).read_text()
