"""Verify public figure lineage before including it in canonical notebooks."""

import hashlib
import json
from pathlib import Path


def verified_system_figure(root: Path, name: str) -> str:
    if name not in {"quality.svg", "documents.svg", "questions.svg"}:
        raise ValueError("Unknown published system figure")
    folder = root / "reports/system"
    manifest = json.loads((folder / "figures.json").read_bytes())
    for path, expected in (
        (folder / "summary.json", manifest["summary_sha256"]),
        (root / "scripts/report_system.py", manifest["source_sha256"]),
        (folder / name, manifest["files"][name]),
    ):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("System figure is stale or corrupt")
    return (folder / name).read_text()
