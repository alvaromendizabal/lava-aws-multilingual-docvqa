"""Checksum-bound, resumable persistence for executed analysis notebooks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import nbformat

from lava.evaluation.submission_store import atomic_write


def analysis_input_digest(root: Path) -> str:
    """Hash public analysis inputs and implementation, excluding runtime artifacts."""
    paths = {root / name for name in ("uv.lock", "pyproject.toml")}
    for directory in ("src", "scripts", "configs", "reports"):
        paths.update(
            path
            for path in (root / directory).rglob("*")
            if path.is_file()
            and not path.is_relative_to(root / "reports/notebooks")
            and path.suffix in {".py", ".json", ".yaml", ".sha256"}
        )
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def execute_and_save(
    source: Path,
    output_dir: Path,
    *,
    input_digest: str,
    code_revision: str,
    execute: Callable[[], nbformat.NotebookNode],
) -> dict[str, Any]:
    """Reuse only a complete matching notebook; never overwrite changed evidence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / source.name
    manifest_path = output_dir / f"{source.stem}.manifest.json"
    if source.resolve() == target.resolve():
        raise ValueError("Executed output must not replace canonical notebook source")
    contract = {
        "schema_version": 1,
        "notebook": source.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "analysis_inputs_sha256": input_digest,
        "code_revision": code_revision,
    }
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_bytes())
        if any(saved.get(key) != value for key, value in contract.items()):
            raise ValueError("Notebook inputs changed; choose a new output directory")
        if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != saved.get(
            "output_sha256"
        ):
            raise ValueError("Saved notebook failed checksum verification")
        return {**saved, "reused": True}
    # A file without its commit manifest can be an interrupted write; do not trust it.
    notebook = execute()
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if any(cell.execution_count is None for cell in code_cells) or any(
        output.output_type == "error" for cell in code_cells for output in cell.outputs
    ):
        raise ValueError("Only fully executed, successful notebooks can be committed")
    payload = nbformat.writes(notebook).encode()
    saved = {
        **contract,
        "output_sha256": hashlib.sha256(payload).hexdigest(),
        "executed_code_cells": len(code_cells),
    }
    atomic_write(target, payload)
    # Publish the manifest last: this is the completion record used by resume.
    atomic_write(manifest_path, (json.dumps(saved, indent=2, sort_keys=True) + "\n").encode())
    return {**saved, "reused": False}
