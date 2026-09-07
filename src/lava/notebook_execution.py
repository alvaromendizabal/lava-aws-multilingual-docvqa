"""Checksum-bound, resumable persistence for executed analysis notebooks."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import nbformat

from lava.evaluation.submission_store import atomic_write

NOTEBOOK_STEMS = (
    "00_reproducibility_and_protocol",
    "01_oracle_reader_benchmark_design",
    "02_verified_gpu_execution",
    "03_model_scaling_and_cost",
    "04_evidence_retrieval",
    "05_end_to_end_system_evaluation",
)


def notebook_source_digest(path: Path) -> str:
    """Hash executable content independently of outputs and automatic kernel metadata."""
    return _source_digest(nbformat.read(path, 4))


def _source_digest(value: nbformat.NotebookNode) -> str:
    notebook = deepcopy(value)
    for key in ("language_info", "widgets", "jupytext"):
        notebook.metadata.pop(key, None)
    for cell in notebook.cells:
        for key in ("trusted", "execution"):
            cell.metadata.pop(key, None)
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []
    return hashlib.sha256(nbformat.writes(notebook).encode()).hexdigest()


def analysis_input_digest(root: Path) -> str:
    """Hash public analysis inputs and implementation, excluding runtime artifacts."""
    paths = {root / name for name in ("uv.lock", "pyproject.toml")}
    for directory in ("src", "scripts", "configs", "reports"):
        paths.update(
            path
            for path in (root / directory).rglob("*")
            if path.is_file()
            and not path.is_relative_to(root / "reports/notebook_execution")
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
        "schema_version": 2,
        "notebook": source.name,
        "source_sha256": notebook_source_digest(source),
        "analysis_inputs_sha256": input_digest,
        "code_revision": code_revision,
    }
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_bytes())
        if any(
            saved.get(key) != value for key, value in contract.items() if key != "code_revision"
        ):
            raise ValueError("Notebook inputs changed; choose a new output directory")
        if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != saved.get(
            "output_sha256"
        ):
            raise ValueError("Saved notebook failed checksum verification")
        return {**saved, "reused": True}
    # A file without its commit manifest can be an interrupted write; do not trust it.
    notebook = execute()
    if _source_digest(notebook) != contract["source_sha256"]:
        raise ValueError("Execution changed notebook source; publication rejected")
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


def publish_notebook(
    root: Path,
    source: Path,
    *,
    input_digest: str,
    code_revision: str,
    execute: Callable[[], nbformat.NotebookNode],
) -> dict[str, Any]:
    """Publish verified output at its canonical path, with resumable immutable staging.

    GitHub checkouts already contain valid publications, so they need no execution.
    When inputs change, staging completes before either canonical file is replaced.
    A crash between notebook and manifest writes recovers from the staging record.
    """
    if source.resolve().parent != (root / "notebooks").resolve():
        raise ValueError("Published notebooks must live directly in notebooks/")
    source_digest = notebook_source_digest(source)
    manifest_path = root / "reports/notebook_execution" / f"{source.stem}.manifest.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_bytes())
        if (
            saved.get("schema_version") == 2
            and saved.get("source_sha256") == source_digest
            and saved.get("analysis_inputs_sha256") == input_digest
            and saved.get("output_sha256") == hashlib.sha256(source.read_bytes()).hexdigest()
        ):
            return {**saved, "reused": True}
    staging = root / "artifacts/notebook_runs" / input_digest / source_digest
    saved = execute_and_save(
        source,
        staging,
        input_digest=input_digest,
        code_revision=code_revision,
        execute=execute,
    )
    atomic_write(source, (staging / source.name).read_bytes())
    manifest = {key: value for key, value in saved.items() if key != "reused"}
    atomic_write(manifest_path, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    return saved


def validate_public_notebook(root: Path, source: Path) -> dict[str, Any]:
    """Check canonical publication integrity, source identity, execution and privacy."""
    manifest = json.loads(
        (root / "reports/notebook_execution" / f"{source.stem}.manifest.json").read_bytes()
    )
    if manifest.get("schema_version") != 2:
        raise ValueError(f"Unsupported notebook manifest: {source.name}")
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["output_sha256"]:
        raise ValueError(f"Notebook output checksum mismatch: {source.name}")
    if notebook_source_digest(source) != manifest["source_sha256"]:
        raise ValueError(f"Notebook source checksum mismatch: {source.name}")
    if analysis_input_digest(root) != manifest["analysis_inputs_sha256"]:
        raise ValueError(f"Notebook analysis inputs changed: {source.name}")
    notebook = nbformat.read(source, 4)
    if notebook.metadata.get("kernelspec", {}).get("name") != "lava":
        raise ValueError(f"Notebook requires the canonical lava kernel: {source.name}")
    if "jupytext" in notebook.metadata:
        raise ValueError(f"Obsolete notebook pairing metadata: {source.name}")
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if len(code) != manifest["executed_code_cells"] or [
        cell.execution_count for cell in code
    ] != list(range(1, len(code) + 1)):
        raise ValueError(f"Incomplete notebook execution: {source.name}")
    if any(
        output.output_type == "error" or output.get("name") == "stderr"
        for cell in code
        for output in cell.outputs
    ):
        raise ValueError(f"Notebook contains failed execution or stderr: {source.name}")
    if re.search(r"arn:aws:|s3://|AKIA[A-Z0-9]{16}|hf_[A-Za-z0-9]{20,}", source.read_text()):
        raise ValueError(f"Notebook contains a private location or credential: {source.name}")
    return manifest
