"""Validate public notebook hygiene, pairing, and safe presentation."""

from __future__ import annotations

import re
from pathlib import Path

import nbformat

_ACCOUNT_ID = re.compile(r"(?<!\d)\d{12}(?!\d)")
_EXPECTED = (
    "00_reproducibility_and_protocol",
    "01_oracle_reader_benchmark_design",
    "02_verified_gpu_execution",
    "03_model_scaling_and_cost",
)


def main() -> int:
    """Fail when a public notebook stores outputs, secrets, or loses its source pair."""
    root = Path(__file__).resolve().parents[1]
    notebook_dir = root / "notebooks"

    observed_ipynb = {path.stem for path in notebook_dir.glob("*.ipynb")}
    observed_py = {path.stem for path in notebook_dir.glob("*.py")}
    expected = set(_EXPECTED)

    if observed_ipynb != expected:
        raise RuntimeError(
            "Unexpected public notebook set: "
            f"expected={sorted(expected)!r}, observed={sorted(observed_ipynb)!r}"
        )
    if observed_py != expected:
        raise RuntimeError(
            "Unexpected paired Jupytext source set: "
            f"expected={sorted(expected)!r}, observed={sorted(observed_py)!r}"
        )

    for stem in _EXPECTED:
        notebook_path = notebook_dir / f"{stem}.ipynb"
        source_path = notebook_dir / f"{stem}.py"

        notebook = nbformat.read(notebook_path, as_version=4)
        kernelspec = notebook.metadata.get("kernelspec", {})
        if kernelspec.get("name") != "lava":
            raise RuntimeError(f"Notebook does not use the canonical lava kernel: {notebook_path}")

        serialized = nbformat.writes(notebook)
        if _ACCOUNT_ID.search(serialized):
            raise RuntimeError(f"Notebook contains a 12-digit account identifier: {notebook_path}")

        source_text = source_path.read_text(encoding="utf-8")
        if _ACCOUNT_ID.search(source_text):
            raise RuntimeError(
                f"Notebook source contains a 12-digit account identifier: {source_path}"
            )

        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            if cell.get("outputs", []):
                raise RuntimeError(f"Notebook contains stored outputs: {notebook_path}")
            if cell.get("execution_count") is not None:
                raise RuntimeError(f"Notebook contains execution counts: {notebook_path}")

    print("PUBLIC_NOTEBOOK_HYGIENE_VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
