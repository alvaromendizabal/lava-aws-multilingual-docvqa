from __future__ import annotations

import re
from pathlib import Path

_FORBIDDEN = re.compile(
    r"(repair|repaired|fixed)",
    flags=re.IGNORECASE,
)

_PROJECT_AREAS = (
    "src",
    "scripts",
    "pipelines",
    "configs",
    "tests",
)


def test_source_tree_has_only_canonical_filenames() -> None:
    """Fixes replace canonical files instead of creating suffixed copies."""
    root = Path(__file__).resolve().parents[2]
    offenders: list[str] = []

    for area in _PROJECT_AREAS:
        area_path = root / area

        if not area_path.exists():
            continue

        for path in area_path.rglob("*"):
            if path.is_file() and _FORBIDDEN.search(path.name):
                offenders.append(str(path.relative_to(root)))

    assert not offenders, f"Forbidden repair/fixed filename variants found: {sorted(offenders)}"
