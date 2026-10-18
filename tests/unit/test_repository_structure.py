"""The public entry points must resolve to the maintained final reference."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_project_entrance_resolves_local_links():
    text = (ROOT / "README.md").read_text()
    for target in re.findall(r"\]\(([^)]+)\)", text):
        if target.startswith(("http:", "https:", "#", "mailto:")):
            continue
        assert (ROOT / target.split("#", 1)[0]).exists(), target


def test_one_canonical_notebook_and_no_operator_controls():
    from lava.notebook_execution import NOTEBOOK_STEMS

    assert {p.stem for p in (ROOT / "notebooks").glob("*.ipynb")} == set(NOTEBOOK_STEMS)
    for stem in NOTEBOOK_STEMS:
        text = (ROOT / "notebooks" / f"{stem}.ipynb").read_text()
        assert "lava.notebook_operations" not in text
    assert not (ROOT / "src/lava/retrieval/domain_features.py").exists()
