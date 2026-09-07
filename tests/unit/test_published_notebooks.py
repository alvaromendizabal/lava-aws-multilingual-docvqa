"""Published notebooks have verified outputs and an explicit project position."""

from pathlib import Path

import pytest

from lava.notebook_execution import NOTEBOOK_STEMS, validate_public_notebook

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("stem", NOTEBOOK_STEMS)
def test_canonical_publication_matches_current_sources_and_verified_inputs(stem):
    path = ROOT / "notebooks" / f"{stem}.ipynb"
    validate_public_notebook(ROOT, path)
    expected = {
        "00": "Data audit complete",
        "01": "Three reader pilots scored",
        "02": "Full pilot scored",
        "03": "Full pilot scored",
        "04": "Full-document retrieval evaluated",
        "05": "Current end-to-end status",
    }[stem[:2]]
    assert expected in path.read_text()
    assert '"contract_current": false' not in path.read_text()
