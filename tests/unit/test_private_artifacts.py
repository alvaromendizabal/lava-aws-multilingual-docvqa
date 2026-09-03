from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

from lava.readers.private_artifacts import persist_raw_response


def test_raw_response_is_exact_atomic_and_private(tmp_path: Path) -> None:
    raw = '  {"answer":"日本語","evidence_pages":[1],"confidence":0.9,"abstain":false}\n'
    record = persist_raw_response(raw, question_id="q/01", root=tmp_path)
    response = Path(record.response_path)
    metadata = Path(record.metadata_path)
    assert response.read_text(encoding="utf-8") == raw
    assert stat.S_IMODE(response.stat().st_mode) == 0o600
    assert stat.S_IMODE(metadata.stat().st_mode) == 0o600
    assert record.sha256 == hashlib.sha256(raw.encode("utf-8")).hexdigest()
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    assert payload["sha256"] == record.sha256
    assert payload["byte_count"] == len(raw.encode("utf-8"))
