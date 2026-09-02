#!/usr/bin/env bash
set -Eeuo pipefail

REPO="$HOME/lava-aws-multilingual-docvqa"
cd "$REPO"

[[ "$(git branch --show-current)" == "feat/evaluation-protocol" ]] || {
    echo "ERROR: current branch must be feat/evaluation-protocol" >&2
    exit 1
}

[[ -z "$(git status --porcelain)" ]] || {
    echo "ERROR: commit the Phase 4 code before freezing the protocol" >&2
    git status --short
    exit 1
}

[[ -f .env ]] || {
    echo "ERROR: .env is missing" >&2
    exit 1
}

uv run ruff check src tests
uv run pytest -q

set -a
source .env
set +a

uv run python -m lava.evaluation.protocol \
    --bucket "$S3_BUCKET" \
    --region "$AWS_REGION"

uv run python - <<'PY'
import json
from pathlib import Path

summary = json.loads(
    Path("reports/evaluation/evaluation_protocol_summary.json").read_text()
)
receipts = json.loads(
    Path("artifacts/evaluation_protocol/s3_upload_receipts.json").read_text()
)
assert summary["question_count"] == 16
assert summary["document_count"] == 5
assert summary["outer_fold_count"] == 5
assert summary["inner_fold_count_total"] == 20
assert len(summary["folds"]) == 5
assert all(fold["inner_fold_count"] == 4 for fold in summary["folds"])
assert sum(summary["language_counts"].values()) == 16
assert sum(summary["answer_format_counts"].values()) == 16
assert len(receipts) == 6
assert summary["judge_specification_boundary"]["status"] == (
    "official-structure-compatible-not-server-identical"
)
print("EVALUATION_PROTOCOL_VERIFIED")
print(f"PROTOCOL_LOCK_ID={summary['protocol_lock_id']}")
PY

uv run ruff check src tests
uv run pytest -q
git diff --check

echo "PHASE_4_EVALUATION_PROTOCOL_FROZEN"
git status --short
