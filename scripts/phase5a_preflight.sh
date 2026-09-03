#!/usr/bin/env bash
set -Eeuo pipefail

REPO="$HOME/lava-aws-multilingual-docvqa"
cd "$REPO"

[[ "$(git branch --show-current)" == "feat/oracle-reader-benchmark" ]] || {
    echo "ERROR: expected branch feat/oracle-reader-benchmark" >&2
    exit 1
}
[[ -f .env ]] || { echo "ERROR: .env is missing" >&2; exit 1; }
[[ -f configs/evaluation_protocol.lock.json ]] || {
    echo "ERROR: frozen evaluation lock is missing" >&2
    exit 1
}
set -a
source .env
set +a

uv run ruff check src tests scripts pipelines
uv run pytest -q
uv run python scripts/validate_sagemaker_contract.py
uv run python scripts/check_sagemaker_quota.py --instance-type ml.g5.2xlarge
uv run python scripts/resolve_oracle_model_revisions.py
uv run python scripts/prepare_oracle_assets.py
uv run python scripts/validate_phase5a.py
uv run python scripts/launch_oracle_reader.py     --model-key qwen35_4b_fused_direct     --limit 1
uv run ruff check src tests scripts pipelines
uv run pytest -q
uv run python -m compileall -q src tests scripts pipelines
git diff --check

echo "PHASE_5A_ORACLE_READER_PREFLIGHT_COMPLETE"
echo "NO_PAID_SAGEMAKER_JOB_WAS_SUBMITTED"
