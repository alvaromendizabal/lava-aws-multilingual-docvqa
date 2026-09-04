#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${LAVA_REPO_ROOT:-$HOME/lava-aws-multilingual-docvqa}"
MODE="${1:-preview}"
cd "$ROOT"

if [[ ! -f .env ]]; then
  printf 'ERROR: %s/.env does not exist.\n' "$ROOT" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

case "$MODE" in
  monitor)
    exec uv run python scripts/monitor_oracle_reader_job.py
    ;;
  stop)
    if [[ "${LAVA_CONFIRM_STOP:-NO}" != "YES" ]]; then
      printf '%s\n' \
        'ERROR: stop is locked.' \
        'Run exactly:' \
        'LAVA_CONFIRM_STOP=YES bash scripts/run_qwen38_smoke.sh stop' >&2
      exit 2
    fi
    exec uv run python scripts/stop_oracle_reader_job.py --confirm YES
    ;;
  preview|submit)
    ;;
  *)
    printf 'Usage: bash scripts/run_qwen38_smoke.sh [preview|submit|monitor|stop]\n' >&2
    exit 2
    ;;
esac

COMMON=(
  uv run python scripts/run_oracle_reader_smoke.py
  --model-key qwen38_27b_fused_direct
  --limit 1
  --hourly-usd-ceiling "${LAVA_HOURLY_USD_CEILING:-10}"
  --maximum-total-usd "${LAVA_MAXIMUM_TOTAL_USD:-12.5}"
  --poll-seconds "${LAVA_POLL_SECONDS:-15}"
  --heartbeat-seconds "${LAVA_HEARTBEAT_SECONDS:-30}"
)

if [[ "$MODE" == "preview" ]]; then
  exec "${COMMON[@]}"
fi

if [[ "${LAVA_ACKNOWLEDGE_CHARGES:-NO}" != "YES" ]]; then
  printf '%s\n' \
    'ERROR: paid submission is locked.' \
    'Run exactly:' \
    'LAVA_ACKNOWLEDGE_CHARGES=YES bash scripts/run_qwen38_smoke.sh submit' >&2
  exit 2
fi

# Deliberately omit a hardware override. The frozen model contract selects
# ml.g7e.12xlarge, the smallest pinned single-GPU-memory-compatible target.
exec "${COMMON[@]}" --submit --wait --acknowledge-charges YES
