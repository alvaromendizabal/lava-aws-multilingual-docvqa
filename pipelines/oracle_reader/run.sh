#!/usr/bin/env bash
# The service entrypoint verifies the complete Git archive before this script runs.
set -Eeuo pipefail
cd /opt/ml/code
START=$(date +%s)
log() {
  printf '[%s] %s total_elapsed_seconds=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*" "$(($(date +%s)-START))"
}
log 'DEPENDENCIES_STARTED'
python -m pip install --disable-pip-version-check --no-input -r pipelines/oracle_reader/requirements-gpu.txt &
PID=$!
(while kill -0 "$PID" 2>/dev/null; do sleep 15; log 'DEPENDENCIES_HEARTBEAT'; done) &
HEARTBEAT=$!
trap 'kill "$HEARTBEAT" 2>/dev/null || true' EXIT
STATUS=0
wait "$PID" || STATUS=$?
kill "$HEARTBEAT" 2>/dev/null || true
wait "$HEARTBEAT" 2>/dev/null || true
if [[ "$STATUS" != 0 ]]; then log 'DEPENDENCIES_FAILED'; exit "$STATUS"; fi
log 'DEPENDENCIES_COMPLETED'
exec python -u pipelines/oracle_reader/job_entry.py --mode system \
  --bucket "$LAVA_BUCKET" --region "$AWS_DEFAULT_REGION" --contract-id "$LAVA_SYSTEM_CONTRACT"
