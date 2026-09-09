#!/usr/bin/env bash
# Additional inference is explicitly authorized and limited by the SageMaker request.
set -Eeuo pipefail
cd /opt/ml/code
START=$(date +%s)
log() {
  printf '[%s] %s total_elapsed_seconds=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*" "$(($(date +%s)-START))"
}
log 'RECOVERY_DEPENDENCIES_STARTED'
python -m pip install --disable-pip-version-check --no-input --only-binary=tesserocr \
  -r pipelines/oracle_reader/requirements-gpu.txt tesserocr==2.9.2 &
PID=$!
(while kill -0 "$PID" 2>/dev/null; do sleep 15; log 'RECOVERY_DEPENDENCIES_HEARTBEAT'; done) &
HEARTBEAT=$!
trap 'kill "$HEARTBEAT" 2>/dev/null || true' EXIT
STATUS=0
wait "$PID" || STATUS=$?
kill "$HEARTBEAT" 2>/dev/null || true
wait "$HEARTBEAT" 2>/dev/null || true
if [[ "$STATUS" != 0 ]]; then log 'RECOVERY_DEPENDENCIES_FAILED'; exit "$STATUS"; fi
log 'RECOVERY_DEPENDENCIES_COMPLETED'
export OMP_THREAD_LIMIT=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/opt/ml/code:/opt/ml/code/src${PYTHONPATH:+:$PYTHONPATH}"
exec python -u -m pipelines.submission.recovery
