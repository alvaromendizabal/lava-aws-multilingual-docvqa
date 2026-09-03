#!/usr/bin/env bash
set -Eeuo pipefail

REPO="${LAVA_REPO:-$HOME/lava-aws-multilingual-docvqa}"
cd "$REPO"
START_EPOCH="$(date +%s)"

log() {
  local level="$1" event="$2"; shift 2
  local now elapsed
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  elapsed="$(( $(date +%s) - START_EPOCH ))"
  printf '[%s] [%s] [phase5c.preflight] %s total_elapsed_seconds=%s %s\n' \
    "$now" "$level" "$event" "$elapsed" "$*"
}

run_stage() {
  local name="$1"; shift
  local stage_start heartbeat_pid rc
  stage_start="$(date +%s)"
  log INFO "$name.started"
  (
    while sleep 15; do
      printf '[%s] [INFO] [phase5c.preflight] %s.heartbeat stage_elapsed_seconds=%s total_elapsed_seconds=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$name" \
        "$(( $(date +%s) - stage_start ))" "$(( $(date +%s) - START_EPOCH ))"
    done
  ) &
  heartbeat_pid=$!
  set +e
  "$@"
  rc=$?
  set -e
  kill "$heartbeat_pid" 2>/dev/null || true
  wait "$heartbeat_pid" 2>/dev/null || true
  if [[ $rc -ne 0 ]]; then
    log ERROR "$name.failed" "exit_code=$rc stage_elapsed_seconds=$(( $(date +%s) - stage_start ))"
    return "$rc"
  fi
  log INFO "$name.completed" "stage_elapsed_seconds=$(( $(date +%s) - stage_start ))"
}

log INFO start "repository=$REPO"
run_stage code_gate uv run python scripts/phase5c_preflight.py --code-only
run_stage ruff uv run ruff check src tests scripts pipelines notebooks/*.py
run_stage pytest uv run pytest -q
run_stage mypy uv run mypy --ignore-missing-imports \
  src/lava/readers/structured_output.py \
  src/lava/readers/private_artifacts.py \
  src/lava/readers/runtime_logging.py \
  src/lava/readers/artifact_gate.py
run_stage compile uv run python -m compileall -q src tests scripts pipelines
run_stage notebooks uv run python scripts/validate_public_notebooks.py
run_stage phase5b bash scripts/phase5b_preflight.sh
run_stage git_diff git diff --check

if [[ -n "$(git status --porcelain)" ]]; then
  log ERROR working_tree.not_clean
  git status --short
  exit 1
fi

log INFO completed
printf 'PHASE_5C_PREFLIGHT_VERIFIED\n'
printf 'NO_PAID_SAGEMAKER_RESOURCE_WAS_CREATED\n'
