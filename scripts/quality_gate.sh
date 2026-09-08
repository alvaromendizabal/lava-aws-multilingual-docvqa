#!/usr/bin/env bash
set -Eeuo pipefail

START_EPOCH=$(date +%s)
HEARTBEAT_SECONDS=${LAVA_QUALITY_GATE_HEARTBEAT_SECONDS:-15}

log() {
  printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

on_exit() {
  local status=$?
  local end_epoch elapsed
  end_epoch=$(date +%s)
  elapsed=$((end_epoch - START_EPOCH))
  if [[ $status -eq 0 ]]; then
    log "QUALITY_GATE_PASSED total_elapsed_seconds=${elapsed}"
  else
    log "QUALITY_GATE_FAILED status=${status} total_elapsed_seconds=${elapsed}"
  fi
}
trap on_exit EXIT

run_stage() {
  local name=$1
  shift

  local stage_start pid heartbeat_pid status stage_end elapsed
  stage_start=$(date +%s)
  log "START: ${name}"

  "$@" &
  pid=$!

  (
    while sleep "$HEARTBEAT_SECONDS"; do
      if ! kill -0 "$pid" 2>/dev/null; then
        break
      fi
      log "HEARTBEAT: ${name} elapsed_seconds=$(($(date +%s)-stage_start))"
    done
  ) &
  heartbeat_pid=$!

  if wait "$pid"; then
    status=0
  else
    status=$?
  fi

  kill "$heartbeat_pid" 2>/dev/null || true
  wait "$heartbeat_pid" 2>/dev/null || true

  stage_end=$(date +%s)
  elapsed=$((stage_end - stage_start))

  if [[ $status -ne 0 ]]; then
    log "FAILED: ${name} status=${status} elapsed_seconds=${elapsed}"
    return "$status"
  fi

  log "DONE: ${name} elapsed_seconds=${elapsed}"
}

if [[ ! -f pyproject.toml || ! -d src/lava ]]; then
  log "ERROR: run scripts/quality_gate.sh from the repository root."
  exit 2
fi

log "LAVA REPOSITORY QUALITY GATE"
log "repository=$(pwd)"
log "branch=$(git branch --show-current)"
log "commit=$(git rev-parse HEAD)"

run_stage "frozen environment" uv sync --frozen --group judge
run_stage "ruff format check" uv run --frozen --group judge ruff format --check src tests scripts pipelines notebooks
run_stage "ruff lint" uv run --frozen --group judge ruff check src tests scripts pipelines notebooks
run_stage "mypy" uv run --frozen --group judge mypy src scripts
run_stage "bash syntax" bash -n scripts/quality_gate.sh scripts/freeze_evaluation_protocol.sh pipelines/oracle_reader/run.sh pipelines/submission/run.sh pipelines/refinement/run.sh
run_stage "pytest" uv run --frozen --group judge pytest -q
run_stage "compileall" uv run --frozen --group judge python -m compileall -q src tests scripts pipelines
run_stage "notebook hygiene" uv run --frozen --group judge python scripts/validate_public_notebooks.py
run_stage "git diff check" git diff --check
