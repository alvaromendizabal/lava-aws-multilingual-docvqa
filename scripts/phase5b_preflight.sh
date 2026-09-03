#!/usr/bin/env bash
set -Eeuo pipefail

START_EPOCH="$(date +%s)"
log() {
  local level="$1"
  shift
  printf '[%s] [%s] [phase5b.preflight] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$level" "$*"
}
finish() {
  local status="$?"
  local elapsed="$(( $(date +%s) - START_EPOCH ))"
  if [[ "$status" -eq 0 ]]; then
    log INFO "completed total_elapsed_seconds=${elapsed}"
  else
    log ERROR "failed exit_code=${status} total_elapsed_seconds=${elapsed}"
  fi
  exit "$status"
}
trap finish EXIT

run_step() {
  local name="$1"
  shift
  local step_start
  step_start="$(date +%s)"
  log INFO "${name}.started"
  "$@" &
  local pid="$!"
  local last_heartbeat="$step_start"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 1
    local now
    now="$(date +%s)"
    if kill -0 "$pid" 2>/dev/null && (( now - last_heartbeat >= 15 )); then
      log INFO "${name}.heartbeat elapsed_seconds=$(( now - step_start ))"
      last_heartbeat="$now"
    fi
  done
  local status=0
  wait "$pid" || status="$?"
  if [[ "$status" -ne 0 ]]; then
    log ERROR "${name}.failed exit_code=${status} elapsed_seconds=$(( $(date +%s) - step_start ))"
    return "$status"
  fi
  log INFO "${name}.completed elapsed_seconds=$(( $(date +%s) - step_start ))"
}

REPO="${LAVA_REPO_ROOT:-$HOME/lava-aws-multilingual-docvqa}"
cd "$REPO"

log INFO "start repository=$REPO branch=$(git branch --show-current)"
run_step "ruff" uv run ruff check src tests scripts pipelines notebooks/*.py
run_step "pytest" uv run pytest -q
run_step "compileall" uv run python -m compileall -q src tests scripts pipelines
run_step "notebook_hygiene" uv run python scripts/validate_public_notebooks.py
run_step "aws_model_data_preflight" uv run python scripts/phase5b_preflight.py
