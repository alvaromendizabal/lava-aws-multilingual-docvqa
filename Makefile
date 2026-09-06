MODEL ?= qwen35_9b_fused_direct
LIMIT ?= 1
JOB ?=
CHARGES ?= NO
CONFIRM ?= NO

.PHONY: quality preflight preview submit monitor verify sync stop notebooks

quality:
	bash scripts/quality_gate.sh

preflight:
	uv run --frozen python scripts/preflight.py --model-key $(MODEL) --limit $(LIMIT)

preview:
	uv run --frozen python scripts/run_oracle_reader.py --model-key $(MODEL) --limit $(LIMIT)

submit:
	@test "$(CHARGES)" = "YES" || (echo "Refusing paid compute. Re-run with CHARGES=YES." >&2; exit 2)
	uv run --frozen python scripts/run_oracle_reader.py --model-key $(MODEL) --limit $(LIMIT) --submit --wait --acknowledge-charges YES

monitor:
	@test -n "$(JOB)" || (echo "JOB is required." >&2; exit 2)
	uv run --frozen python scripts/monitor_oracle_reader_job.py --job-name $(JOB)

verify:
	@test -n "$(JOB)" || (echo "JOB is required." >&2; exit 2)
	uv run --frozen python scripts/inspect_oracle_reader_artifact.py --job-name $(JOB)

sync:
	@test -n "$(JOB)" || (echo "JOB is required." >&2; exit 2)
	uv run --frozen python scripts/sync_oracle_reader_results.py --job-name $(JOB)

stop:
	@test -n "$(JOB)" || (echo "JOB is required." >&2; exit 2)
	@test "$(CONFIRM)" = "YES" || (echo "Refusing to stop compute. Re-run with CONFIRM=YES." >&2; exit 2)
	uv run --frozen python scripts/stop_oracle_reader_job.py --job-name $(JOB) --confirm YES

notebooks:
	uv run --frozen jupytext --sync notebooks/*.py

.PHONY: benchmark-preview benchmark-submit report
benchmark-preview:
	uv run --frozen python scripts/run_oracle_reader.py --mode benchmark --model-key $(MODEL)

benchmark-submit:
	@test "$(CHARGES)" = "YES" || (echo "Refusing paid compute. Re-run with CHARGES=YES." >&2; exit 2)
	uv run --frozen python scripts/run_oracle_reader.py --mode benchmark --model-key $(MODEL) --submit --wait --acknowledge-charges YES

report:
	uv run --frozen python scripts/report_oracle_reader.py

.PHONY: benchmark-resume-preview benchmark-resume
benchmark-resume-preview:
	@test -n "$(JOB)" || (echo "JOB is required." >&2; exit 2)
	uv run --frozen python scripts/run_oracle_reader.py --mode benchmark --model-key $(MODEL) --resume-job $(JOB)

benchmark-resume:
	@test -n "$(JOB)" || (echo "JOB is required." >&2; exit 2)
	@test "$(CHARGES)" = "YES" || (echo "Refusing paid compute. Re-run with CHARGES=YES." >&2; exit 2)
	uv run --frozen python scripts/run_oracle_reader.py --mode benchmark --model-key $(MODEL) --resume-job $(JOB) --submit --wait --acknowledge-charges YES
