# Phase 5B — Observable, charge-bounded oracle-reader GPU smoke

## Purpose

Phase 5A established immutable reader revisions, private oracle-evidence assets, and a
one-question SageMaker plan. Phase 5B makes the first paid execution observable,
reconnectable, privacy-aware, and fail-closed.

The first paid run is infrastructure validation, not model selection. It is restricted to:

- one question;
- Qwen3.5-4B fused image-plus-text direct decoding;
- one `ml.g6e.2xlarge` training instance;
- at most one hour;
- no managed endpoint;
- no parallel LAVA training job;
- no reuse of an already-populated immutable output prefix.

## Telemetry contract

Every local orchestration event includes:

- UTC timestamp;
- run identifier;
- component and event name;
- elapsed seconds;
- stage duration for terminal events;
- status changes and 30-second heartbeats;
- SageMaker training and billable seconds when AWS returns them.

Raw questions, reference answers, extracted page text, prompts, model responses, bucket
names, credential-like values, and 12-digit AWS account identifiers are redacted or
suppressed. Arbitrary CloudWatch lines are not surfaced by the public monitor; only
structured LAVA events are eligible for display.

## No-cost preflight

```bash
bash scripts/phase5b_preflight.sh
```

The preflight runs Ruff, all tests, source compilation, notebook hygiene, two headless
notebook executions, protocol/model/asset checks, private S3 object checks, active-job
checks, the one-question plan guard, an operator-supplied cost ceiling, and the SageMaker
training quota lookup. It creates no paid resource.

## Preview

```bash
uv run python scripts/run_oracle_reader_smoke.py
```

Preview is the default. The command prints a sanitized plan and exits with:

```text
SMOKE_PLAN_PREVIEWED
NO_PAID_SAGEMAKER_RESOURCE_WAS_CREATED
```

## Submit exactly one paid smoke job

Run only after the preflight confirms quota of at least one and Git is clean:

```bash
uv run python scripts/run_oracle_reader_smoke.py \
  --submit \
  --wait \
  --acknowledge-charges YES
```

The wrapper launches the already-tested Phase 5A SageMaker job, discovers its AWS job
name, writes atomic reconnect state, polls `DescribeTrainingJob`, emits status changes and
heartbeats, requests a stop if the bounded monitor times out, and reports total and AWS
billable time.

## Reconnect after a browser or terminal interruption

```bash
uv run python scripts/monitor_oracle_reader_job.py
```

The command reads `artifacts/oracle_reader/runtime/latest.json`. A specific job can also
be supplied:

```bash
uv run python scripts/monitor_oracle_reader_job.py --job-name <training-job-name>
```

## Result synchronization

After a completed job:

```bash
uv run python scripts/sync_oracle_reader_results.py
```

Only checksum-verified, sanitized aggregate results belong in GitHub. Private questions,
answers, page images, text, per-question traces, and model responses remain in versioned
private S3 paths.

## Scientific sequence after infrastructure validation

1. Repeatability check on the same deterministic one-question plan.
2. Full 16-question Qwen3.5-4B fused-direct oracle benchmark.
3. Image-only and text-only modality controls.
4. Bounded-thinking ablation.
5. Qwen3.5-9B challenger after measured memory and latency safety.
6. Reader error taxonomy by language, answer format, document, page complexity, and
   evidence cardinality.
7. Nested-fold model/input selection under the frozen Phase 4 protocol.
8. Retrieval development only after reader capability is established.

No final system is described as state of the art unless locked evaluation and external
benchmark evidence support that claim.
