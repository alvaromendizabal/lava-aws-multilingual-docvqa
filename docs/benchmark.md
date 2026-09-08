# Oracle-evidence reader benchmark

## Research question

How much reader quality is available when the correct evidence pages are held fixed, before retrieval quality is allowed to influence the answer?

This decomposition prevents page-selection errors from being misclassified as model-reading errors. It also makes later retrieval and reranking experiments interpretable.

## Frozen evaluation boundary

The benchmark is tied to the immutable evaluation protocol and pinned model revisions. Private questions, answers, page images, extracted text, and per-example traces remain in versioned S3. Git contains code, configuration locks, checksums, sanitized aggregate summaries, and public notebooks.

## Controlled reader ladder

1. Qwen3.5-4B fused direct baseline on `ml.g5.2xlarge`.
2. Qwen3.5-4B image-only and text-only modality controls.
3. Qwen3.5-4B bounded thinking-mode ablation.
4. Qwen3.5-9B fused direct challenger on the **verified LAVA `ml.g6e.2xlarge` path**.
5. Qwen3.8-27B NF4 fused direct challenger on its verified `ml.g5.2xlarge` path. The unquantized high-memory variant remains a separate research candidate.
6. Multilingual slices, repeated-run stability, error taxonomy, latency, throughput, peak VRAM, and cost-quality Pareto analysis.
7. Retrieval and reranking only after the reader ladder is characterized.

## Acceptance gate

A SageMaker job reaching `Completed` is not enough. A run is accepted only after the artifact gate verifies lineage, raw-response count, structured-output validity, parser errors, and the canonical S3 output prefix. Public synchronization re-runs verification before writing sanitized result manifests to Git.

## Claims discipline

Model size, GPU size, and code complexity are not treated as evidence of benchmark leadership. Strong claims are reserved for results produced under the frozen protocol and, where relevant, an external benchmark.

## Complete descriptive pilot

See [evaluation.md](evaluation.md) for the 16-question workflow and reporting boundaries. A malformed answer is a measured reader failure and remains in the full-pilot denominator. Missing records or inconsistent hashes fail artifact verification.


## One comparable 27B pilot

Only one reader is needed for deployment. The 27B run is a candidate comparison,
not a requirement to operate three models. Reuse completed 4B/9B outputs; evaluate
27B on the same frozen 16 questions, then inspect local LAVA overall, its answer
and citation components, document regressions, latency, memory and compute cost.
Qwen3.8-27B NF4 differs in model generation and precision from Qwen3.5 BF16.
It cannot isolate the effect of parameter count.

The verified path is `qwen38_27b_nf4_g5_fused_direct` on `ml.g5.2xlarge`.
Use the canonical full-pilot readiness command (the default preflight remains a
one-question smoke check):

```bash
make benchmark-preflight MODEL=qwen38_27b_nf4_g5_fused_direct
make benchmark-preview MODEL=qwen38_27b_nf4_g5_fused_direct
```

For a deliberately authorized new run, with a $2/hour planning ceiling and a
$2.50 compute allowance:

```bash
uv run --frozen python scripts/run_oracle_reader.py \
  --mode benchmark --model-key qwen38_27b_nf4_g5_fused_direct \
  --hourly-usd-ceiling 2 --maximum-total-usd 2.50 \
  --submit --wait --acknowledge-charges YES
```

The dated [Training price snapshot](../reports/aws/training_prices.json) records
$1.515/hour in Oregon on September 6, 2026. Check current rates before future
paid runs. The allowance is a planning check, not an AWS account spending cap;
Studio, storage, logs, transfer, taxes and discounts are outside it. AWS enforces
one hour runtime separately from 24 hours pending. The managed job survives a
browser disconnect. The local monitor is reconnectable and does not cancel the
job merely because a terminal disappears.

A completed run must pass `make verify JOB=<job-name>` and `make sync JOB=<job-name>`.
Then use `make evaluate JOB=<job-name>` to judge its saved answers and `make report`
to rebuild public views. A Failed/Stopped run can use the documented benchmark
resume command; an active job should be monitored, not duplicated.

## Read the process in notebooks

All six executed notebooks live in [notebooks/](../notebooks/README.md).
Read 00 through 04 to understand the data, design, verified execution, model
comparison, and retrieval results. Each runs independently against local public
aggregates. It creates no GPU job and repeats no model inference.

```bash
make notebooks
```

The command publishes results directly to the canonical notebook files and
records source, input, and output hashes in `reports/notebook_execution/`.
Current publications are reused even in a fresh clone. Changed inputs execute
into immutable staging before publication; failures preserve previous outputs,
and interrupted publication resumes without repeating completed execution.
UTC stage events, progress, total time, and heartbeats are recorded under
`artifacts/notebook_runs/`. Git retains the reviewed outputs in `notebooks/`.
The former duplicate notebook folder and paired Python files are removed.

For a separate private execution archive, the existing runner also supports
`--output-dir artifacts/notebook_runs/<attempt>`. Archive important private
artifacts to the approved S3 prefix and verify read-back before removing a
working copy. Completed GPU responses and checkpoints already persist in S3.
