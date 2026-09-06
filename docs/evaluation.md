# Complete oracle-reader pilot

The next experiment evaluates all 16 frozen questions, across five documents,
on the verified Qwen3.5-4B `ml.g5.2xlarge` path. It uses the current prompt and
pinned model revision. No tuning, model promotion, semantic-judge implementation,
or new GPU experiment has been performed as part of this code change.

## 1. Validate and preview

Run from a clean checkout of this branch in SageMaker Studio:

```bash
make quality
make report
make benchmark-preview MODEL=qwen35_4b_fused_direct
```

The preview launches no GPU. It validates local configuration and uses the
project's existing `.env` for `S3_BUCKET` and `AWS_REGION`. Open
`reports/oracle_reader/evaluation/index.html` in a browser or execute notebook 03.

## 2. Submit one complete pilot

```bash
make benchmark-submit MODEL=qwen35_4b_fused_direct CHARGES=YES
```

This is the paid step. Submission requires a clean Git working tree, the existing
cost acknowledgement, a sufficient training quota, no active LAVA job, an unused
output prefix, and the exact frozen manifest version and SHA-256. It creates one
on-demand training job, with a one-hour runtime limit and no endpoint. The
existing 24-hour server-side capacity-acquisition window is retained; pending
capacity is distinct from active inference. Cost checks use a supplied hourly
ceiling and contingency, not a live quote or guaranteed AWS invoice cap.

UTC timestamps, elapsed time, monitor heartbeats, and per-question progress make
execution observable. The GPU job saves an ignored/private local record checkpoint
after each question. A failed job does not produce an accepted complete result.
Checkpoints aid diagnosis; automatic resumption is not implemented.

If the terminal disconnects, the cloud job can continue within its server limits.
Use the printed job name to reconnect:

```bash
make monitor JOB=your-training-job-name
```

## 3. Verify, synchronize, and rebuild

The submit command already verifies the completed result. Synchronization repeats
verification before saving public aggregates:

```bash
make sync JOB=your-training-job-name
make report
uv run --frozen python scripts/execute_notebook_smoke.py notebooks/03_model_scaling_and_cost.ipynb
```

The expected markers are `ORACLE_READER_BENCHMARK_VERIFIED`,
`ORACLE_READER_RESULTS_SYNCED`, and `ORACLE_READER_REPORT_VERIFIED`.
Install the canonical notebook kernel once if needed:

```bash
uv run --frozen python -m ipykernel install --user --name lava --display-name 'Python (lava)'
```

Review and commit the sanitized report changes before the next paid experiment;
submission requires code and result lineage to be committed. The full-pilot output
prefix includes `/benchmark`, so it cannot overwrite the one-question smoke result.
A repeat at the same model/code/output prefix is refused.

## 4. Continue the controlled reader comparison

After verifying the 4B run, the same preview and submit commands accept
`MODEL=qwen35_9b_fused_direct` and `MODEL=qwen38_27b_nf4_g5_fused_direct`.
Run one model at a time. The 9B path remains `ml.g6e.2xlarge`; 27B NF4 remains
`ml.g5.2xlarge`. A one-question smoke does not prove that every multi-page input
will fit or that cloud capacity will be immediately available.

## Interpretation and next research gate

- Questions are not independent samples: report question-micro and document-macro
  answer diagnostics, plus all five document scores.
- The set contains 15 Japanese and one Vietnamese question. Document and language
  are confounded. Do not characterize Vietnamese performance from a single example.
- The pinned reader is evaluated descriptively on the complete pilot. Existing
  nested fold manifests do not establish that training or nested tuning occurred.
- The evaluator uses normalized-exact scalar equivalence, list matching/LCS, and
  evidence-set F1. It is not an organizer-server semantic score. Invalid output
  and abstention remain in the denominator; infrastructure failures invalidate
  completeness instead of silently dropping questions.
- Oracle-fixed overall diagnostics include a constant grounding term. Keep answer
  quality separate from that diagnostic when interpreting reader capability.
- Compatible complete runs receive document-paired deltas, an exploratory cluster
  bootstrap interval, and an exact sign-flip p-value. With five nonzero document
  deltas the smallest two-sided p-value is 0.0625. These comparisons cannot establish
  conventional 5% significance or justify model promotion alone.
- Generation time excludes model loading. Billable seconds are not dollar costs;
  hardware and quantization differ. Repeated runs and current rate provenance are
  required before asserting a cost-quality frontier.

The next research gate is a pinned, documented semantic judge validated on explicit
multilingual equivalence cases, followed by modality controls and retrieval/reranking
experiments. Preserve this pilot's exact diagnostics as a reproducible baseline.

## Canonical interface

`scripts/run_oracle_reader.py` replaces `scripts/run_oracle_reader_smoke.py`.
It supports `--mode smoke` (default, exactly one question) and `--mode benchmark`
(exactly the complete frozen set). No repair, fixed, or duplicate runner is retained.
The original model registry, protocol locks, and dependency lock are unchanged.

## Validation of this implementation

Validated on 2026-09-06 in the locked Python 3.12 environment:

- Canonical `scripts/quality_gate.sh`: passed, six seconds with a warm environment.
- Pytest: **205 passed**, including 17 new complete-pilot tests.
- Mypy: no issues in 57 source files; Ruff, shell syntax, compileall, notebook hygiene,
  and Git whitespace checks passed.
- Complete 4B preview: `BENCHMARK_PLAN_PREVIEWED`; no paid resource created.
- Notebook 03 code executed successfully in-process through IPython. A separate
  Jupyter kernel could not start because this execution environment denies its
  socket bindings; the SageMaker notebook smoke command remains documented above.
- Browser screenshot verification was unavailable because the local browser binary
  could not be downloaded. HTML generation, escaping, checksum checks, and coverage
  behavior were tested; browser visual inspection remains pending.
- New GPU inference has not been run. The existing three synchronized one-question
  results are genuine prior run artifacts; the new complete-pilot tests use synthetic
  data and an injected reader/cloud adapter.

A generated local Mypy cache was cleared after SQLite reported corruption. The
canonical Mypy command then passed without changing dependencies or skipping checks.
