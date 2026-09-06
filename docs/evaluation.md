# Oracle-reader evaluation and recovery

The **4B complete pilot is verified**: 16 questions across five documents, 380
billable seconds, 100% valid output, and no parser errors. Its normalized-exact
answer diagnostic is **0.38125 question-average** and **0.43833 document-average**.
Formatting validity and answer quality measure different things. The public report
contains the actual synchronized results; do not rerun this completed experiment.

Run: `lava-oracle-qwen35-4b-fused-direct-20260906180137`.
Code: `a4a5b8b1271cef96866f6c16134670cac78867d7`.
Public-summary SHA-256: `ccbfc5f8f3c43549a3c559715fd06e5239316178f08d5a763b2d8e3d749a3926`.

## Next experiment

One deployed reader is sufficient. The model sizes are comparison candidates, not
three required production models or three training stages. The 4B and 9B candidates
are Qwen3.5 models; the configured 27B candidate is Qwen3.8 with NF4 quantization.
The last comparison therefore changes model generation and numerical precision as
well as parameter count. It cannot isolate parameter scaling alone.

The next optional comparison is 9B on the same 16 questions. Defer 27B until the
4B/9B results establish a reason to spend more. None of these pilot results alone
establishes state-of-the-art performance. Semantic evaluation and a larger,
representative evaluation set remain research gates.

From the project checkout in SageMaker Studio:

```bash
make quality
make report
make benchmark-preview MODEL=qwen35_9b_fused_direct
```

Preview starts no GPU. Open `reports/oracle_reader/evaluation/index.html`, or run
notebook 03, to inspect answer scores by document and format, runtime, memory,
coverage, and provenance. Smoke results remain explicitly labeled.

The paid step, when ready:

```bash
make benchmark-submit MODEL=qwen35_9b_fused_direct CHARGES=YES
```

Submission requires committed code, explicit charge acknowledgement, a sufficient
quota, no active LAVA job, an unused output prefix, and the pinned manifest version
and SHA-256. It creates one on-demand `ml.g6e.2xlarge` training job, with a one-hour
compute limit, a separate 24-hour capacity-acquisition limit, and no endpoint.
The cost check uses a supplied hourly ceiling and contingency; it is not a live
price quote or guaranteed invoice cap. Existing one-question execution proves the
model can load; it does not guarantee capacity or memory fit for every pilot input.

## Durable recovery

For new benchmark jobs, every completed question is saved privately to S3 as one
immutable object containing its exact raw generation, parsed record, scores, and
compatibility contract. S3 SHA-256 checksums and conditional writes protect those
objects. The `question.completed` event follows successful durable persistence.
Records also remain in a private atomic local checkpoint.

UTC timestamps, stage duration, total elapsed time, question counts, and heartbeats
remain visible during inference, checkpoint writes, monitoring, and verification.
New and reused question counts appear in the final public summary. Runtime telemetry
on a reused answer describes its original inference, not the time spent restoring
it. SageMaker billable duration refers to each attempt separately.

If only the terminal disconnects, the cloud job may still be running. Reattach:

```bash
make monitor JOB=your-training-job-name
```

If AWS reports **Failed** or **Stopped**, keep the exact source Git checkout and
preview recovery before creating replacement compute:

```bash
make benchmark-resume-preview MODEL=qwen35_9b_fused_direct JOB=your-failed-job-name
make benchmark-resume MODEL=qwen35_9b_fused_direct JOB=your-failed-job-name CHARGES=YES
```

Recovery validates checkpoint checksums, question identity, code/model revisions,
prompt, data, generation configuration, hardware, parsed answers, and recomputed
scores before submission. The GPU process independently repeats compatibility
checks. A new attempt has a distinct output prefix; prior outputs are preserved.
An immutable parent link retains earlier checkpoints even if recovery itself is
interrupted while copying saved work. Cyclic ancestry and conflicting answers are
rejected. All parent attempts must remain available until evaluation is complete.

Only unfinished questions run inference again. Invalid model outputs and
abstentions are completed observations and are reused too. A question interrupted
before its S3 checkpoint is acknowledged may need to run again. Replacement compute
must provision a new instance and reload weights if any inference remains. If all
16 answers were already checkpointed, the reader is never constructed, although
the current resume command still provisions a billed job to finalize and verify
its SageMaker artifact. Preview shows the reusable count before that paid choice.

Automatic reuse is deliberately limited to unchanged code and deterministic
benchmark decoding. Changed implementations must not mix old and new predictions.
Jobs created before this checkpoint implementation cannot acquire resumability
retroactively. The successful 4B run needs only synchronization, which is complete.

## Verify, synchronize, and rebuild

Submission already verifies completed outputs. Synchronization repeats verification
before writing public aggregates:

```bash
make sync JOB=your-completed-job-name
make report
```

Expected markers: `ORACLE_READER_BENCHMARK_VERIFIED`,
`ORACLE_READER_RESULTS_SYNCED`, and `ORACLE_READER_REPORT_VERIFIED`.
Review and commit result changes before another paid experiment. Public reports
contain aggregates and hashes; private questions, answers, images, and generations
remain outside Git. The normal runner is `scripts/run_oracle_reader.py`; corrections
are made in canonical sources with no duplicate repair or fixed variants.

## Interpretation

- Report both question-average and document-average scores. Questions from one
  document are related observations, not independent experimental units.
- The set contains 15 Japanese questions and one Vietnamese question. Language and
  document identity are confounded; one example cannot establish Vietnamese quality.
- List answers can receive partial credit. These are normalized-exact diagnostics,
  not organizer-server semantic scores. Numeric answers scored zero on this 4B run;
  diagnosis must distinguish model errors from equivalent answer representations.
- Oracle-fixed overall diagnostics include a constant grounding contribution.
  Interpret answer quality separately. Retrieval and reranking are not evaluated here.
- Compatible complete runs receive document-paired deltas, exploratory cluster
  bootstrap intervals, and exact sign-flip p-values. With five nonzero document
  deltas, the smallest two-sided p-value is 0.0625; this pilot cannot support a 5%
  significance claim or model promotion by itself.
- Hardware, quantization, startup overhead, retries, and per-question generation
  time must be reported separately. Billable seconds are not dollar costs. Account
  for failed and resumed attempts when calculating the cost of an experiment.

## Validation

On 2026-09-06, the canonical gate passed all 226 tests and Mypy across 59 source
files. The canonical `make quality` gate runs Ruff, Mypy, Pytest, shell syntax, compilation,
notebook hygiene, and Git whitespace checks with timestamps, heartbeats, and total
duration. Explicit synthetic interruption tests cover multiple restarts, disk loss,
interruption during restoration, corrupt or incompatible checkpoints, rejected
writes, immutable idempotent writes, parser failures, and complete artifact audits.
Tests use injected model and cloud adapters; they do not claim GPU execution.

The original 4B run is real AWS evidence and was independently checked against all
16 downloaded raw generations and the pinned manifest before synchronizing its
public summary. The resume implementation has not yet been exercised by an
interrupted paid AWS job. No new paid job was launched while implementing it.

Notebook 03 supports offline report viewing. This editing environment blocks the
socket bindings needed by a separate Jupyter kernel; in-process cell execution is
used here. Browser screenshot inspection is unavailable in this environment; report
generation, escaping, and numeric output are tested. To exercise a separate kernel in SageMaker Studio:

```bash
uv run --frozen python -m ipykernel install --user --name lava --display-name 'Python (lava)'
uv run --frozen python scripts/execute_notebook_smoke.py notebooks/03_model_scaling_and_cost.ipynb
```
