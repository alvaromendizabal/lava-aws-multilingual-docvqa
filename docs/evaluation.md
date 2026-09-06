# Oracle-reader evaluation and recovery

The **4B and 9B complete pilots are verified**: each covers the same 16 questions
across five documents, with 100% valid output and no parser errors. Their
normalized-exact diagnostics are:

| Measure | 4B | 9B |
| --- | ---: | ---: |
| Question-average answer score | 38.125% | 45.774% |
| Document-average answer score | 43.833% | 39.714% |
| Mean generation time | 5.517 s | 3.610 s |
| Peak allocated GPU memory | 10.951 GiB | 20.149 GiB |
| AWS billable time | 6m 20s | 6m 35s |
| Capacity wait | 57s | 26m 06s |
| Submission to completion | 7m 18s | 32m 42s |

The last row measures AWS creation to completion, excluding local preparation and
monitor polling. Job lifecycle timestamps and phase durations are retained in each
full run's public synchronization manifest. Missing legacy telemetry is shown as
unknown. Generation time excludes model loading and uses different GPUs, so it is
not an isolated model-speed comparison. Formatting validity and answer quality
measure different things. Do not rerun either completed experiment.

Run: `lava-oracle-qwen35-4b-fused-direct-20260906180137`.
Code: `a4a5b8b1271cef96866f6c16134670cac78867d7`.
Public-summary SHA-256: `ccbfc5f8f3c43549a3c559715fd06e5239316178f08d5a763b2d8e3d749a3926`.

9B run: `lava-oracle-qwen35-9b-fused-direct-20260906190503`.
Code: `5fcc7a35ac39c7223810004042b055abc79bf14b`.
Public-summary SHA-256: `dd60d8668b69ec82e5aaa90b4311c12534852b43112e36e687aeb5c3db3ad0cc`.

9B improves two documents, ties two, and regresses on one. Its question-average
gain is 7.65 percentage points, but its document-average change is −4.12 points.
The exploratory 95% document-bootstrap interval is −43.17 to +25.98 points; the
exact paired two-sided sign-flip p-value is 1.000. These results do not select a
winning model. The dashboard presents signed document deltas, a numeric table,
both weighting schemes, and uncertainty directly.

## Next experiment

One deployed reader is sufficient. The model sizes are comparison candidates, not
three required production models or three training stages. The 4B and 9B candidates
are Qwen3.5 models; the configured 27B candidate is Qwen3.8 with NF4 quantization.
The last comparison therefore changes model generation and numerical precision as
well as parameter count. It cannot isolate parameter scaling alone.

The next research step is failure analysis and evaluation expansion. Both readers
score zero on doc-02; 9B regresses on the sole Vietnamese question in doc-05.
Review those raw answers privately against the source pages and reference answers,
distinguishing reading errors, evidence selection, and equivalent representations.
Keep this frozen diagnostic unchanged; any new semantic metric needs a versioned
contract and independent validation. Add representative documents and languages
with a held-out evaluation split before selecting one reader.

27B can optionally answer whether a different reader resolves the observed failures
on the same frozen pilot. Its one-question smoke is already verified; a complete
16-question run has not yet been submitted. None of these pilot results alone
establishes state-of-the-art performance.

From the project checkout in SageMaker Studio:

```bash
make quality
make report
make benchmark-preview MODEL=qwen38_27b_nf4_g5_fused_direct
```

Preview starts no GPU. Open `reports/oracle_reader/evaluation/index.html`, or run
notebook 03, to inspect answer scores by document and format, runtime, memory,
coverage, and provenance. Smoke results remain explicitly labeled.

The optional paid 27B comparison, after reviewing its preview:

```bash
make benchmark-submit MODEL=qwen38_27b_nf4_g5_fused_direct CHARGES=YES
```

Submission requires committed code, explicit charge acknowledgement, a sufficient
quota, no active LAVA job, an unused output prefix, and the pinned manifest version
and SHA-256. This 27B plan creates one on-demand `ml.g5.2xlarge` training job, with a one-hour
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

Once AWS accepts a job, execution is independent of the terminal, browser, or
ChatGPT session. Signing out does not cancel it. Server-side runtime and pending
limits still apply; logging and S3 checkpoint writes continue in the cloud.
Local verification and report synchronization can be run when the operator returns.
The current runner still permits only one active LAVA job. Optional two-job
concurrency needs separate state isolation, duplicate submission protection, quota
checks, combined cost review, and explicit tests; it is not needed to preserve
either completed result.

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

On 2026-09-06, the canonical gate passed all 234 tests and Mypy across 59 source
files. The canonical `make quality` gate runs Ruff, Mypy, Pytest, shell syntax, compilation,
notebook hygiene, and Git whitespace checks with timestamps, heartbeats, and total
duration. Explicit synthetic interruption tests cover multiple restarts, disk loss,
interruption during restoration, corrupt or incompatible checkpoints, rejected
writes, immutable idempotent writes, parser failures, and complete artifact audits.
Tests use injected model and cloud adapters; they do not claim GPU execution.
Eight additional checks cover UTC lifecycle arithmetic and invalid timestamps,
missing historical telemetry, the verified mixed 4B/9B result, accessible signed
chart geometry, and HTML escaping. The quality gate completed in seven seconds.

The 4B and 9B runs are real AWS evidence. Each was independently checked against all
16 downloaded raw generations and the pinned manifest before synchronizing its
public summary. The completed 9B run also verifies all 16 immutable S3 question
checkpoints against final records, raw text, and recomputed scores. This proves
successful durable writes and readback on AWS. Recovery after interruption remains
covered by explicit synthetic tests; it has not yet been exercised by deliberately
interrupting a paid AWS job. No additional paid job was launched for this report.

Notebook 03 supports offline report viewing. This editing environment blocks the
socket bindings needed by a separate Jupyter kernel; in-process cell execution is
used here. The signed comparison SVG was rendered and visually inspected; full
browser screenshot inspection is unavailable in this environment. Report generation,
escaping, and numeric output are tested. To exercise a separate kernel in SageMaker Studio:

```bash
uv run --frozen python -m ipykernel install --user --name lava --display-name 'Python (lava)'
uv run --frozen python scripts/execute_notebook_smoke.py notebooks/03_model_scaling_and_cost.ipynb
```
