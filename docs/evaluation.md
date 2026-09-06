# Oracle-reader evaluation and recovery

## Operator steps: Hugging Face login and saved-answer scoring

No Claude account, Claude installation, or Claude API key is used by this project.
The Hugging Face CLI may print `hf skills add -g --claude`. This is an optional
AI-assistant skill suggestion; ignore it. The login menu that follows belongs to
**Hugging Face**. See the [official CLI guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli#hf-auth-login).

GitHub stores reviewed source and reports. AWS runs readers and retains private
results in S3. Hugging Face provides model files, including Google's Gemma judge.
Linking these services to ChatGPT does not automatically authenticate the CLI
running inside your separate SageMaker terminal. Model-term acceptance is also
separate from signing in.

Perform these steps in order. Paste an interactive login command **by itself**.

1. If the terminal is currently showing the login menu, leave **Log in with your
   browser** selected and press **Enter**. Open the Hugging Face URL it prints in
   your desktop browser, sign in, enter the displayed one-time code if requested,
   and approve the device authorization. Wait for **Login successful** and the
   ordinary `sagemaker-user@...$` prompt. If a previously pasted `make evaluate`
   starts automatically on the old 4 GiB space, press **Ctrl+C** to stop the CPU
   evaluation before proceeding. Completed reader answers remain in S3.

2. In that same browser account, open
   [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it). If prompted,
   review and accept Google's terms or request access. Continue once access is
   granted. You do not need another model-provider account.

3. The verified `lava-dev` app was on **ml.t3.medium (4 GiB)** on 2026-09-06.
   This is insufficient headroom for the pinned float32 CPU judge. Save notebook
   files, open **SageMaker Studio > JupyterLab > lava-dev**, choose **Stop space**,
   wait until stopped, select **ml.m7i.2xlarge (8 vCPUs / 32 GiB)**, and choose
   **Run space**, then **Open JupyterLab**. Review the displayed CPU price before
   starting it. Keep the existing space/storage and do not delete them: its EBS
   volume persists across instance changes. This is a billed CPU size change,
   not a GPU job. See the [AWS JupyterLab guide](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-jl-user-guide.html).

4. At an idle terminal prompt, update the merged main branch. This preserves
   original experiment commits in Git history. Do not use force/reset commands
   to discard local work if Git reports uncommitted changes.

   ```bash
   cd "$HOME/lava-aws-multilingual-docvqa"
   git fetch origin &&
   git switch main &&
   git merge --ff-only origin/main
   ```

5. If login has not been completed, run just this command, then complete step 1:

   ```bash
   uv run --frozen --group judge hf auth login
   ```

6. Check memory, the terminal's account and the pinned Gemma resource. This downloads no
   weights, reads no private predictions, and creates no AWS resources:

   ```bash
   make evaluation-check
   ```

   Continue only after **SEMANTIC_JUDGE_ACCESS_VERIFIED**. The output identifies
   available RAM, the account and pinned model revision. A memory/login/access/network problem returns
   a nonzero exit code with a specific next action, not an ambiguous success.

7. Evaluate the saved complete pilots:

   ```bash
   make evaluate
   ```

   Expect source verification, `judge.load` heartbeats, acceptance probes, and
   `judge.decision.completed` events with new/reused counts. CPU model loading or
   judging can take time; a heartbeat is a liveness signal, not a quality score.
   Completion is **SAVED_PREDICTIONS_EVALUATION_COMPLETED**. This creates no new
   reader GPU job. Re-running the same command reuses compatible S3 decisions.

8. Open `notebooks/03_model_scaling_and_cost.ipynb` in JupyterLab and select
   **Run > Run All Cells**, or open the generated
   `reports/oracle_reader/evaluation/index.html`. Notebook 02 shows current
   coverage; notebook 03 embeds the report. The semantic score remains a clearly
   labeled local implementation of the published formulas.

Each evaluation/check invocation prints its log path under
`artifacts/semantic_judge/runtime/<attempt>/events.jsonl`. These UTF-8 logs append
UTC timestamps, stage durations, total elapsed time, heartbeats, and progress;
each event is flushed to disk before it is printed. Keep these private workspace
logs with the run. Accepted answer-pair decisions are separately durable in S3,
including after loss of the local workspace. CPU judging runs on the current
host: if that host stops, rerun the command to resume.

The memory gate requires at least 8 GiB currently available, bounded by Linux
container limits as well as host RAM. It repeats immediately before uncached
model loading; the threshold is conservative operating headroom, not a measured
peak-memory claim. Fully cached judge decisions can still be reused without
loading the model. The authorized Studio host completed model loading, all 28
controls, both full evaluations, and a second run using only cached decisions on
2026-09-06. Runtime artifacts remain available in S3.

After successful scoring, review and commit only public result aggregates and
the report on a results branch, run `make quality`, and merge through a reviewed
pull request. Questions, answers, credentials, weights, and logs remain outside
Git. A new implementation or judge contract uses a separate scoring cache.

## Judge validation and notebook update recovery

The 2026-09-06 operator run passed access checks with 28.16 GiB available, loaded
Gemma successfully, and stopped because the original prompt judged `1000` and
`1001` equivalent. This was a judge-quality failure. The earlier gated-model 403
was resolved by the operator; repeated login or reader GPU runs cannot correct
an inaccurate judge.

The revised prompt explicitly treats the ground truth as authoritative and grades
a quoted submitted answer against it. It preserves the pinned Gemma revision,
float32 CPU runtime, strict YES/NO parsing, and all published score formulas.
No probe answers are inserted into the prompt. The original eight controls remain;
20 additional controls cover decimal precision, dates, opposite meanings, empty
answers, multilingual names and negation, formatting, and injected instructions.
All 28 passed on the real cached model in `lava-dev`; see the
[validation evidence](../reports/oracle_reader/judge_validation.json). These are public development
controls, not a held-out estimate of general judge accuracy or organizer parity.

Every acceptance run evaluates all controls, logs expected/observed results with
UTC timestamps, and saves a content-addressed report in S3 before reporting pass
or failure. Malformed output is not converted to a negative vote. A rejection
exits with status 3 and `SEMANTIC_JUDGE_REJECTED`; saved reader answers remain valid.
The changed scoring contract isolates earlier decisions, including the incorrect
old numeric decision. Compatible decisions within the new contract are reusable.

If an executed notebook blocks `git switch main`, first copy its exact bytes to
`artifacts/notebook_runs/<UTC-attempt>/`, upload the copy to the private S3 run
prefix, and verify the checksum. A Git stash alone is **not an output backup**:
the notebook clean filter strips outputs even when stashing. Preserve source
edits with a named stash and reapply those edits deliberately after updating.
Only restore the canonical notebook after verifying that its source is unchanged
and its executed copy is safely preserved.

The Studio checkout's old filter also renumbered cell IDs, making a freshly
restored notebook appear modified. Preserve IDs while continuing to strip outputs:

```bash
git config --local filter.nbstripout.clean 'uv run --frozen nbstripout --keep-id'
git config --local filter.nbstripout.smudge cat
git config --local filter.nbstripout.required true
git diff -- notebooks/02_verified_gpu_execution.ipynb
```

On 2026-09-06 this configuration removed the false ID-only difference in Studio.
A regression verifies preserved IDs and source, removed outputs and execution
counts, and an idempotent second clean pass. Do not paste independent update
commands that continue after a failed switch. The `&&` chain in step 4 stops at
the first error. Never use `reset --hard` to resolve this. Notebook outputs do
not imply that reader inference needs repeating; the paired public sources stay
clean and reviewable.

## Verified reader results

The **4B and 9B complete pilots are verified**: each covers the same 16 questions
across five documents, with 100% valid output and no parser errors. Their
normalized-exact diagnostics are:

| Measure | 4B | 9B |
| --- | ---: | ---: |
| Question-average answer score | 38.125% | 45.774% |
| Document-average answer score | 43.833% | 39.714% |
| Evidence-page F1 (oracle pages supplied) | 97.024% | 93.899% |
| Exact evidence-page set | 87.50% | 81.25% |
| Normalized-exact full-credit answers | 18.75% | 31.25% |
| Generation p50 / p95 | 5.286 / 8.910 s | 3.422 / 5.157 s |
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

## Completed local semantic evaluation

| Reader | Semantic VQA / question | Semantic VQA / document | Grounding F1 | Local LAVA overall |
| --- | ---: | ---: | ---: | ---: |
| 4B | 50.625% | 53.833% | 97.024% | 73.824% |
| 9B | 80.149% | 76.381% | 93.899% | 87.024% |

Both rows cover the same 16 questions and five documents. The combined column is
question-weighted, as specified by LAVA. These are local published-formula scores,
not organizer-server results. The judge contract is
`6f3e8ea4f81bf99601d0a427bd541b27020e1668741d739ac97bde7e738222cc`.
Original reader hashes are unchanged. Public semantic summaries include per-format,
per-language, and per-document results with exact source and judge provenance.

9B improves semantic VQA on three documents, ties one, and regresses on doc-05,
the sole Vietnamese question. The semantic and normalized-exact comparisons answer
different evaluation questions; the dashboard labels them separately. The small
pilot and oracle evidence setting do not support broad performance claims.

## Next experiment

Semantic evaluation of the **already saved** 4B and 9B answers is complete.
Their GPU inference and saved-answer scoring are both verified.
Notebook 02 and the report show 16/16 coverage for both, separate from historical
one-question smoke runs. No reader rerun is required to add semantic scores.

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

Use **one** `qwen38_27b_nf4_g5_fused_direct` full pilot on `ml.g5.2xlarge`, the exact
27B path that passed its smoke. G6e was verified for 9B, not this 27B configuration.
Duplicating 27B across G5 and G6e would test hardware, not add another reader
candidate. A controlled hardware experiment can follow if latency/cost is the
research question. Do not assume that similarly named G6 and G6e have the same GPU.

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

## Published metric and saved-answer evaluation

The [LAVA organizers](https://lava-workshop.github.io/#evaluation) define:

- String/number answers: binary semantic equivalence judged by Gemma-3 1B.
- Unordered lists: F1 after semantic item matching without double-counting.
- Ordered lists: semantic longest-common-subsequence length divided by the longer list.
- Grounding: exact F1 between predicted and reference evidence-page sets.
- Overall: average `(answer score + grounding F1) / 2` over questions.

Document averages and language/format slices supplement the question-average score.
The report also includes evidence precision, recall and exact page-set rate;
normalized-exact full/zero-credit answer rates; schema validity and abstention;
generation p50/p95, throughput, GPU memory, provisioning wait and billable duration.
These are distinct measurements, not interchangeable definitions of accuracy.

`scripts/evaluate_oracle_reader.py` audits each completed job, then scores the exact
saved private records and pinned reference manifest. Its snapshot prevents scoring
different bytes from those verified. Evidence comes from the reader's **predicted
pages**, never the constant gold-page contribution used by the old oracle-fixed
diagnostic. Since the reader receives oracle pages, this still does not evaluate
retrieval or establish end-to-end challenge performance.

```bash
make evaluation-preview
make metrics
```

Preview creates no compute and reads no private predictions. Diagnostics read
existing S3 results and update public aggregate metrics. `make evaluate` runs the
pinned Gemma-3 1B judge on the current machine's CPU; it creates no SageMaker job.
S3 requests and the existing host retain their normal costs. The `judge`
dependency group uses the official PyTorch CPU wheel index and is included in the
default development environment so ordinary commands do not remove its packages.
GPU container dependencies remain unchanged.

Google requires accepting the [Gemma terms](https://huggingface.co/google/gemma-3-1b-it)
and authenticating an authorized Hugging Face account. Configure access locally;
never paste a token into ChatGPT, a notebook, or Git:

Run `uv run --frozen --group judge hf auth login` by itself and finish browser
authorization. Then run `make evaluation-check`, followed by `make evaluate`
after the access check succeeds. The operator steps above explain each prompt.

The operator resolved model access and resized `lava-dev` before real-model
validation. On 2026-09-06 all 28 public controls and both 16-question evaluations
passed on the existing CPU host. A second run reused all 129 decision requests
with no model loading or inference. These public development controls are
independent of the private pilot answers; they are not proof of broad judge quality. The exact
organizer prompt and decoding runtime are not published. The dashboard labels
results **local published-formula scores**, with `official_server_score: null`.
An organizer-server score requires an actual organizer evaluation result.

The judge contract fingerprints the model/tokenizer revisions, prompt, probes,
runtime lock and metric sources. Every YES/NO decision is checksummed and stored
privately with S3 conditional writes before progress is reported. Cache keys bind
the contract, language and exact answer pair. Both positive and negative decisions
are reusable across model candidates; ambiguous output, corruption and conflicting
writes fail closed. Re-running `make evaluate` reuses completed decisions and
continues only missing work. Changed contracts use a separate cache; old results
remain in S3 and are labeled stale locally until rescored. No reader inference is
repeated. The same original reader summary hashes remain unchanged.

Evaluation emits UTC timestamps, stage and total elapsed time, 15-second
heartbeats, and new/reused decision counts. CPU evaluation runs in the current
process: shutting down that host stops it. Rerun the command to resume from S3.
Accepted SageMaker GPU jobs have the separate session-independent behavior below.
After scoring, review and commit aggregate result files and the rebuilt report.

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

On 2026-09-06, the canonical gate passed all 369 tests and Mypy across 67 source
files in 45 seconds. The canonical `make quality` gate runs Ruff, Mypy, Pytest, shell syntax, compilation,
notebook hygiene, and Git whitespace checks with timestamps, heartbeats, and total
duration. Explicit synthetic interruption tests cover multiple restarts, disk loss,
interruption during restoration, corrupt or incompatible checkpoints, rejected
writes, immutable idempotent writes, parser failures, and complete artifact audits.
Tests use injected model and cloud adapters; they do not claim GPU execution.
Checks also cover UTC lifecycle arithmetic and invalid timestamps,
missing historical telemetry, the verified mixed 4B/9B result, accessible signed
chart geometry, and HTML escaping. Two presentation
regressions verify that current semantic results drive semantic charts and stale
results retain diagnostic labels.
The notebook-filter regression preserves IDs while removing execution output.
The semantic-analysis and submission update adds 63 tests. These cover paired
document comparisons, additive score gaps, incompatible or invalid metrics,
model-free report imports, and the strict, resumable CSV export described in
[the submission guide](submission.md). Synthetic CSV tests verify the exporter;
they do not establish test-set model quality or competition eligibility.
Three deterministic concurrency regressions delay a heartbeat until after stage
shutdown and cover successful, failed and interrupted stages. They fail against
the earlier logger and pass with serialized terminal events. Heartbeat output
cannot appear after a stage's final completion or failure event.
The semantic evaluation stage adds 34 tests covering immutable decision reuse,
negative decisions, rejected writes, ambiguous output, corruption, stale contracts,
source hashes, actual predicted-page scoring, missing access, metric denominators,
latency percentiles, and notebook execution from its own directory. A tiny Gemma
model with synthetic weights exercises the real pinned CPU generation path; it
does not validate pretrained semantic judgment quality. No gated weights are
downloaded in CI.

The operator-readiness update adds 25 tests covering terminal authentication,
gated permissions, network failures, secret-safe error messages, metadata-only
access checks, host and cgroup memory limits, pre-load memory rejection, persisted
event logs, interruption, and type checking with a corrupt old SQLite cache.
Mypy uses its supported file-cache backend; actual type errors still fail the gate.
The operator resized `lava-dev` from `ml.t3.medium` to `ml.m7i.2xlarge`.
The real canonical evaluator completed in 65.13 seconds, creating 82 unique
Gemma decisions and reusing 47 requests across readers. A second evaluation
completed in 7.71 seconds with 129 reused requests, zero new decisions, and no
model loading. Both notebooks 02 and 03 executed in real Jupyter kernels in
2.93 seconds total. No reader inference or new GPU job was required.

The 4B and 9B runs are real AWS evidence. Each was independently checked against all
16 downloaded raw generations and the pinned manifest before synchronizing its
public summary. The completed 9B run also verifies all 16 immutable S3 question
checkpoints against final records, raw text, and recomputed scores. This proves
successful durable writes and readback on AWS. Recovery after interruption remains
covered by explicit synthetic tests; it has not yet been exercised by deliberately
interrupting a paid AWS job. No additional paid job was launched for this report.

Notebook 03 supports offline report viewing. Both notebooks 02 and 03 executed
successfully in real SageMaker Jupyter kernels after scoring. Public notebooks
remain output-free and paired with reviewable Python sources. The generated HTML
report is self-contained; its semantic charts use the current judge contract and
its normalized-exact comparisons remain explicitly labeled. To execute the
notebooks again in SageMaker Studio:

```bash
uv run --frozen python scripts/execute_notebook_smoke.py notebooks/02_verified_gpu_execution.ipynb notebooks/03_model_scaling_and_cost.ipynb
```
