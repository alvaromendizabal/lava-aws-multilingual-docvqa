# Final retrieved-evidence pilot

This is the final **scoped research-portfolio** milestone. It is not a trained model,
hidden-test benchmark, production deployment or Kaggle submission. Keep the measured
oracle results and the completed retriever; do not repeat the model sweep.

## The normal finish command

Use the existing Linux SageMaker `lava-dev` workspace and canonical repository.
Its `.env`, AWS execution role and Hugging Face model access must remain configured.
The linked ChatGPT account is not a replacement for this terminal's credentials.
The command checks Gemma access and available CPU memory before any new GPU launch.

```bash
cd /home/sagemaker-user/lava-aws-multilingual-docvqa
make system-preview
make finish CHARGES=YES
```

`system-preview` is offline and creates no paid resource. `finish` explicitly approves
one bounded GPU attempt, uses the frozen 9B/BM25 five-page configuration on all 16
questions, scores real answers with the unchanged semantic judge, executes all six
notebooks, runs the full quality gate, and read-back verifies successful notebook
publications in private S3. It never uploads a Kaggle submission.

The AWS training job uses one `ml.g6e.2xlarge`, the existing image pinned by digest,
a 1,800-second runtime cap and a 24-hour server-side capacity-acquisition limit.
Default cost guard: $5/hour ceiling × 0.5 runtime hours × 1.25 contingency = $3.125,
against a $5 **per-attempt estimate limit**. The recorded September 6 training price
is $2.80/hour; it is not a live billing quote. These controls are not an account-wide
hard dollar cap. Existing Studio, storage, logs, transfer, taxes and other attempts
are separate. No resize, endpoint or IAM change is performed.

The service limits each container argument to 256 characters. A short entrypoint
verifies the SHA-256 Git archive before invoking `pipelines/oracle_reader/run.sh`.
The bootstrap installs the pinned GPU requirements and then runs the existing
canonical `job_entry.py` in `system` mode. Source input uses the already-authorized
oracle-reader prefix; checkpoints use the existing submission prefix. New page
assets use content-addressed keys and exact-byte verification, without requiring
ungranted `GetObjectVersion` or `ListBucket` permissions in that prefix.

## See progress and recover

The terminal prints UTC events with stage/total elapsed time and 15-second
heartbeats. Reader events and dependency/model logs are also retained by SageMaker
in CloudWatch under `/aws/sagemaker/TrainingJobs` for the printed job name. Local
operator logs are in `artifacts/system/runtime/`; they are archived to S3 on exit.
Successful notebook outputs are published in `notebooks/`, not a duplicate folder.

A browser or monitor interruption does not cancel an accepted training job. Repeat
**the same command with the same attempt number** to reconnect. An ambiguous create
response is resolved by the same deterministic job name, never a random second job.
Each compatible completed answer is read back and independently parsed before reuse.
A question interrupted before its checkpoint commits may need to run again.

A **Failed/Stopped** attempt is not automatically relaunched. Inspect its CloudWatch
log first. A deliberately approved second attempt reuses the first attempt's saved
answers under the unchanged input contract:

```bash
make finish CHARGES=YES ATTEMPT=2 RETRY=YES
```

Changing inference source, model revision, source manifest, page budget or rendering
changes the contract and correctly invalidates incompatible results. Report-only
commits and merges do not invalidate compatible answers. Changes to the judge or
scoring code cannot silently reuse a current public score. Never delete existing
checkpoints to make a gate pass.

When only semantic scoring or notebook publication was interrupted, use:

```bash
make system-evaluate
make notebooks
make quality
```

These commands create no new GPU resource. `finish` also detects completed inference
and skips allocation, but its explicit `CHARGES=YES` guard remains visible.

## Publish the measured result

Only after `system.portfolio.ready` and `QUALITY_GATE_PASSED`, review Notebook 05's
actual metrics and failure table. Do not mark this milestone complete while the
notebook says that inference is unmeasured. No minimum score is used to suppress
unfavorable results; all 16 questions and any invalid outputs remain counted.

Create one results branch and commit **only** public aggregates and reviewed outputs:

```bash
git switch -c feat/system-results
git add reports/system notebooks reports/notebook_execution
git diff --cached --check
git commit -m "feat: publish measured retrieved-evidence 9B evaluation" \
  -m "Evaluate all 16 labeled questions with frozen BM25-selected pages and the unchanged local semantic judge. Preserve failures, paired document diagnostics, resource telemetry and executed notebook provenance. Kaggle submission remains optional."
git push -u origin feat/system-results
```

If that branch already exists after an interrupted publication, use
`git switch feat/system-results` rather than creating a duplicate branch.
Open its pull request in GitHub, require CI to pass on the final head, review the
notebook outputs and merge using a merge commit to retain experiment ancestry.
After the measured results merge, tag that **merged commit** as `v1.0.0`:

```bash
git switch main
git pull --ff-only
git tag -a v1.0.0 -m "Measured retrieved-evidence document-QA portfolio"
git push origin v1.0.0
```

Do not create the tag for the implementation-only change. A Kaggle extension still
needs all 624 test predictions, complete container/runtime validation and verified
submission eligibility; none is claimed by this portfolio milestone.

## Evaluation and privacy

Primary metric: question-average `(semantic answer credit + evidence-page F1) / 2`.
Supporting views: equal-document score, answer/evidence components, precision and
recall, complete evidence coverage, validity/abstention, per-document and answer-format
slices, failure categories and measured reader latency/memory. All five PDFs were
previously examined; intervals over five document clusters are exploratory.

`ReaderInput` contains no reference answer or gold-page field. The original
`OracleExample` keeps its gold-alignment checks. Physical page IDs are retained
through retrieval, rendering, prompting and parsing. References enter only after
complete inference. Exact generations remain private; public diagnostics use stable
question/document aliases and contain no questions, answers or PDF content.

The published LAVA formula is reproduced locally; organizer prompt/runtime parity
and official server scores are not claimed. The existing 28 judge controls remain
mandatory and unchanged.

References: [LAVA evaluation](https://lava-workshop.github.io/#evaluation),
[SageMaker container specification](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_AlgorithmSpecification.html),
[server stopping conditions](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_StoppingCondition.html).
