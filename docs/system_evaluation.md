# Complete-system evaluation and reproduction

The release includes two completed 16-question inference conditions: full-PDF BM25 retrieval followed by Qwen3.5 9B answering, and a second read of that model’s own cited pages. Both use the original pinned reader, deterministic decoding, image assets and semantic judge.

| Condition | Answer credit | Evidence F1 | Local LAVA |
| --- | ---: | ---: | ---: |
| Oracle pages | 80.15% | 93.90% | 87.02% |
| Retrieved pages | 48.90% | 86.25% | 67.57% |
| Self-cited reread | 67.65% | 88.33% | 77.99% |

The second read improves the question average by 10.42 percentage points and the equal-document average by 9.17 points. Three documents improved, two tied. This hypothesis was designed after inspecting first-pass errors on the training pilot; it is not independent validation. The exact two-sided document sign-flip p-value is 0.25. Five clusters support descriptive, exploratory uncertainty only.

## Fixed refinement policy

The first pass receives up to five BM25-selected pages. The second pass keeps its valid cited pages in physical order. Empty or invalid citations retain the original input. The final answer is always the second output. No answer label, gold page or per-question score selects the page subset or final prediction.

The second pass presents 27 pages across 16 questions, versus 76 in the first pass. It cannot recover evidence omitted upstream. Complete input-evidence coverage falls from 14/16 to 12/16 because some self-citations omit relevant pages. The failure table and all predictions remain counted.

## Verified lineage

| Artifact | Contract or checksum |
| --- | --- |
| First-pass inference contract | `fd24e61f112be2eee0cff954cb8b6239c8ab25d14efeeac1b25599ca8dd4753f` |
| First-pass public summary SHA-256 | `05fd5b926205307cb7c5f078227a9212d1eb75372da7cd4e7c8e17a0eb912313` |
| Refinement inference contract | `b9d23cfe228e248abcefb91e71b7d57bb0822f2f6a17285f709098c911e911ea` |
| Refinement public summary SHA-256 | `545691e6d35815fd5bdf89c91cc834ae20d78060084e5817ab151f437da66913` |

Public results are in `reports/system/summary.json` and `refinement.json`, with SHA-256 sidecars. Job receipts are `attempt-3.json` and `refinement/attempt-1.json`. Source-bound loaders reject stale inference, judge or scoring lineage. The public reports use stable anonymous question/document aliases; exact questions, generations, answers and PDF content remain private.

## Runtime and cost

| Completed pass | Instance | Billable seconds | Estimated Training USD |
| --- | --- | ---: | ---: |
| First pass, attempt 3 | `ml.g6e.8xlarge` | 409 | 0.6430 |
| Reread, attempt 1 | `ml.g6e.8xlarge` | 355 | 0.5581 |
| Both completed passes | One GPU per job | 764 | 1.2012 |

The dated AWS Price List Training rate is $5.66/hour in us-west-2. These estimates exclude Studio, storage, logs, transfers, taxes, discounts and earlier attempts. They are not an invoice or an account-wide cap.

Mean generation was 6.357 seconds in the first pass and 3.768 seconds in the reread; p95 was 7.751 and 6.493 seconds. Peak allocated GPU memory was 21.48 and 20.65 GiB. The two-pass system incurs both passes. Reader telemetry excludes retrieval, provisioning and checkpoint I/O, and the first uncached question includes model load. These numbers do not establish full competition-runtime compliance.

The initial first-pass attempts were stopped during capacity acquisition. The successful third attempt used another available host size with the same single L40S GPU class. No endpoint or larger reader was needed.

## Inspect without cloud access

Read [Notebook 05](../notebooks/05_end_to_end_system_evaluation.ipynb) with its saved outputs. All notebook run controls default to false. To verify the current public contracts locally:

```bash
make system-preview
make refine-preview
make notebooks
make quality
```

None of these commands allocates a GPU. To regenerate figures, `make system-report` uses an isolated, pinned Matplotlib environment so the frozen semantic-judge dependency lock remains unchanged. It produces three SVGs, a Plotly HTML report with static fallbacks, and a checksum manifest.

## Reproduce on the configured Studio host

Use `/home/sagemaker-user/lava-aws-multilingual-docvqa` with its existing AWS role, storage configuration and Hugging Face judge access. The operator checks judge access and CPU memory before new paid inference.

```bash
make system-evaluate
make refine-evaluate
make system-report
make notebooks
make quality
```

The scoring commands reuse completed inference and verified semantic decisions. They do not create a new GPU job. The fixed research audit can also be repeated with `make research-evaluate`; complete families are reused after verification.

For intentional new inference, `make finish CHARGES=YES` operates the base condition, and `make refine-finish CHARGES=YES` operates the frozen second read. Both skip allocation when compatible inference is already complete. Refinement requires the saved first pass.

The base launcher supports `ml.g6e.2xlarge` and `ml.g6e.8xlarge`; the latter is exposed by `scripts/evaluate_system.py --instance-type`. Refinement uses `ml.g6e.8xlarge`. Each new attempt has a 1,800-second runtime cap, a server pending limit of 24 hours and an explicit $5 per-attempt estimate guard. The refinement hourly estimate ceiling is $6, with 25% contingency. These are runtime and estimate controls, not account-wide spending limits.

## Progress and recovery

Operator logs emit UTC events with stage/total elapsed time and 15-second heartbeats. SageMaker retains container logs in CloudWatch. Per-question answer checkpoints, manifests, job requests, completed receipts and operator logs persist in S3.

Repeat the same command and attempt number after a monitor interruption. A deterministic job name resolves an ambiguous create response without creating another job. Completed compatible answers are read back and independently parsed. A question interrupted before checkpoint commit may need to run again.

Failed or stopped attempts are never relaunched automatically. Inspect the recorded failure, then explicitly request a new attempt with `ATTEMPT=2 RETRY=YES` (or the next unused number). Existing completed answers are retained.

Changes to model revision, page policy, source manifest, inference implementation or dependency lock invalidate incompatible inference. Changes to the judge or scoring code invalidate stale scores. Report-only changes do not require new GPU inference. Never delete checkpoints to make a gate pass.

## Evaluation boundary

`ReaderInput` has no reference answer or gold-evidence fields. Citations must belong to supplied physical pages, not to a reference set. References enter only after inference is complete.

The primary score averages semantic answer credit and evidence-page F1 per question. Supporting views include equal-document scores, language and answer-format slices, precision/recall, input-evidence coverage, validity, abstention, failure categories and runtime. The unchanged pinned judge passes 28 development controls.

The [published LAVA formula](https://lava-workshop.github.io/#evaluation) is implemented locally. Exact organizer prompt/runtime parity, hidden-test accuracy, full-test throughput and leaderboard performance are not established. The separate [CSV export workflow](submission.md) remains an optional operator action and never automatically uploads to Kaggle.
