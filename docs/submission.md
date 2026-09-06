# From verified reader results to a LAVA submission

The reader pilot is complete. Its 16 questions are all supplied training labels,
covering five documents. The separate test set has 624 questions from 200 PDFs,
with 587 Japanese and 37 Vietnamese questions. Test answers are unavailable.
Do not turn test predictions or leaderboard feedback into a training-label source.

The current reader sees the known evidence pages. A competition prediction must
start from the full PDF and find its evidence. The existing 4B and 9B results
remain useful reader baselines, but are not competition test predictions.

## Current competition status

The [organizer schedule](https://lava-workshop.github.io/) lists May 31, 2026 as
the normal closing date. On September 6, 2026, the public
[Kaggle competition page](https://www.kaggle.com/competitions/lava-challenge-2026)
showed a **Late Submission** button, disabled in the signed-out view.
Account eligibility has not been verified. Do not claim a leaderboard submission
or score until Kaggle actually accepts and evaluates a file. This community
competition lists Kudos and does not award Kaggle points or medals.

The organizer specifies open models/data, deterministic seeds, a Dockerfile for
reproducibility verification, and full inference within two hours on one A100
40GB GPU. The pilot's generation-only times do not verify the end-to-end budget.
New competition-specific datasets also have publication and announcement rules;
review those before adding or distributing external training material.

## The next model experiment

1. Freeze a full-document page-retrieval baseline using only the PDF and question.
   Evaluate evidence recall at k, all-evidence coverage, MRR, MAP and nDCG on the
   labeled training documents. Keep failed or textless pages in coverage reports.
2. Compare a multilingual text-retrieval baseline with an appropriate visual
   retrieval method for scanned pages, tables and figures. Use separate public
   labeled development data and document-isolated evaluation when expanding
   model development; the 624 hidden-label test questions cannot provide this.
3. Reuse the existing reader implementation with retrieved pages. Compare the
   resulting semantic answer, grounding and combined scores against the oracle
   baseline. This measures the performance lost when evidence must be found.
4. Freeze one complete inference configuration, generate all 624 test predictions
   with durable per-question checkpoints, and measure complete runtime and GPU
   memory. Preserve the existing 4B and 9B experiments. A 27B comparison is optional.
5. Build and validate the CSV below, confirm authenticated late-submission
   availability, and upload the reviewed file. No upload is performed by this code.

These steps describe outstanding model work. Retrieval, complete test inference,
organizer-hardware verification and an accepted submission are not yet completed.

## Inspect and verify the submission inputs

The September 6 live check verified both pinned source files and all 624 IDs,
then failed while archiving its log: the Studio execution role lacks write access
to `experiments/submissions/*`. Input caching succeeded and remains reusable.
The IAM change is pending explicit approval; do not run the full check/build
expecting successful archival until it is applied and verified.

The reviewable policy is
[`infra/iam/submission.template.json`](../infra/iam/submission.template.json).
Replace the literal `${S3_BUCKET}` placeholder with the configured project bucket
before applying it as `LavaSubmissionS3Access` on the existing Studio execution
role. Its only statement permits `s3:GetObject` and `s3:PutObject` for that
bucket's `experiments/submissions/*` objects. It adds no delete, public-access,
raw-data write, compute, or unrelated-bucket permission. Existing source-read
permissions remain in place. No trust policy change is proposed.

The rendered policy passed AWS Access Analyzer with no findings. IAM simulation
allowed its two intended object actions and denied deletion and access to raw
inputs or an unrelated bucket under this policy. Automatic approval review
rejected applying the new permission without explicit user approval; the role
has not been changed. A failed log upload now produces a local error event and
a nonzero exit, while preserving an earlier input error if one occurred.

The IAM policy generator could not be installed through the available network.
The minimal action mapping was checked against the
[AWS S3 permission reference](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-with-s3-policy-actions.html),
then validated and simulated through AWS before proposing it.

The unaffected Studio validation completed: both cached inputs were reused with
zero S3 downloads, and notebook 03 executed in 1.080 seconds with no progress-widget
warning. Its executed copy and validation logs remain on the Studio EBS volume.
The [public validation record](../reports/submission/validation.json) distinguishes
these completed checks from the blocked S3 log-archival check.

From the repository root:

```bash
make submission-preview
make submission-check
```

Preview is offline. Check reads the exact S3 versions pinned in
`configs/submission.json`, verifies SHA-256 checksums, CSV columns, unique IDs,
template/test coverage, document counts, answer formats and languages. It writes
`reports/submission/readiness.json` with explicit outstanding requirements.

Inputs are cached under `artifacts/submission/inputs/`. A valid cached file is
reused; a missing or corrupted copy is restored from the pinned S3 source. The
source files already remain durable in S3. Every attempt has a distinct JSONL log
with UTC timestamps, total elapsed time, stage events and 15-second heartbeats.
Check and build copy their completed or failed attempt logs to S3. A rerun does
not redo reader inference, and no new GPU job is launched.

## Prediction and provenance contracts

The future test runner must supply a private JSONL file containing exactly one
record per test ID:

```json
{"question_id":"synthetic-example","answer":"[\"item one\",\"item two\"]","evidence_pages":[2,5]}
```

This example is documentation only, not a real prediction. The `answer` field is
a string. List answers contain a JSON or Python list literal; their item order
and repeated items are preserved. Scalar answers retain Unicode, commas, quotes
and newlines through proper CSV quoting. Pages must be distinct positive integers
within that question's PDF page count. The exporter does not invent answers,
silently fill missing rows, or substitute template placeholder values.

Provide `page_counts.json` as a mapping from each test `file_id` to its verified
positive PDF page count. Provide `provenance.json` with these fields:

| Field | Required value |
| --- | --- |
| `split` | `test` |
| `evidence_scope` | `retrieved_full_document_pages` |
| `test_csv_sha256` | Pinned test CSV SHA-256 from the submission configuration |
| `question_count` | 624 |
| `model_id` | The actual public model identifier |
| `model_revision` | Actual 40-character model commit |
| `code_commit` | Actual 40-character inference code commit |
| `predictions_sha256` | SHA-256 of the exact prediction JSONL bytes |
| `page_counts_sha256` | SHA-256 of `lava.evaluation.semantic.encode(page_counts)` |

This record binds the files and declares their origin. It is not independent
proof of model quality or hardware compliance; retain the actual inference logs,
checkpoint manifest and measurements as evidence. Oracle or training provenance
is rejected, as are missing, duplicate or extra prediction IDs.

After the actual runner has created those three files:

```bash
uv run --frozen python scripts/prepare_submission.py --mode build \
  --predictions artifacts/test_inference/predictions.jsonl \
  --page-counts artifacts/test_inference/page_counts.json \
  --provenance artifacts/test_inference/provenance.json
```

The command verifies all rows, restores exact template order and independently
validates the resulting CSV. Its exact columns are `id,answer,evidence_page_number`.
Schema validity does not mean model quality or organizer runtime has been verified.

The CSV is written privately to a content-addressed S3 bundle. Its immutable
manifest is written last and serves as the completion record. Conditional writes
prevent overwriting earlier work. If interrupted after the CSV write, an identical
rerun verifies and reuses it before finishing the manifest. Only then is the
complete bundle materialized under `artifacts/submission/<bundle-id>/`.
Do not commit private predictions or the submission CSV to the public repository.

## Validation

Tests cover Unicode/CSV quoting, exact template ordering, list semantics, bad or
missing IDs, page boundaries, oracle provenance rejection, source hashes,
corrupted caches, interrupted S3 publication, conflicting immutable objects,
closed streaming bodies and persisted terminal log events. Model quality and
real A100 runtime require actual inference; synthetic exporter tests do not
claim to establish them.
