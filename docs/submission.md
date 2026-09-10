# Generate a submission in your notebook

The optional test exporter uses the first-pass BM25/9B configuration. The citation-guided reread in Notebook 05 is measured on the training diagnostic; no full-test reread result is claimed.

Open `notebooks/05_end_to_end_system_evaluation.ipynb` using the locked LAVA kernel.
Its **Generate and download your own submission** section runs the normal source
implementation; you do not paste an assistant-provided predictions file.

The default `RUN_PILOT`, `RUN_TEST_INFERENCE`, and `EXPORT_SAVED_TEST` controls are
all `False`. Normal notebook publication and CI additionally set
`LAVA_NOTEBOOK_PUBLICATION=1`, so even accidentally committed enabled controls cannot
start a cloud job or export private data. They display the public analysis only.

## Your two independent actions

For the 16-question development evaluation, set `RUN_PILOT=True` and
`ACKNOWLEDGE_AWS_CHARGES="YES"`. Run the control cell and the following analysis.
This reuses the existing pilot inputs and prior compatible checkpoints. It does not
recursively execute or publish the notebook that is currently running.

For the actual test CSV, set `RUN_TEST_INFERENCE=True` and explicitly acknowledge
charges. Run the controls, then the **Generate and download** section. Full test
inference processes all 624 pinned test questions across 200 PDFs with BM25-selected
pages and the pinned 9B reader. A distinct test contract and job namespace prevent
training diagnostics from being exported as test predictions. Questions, answers,
PDF contents, and generated files remain private.

Test compute is separate from the pilot: one `ml.g6e.2xlarge`, a 7,200-second runtime
limit, and a default $15 per-attempt training estimate guard. The estimate is not an
account-wide spending cap or a quote; existing Studio, storage, logging, and other
attempts are separate. The runtime limit does not guarantee that all questions
finish in a single attempt. No automatic paid retries, endpoint, or IAM changes.

After inference completes, set `RUN_TEST_INFERENCE=False` and
`EXPORT_SAVED_TEST=True` to recreate the CSV from verified saved responses without
new GPU allocation. The notebook displays a local FileLink after checking the
CSV hash and manifest. The canonical path is
`artifacts/submission/submission.csv`; in Studio's file browser, right-click the
file and choose **Download**. No CSV bytes or answers are embedded in public outputs.
Upload to Kaggle yourself, under your account and any applicable competition rules.
**This code never calls a Kaggle upload API.** Generating a structurally valid CSV
is not evidence of eligibility, official runtime compliance, or model quality.

## Resume and validate

### Recover unanswered test questions

`pipelines/submission/recovery.py` adds a separate, explicitly authorized recovery
pass after complete first-pass inference. It preserves every structurally complete
base answer. Only invalid, empty, or abstained questions receive additional model
calls. It uses the same pinned Qwen3.5-9B weights and decoding configuration.

For affected PDFs, it extracts native text and runs Tesseract OCR on every physical
page. The OCR adapter (`tesserocr==2.9.2`, Tesseract 5.5.1) and English, Japanese,
and Vietnamese language models are pinned; downloaded model bytes are verified by
SHA-256. OCR runs in eight isolated processes and commits each page separately.
BM25 ranks the combined text. The reader then considers four-page context,
adjacent pages, focused single pages, and remaining ranked page groups until it
produces a complete non-abstaining response. OCR text is labelled as fallible;
the page image remains authoritative. See the upstream
[OCR API](https://github.com/sirfz/tesserocr) and
[language model repository](https://github.com/tesseract-ocr/tessdata_fast).

The recovery contract binds the original inference checksum, source files, OCR
models, reader, and fixed search policy. Every model attempt and accepted decision
is private and immutable. The export independently reparses chosen raw responses,
checks unchanged questions and PDF sources, and applies the canonical 624-row CSV
validator. Remaining abstentions still block export. No missing answer is filled
with a placeholder. Recovery is operational completion work; its quality is not
established by the earlier 16-question development scores.

The managed entrypoint is `pipelines/submission/run_recovery.sh`. A reviewed launch
must supply `LAVA_BASE_CONTRACT`, `LAVA_BASE_INFERENCE_SHA256`,
`LAVA_RECOVERY_CONTRACT`, `LAVA_GIT_COMMIT_SHA`, `LAVA_BUCKET`, and
`AWS_DEFAULT_REGION`. Use the same one-device/36-GiB guard, a 7,200-second managed
runtime limit, and a recorded cost estimate. The worker additionally caps new model
calls at 700 and work time at 6,600 seconds. Completed attempts resume without
regeneration. The final `export.json` resides under the **recovery contract**;
its immutable CSV manifest records both inference revisions and the numbers of
preserved and recovered answers. Original manifests retain their generation-state
`uploaded_to_kaggle:false`; a real upload receipt is recorded separately.

### Targeted review after an exhausted search

`pipelines/submission/targeted.py` is a separate, explicitly operator-routed
diagnostic stage. Its private, checksum-pinned plan selects physical pages and
optional geometric detail crops after reviewing failed evidence searches. It
contains no candidate answers and does not use hidden labels. This is not the
original automated BM25 retrieval policy and must be identified separately in
any report.

Before loading the unchanged 9B reader, the stage independently reparses every
inherited base and recovery prediction and checks the original question, PDF,
OCR context, source revision, object version and checksum. Complete answers are
never overwritten. New routes and source code receive a new immutable contract.
Every attempted generation and accepted decision is saved and read back.

The stage allows at most five questions, two candidate views per question, four
pages per view, ten model calls and 1,200 seconds of work. The managed job imposes
an additional wall-clock cap. A document mismatch is recorded as an explicit
blocker, not resolved by substituting another PDF or manufacturing an answer.
Its report distinguishes inherited answers, additional answers and unresolved
IDs. It deliberately does not export a CSV or upload to Kaggle. A complete
624-row submission still requires the canonical validator and verified
provenance for every answer.

Before allocating a recovery GPU, run `scripts/verify_targeted_access.py` in a
bounded CPU job using the same SageMaker execution role and the exact staged
inference source. The probe rejects an administrator or a different role,
revalidates inherited answers, reads the pinned PDFs, prepares every routed page
and crop, checks versioned image/text reads, and verifies a private receipt write
and read-back. It makes no model calls. A successful administrator-side S3 read or
IAM simulation alone does not replace this runtime check.

The role needs `s3:GetObjectVersion` for version-pinned reads in addition to the
existing `s3:GetObject` and `s3:PutObject` submission permissions. The canonical
`infra/iam/submission.template.json` includes these actions within the submission
prefix. The first targeted attempt on September 10, 2026 UTC failed while reading
its pinned plan because that action was missing; it generated no new answers.
All 619 previously complete answers remained preserved, with five unresolved and
no exported CSV or Kaggle submission at that checkpoint.

### Resume the frozen first pass

For an explicitly reviewed hardware fallback, the canonical GPU entrypoint is
`pipelines/submission/inference.py`. With `CUDA_VISIBLE_DEVICES=0`, it requires
exactly one visible accelerator and caps PyTorch's allocator at 36 GiB before
loading the unchanged reader. The launch registers the module namespace so that
Pydantic resolves the test page schemas correctly. A fresh-interpreter regression
test checks that startup path. This memory boundary does not certify A100 runtime
equivalence. Source archives and job requests must record the actual hardware,
runtime limit, source revision, and archive checksum.
After inference succeeds, this entrypoint runs the same strict exporter and saves
the CSV plus an immutable `export.json` receipt under its private test contract.
Incomplete or invalid predictions block export. This avoids a separate manual
export step after the managed GPU job; it still never uploads to Kaggle.

Keep the same attempt number after browser disconnection. The stored launch intent,
source archive, per-document extraction, per-query ranking, selected-page images,
and every committed answer are verified and reused. A question whose checkpoint
never committed may run again. Failed/stopped managed jobs require deliberate
`ALLOW_PAID_RETRY=True` and a new attempt number; inspect the recorded failure first.

CSV validation checks exact template order, unique and complete IDs, UTF-8/list
serialization, nonempty real answers, physical page bounds, source versions, model
revision, and prediction/provenance hashes. Invalid/abstained responses remain
stored and block a misleading complete export. No placeholders, gold-page fallbacks,
or dropped failure rows are used. Every compatible inference commit is retained
in provenance across resumptions. Private bundles are persisted in S3 before the
canonical local CSV is exposed.

The same implementation is available without a notebook:

```bash
# Offline preview; no private input reads or paid resource.
uv run --frozen python scripts/prepare_submission.py --mode preview

# Deliberate full-test inference and local export. This authorizes AWS spending.
uv run --frozen python scripts/prepare_submission.py --mode test \
    --acknowledge-charges YES --attempt 1 --maximum-training-usd 15

# Reconstruct the local file from completed inference; no new GPU.
uv run --frozen python scripts/prepare_submission.py --mode export
```

The existing `--mode build --predictions ... --page-counts ... --provenance ...`
interface also accepts your own fully reviewed prediction files. It applies the
same complete-coverage and provenance checks; it does not run a model or upload.
Reset notebook controls to their safe defaults before `make notebooks` and
`make quality`. Commit only reviewed public reports and executed notebook outputs,
never `.env`, input documents, predictions, or `artifacts/`.
