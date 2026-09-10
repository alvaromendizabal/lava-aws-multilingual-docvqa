# Generate a submission in your notebook

**Verified project state, September 10, 2026:** 550 first-pass answers and 69
OCR-recovered answers and three targeted recoveries are preserved, giving 622 of
624 structurally complete answers. The targeted run finished. Two questions
have suspected question/PDF mismatches; these are not organizer-confirmed data
corrections. No complete CSV or verified Kaggle upload receipt is available.

**Source audit, September 10, 2026:** a model-free follow-up independently
verified all 200 frozen test-document extraction checkpoints and searched their
4,698 physical pages. It also reverified the 63 saved OCR pages covering the two
assigned PDFs, including byte hashes, object versions and complete page coverage.
No supporting source correction was established. Eight pages elsewhere matched
the financial query's two anchor terms; none establishes an authoritative remap.
No native-text page matched both flood-table anchor terms. The corpus contains
798 pages with no native text, so this is not an exhaustive visual search.

The assigned PDFs' checksums and page counts still match the frozen inputs.
Their visible subject matter remains inconsistent with the questions. This is
evidence for suspected source mismatch, not proof of a dataset error or evidence
that the questions intentionally lack answers. The current Kaggle data description
still maps each `file_id` to its corresponding PDF. No correction was found in
the inspected discussion list and `data`/`question` searches. A fresh browser
download did not complete, so no new Kaggle-download checksum is claimed.

Coverage remains **622/624**, with zero new model calls, OCR runs or GPU jobs in
this follow-up. No predictions or immutable inference receipts were changed.
Completion needs valid evidence in the assigned sources or authoritative
clarification of the source mapping or unanswerable-question policy. No test questions or row-level predictions are included in this repository.
See [the source-audit protocol](DATA_AUDIT_PROTOCOL.md#source-consistency-follow-up).

### Independent source-association audit

A further read-only audit at **2026-09-10 19:13 UTC** searched all **1,739 saved
OCR checkpoints across 48 documents**, including OCR for 639 of the 798
native-textless test pages. All requested objects were read; record sizes,
versions, S3 checksums and checksum metadata were reconciled. Candidate bodies
were independently checked against their returned SHA-256 checksums.

The audit also checked the raw inventory against all 200 frozen test-document
IDs, found no duplicate PDF contents, searched native text in all five supplied
training PDFs (74 pages), and inspected page overviews of all 63 pages in the two
assigned PDFs. Neither assigned PDF has an embedded attachment. Each unresolved
question has three sibling questions whose subject matter agrees with its
assigned PDF, supporting an isolated question/source inconsistency rather than a
wholesale PDF swap.

Ten saved OCR pages matched both financial anchor terms. Review identified
special-account statements, transfers and health-insurance finances; no
supporting source replacement was established. No saved OCR page matched
`water-defense law`, `inundation assumption` or `rainwater flooding` in the
Japanese terms specified in the private audit. Four additional scanned pages in
three relevant hazard-map PDFs were visually reviewed and did not contain the
requested designation-status table.

**Remaining-page review, September 10 at 19:55 UTC:** all 155 previously
unreviewed native-textless pages were rendered from 34 exact-version PDFs. Each
PDF matched its frozen SHA-256 and page count. All 26 contact sheets were visually
screened and reconciled to the 155-page queue, with no omitted or duplicate pages.
Eleven pages were blank. The remaining pages contained questionnaires, transport
schedules, construction forms, laboratory water-quality tables, education and
health material, covers and corporate proceedings. No supporting revenue chart
or flood-designation table was established. This review recovered **zero** new
answers; **622/624** complete predictions and both abstentions remain preserved.
Preparation took 61.93 seconds on local CPU; that excludes interactive visual
review time. No new OCR, deployed-reader calls or paid training jobs were used.
Private page-review outcome SHA-256:
`1198ab05c2abe6ecbc6a1fca0de11d1b67415504a520f33605977fd9c191b283`.

**Limits:** the remaining queue is now visually screened, but contact-sheet
inspection is not full OCR or character-by-character transcription. Existing OCR
can corrupt or omit text. This is not a claim that every pixel in the corpus was
read, that the dataset has confirmed errors, or that the two questions are
intentionally unanswerable. Cross-document topic matches are diagnostic leads,
not verified source substitutions.

**Decision:** preserve all **622** complete predictions and both abstentions.
This audit recovered **zero** additional answers and used **zero** new reader
calls or paid training jobs. Repeating inference on the same unrelated assigned
sources is not justified by this evidence. No CSV was exported or uploaded.
Private outcome SHA-256:
`85a8388009c5c8074eb7740d80c03696fe64a78b857f854638974087b253a77c`.

A submission remains an acceptance requirement. The project's current **strict
export policy** requires all 624 answers to be non-abstaining, with exact template
IDs/order, valid answer serialization and evidence-page bounds. This is a project
quality gate; it is **not evidence that Kaggle prohibits abstentions**. The
inspected competition instructions do not establish how empty-answer/evidence
rows are accepted and scored. Any best-effort export must explicitly identify its
abstention policy and coverage instead of being presented as 624 answered
questions. The training feature ablations in Notebook 04 do not fill missing
test answers or certify their accuracy.

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

The permission was corrected without expanding the submission resource prefix.
At 03:36 UTC on September 10, 2026, a bounded CPU job completed under the actual
SageMaker execution role: it revalidated all 619 inherited answers, read the three
routed PDFs, verified eight page views including the detail crop, and saved and
read back its access receipt. It made zero model calls. The private receipt's
SHA-256 is `2f58c4ec476616a15efd4e7651bf4009fbdbd22e664c1757a987d97001d80ae1`.
The inference revision remains `1fd9ec645e7cf8697ee163b725da645e369e19be`;
the executed access probe was committed at
`6d8ce7ffe511b497049bcc570e7ca6cba1e2ce23`. These are access-verification results,
not recovered answers.

The GPU retry completed at **04:36:51 UTC on September 10, 2026**. It generated
three new complete responses in three model calls and preserved all 619 earlier
answers. Independent replay of `check_targeted` and `validate_prediction` matched
the saved raw generations, original questions, PDF sources, planned routes and
actual inference revision. The worker reported 67.051 seconds; SageMaker recorded
315 billable seconds including setup. These are different timing scopes.

The resulting coverage is **622/624**. The two suspected question/document
mismatches still require valid documentary evidence or authoritative source
clarification. The run exported no CSV and made no Kaggle upload. The sanitized
[recovery receipt](../reports/submission/targeted_recovery.json) records the actual
outcome. The result is structural completeness, not measured answer accuracy.

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
