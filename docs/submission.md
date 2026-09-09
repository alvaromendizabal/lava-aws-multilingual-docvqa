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

For an explicitly reviewed hardware fallback, the canonical GPU entrypoint is
`pipelines/submission/inference.py`. With `CUDA_VISIBLE_DEVICES=0`, it requires
exactly one visible accelerator and caps PyTorch's allocator at 36 GiB before
loading the unchanged reader. The launch registers the module namespace so that
Pydantic resolves the test page schemas correctly. A fresh-interpreter regression
test checks that startup path. This memory boundary does not certify A100 runtime
equivalence. Source archives and job requests must record the actual hardware,
runtime limit, source revision, and archive checksum.

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
