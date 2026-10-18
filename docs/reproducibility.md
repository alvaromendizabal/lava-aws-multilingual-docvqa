# Reproduce the public reference

## No account or model download

Python 3.11+ runs the supplied-response parser and closeout arithmetic. Add Node.js 22 or 24 for the browser engine and export replay.

```bash
python3 -S review/run.py --output /tmp/lava-review --with-node
python3 -S review/run.py --output /tmp/lava-review --with-node --check
```

Open the generated `index.html`. The second command verifies the saved bundle without rewriting it. On Windows, use a writable directory outside the checkout.

```bash
python3 examples/run_demo.py
python3 examples/verify_submission_closeout.py
node tools/replay_public_demo.mjs --output /tmp/lava-export
node tools/replay_public_demo.mjs --output /tmp/lava-export --check
```

These execute the actual retained public components. They do not train a neural reader or reproduce private full-test predictions.

## Locked source and notebook checks

Use the committed Python 3.12 environment and lock. The notebook replays historical aggregate results in a fresh kernel; no cloud or model call is made.

```bash
uv sync --frozen --group judge
uv run --frozen python -m ipykernel install --user --name lava
make notebooks
make quality
```

The gate runs lint, formatting, type checks, tests, compilation and checksum-bound notebook validation. [The integrated notebook](../notebooks/05_end_to_end_system_evaluation.ipynb) keeps all 16 reported development questions visible.

## Boundary

Full model-backed reruns require authorized source documents, pinned model artifacts and an independently provisioned environment. Private full-test inputs, generations and routing rules are excluded. Project AWS spaces and buckets are retired; no public command depends on those deleted resources. Published local and external metrics have different populations and cannot be interchanged. [Closeout](submission_closeout.md).
