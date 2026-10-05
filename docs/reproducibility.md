# Reproducibility guide

This repository is deliberately **semi-reproducible**: the public research contract, source code, frozen configurations, executed notebooks, aggregate evidence, and quality gates are reproducible; private competition inputs, private predictions, exact routing rules, credentials, and cloud-return bundles are not published.

The design goal is to let a reviewer verify the engineering and scientific process without releasing competition-sensitive artifacts.

## Reproduction levels

### Level 1 — inspect the executed evidence

No account, GPU, or model download is required.

Review:
- the six executed notebooks in `notebooks/`;
- frozen configuration files in `configs/`;
- aggregate reports in `reports/`;
- public-safe frontier artifacts in `research/`.

Each canonical notebook has a publication manifest that binds source content, analysis inputs, output bytes, and execution state.

### Level 2 — run public CPU checks

Use Python 3.12 and the frozen uv environment:

```bash
uv sync --frozen --group judge
make quality
make notebooks
```

These commands exercise the public software-quality and notebook-publication contracts. Existing verified notebook outputs are reused when their source/input identity still matches.

Useful targeted checks:

```bash
uv run --frozen pytest -q
uv run --frozen ruff check .
uv run --frozen mypy src
```

### Level 3 — reproduce public model evaluation

Model-backed evaluation requires access to the pinned open-model checkpoints documented in `configs/`.

Start with:

```bash
make evaluation-check
make evaluation-preview
```

The check identifies missing model access or runtime prerequisites before expensive work begins. The evaluation stack uses frozen model revisions, deterministic scoring settings where supported, and contract-specific caches.

### Level 4 — private full-test workflow

The private competition workflow is intentionally **not** reproduced from public Git alone.

Excluded inputs include:
- private test predictions and raw generations;
- exact competition routing logic;
- private source documents where redistribution is inappropriate;
- private cloud paths, credentials, and return bundles;
- durable model/runtime caches.

The public repository still exposes the reusable architecture: typed prediction schemas, retrieval, evaluation, checkpointing, observability, and submission validation.

## Reproducibility contract

A result is treated as reproducible only when its identity includes the relevant combination of:

- source revision or source hash;
- model ID and pinned model revision;
- input/data identity;
- prompt or inference-policy identity;
- evaluation/judge contract;
- dependency/runtime contract where material;
- output checksum or immutable artifact identity.

This prevents a filename such as `results.json` from being treated as sufficient provenance.

## Notebook policy

The notebooks are both source and publication artifacts.

The workflow:
1. computes the notebook source identity;
2. computes identities for declared analysis inputs;
3. reuses a verified publication when those identities match;
4. otherwise executes in a content-addressed staging directory;
5. requires successful completion and unchanged source content;
6. publishes atomically;
7. records a completion manifest.

A failed refresh does not destroy a previously verified notebook.

## Resumable GPU work

Private GPU execution uses the same public engineering principles:

- deterministic run identities;
- per-question checkpoints;
- immutable model and input contracts;
- heartbeat and progress reporting;
- bounded retries and hard-stop windows;
- failure bundles that preserve completed work;
- process cleanup and resource telemetry;
- checkpoint compatibility checks before reuse.

The public source shows the abstractions and validation logic; private competition outputs stay outside Git.

## Expected public verification

A successful public review should establish that:

- notebooks are readable and executed;
- reported aggregate metrics match published reports;
- evaluation boundaries are explicit;
- source code is linted, typed, and tested;
- private artifacts are excluded from Git;
- results are not silently promoted across incompatible evaluation scopes.

## What is intentionally not guaranteed

The repository does not claim:
- bit-for-bit reproduction of organizer infrastructure;
- organizer-server-identical semantic judging;
- free access to all third-party model weights;
- reproduction of private competition outputs from public data alone.

Those limitations are documented rather than hidden.
