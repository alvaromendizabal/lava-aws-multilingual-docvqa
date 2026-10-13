# Answers that can point to their evidence.

**Alvaro Mendizabal · Document intelligence / retrieval / reliable ML**

A multilingual document-QA research system that separates finding evidence, reading it, and validating the answer. The engineering story is inspectable: explicit contracts, controlled comparisons, and recoverable execution.

[Project overview](../README.md) · [Printable inspection brief](index.html) · [Review configuration](project.json)

## Review the engineering, not just the headline

| Focus | Decision to inspect | Follow the proof |
|---|---|---|
| Applied AI | **Separate format validity from factual correctness.** Parse structured answers, constrain cited physical pages, and demonstrate that a schema-valid answer can still be wrong. | [Code](../src/lava/readers/structured_output.py) · [Tests](../tests/unit/test_structured_output.py) · [Evidence](../examples/fixture.json) |
| Data science | **Diagnose retrieval before blaming the reader.** Evaluate page retrieval separately from supplied-evidence reading and end-to-end answer quality. | [Code](../src/lava/retrieval/pipeline.py) · [Tests](../tests/unit/test_retrieval_pipeline.py) · [Evidence](../reports/retrieval/summary.json) |
| ML engineering | **Make interrupted work recoverable.** Use durable reader checkpoints and artifact identities so completed questions need not be repeated after a failure. | [Code](../src/lava/readers/checkpoints.py) · [Tests](../tests/unit/test_private_artifacts.py) · [Evidence](../docs/reliability_recovery.md) |
| Data science | **Preserve negative and incomplete results.** Keep official, development, structural-coverage, and factual-correctness claims separate in a machine-checkable closeout. | [Code](../examples/verify_submission_closeout.py) · [Tests](../tests/unit/test_submission_closeout_public.py) · [Evidence](../reports/portfolio/closeout.json) |

The linked tests are review entry points, not a claim that this compact command executes the full research suite. The project’s canonical reproducibility guide describes the broader environment and tests.

## One command, an inspectable result

From a checkout with Python 3.11 or newer:

```bash
python3 -S review/run.py --output /tmp/ml-review
```

Open `/tmp/ml-review/index.html`. The directory contains an accessible HTML brief, `receipt.json`, `manifest.json`, and logs for each executed command. No Python package installation, model download, cloud credentials, or external dataset is required. On Windows, substitute a writable directory outside the checkout for `/tmp/ml-review`.

To execute the browser engine checks as well, install Node.js and use a separate output directory:

```bash
python3 -S review/run.py --output /tmp/ml-review-node --with-node
python3 -S review/run.py --output /tmp/ml-review-node --with-node --check
```

The default path does not execute Node checks and never labels them as passed. CI runs both the Python and browser checks on Python 3.11 and 3.13. It retains the exact review artifacts with the workflow run.

| Check | Runtime | Inspected entry point |
|---|---|---|
| `citation-contract` | python | [`examples/run_demo.py`](../examples/run_demo.py) |
| `closeout-arithmetic` | python | [`examples/verify_submission_closeout.py`](../examples/verify_submission_closeout.py) |
| `browser-evidence` | node | [`tools/test_public_demo.mjs`](../tools/test_public_demo.mjs) |

The optional Node path also recomputes the [public challenge evaluation](../docs/PUBLIC_CHALLENGES.md), including its known failures. A passed verification means the result and its source identities reproduce; it does not mean every authored case succeeds.

## What is actually reproduced

The public checks execute authored response fixtures through the actual parsing and citation validators, and recompute aggregate closeout consistency. The optional browser checks exercise lexical retrieval on synthetic documents. These are different from model-backed question answering.

A repeat invocation verifies the saved bundle rather than rerunning completed checks. `--check` performs verification only. A changed linked source file, missing evidence, changed artifact, failed command, invalid path, or mismatched check mode causes a nonzero exit. Use a fresh output directory after an intentional source change; stale evidence is never silently relabeled as current.

Each command has a default 90-second deadline, configurable with `--timeout`. The runner records UTC time, interpreter version, elapsed command times, command identities, and SHA-256 hashes for its configuration, template, invoked entry points, linked implementations, tests, and evidence. Failed runs keep diagnostic logs in a separately named failed-run directory and never produce a success receipt.

The fingerprint also binds the explicit `source_dependencies` list, including browser assets, authored challenge inputs and result files. Changing one invalidates reuse. The fingerprint covers this curated review surface, **not every transitive dependency or every repository file**. Checksums establish consistency, not independent authentication of historical scores. HTML links follow the current default branch; the receipt is the record of the local bytes inspected. The static page is explicitly an inspection brief, not an execution receipt.

## Public/private boundary

Full-test documents, generated answers, exact routing rules, private checkpoints, model caches, and operational cloud locations remain excluded. The two unresolved historical answers remain unresolved.

This review adds no training result and changes no historical score, selection decision, notebook output, source attribution, or license. A synthetic demonstration, an aggregate evidence check, public reference training, and exact private-model replay are different reproducibility levels.

## Verify the review kit itself

```bash
python3 -B -S -m unittest discover -s review -p test_run.py -v
python3 -B -S -O -m unittest discover -s review -p test_run.py
python3 -S review/run.py --check-page
```

After an intentional configuration or template change, regenerate the static brief with `python3 -S review/run.py --render`, inspect it, then run the checks above. The kit needs no JavaScript or network access to render its HTML. It uses semantic headings, keyboard-visible links, a skip link, responsive layouts, and print styling.
