# Answers that can point to their evidence.

**Alvaro Mendizabal · Document intelligence / retrieval / reliable ML**

A multilingual document-QA research system that separates finding evidence, reading it, and validating the answer. The engineering story is inspectable: explicit contracts, controlled comparisons, and recoverable execution.

[Project overview](../README.md) · [Printable technical brief](index.html) · [Review configuration](project.json)

## Follow an engineering decision to its evidence

| Focus | Engineering decision | Inspect |
|---|---|---|
| Applied AI | **Separate format validity from factual correctness.** Parse structured answers, constrain cited physical pages, and demonstrate that a schema-valid answer can still be wrong. | [Implementation](../src/lava/readers/structured_output.py) · [Tests](../tests/unit/test_structured_output.py) · [Evidence](../examples/fixture.json) |
| Data science | **Diagnose retrieval before blaming the reader.** Evaluate page retrieval separately from supplied-evidence reading and end-to-end answer quality. | [Implementation](../src/lava/retrieval/pipeline.py) · [Tests](../tests/unit/test_retrieval_pipeline.py) · [Evidence](../reports/retrieval/summary.json) |
| ML engineering | **Make interrupted work recoverable.** Use durable reader checkpoints and artifact identities so completed questions need not be repeated after a failure. | [Implementation](../src/lava/readers/checkpoints.py) · [Tests](../tests/unit/test_private_artifacts.py) · [Evidence](../docs/reliability_recovery.md) |
| Data science | **Preserve negative and incomplete results.** Keep official, development, structural-coverage, and factual-correctness claims separate in a machine-checkable closeout. | [Implementation](../examples/verify_submission_closeout.py) · [Tests](../tests/unit/test_submission_closeout_public.py) · [Evidence](../reports/portfolio/closeout.json) |
| Data science | **Test the failure boundary, not only the happy path.** Execute visible authored challenges against the shipped engine; separate retrieval or decision coverage from successful outcomes and retain every failure. | [Implementation](../tools/evaluate_public_challenges.mjs) · [Tests](../tools/test_public_challenges.mjs) · [Evidence](../review/public_challenges.json) |
| ML engineering | **Make an exported answer independently replayable.** Rerun retrieval and validate separate human edits, source passages and receipts against the actual browser implementation. | [Implementation](../tools/replay_public_demo.mjs) · [Tests](../tools/test_replay_public_demo.mjs) · [Evidence](../docs/DEMO_REPLAY.md) |

## One command, an inspectable result

From a checkout with Python 3.11 or newer and Node.js 24.19.0:

```bash
python3 -S review/run.py --output /tmp/ml-review --with-node
python3 -S review/run.py --output /tmp/ml-review --with-node --check
```

Open `/tmp/ml-review/index.html` for the rendered brief. Logs, `receipt.json` and `manifest.json` record the actual checks, source identities and output hashes. On Windows, choose a writable directory outside the checkout. Repeating the command reuses a verified result; changed sources or damaged evidence fail explicitly.

Omit `--with-node` to run only the Python checks; omitted browser checks are never reported as passed. CI runs both runtimes on Python 3.11 and 3.13.

## Verification scope

The public checks execute authored response fixtures through the actual parsing and citation validators, and recompute aggregate closeout consistency. The optional browser checks exercise lexical retrieval on synthetic documents. These are different from model-backed question answering.

Full-test documents, generated answers, exact routing rules, private checkpoints, model caches, and operational cloud locations remain excluded. The two unresolved historical answers remain unresolved.

[Case study](../docs/case_study.md) · [Reproducibility levels](../docs/reproducibility.md) · [Technical review](../docs/reviewer_guide.md) · [Challenge results and failure analysis](../docs/PUBLIC_CHALLENGES.md) · [Export replay](../docs/DEMO_REPLAY.md) · [Repository map](../docs/repository_guide.md)
