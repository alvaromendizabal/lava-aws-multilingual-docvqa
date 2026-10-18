# Repository guide

Start with [Document Desk](../public-demo/index.html), the [case study](case_study.md), and [reproducibility levels](reproducibility.md). The public demo is a complete lexical-retrieval example; the research stack adds separate model-backed reading and evaluation components.

| Directory | Maintained purpose |
|---|---|
| `public-demo/` | Authored multilingual documents, lexical retrieval, passage extraction, citation checks and the shared export builder |
| `examples/` | Small Python contract examples and visible challenge questions |
| `src/lava/` | Public retrieval, reader interfaces, evaluation, checkpointing and observability |
| `tests/`, `tools/` | Research regression tests, browser behavior, authored challenge evaluation and export replay |
| `reports/`, `research/` | Aggregate results, execution identities, figures and recorded research decisions |
| `notebooks/` | Six canonical executed notebooks and their reading guide |
| `configs/`, `pipelines/`, `scripts/` | Reusable evaluation and bounded execution entry points |
| `review/` | Reproducible technical brief and challenge report |
| `infra/` | Public IAM template with configurable account-specific values |

## Reproducible public review

```bash
python3 -S review/run.py --output /tmp/document-review --with-node
python3 -S review/run.py --output /tmp/document-review --with-node --check
```

This executes the declared Python and JavaScript checks, including the actual export-replay tests. It produces an HTML brief, logs and a source-bound receipt. The [model-backed reproduction guide](reproducibility.md) documents the broader environment separately.

## Closeout cleanup

The unreferenced one-line historical Kaggle CLI version stamp was removed. Runtime versions remain in pinned dependency files and execution evidence. The original and reviewed export formatting now share one implementation used by the browser and replay, eliminating a separate copy of the payload-building logic.

Recorded evaluation inputs, checkpoint summaries and notebook manifests remain because the verification tools consume them. The identical latest-reader summary is a documented latest-run interface used by analysis code; deleting that file would break a working path. Private documents, generations, exact routing rules, model weights and cloud object locations are excluded as documented in the [closeout](closeout.md).
