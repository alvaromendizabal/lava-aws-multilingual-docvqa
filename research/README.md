# Frontier research artifacts

This folder contains public-safe aggregate evidence for the later LAVA research program.

It is intentionally separate from the six canonical executed notebooks. Those notebooks are checksum-bound to the exact source and public inputs they actually executed; later frontier work is not retroactively inserted without a legitimate rerun.

## Current public state

The published research now spans five layers:

1. **Retrieval:** lexical, multilingual dense, and page-image retrieval.
2. **Reader evaluation:** open multimodal readers under one pinned semantic contract.
3. **Reasoning/perception ablations:** self-consistency, targeted visual perception, exhaustive screening, and explicit reasoning.
4. **Heterogeneous validation:** complementary reader families evaluated with nested held-out-document selection.
5. **Reliability / systems research:** checkpoint reuse, runtime qualification, bounded failure recovery, and publication integrity.

Key current aggregate measurements:
- strongest supplied-evidence reader: **87.02% local LAVA**
- measured two-pass incumbent: **77.99%**
- original heterogeneous route, small-panel estimate: **82.68%**
- strongest measured recall@5 retrieval policies: **98.44%**

All local quality measurements use a reused **16-question / five-PDF** panel. The 82.68% route score does not apply to the later recovered full-test composite. Best official scores are **0.49 public / 0.53 private**; the later diagnostic scored **0.48 / 0.49**. Current structural acceptance is **622/624**, not an accuracy count.

## Files

- [submission_closeout.json](submission_closeout.json) — sanitized official outcomes and completion scope
- [../docs/submission_closeout.md](../docs/submission_closeout.md) — provenance and public verification command
- [frontier_status.json](frontier_status.json) — machine-readable current research state
- [frontier_results.csv](frontier_results.csv) — aggregate retrieval, reader, reasoning, and validation measurements
- [../docs/frontier_research_update.md](../docs/frontier_research_update.md) — employer-facing interpretation and research decisions
- [../docs/portfolio.md](../docs/portfolio.md) — concise portfolio review
- [reliability_frontier.json](reliability_frontier.json) — public-safe reliability mechanisms and failure taxonomy
- [../docs/reliability_recovery.md](../docs/reliability_recovery.md) — employer-facing recovery engineering case study

## Reproduction boundary

The public artifacts include model/revision identities, aggregate metrics, evaluation boundaries, and promotion logic. They exclude private questions, reference answers, raw generations, private test predictions, credentials, exact cloud locations, return bundles, and private routing rules.

The original document-disjoint route remains a development result. Official outcomes and completion checks are reported separately; they limit the conclusions that follow from the small-panel experiments.
