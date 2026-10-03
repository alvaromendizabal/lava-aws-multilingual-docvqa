# Frontier research artifacts

This folder contains public-safe aggregate evidence for the later LAVA research program.

It is intentionally separate from the six canonical executed notebooks. Those notebooks are checksum-bound to the exact source and public inputs they actually executed; later frontier work is not retroactively inserted without a legitimate rerun.

## Current public state

The published research now spans four layers:

1. **Retrieval:** lexical, multilingual dense, and page-image retrieval.
2. **Reader evaluation:** open multimodal readers under one pinned semantic contract.
3. **Reasoning/perception ablations:** self-consistency, targeted visual perception, exhaustive screening, and explicit reasoning.
4. **Heterogeneous validation:** complementary reader families evaluated with nested held-out-document selection.

Key current aggregate measurements:
- strongest supplied-evidence reader: **87.02% local LAVA**
- measured two-pass incumbent: **77.99%**
- validated heterogeneous routed challenger: **82.68%**
- strongest measured recall@5 retrieval policies: **98.44%**

## Files

- [frontier_status.json](frontier_status.json) — machine-readable current research state
- [frontier_results.csv](frontier_results.csv) — aggregate retrieval, reader, reasoning, and validation measurements
- [../docs/frontier_research_update.md](../docs/frontier_research_update.md) — employer-facing interpretation and research decisions
- [../docs/portfolio.md](../docs/portfolio.md) — concise portfolio review

## Reproduction boundary

The public artifacts include model/revision identities, aggregate metrics, evaluation boundaries, and promotion logic. They exclude private questions, reference answers, raw generations, private test predictions, credentials, exact cloud locations, return bundles, and private routing rules.

The current research frontier is a **document-disjoint heterogeneous routed challenger**, not a private test-result dump. The public value is the experiment design and validation evidence.
