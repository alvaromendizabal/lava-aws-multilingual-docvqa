# Frontier research artifacts

This folder contains aggregate, public-safe evidence for the later LAVA frontier research.

It is intentionally separate from the six canonical executed notebooks. The notebook publication pipeline binds those notebooks to the exact source/input state they actually executed, so later frontier research is not retroactively inserted into them without a legitimate rerun.

## Files

- [frontier_status.json](frontier_status.json) — machine-readable current research state
- [frontier_results.csv](frontier_results.csv) — aggregate retrieval and reader measurements
- [../docs/frontier_research_update.md](../docs/frontier_research_update.md) — employer-facing interpretation and research decisions

## Reproduction boundary

The public artifacts include model/revision identities, aggregate metrics, evaluation boundaries, and promotion logic. They exclude private questions, reference answers, raw predictions, credentials, private cloud locations, return bundles, and exact internal prompts.

The current strongest reader remains the pinned Qwen3.5-9B result on the 16-question oracle-evidence development diagnostic. The one-shot Qwen3.6-27B branch was not promoted after apples-to-apples semantic rescoring.
