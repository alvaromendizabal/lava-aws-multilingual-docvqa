# Frontier research artifacts

This folder contains aggregate, public-safe evidence for the later LAVA research program.

It is intentionally separate from the six canonical executed notebooks. Those notebooks remain checksum-bound to the source/input state they actually executed; later frontier work is documented here until a legitimate notebook refresh is warranted.

## Files

- [frontier_status.json](frontier_status.json) — machine-readable current research state
- [frontier_results.csv](frontier_results.csv) — aggregate retrieval, reader, reasoning, and routed-system measurements
- [../docs/frontier_research_update.md](../docs/frontier_research_update.md) — chronological research decisions
- [../docs/heterogeneous_routing_update.md](../docs/heterogeneous_routing_update.md) — validated multimodel routing and inference-engineering case study

## Current public result

The original two-pass Qwen system measures **77.99% local LAVA** on the supplied development panel.

Later work tested larger readers, self-consistency, targeted visual perception, exhaustive screening, explicit reasoning, and a genuinely different Gemma reader family. The strongest validated development challenger is a **heterogeneous routed system at 82.68% local LAVA under nested held-out-document evaluation**, improving two held-out documents and regressing none.

The routing result is published as an aggregate research finding. The exact competition routing implementation remains private.

## Reproduction boundary

Public artifacts include:

- model/revision identities where appropriate;
- aggregate metrics;
- evaluation boundaries;
- held-out-document validation design;
- experiment promotion/kill decisions;
- systems architecture;
- reproducibility and recovery contracts.

They exclude:

- private questions and reference answers;
- raw generations;
- test predictions;
- credentials;
- private cloud object locations;
- exact internal prompts;
- exact competition routing/recovery heuristics;
- execution return bundles.

The repository is designed to demonstrate rigorous ML experimentation and systems engineering without publishing a turnkey competition solution.
