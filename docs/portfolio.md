# Portfolio overview

LAVA demonstrates ownership of an applied document-AI project from problem framing through AWS execution and honest evaluation of its limits.

| Capability | Concrete evidence | Review |
| --- | --- | --- |
| Multimodal retrieval | Lexical, multilingual dense, and page-image methods evaluated separately | [Notebook 04](../notebooks/04_evidence_retrieval.ipynb) |
| Vision-language reading | Comparable readers, structured output, and citation-guided rereading | [Notebook 05](../notebooks/05_end_to_end_system_evaluation.ipynb) |
| Experiment design | Supplied-evidence controls, document holdout, ablations, and retained negative results | [Case study](case_study.md) |
| AWS GPU engineering | Per-question checkpoints, immutable identities, failure packaging, and recovery | [Reliability](reliability_recovery.md) |
| Evaluation integrity | Local and official scores separated; incomplete answer coverage disclosed | [Closeout](submission_closeout.md) |
| Software delivery | Public package, tests, typing, CI, frozen environment, and saved notebook outputs | [Reproduction guide](reproducibility.md) |

The local 82.68% route result uses a reused 16-question/five-PDF panel. The best official result is 0.49 public / 0.53 private; the later diagnostic scored 0.48 / 0.49. The full-test supported-answer objective remains unresolved, with 622 structurally accepted predictions out of 624. These are different measurements, not interchangeable claims of accuracy.

Start with the [case study](case_study.md), run the [synthetic public example](../examples/README.md), then follow the [technical review guide](reviewer_guide.md). The repository demonstrates implemented research and engineering, with no claim of production deployment or winner reproduction.
