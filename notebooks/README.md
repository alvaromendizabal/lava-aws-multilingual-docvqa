# Read the completed research benchmark

All seven canonical notebooks include verified execution outputs. **Open and read; no setup or rerun is required.**

| Notebook | Purpose |
| --- | --- |
| [00 — Research overview](00_reproducibility_and_protocol.ipynb) | Start here: scope, findings, and conclusions |
| [01 — Experiment design](01_oracle_reader_benchmark_design.ipynb) | Model choices and comparison methodology |
| [02 — Cloud execution](02_verified_gpu_execution.ipynb) | Completed runs, observability, and recovery |
| [03 — Model quality and cost](03_model_scaling_and_cost.ipynb) | Actual scores, visual analysis, and resource tradeoffs |
| [04 — Evidence retrieval](04_evidence_retrieval.ipynb) | Baseline, lexical feature research, visual challenger, and recovery |
| [05 — Optional integrated workflow](05_end_to_end_system_evaluation.ipynb) | Implemented pipeline and user-operated export; integrated score unmeasured |
| [06 — Frontier reproduction](06_frontier_reproduction_and_reader_benchmark.ipynb) | Aggregate-only reconstruction evidence, reader GPU contracts, and resumable six-arm benchmark status |

**Short employer review: 00 → 03 → 04 → 06.** Read 01 and 02 for additional technical detail.

The component benchmark is complete. Integrated reader/retrieval evaluation, deployment, and Kaggle submission are optional extensions; no end-to-end or leaderboard score is claimed. See [scope and limitations](../README.md#scope-and-limitations).

In Studio, this folder is `/home/sagemaker-user/lava-aws-multilingual-docvqa/notebooks/`. For reproduction, `make notebooks` verifies/reuses outputs or refreshes changed inputs, and `make quality` checks tests and publication integrity. Both run from the repository root and create no GPU job.


## Optional integrated evaluation

[05 — Retrieved-evidence evaluation](05_end_to_end_system_evaluation.ipynb) explains the label-blind input design, actual measurement status, failure analysis and recovery. It can be viewed without a GPU. The measured component benchmark remains complete. Its optional GPU attempt was stopped at closeout. No further run is required; the [operator guide](../docs/system_evaluation.md) is retained for deliberate future reproduction.


Notebook 06 is an executed aggregate-only snapshot. It deliberately excludes private questions, answers, test predictions, credentials, raw return bundles, and model caches. Its 33/96 reader-progress count is execution status, not a score or promotion claim.
