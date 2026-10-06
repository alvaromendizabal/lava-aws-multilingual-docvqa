# Project status — October 6, 2026

The implemented research pipeline and public evidence are available for review. The competition improvement and complete-answer objectives remain open.

| Milestone | Verified state |
| --- | --- |
| Development research | Reader, retrieval, and end-to-end comparisons on a reused 16-question/five-PDF panel |
| Original routed development candidate | 82.68% local LAVA; limited to its original evaluation scope |
| Best recorded submission | 0.49 public / 0.53 private |
| Later recovered diagnostic | 0.48 public / 0.49 private; no improvement |
| Recovered prediction acceptance | 622/624 structurally accepted non-abstaining predictions; two unresolved |
| Checkpoint preservation | 1,081 unchanged inference checkpoints verified in AWS |
| Closeout execution | 14/14 audit tasks, 21.88 seconds, about $0.01 recorded compute estimate |
| Notebook and backup | Real Jupyter execution, saved/reopened Plotly outputs, and versioned backup byte verification |
| New inference or uploads during closeout | None |

[Sanitized closeout evidence and verification](submission_closeout.md) distinguish archive/format completion from factual answer support. The public verifier checks aggregate consistency; private receipts and predictions are not released.

## What remains

1. Resolve the two unsupported answers using actual source evidence. Schema-valid placeholders are not recovered answers.
2. Evaluate new model or selection changes on independent documents. Reusing the small development panel limits confidence in generalization.
3. Produce a genuinely new, validated full-test candidate before another submission. Existing scores are retained; private scores were not used for tuning.
4. Verify a score-to-code connection before claiming reproduction of a leading public solution.

## Public release boundary

Public review and the synthetic example require no account or GPU. The repository contains reusable code, tests, configuration, executed development notebooks, aggregate results, and a documented reproduction path. Exact routing, private predictions, raw generations, credentials, cloud paths, caches, and private return bundles are excluded.

Software tests and successful recovery checks support the engineering account. They do not establish production deployment, service availability, factual correctness of all answers, or winner-level performance.

[Case study](case_study.md) · [Review guide](reviewer_guide.md) · [Reproducibility](reproducibility.md)
