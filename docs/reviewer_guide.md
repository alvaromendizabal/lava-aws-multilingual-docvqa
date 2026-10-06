# Technical review guide

## Five-minute review

1. Read the [case study](case_study.md): problem, ownership, decisions, and measured outcomes.
2. Run the [synthetic example](../examples/README.md): inspect a response and see an invalid citation rejected by the public implementation.
3. Check the [submission closeout](submission_closeout.md): official scores, completion limits, and provenance.

## Fifteen-minute review

| Question | Evidence |
| --- | --- |
| How are missing evidence and reader errors separated? | [Retrieval notebook](../notebooks/04_evidence_retrieval.ipynb), [end-to-end notebook](../notebooks/05_end_to_end_system_evaluation.ipynb) |
| What did larger readers and rereading change? | [Quality/cost notebook](../notebooks/03_model_scaling_and_cost.ipynb), [frontier decisions](frontier_research_update.md) |
| How was model-family routing selected? | [Heterogeneous routing study](heterogeneous_routing_update.md); reused 16-question/five-document panel |
| What survives a GPU interruption? | [Cloud notebook](../notebooks/02_verified_gpu_execution.ipynb), [recovery design](reliability_recovery.md) |
| What is actually reproducible from Git? | [Reproduction levels](reproducibility.md), [public verification script](../examples/verify_submission_closeout.py) |

## Questions worth discussing

- Why does high evidence recall not guarantee correct answers?
- When is a second read more useful than a larger model?
- How much confidence is justified by nested selection on only five documents?
- Why did the later full-test recovery fail to improve the official result?
- How are checkpoint compatibility, citation bounds, and factual support different checks?
- What evidence would justify the next model promotion?

The original local route result is exploratory; the later recovered composite does not inherit that score. The current 622/624 structural acceptance count is not accuracy. These limits are explicit so a reviewer can judge the work from the evidence.

Private predictions, raw generations, exact routing rules, cloud locations, and credentials are excluded. The public implementation, synthetic contracts, executed research notebooks, and aggregate reports remain inspectable.
