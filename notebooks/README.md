# Read the project here

This is the only canonical notebook folder. Each notebook includes verified execution outputs and can run independently.

1. [00 — Reproducibility and protocol](00_reproducibility_and_protocol.ipynb): start here for the data and progress map.
2. [01 — Reader benchmark design](01_oracle_reader_benchmark_design.ipynb): understand what was compared.
3. [02 — Verified GPU execution](02_verified_gpu_execution.ipynb): inspect all three complete pilots.
4. [03 — Model scaling and cost](03_model_scaling_and_cost.ipynb): compare answer quality, evidence, runtime, memory, and cost.
5. [04 — Evidence retrieval](04_evidence_retrieval.ipynb): inspect full-document retrieval.
6. [05 — End-to-end system evaluation](05_end_to_end_system_evaluation.ipynb): inspect the integrated pipeline, actual completion status, metrics and failure analysis.

**Current position:** the three reader pilots and retrieval pilot are complete. Notebook 05 is the integrated evaluation and explicitly shows whether its GPU inference is pending or measured. Kaggle test inference and submission are optional extensions.

Open this folder in SageMaker at:
`/home/sagemaker-user/lava-aws-multilingual-docvqa/notebooks/`

Run `make notebooks` from the repository root to verify/reuse current outputs or refresh changed inputs. This runs local analysis and creates no GPU job. Logs are timestamped, and completed notebook execution can resume. Published output integrity is checked by `make quality`.

For precise completion criteria, see [Remaining delivery milestones](../README.md#remaining-delivery-milestones).
