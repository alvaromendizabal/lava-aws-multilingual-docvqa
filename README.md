# LAVA — Evidence-Grounded Multilingual Document Intelligence on AWS

[![CI](https://github.com/alvaromendizabal/lava-aws-multilingual-docvqa/actions/workflows/ci.yml/badge.svg)](https://github.com/alvaromendizabal/lava-aws-multilingual-docvqa/actions/workflows/ci.yml)

LAVA is a research-grade multilingual document-intelligence system built to separate **reader quality**, **retrieval quality**, and **systems cost** under a frozen, leakage-resistant evaluation protocol. The project combines open vision-language models, immutable model/data lineage, AWS SageMaker GPU execution, structured artifact verification, and reproducible public analysis notebooks.

## Current verified reader ladder

| Reader | SageMaker target | Verified scope | Billable seconds |
| --- | --- | --- | --- |
| Qwen3.5-4B fused direct | `ml.g5.2xlarge` | One-question smoke | 336 |
| Qwen3.5-9B fused direct | `ml.g6e.2xlarge` | One-question smoke | 350 |
| Qwen3.8-27B NF4 fused direct | `ml.g5.2xlarge` | One-question smoke | 783 |

These three synchronized runs passed structured-output and artifact checks. Their
one-question exact scores are diagnostic results, not evidence of model superiority.
The completed 27B path uses NF4 quantization. The unquantized high-memory candidate
is a separate configuration and is not required for this pilot.

The next stage is a descriptive evaluation on **all 16 frozen questions across five
documents**. The complete-pilot command verifies manifest version, checksum, question
identity, and coverage. It preserves parsing failures and abstentions, then independently
recomputes every score during artifact verification.

[View the report source](reports/oracle_reader/evaluation/index.html) ·
[Evaluation workflow](docs/evaluation.md) · [Aggregate results](reports/oracle_reader/evaluation/summary.json)

## Architecture

```mermaid
flowchart TD
    A["Frozen protocol and model revisions"] --> B["Oracle evidence assets"]
    B --> C["Bounded SageMaker reader job"]
    C --> D["Private responses and checkpoints"]
    D --> E["Independent artifact and score checks"]
    E --> F["Public reports and notebooks"]
    F --> G["Semantic judge, then retrieval experiments"]
```

See [`docs/architecture.md`](docs/architecture.md) for the execution and lineage model.

## Reproducible workflow

The public interface is intentionally small. Historical phase-specific wrappers have been removed.

```bash
# Local quality and reproducibility gate
make quality

# No-cost AWS/data/quota/cost preflight for the already-verified 9B path
make preflight MODEL=qwen35_9b_fused_direct

# Preview a one-question plan; creates no paid GPU resource
make preview MODEL=qwen35_9b_fused_direct

# Paid submission is explicitly locked
make submit MODEL=qwen35_9b_fused_direct CHARGES=YES

# Complete-pilot preview; no GPU is launched
make benchmark-preview MODEL=qwen35_4b_fused_direct

# Rebuild the offline public dashboard
make report
```

Completed jobs are reconnectable and independently verifiable:

```bash
make monitor JOB=<sagemaker-job-name>
make verify JOB=<sagemaker-job-name>
make sync JOB=<sagemaker-job-name>
make stop JOB=<sagemaker-job-name> CONFIRM=YES
```

## Engineering controls

- Python 3.12 environment locked with `uv.lock` and `uv sync --frozen`.
- Ruff formatting/linting, repo-wide Mypy, Pytest, compile checks, shell syntax checks, notebook hygiene, and Git diff validation run through one fail-closed quality gate.
- GitHub Actions runs that same quality gate instead of maintaining a second CI implementation.
- Model repositories and revisions are pinned; benchmark protocol and oracle assets carry immutable lineage identifiers.
- Paid SageMaker runs require explicit operator acknowledgement and conservative cost ceilings.
- Runtime events include UTC timestamps, stage timings, and heartbeats for long operations.
- Public reports contain sanitized aggregate metadata; private questions, answers, page images, raw model responses, bucket names, and identifiers remain outside Git.

## Notebooks

The notebooks are paired with Jupytext so the `.ipynb` files remain convenient for readers while `.py` sources remain reviewable and testable.

1. `00_reproducibility_and_protocol` — frozen protocol, model registry, deterministic experiment contract.
2. `01_oracle_reader_benchmark_design` — controlled reader ladder and ablation design.
3. `02_verified_gpu_execution` — verified SageMaker run lineage and artifact-gate results.
4. `03_model_scaling_and_cost` — checksum-verified offline dashboard with coverage, latency, VRAM, billable time, document comparisons, and lineage.

## Research program

The current milestone isolates reader capability with oracle evidence. The next benchmark layer evaluates 4B → 9B → 27B scaling, modality ablations, multilingual slices, error taxonomy, latency/throughput/VRAM, and cost-quality Pareto behavior. Retrieval and reranking are then introduced under the same frozen document-isolated protocol so retrieval failures cannot be confused with reader failures.

## Repository layout

```text
configs/           frozen benchmark and model contracts
docs/              architecture and methodology
infra/terraform/   reproducible AWS infrastructure
notebooks/         public Jupytext-paired analysis notebooks
pipelines/         GPU job entrypoints and pinned runtime dependencies
reports/           sanitized public manifests, metrics, and figures
scripts/           small canonical operational interface
src/lava/          tested application and research library
tests/             unit, integration, and regression contracts
```

The repository intentionally avoids historical `fix`, `repair`, `patched`, and phase-specific duplicate implementations. Corrections are made in the canonical source files and validated by tests and CI.
