# Evaluation, notebooks, and recovery

## Read the current project

Start with [Notebook 00](../notebooks/00_reproducibility_and_protocol.ipynb) and continue through 04.
The only canonical notebook folder is `notebooks/`. It includes verified outputs, so reading the project does not require cloud execution, another login, or another model run.

This release completes the data audit, 4B/9B/27B oracle-reader comparison, and full-document retrieval benchmark. Read 00 → 03 → 04 for a short employer review; 01 and 02 provide technical detail. Integrated reader/retrieval inference and Kaggle submission are optional extensions outside the [completed scope](../README.md#scope-and-limitations).

## Metric and interpretation

The [published LAVA metric](https://lava-workshop.github.io/#evaluation) averages answer credit and evidence-page F1 for each question, then averages over questions. Scalar answers use semantic equivalence; ordered lists use semantic longest-common-subsequence credit; unordered lists use maximum one-to-one semantic matching F1.

Our pinned Gemma-3 1B judge implements these formulas with a validated local prompt and deterministic float32 CPU inference. The organizer's exact prompt/runtime is unpublished. Local scores are labeled accordingly and are not represented as server-identical leaderboard results.

| Reader | Semantic answer credit | Evidence-page F1 | Local LAVA overall |
| --- | ---: | ---: | ---: |
| 4B | 50.63% | 97.02% | 73.82% |
| 9B | 80.15% | 93.90% | 87.02% |
| 27B NF4 | 70.98% | 89.73% | 80.36% |

All three pilots contain the same 16 questions and five PDFs with correct evidence supplied. 4B and 9B each produced 16 valid responses. 27B produced 15 valid responses; its contradictory abstention remains a counted model failure. The 27B candidate also differs in model generation, quantization, and hardware.

9B has the highest observed local score among these configurations. Inspect question averages, equal-document averages, per-document effects, answer formats, language slices, validity, latency, memory, and cost together. Five PDFs provide limited evidence, and there is only one Vietnamese training question. Document-bootstrap intervals and exact paired sign-flip tests are exploratory. The frozen nested document-isolation protocol is a constraint for future tuning, not a claim that nested cross-validation has already been completed.

The initial normalized-exact answer diagnostics remain available for comparison; they are not semantic LAVA scores. Reader citation F1 with oracle pages also does not measure retrieval. [Notebook 04](../notebooks/04_evidence_retrieval.ipynb) reports retrieval recall, all-evidence coverage, MRR, MAP, and nDCG separately.

## Judge acceptance and provenance

The current prompt passed all 28 public development controls, including numeric precision, dates, negation, multilingual names, answer formatting, and instruction-injection cases. These controls are not a held-out accuracy estimate. See [the real-model validation record](../reports/oracle_reader/judge_validation.json).

Model checkpoint, revision, prompt, decoding, dependencies, input versions, and code hashes form the scoring contract. Saved decisions are immutable and verified on read-back. Changed scoring contracts use separate caches. Malformed judge output is an error rather than an automatic negative vote. A rejected judge leaves reader predictions intact.

Source predictions, raw responses, reference answers, and judge decisions stay private. Public reports contain sanitized aggregates and provenance. Neither authentication tokens nor model weights belong in Git.

## Canonical commands

Run these from the repository root. Review each command's scope before using it.

| Command | What it does |
| --- | --- |
| `make notebooks` | Verify/reuse or refresh all five canonical executed notebooks |
| `make quality` | Explicit tests, lint, format, types, compilation, and publication integrity |
| `make evaluation-preview` | Inspect the saved-answer evaluation plan without loading a model |
| `make evaluation-check` | Verify CPU memory, terminal authentication, and pinned Gemma access |
| `make metrics` | Recompute supporting diagnostics from verified saved predictions |
| `make evaluate` | Judge compatible saved predictions on the current CPU, resuming S3 decisions |
| `make retrieval-preview` | Inspect the retrieval configuration without private downloads |
| `make retrieval-evaluate` | Resume full-document retrieval from S3 checkpoints |

Optional `JOB=<sagemaker-job-name>` narrows the applicable evaluator command. Existing scored pilots do not need another GPU run.

GPU submission, monitoring, verification, and checkpoint resume are documented in [the reader execution guide](benchmark.md). `make submit` is an AWS reader-job command; it does not submit to Kaggle.

## Notebook persistence

The notebook itself is the editable source and the readable output. There are no paired Python notebook files or duplicate publication notebooks. Reusable implementation belongs in `src/lava/`.

Each canonical notebook has a manifest under `reports/notebook_execution/`, binding its source content, analysis inputs, exact output bytes, executed-cell count, and execution commit. Source identity ignores output values and automatic kernel metadata while retaining code, markdown, cell IDs, and meaningful metadata.

`make notebooks`:

1. Verifies a matching canonical publication and reuses it without execution.
2. If inputs changed, executes into a content-addressed staging directory.
3. Requires complete successful execution with unchanged source content.
4. Publishes the notebook atomically, then its completion manifest.
5. Recovers from a publication interruption using the completed staging record.

A failed refresh preserves the previous publication. Merging unchanged code does not invalidate analysis results merely because the commit SHA changed; implementation and data hashes govern reuse. Publication tests verify all five notebooks, privacy, complete execution, absence of error/stderr outputs, and the canonical folder layout.

For a private standalone archive, `scripts/execute_notebooks.py` retains its `--output-dir artifacts/notebook_runs/<attempt>` interface. Runtime logs include UTC timestamps, total/stage time, progress, and heartbeats.

## Updating a working copy

Close the notebooks after saving intentional edits. Run Git from the project root and inspect its status before switching or merging. Preserve any meaningful local source or result changes first. Never use a blanket reset or deletion to resolve an update conflict.

Canonical notebooks explicitly disable Git's output-stripping filter. If automatic Jupyter metadata makes a verified notebook appear modified, `make notebooks` restores its verified publication when the source and data are unchanged. If the source changed, it executes and publishes the changed analysis.

The former validation checkouts and legacy bundles were archived with checksums and S3 read-back before removal. There is one active checkout at `/home/sagemaker-user/lava-aws-multilingual-docvqa`. You do not need to navigate into runtime artifacts to find notebooks.

## Authentication when actually needed

Viewing or refreshing public analysis notebooks needs no model-provider login. The existing Studio environment already completed authenticated scoring.

On a new environment, `make evaluation-check` identifies missing access before CPU judging. If it reports missing Hugging Face authentication, run the interactive `hf auth login` command by itself and complete its browser/device flow. The same account must have access to [Gemma](https://huggingface.co/google/gemma-3-1b-it). No Claude account is used.

GitHub, AWS, and Hugging Face connectors do not automatically configure a separate terminal's credentials. Use the check output to determine whether setup is needed; do not repeat login or recreate the Studio space after successful authentication.

## Runtime and resumption boundaries

Accepted SageMaker jobs continue independently of a terminal or browser. A monitor reconnects by job name. Replacement jobs still incur provisioning and model-loading time, even when completed question checkpoints are reused.

CPU judging and retrieval run on the current host. If Studio stops, the process stops; completed S3 checkpoints remain. Restarting the same command repeats only unfinished or incompatible work. A heartbeat reports liveness, while completed/total counts report progress.

Cloud runtime limits and capacity constraints still apply. Two-job concurrent orchestration remains optional future work; the current job guard allows one active LAVA reader job.

[Model comparison](../notebooks/03_model_scaling_and_cost.ipynb) · [Retrieval](retrieval.md) · [Optional submission extension](submission.md)
