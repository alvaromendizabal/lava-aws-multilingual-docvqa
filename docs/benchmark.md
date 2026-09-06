# Oracle-evidence reader benchmark

## Research question

How much reader quality is available when the correct evidence pages are held fixed, before retrieval quality is allowed to influence the answer?

This decomposition prevents page-selection errors from being misclassified as model-reading errors. It also makes later retrieval and reranking experiments interpretable.

## Frozen evaluation boundary

The benchmark is tied to the immutable evaluation protocol and pinned model revisions. Private questions, answers, page images, extracted text, and per-example traces remain in versioned S3. Git contains code, configuration locks, checksums, sanitized aggregate summaries, and public notebooks.

## Controlled reader ladder

1. Qwen3.5-4B fused direct baseline on `ml.g5.2xlarge`.
2. Qwen3.5-4B image-only and text-only modality controls.
3. Qwen3.5-4B bounded thinking-mode ablation.
4. Qwen3.5-9B fused direct challenger on the **verified LAVA `ml.g6e.2xlarge` path**.
5. Qwen3.8-27B fused direct challenger on `ml.g7e.12xlarge`, whose frozen contract requires at least 80 GiB of CUDA memory on one device.
6. Multilingual slices, repeated-run stability, error taxonomy, latency, throughput, peak VRAM, and cost-quality Pareto analysis.
7. Retrieval and reranking only after the reader ladder is characterized.

## Acceptance gate

A SageMaker job reaching `Completed` is not enough. A run is accepted only after the artifact gate verifies lineage, raw-response count, structured-output validity, parser errors, and the canonical S3 output prefix. Public synchronization re-runs verification before writing sanitized result manifests to Git.

## Claims discipline

Model size, GPU size, and code complexity are not treated as evidence of benchmark leadership. Strong claims are reserved for results produced under the frozen protocol and, where relevant, an external benchmark.
