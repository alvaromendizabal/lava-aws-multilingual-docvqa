# Oracle-Evidence Multimodal Reader Benchmark

## Research question

Which open multimodal reader most accurately answers Japanese and Vietnamese LAVA questions when the correct evidence pages are already known?

Holding retrieval constant is essential. It separates reader errors from page-selection errors before sparse, dense, visual, and hybrid retrieval are introduced.

## Evaluation boundary

The benchmark is bound to protocol lock
`23cf9a605fcdc553c2948b1fd1001d8e300187f35a43f46083a3396cc7ac2b61`.
The corpus has only 16 labeled questions across five source documents. No prompt, render, decoding, or model choice may be selected from an outer validation document. The current normalized-exact score is a deterministic engineering diagnostic, not the organizer server's semantic score. A pinned semantic judge is a separate gate.

## Reader ladder

1. Qwen3.5-4B, fused page image plus native text, direct deterministic decoding.
2. Image-only and text-only controls using the same model revision.
3. Thinking-mode ablation with bounded seeded sampling.
4. Qwen3.5-9B fused challenger after the 4B path and GPU-memory telemetry are verified.
5. Standard versus high-resolution rendering.
6. Structure-aware parsing and answer-blind region crops.
7. Bounded verification and abstention.

The code resolves every mutable Hugging Face model name to a full commit SHA before the first score. Public, ungated, Apache-2.0 checkpoints are required for the competition-comparable track.

## Evidence assets

For each unique gold evidence page, the asset builder creates:

- a lossless RGB PNG using a named render profile;
- sorted native PDF text;
- word boxes, text blocks, image blocks, reading order, rotation, and page geometry;
- SHA-256 checksums and S3 version IDs;
- a private question-to-page manifest tied to the frozen evaluation lock.

Questions, answers, source IDs, page images, text, and layout remain in private S3. GitHub contains aggregate summaries, code, notebooks without outputs, and immutable configuration.

## SageMaker execution

SageMaker Studio JupyterLab is the interactive control plane. GPU work runs in an ephemeral SageMaker Training Job through SDK V3 `ModelTrainer`.

The first paid run is limited to:

- `Qwen/Qwen3.5-4B` at a pinned model revision;
- one question;
- one `ml.g6e.2xlarge` instance with one 48 GB L40S GPU;
- one-hour runtime and wait limits;
- no endpoint;
- no automatic submission;
- explicit `--acknowledge-charges YES`.

The launcher previews the complete plan without constructing a trainer or creating a billable resource. A contract test inspects the installed SageMaker SDK signatures before submission, preventing silent API drift.

## Reproducibility record

Each run records:

- evaluation protocol lock, Git SHA, model revision, prompt version, and asset-manifest hash;
- training image URI and published image digest;
- render profile, input mode, decoding configuration, and random seed;
- schema validity, abstention, local evidence attribution, and diagnostic answer score;
- model load, preprocessing, generation, and total latency;
- prompt and generated tokens;
- image count and pixel count;
- peak allocated and reserved CUDA memory;
- GPU name, compute capability, PyTorch and Transformers versions;
- private per-question records and sanitized public summaries in versioned S3.

## Advancement criteria

A candidate advances only when it:

- completes the one-question smoke job;
- produces valid structured output;
- uploads verifiable artifacts;
- stays within the runtime and memory guardrails;
- improves nested document-isolated validation or provides a justified efficiency advantage;
- survives repeated-run and failure-slice analysis.

A benchmark-leading claim is reserved for evidence from the frozen protocol and an external benchmark, not for model size or novelty alone.
