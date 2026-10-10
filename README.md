# LAVA · Multilingual Document Intelligence

**Alvaro Mendizabal · Retrieval · Document QA · Evaluation · AWS GPU engineering**

[![CI](https://github.com/alvaromendizabal/lava-aws-multilingual-docvqa/actions/workflows/ci.yml/badge.svg)](https://github.com/alvaromendizabal/lava-aws-multilingual-docvqa/actions/workflows/ci.yml)

![Document intelligence: find the evidence, show the source](docs/assets/public-demo-hero.svg)

I built an **Applied ML research system** that answers questions from Japanese and Vietnamese PDFs and returns the physical pages supporting each answer. My work covers page retrieval, vision-language reading, structured output validation, model comparison and recovery from interrupted GPU runs.

**Engineering ownership:** I designed the research workflow, implemented the document-processing and answer-validation interfaces, evaluated reader configurations, and built checkpointed AWS execution. The public review connects those decisions to inspectable code, recorded measurements and an interactive evidence viewer.

**Historical recorded best: 0.49 public / 0.53 private.** The later diagnostic scored 0.48 / 0.49 and was not an improvement. I kept retrieval quality, answer quality, citation validity and operational completeness separate throughout the evaluation.

**Start here:** [Document Desk demo](https://alvaro-document-evidence.tartmacaw2.chatgpt.site) · [Case study](docs/case_study.md) · [Three-minute review](docs/reviewer_guide.md) · [Run locally](docs/reproducibility.md)

## Try Document Desk

[Open the public demo](https://alvaro-document-evidence.tartmacaw2.chatgpt.site) without installation. To run the same source locally, open `public-demo/index.html` from a checkout, or serve the checkout:

```bash
python -m http.server 8000
```

Visit `http://localhost:8000/public-demo/`. Ask a question about the authored Japanese or Vietnamese document, change retrieval settings and inspect the selected pages. The browser ranks pages with BM25, selects an extractive answer, highlights its copied support and validates the physical-page citation. Unsupported queries can abstain; the result is exportable.

This is actual lexical retrieval and extractive answering over synthetic documents, with no LLM, OCR service, backend or model download. The lexical support measure is not calibrated confidence. The historical research used a separate multimodal reading stack.

```bash
node tools/test_public_demo.mjs
```

## Research evidence

| Measured result | Scope |
|---|---|
| Citation-guided reread: **67.57% → 77.99% local LAVA** | Reused 16-question/five-PDF development panel |
| Original heterogeneous route: **82.68% local LAVA** | Original frozen route with document-held-out selection; not the later recovered composite |
| Evidence recall@5: **98.44%** | Retrieval only on the same small development panel |
| **1,081** preserved checkpoints | Byte-identical in the recorded AWS closeout |

Local and organizer scores use different populations and runtimes. The recovered candidate has **622 structurally accepted predictions out of 624**, with two unresolved answers. Structural acceptance is not factual correctness. The historical best used two template-derived values without verified support; the diagnostic used two compatibility abstentions. [Complete evidence and limits](docs/submission_closeout.md)

## Development benchmarks

These measurements use the reused 16-question/five-PDF panel and are **not presented as organizer-server-identical competition scores**. The citation-guided reread is a **post-hoc development finding**. Fixed BM25 recall@5 was **95.31%**.

| Reader / system | Answer | Evidence F1 | Local LAVA |
|---|---:|---:|---:|
| Supplied evidence · Qwen3.5-4B BF16 | 50.62% | 97.02% | 73.82% |
| Supplied evidence · Qwen3.5-9B BF16 | 80.15% | 93.90% | 87.02% |
| Supplied evidence · Qwen3.8-27B NF4 | 70.98% | 89.73% | 80.36% |
| BM25 → Qwen3.5-9B | 48.90% | 86.25% | 67.57% |
| Citation-guided reread | 67.65% | 88.33% | 77.99% |

## What I built

- **Separated evaluation stages:** evidence retrieval, supplied-evidence reading and complete answer/citation behavior.
- **Controlled reader comparisons:** model scaling, citation-guided rereading and document-level routing with explicit promotion gates.
- **Reliable execution:** per-question checkpoints, source/model/input identities, failure diagnostics and verified saved notebooks.
- **Inspectable interfaces:** typed outputs, citation bounds, public parsing utilities, tests and an interactive evidence viewer.

[Architecture](docs/architecture.md) · [Case study](docs/case_study.md) · [Recovery design](docs/reliability_recovery.md)

## Run public Python checks

Python 3.12; these standard-library commands need no model or account:

```bash
python3 examples/run_demo.py
python3 examples/verify_submission_closeout.py
```

The first exercises supplied-response parsing and citation checks. The second verifies aggregate closeout consistency. For the pinned research environment and notebook checks:

```bash
uv sync --frozen --group judge
make quality
make notebooks
```

[Verification levels](docs/reproducibility.md) explain the additional requirements for model-backed evaluation. These instructions concern running this project's components and checking its evidence.

## Scope and limitations

This release contains implemented research, executable public demos and dated evidence. The small reused development panel limits generalization claims; the two unresolved answers remain unresolved. Production service quality is not claimed.

The private full-test and competition layer is intentionally excluded: source documents, predictions, generations, exact routing rules, model caches, credentials and cloud object locations remain private. Historical notebook outputs are preserved. [Project closeout](docs/closeout.md) · [Executed research guide](notebooks/README.md)

## Executed research notebooks

| Notebook | Review focus |
|---|---|
| [00 · Overview](notebooks/00_reproducibility_and_protocol.ipynb) | Contracts and measured findings |
| [01 · Reader comparison](notebooks/01_oracle_reader_benchmark_design.ipynb) | Comparable supplied evidence |
| [02 · GPU execution](notebooks/02_verified_gpu_execution.ipynb) | Checkpoints and recovery |
| [03 · Quality and cost](notebooks/03_model_scaling_and_cost.ipynb) | Model trade-offs |
| [04 · Retrieval](notebooks/04_evidence_retrieval.ipynb) | Page selection |
| [05 · End-to-end](notebooks/05_end_to_end_system_evaluation.ipynb) | Answer and citation evaluation |
