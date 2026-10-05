# Engineering case study — evidence-grounded multilingual document QA

## Executive summary

I built LAVA as an end-to-end multilingual document-intelligence system rather than a single model benchmark. Given a complete Japanese or Vietnamese PDF and a question, the system must find the relevant physical pages, reason over native text and page images, generate a structured answer, cite supporting pages, and survive the operational realities of GPU inference.

The project demonstrates three things at once:

1. **Applied ML research:** controlled reader, retrieval, reasoning, and routing experiments.
2. **ML engineering:** typed interfaces, checkpoints, immutable lineage, CI, and cloud execution.
3. **Decision discipline:** explicit promotion gates, document-disjoint validation, and retained negative results.

## The challenge

A document-VQA system can look strong in an oracle setting while failing in production for several different reasons:

- the evidence retriever misses the required page;
- the reader sees the page but misinterprets the chart, table, or text;
- a larger model is more expensive without being better;
- a second pass improves some questions while regressing others;
- routing overfits a small development set;
- an interrupted GPU run loses expensive completed work;
- a valid answer is paired with invalid or weak grounding.

I treated those as separate engineering and research problems rather than collapsing them into one score.

## System architecture

The public pipeline is organized around explicit contracts:

**PDFs + questions → page extraction → lexical/dense/visual retrieval → multimodal readers → structured answer + citations → schema/evidence validation → semantic + grounding evaluation → held-out-document selection**

The public repository includes the reusable implementation for retrieval, readers, scoring, schemas, checkpoints, observability, and publication. Competition-sensitive predictions and routing details remain private.

## Key technical decisions

### 1. Measure retrieval independently from reading

The project evaluates evidence discovery directly instead of assuming an answer error is a reader error.

Measured development evidence:
- fixed BM25 recall@5: **95.31%**
- strongest measured visual/lexical recall@5: **98.44%**
- BM25 with a larger page budget reaches complete evidence coverage on the small development panel

This exposed an important tradeoff: more evidence availability does not automatically improve answer quality because irrelevant context can dilute reasoning and grounding.

### 2. Isolate reader capability before optimizing the full system

Readers were first compared with the correct evidence supplied.

The strongest verified supplied-evidence reader achieved:
- semantic answer credit: **80.15%**
- evidence F1: **93.90%**
- local LAVA: **87.02%**

A substantially larger one-shot reader failed to beat that frontier. The project therefore stopped treating parameter count as a proxy for quality.

### 3. Use the model's own citations for a second read

The retrieved-evidence first pass exposed questions where evidence was present but the answer was weak. A citation-guided reread improved the end-to-end development score from **67.57% to 77.99% local LAVA** using the same 9B reader.

That result was useful but still post-hoc on a reused panel, so it was not treated as sufficient evidence for more complex routing.

### 4. Combine model families only when their errors are complementary

A compressed Gemma-family reader did not win as a standalone model, but its errors differed enough from Qwen to justify a small heterogeneous candidate family.

The route was then evaluated with nested held-out-document selection:
- incumbent: **77.99%**
- heterogeneous routed challenger: **82.68%**
- improvement: **+4.69 percentage points**
- held-out documents improved: 2
- held-out documents regressed: 0

The exact private routing rule is not published. The validation design and aggregate result are.

### 5. Make failed runs resumable

GPU work is treated as durable state, not a disposable process.

The execution architecture includes:
- deterministic run identities;
- per-question checkpoints;
- model/data/source hashes;
- immutable artifact identities;
- heartbeats and progress counters;
- resource telemetry;
- bounded runtime and cost gates;
- process cleanup;
- failure bundles that preserve completed work.

This prevents one late failure from turning an otherwise useful GPU run into total loss.

## Validation discipline

The project separates several evaluation scopes that are easy to confuse:

- **reader isolation:** correct evidence is supplied;
- **retrieval evaluation:** evidence discovery is scored directly;
- **complete-system evaluation:** retrieved evidence is passed to the reader;
- **reasoning ablations:** alternate reading policies are compared;
- **document-disjoint selection:** candidate policies are selected without the held-out document's labels.

Public claims are tied to the scope in which they were measured.

## What did not work

Negative results are part of the portfolio because they show decision quality.

Examples:
- a larger one-shot reader was not better;
- self-consistency improved a same-run control but failed the cross-document promotion rule;
- exhaustive page screening increased context without producing the best complete-system result;
- active perception did not justify global promotion;
- a strong numeric reasoning signal was kept as a specialization finding instead of being generalized beyond the evidence.

## Engineering evidence

The repository contains:
- six executed, checksum-bound notebooks;
- frozen configs and model revisions;
- reusable Python packages under `src/lava/`;
- unit and integration tests;
- Ruff and mypy;
- GitHub Actions CI;
- public-safe aggregate reports;
- explicit public/private artifact boundaries;
- reproducibility and reviewer guides.

## What I would discuss in an interview

The most representative engineering questions are:

- How do you tell whether an answer failure came from retrieval or reasoning?
- When should a larger model be rejected despite higher capacity?
- How do you validate a routing policy without leaking the held-out document?
- What should be checkpointed in a multimodal GPU pipeline?
- How do you make experiment results inspectable without publishing sensitive evaluation artifacts?

## Scope

The public results use a small released development panel and a pinned local implementation of the published metric structure. They are not represented as organizer-server-identical scores.

The point of the public case study is the system design, evaluation discipline, and production-oriented research workflow—not a claim that a small development set is a universal benchmark.
