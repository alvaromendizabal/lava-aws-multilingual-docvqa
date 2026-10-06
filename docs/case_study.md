# Case study: answering questions with inspectable document evidence

**Role:** sole project owner for research design, Python implementation, AWS GPU execution, evaluation, and public reporting.

**Problem:** a useful PDF answer needs both the answer and a reliable way to locate its evidence. Japanese and Vietnamese documents add language variation; charts, tables, scanned pages, and long files add retrieval and reading errors.

I implemented a pipeline that extracts page text and images, retrieves relevant pages, runs open vision-language readers, parses structured answers, validates physical-page citations, and scores answer semantics and grounding separately. The use case is document review with traceable sources. No production deployment or business savings are claimed.

## The work and the evidence

| Decision | Observation | Consequence |
| --- | --- | --- |
| Separate retrieval from reading | BM25 recall@5 was 95.31%; visual/lexical retrieval reached 98.44% | Diagnose evidence discovery separately from reader behavior |
| Supply gold evidence to isolate readers | Qwen3.5-9B reached 87.02% local LAVA | Establish reader capability before optimizing the complete pipeline |
| Reread cited pages | Local end-to-end score rose from 67.57% to 77.99% | Retain a useful two-pass research baseline |
| Evaluate complementary reader families | Original route reached 82.68% with held-out-document selection | Retain a small-panel lead requiring independent confirmation |
| Compare actual submissions | Best 0.49 public / 0.53 private; later diagnostic 0.48 / 0.49 | Preserve the best; do not equate development gains with competition gains |

**All local results above use a reused panel of 16 questions across five PDFs.** The 82.68% result belongs to the original frozen route. It does not validate the later full-test composite assembled with legacy recoveries. Neither supplied-evidence nor local development results are substitutes for official scores.

## Why the system is organized this way

**A missing page and a misread page need different fixes.** Retrieval is scored directly, then readers are compared on known evidence, and finally the combined pipeline is evaluated. That separation makes ablations interpretable.

**A larger model needs to justify its cost.** The larger one-shot reader, self-consistency, active-perception, and exhaustive-screening branches did not consistently satisfy promotion gates. Their negative results remain in the record. Model-family diversity helped on the small development panel, but that does not establish an improvement across the full test set.

**Interrupted inference should remain usable.** Per-question checkpoints and model/data/source hashes make reuse explicit. The latest AWS closeout verified 1,081 unchanged checkpoints, executed and reopened its notebook, and verified a versioned backup by reading its bytes back. This demonstrates recovery mechanisms; it is not a production reliability or uptime measurement.

## Current completion boundary

The system has records for all 624 questions. The recovered candidate has 622 structurally accepted predictions and two unresolved answers. Structural acceptance checks formats and citation bounds; it does not establish factual correctness. The historical best used two template-derived values without verified support, while the later diagnostic used two compatibility abstentions. Neither satisfies the project's strict supported-answer completion objective.

The closeout confirmed the recorded best and retained the later negative result. It made no new model calls or submissions. The next scientific requirement is broader independent evaluation and genuine evidence for the two unresolved answers. Documentation alone cannot close those gaps.

## Inspect the implementation

- [Synthetic example](../examples/README.md): run the real public parsing and citation checks without downloading a model.
- [Notebook 05](../notebooks/05_end_to_end_system_evaluation.ipynb): executed end-to-end development evaluation.
- [Submission closeout](submission_closeout.md): official outcomes, exact scope, and public verification.
- [Architecture](architecture.md): subsystem boundaries and lineage.
- [Reproducibility](reproducibility.md): what is runnable and what remains private.

The public repository exposes reusable implementation and aggregate evidence. Exact routing, private predictions, raw generations, credentials, and cloud object locations remain outside Git.
