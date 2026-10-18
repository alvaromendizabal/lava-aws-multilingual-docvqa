# Evidence-page retrieval

The public reference is [BM25 retrieval](../src/lava/retrieval/lexical.py) with Unicode normalization, word and character n-gram features, stable physical page numbers and bounded page budgets. [The pipeline](../src/lava/retrieval/pipeline.py) binds extraction and ranked outputs to input identities and rejects partial or changed cache state.

[The recorded report](../reports/retrieval/summary.json) measures **95.31% evidence recall@5** on a reused 16-question, five-document development panel. The historical multimodal retrieval comparison is retained only as compact aggregate context in [final development evidence](../research/final_development.json). It is not the browser's implementation or a newly rerun benchmark.

Correct retrieval does not imply a correct answer. The [integrated notebook](../notebooks/05_end_to_end_system_evaluation.ipynb) retains the actual first-pass/reread comparison, including errors despite complete evidence and the five-document uncertainty limitation. Unpromoted feature-search campaigns and their operator wrappers are not part of the maintained public interface.
