"""Selection boundaries, numerical equivalence, and interrupted-family recovery."""

from dataclasses import replace

import pytest

from lava.evaluation.schemas import AnswerFormat, ReferenceRecord
from lava.readers.runtime_logging import RuntimeEventLogger
from lava.retrieval.feature_research import bm25_feature_grid, bm25_scores
from lava.retrieval.lexical import PageText, RetrievalQuery
from lava.retrieval.research import (
    BASELINE,
    PreparedBM25,
    build_rankings,
    evaluate_candidates,
    select_candidate,
)


def test_cached_features_preserve_all_catalog_rankings():
    pages = (PageText(1, "東京 １２３ ab cd"), PageText(2, "Hà Nội 123"), PageText(3, ""))
    indexes = {}
    for spec in bm25_feature_grid():
        index = indexes.setdefault(spec.representation, PreparedBM25(pages, spec.representation))
        assert index.rank("東京 123 abcd", spec.k1, spec.b) == [
            page
            for page, _ in bm25_scores("東京 123 abcd", [(p.number, p.text) for p in pages], spec)
        ]


def reference(question, document, gold):
    return ReferenceRecord(
        question_id=question,
        document_id=document,
        question="question",
        answer="answer",
        evidence_pages=gold,
        language="ja",
        answer_format=AnswerFormat.STRING,
    )


def test_selector_sees_training_labels_only_and_deduplicates_on_training():
    # Test-document rankings differ; selection receives only the two training documents.
    refs = [reference("a", "a", [1]), reference("b", "b", [1])]
    ranks = {BASELINE: [[1, 2, 3, 4, 5, 6]] * 2, "same_on_training": [[1, 2, 3, 4, 5, 6]] * 2}
    selected, audit = select_candidate(["same_on_training"], ranks, refs, conservative=True)
    assert selected == BASELINE
    assert audit["duplicates_rejected_on_training_only"] == 1
    # Passing an extra held-out ranking alongside only training references is rejected.
    with pytest.raises(ValueError, match="coverage"):
        select_candidate(
            [BASELINE], {BASELINE: ranks[BASELINE] + [[6, 5, 4, 3, 2, 1]]}, refs, conservative=True
        )


def test_conservative_selection_rejects_single_document_overfit():
    refs = [reference("a", "a", [6]), reference("b", "b", [1])]
    ranks = {
        BASELINE: [[1, 2, 3, 4, 5, 6]] * 2,
        "one_document_gain": [[6, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]],
    }
    assert select_candidate(["one_document_gain"], ranks, refs, conservative=True)[0] == BASELINE


def test_changing_held_out_labels_cannot_change_its_selected_candidate():
    class Store:
        def __init__(self):
            self.values = {}

        def read(self, key):
            return self.values.get(key)

        def write(self, key, value):
            self.values[key] = value

    refs = [reference(doc, doc, [6]) for doc in ("a", "b", "c")]
    queries = [RetrievalQuery(ref.question_id, ref.document_id, "needle") for ref in refs]
    pages = tuple(PageText(i, "needle" if i == 6 else "other") for i in range(1, 7))
    rankings, _ = build_rankings(
        queries, {doc: pages for doc in ("a", "b", "c")}, Store(), {}, RuntimeEventLogger("test")
    )
    first = evaluate_candidates(rankings, refs)
    changed = evaluate_candidates(
        rankings, [refs[0].model_copy(update={"evidence_pages": (1,)}), *refs[1:]]
    )
    original_folds = [f for f in first["document_folds"] if f["held_out_document"] == "doc-01"]
    changed_folds = [f for f in changed["document_folds"] if f["held_out_document"] == "doc-01"]
    for before, after in zip(original_folds, changed_folds, strict=True):
        assert before["selected_candidate"] == after["selected_candidate"]
        assert before["selection_audit"] == after["selection_audit"]
        assert before["metrics"] != after["metrics"]


def test_completed_feature_families_survive_an_interruption():
    class Store:
        def __init__(self):
            self.values = {}
            self.interrupt = True

        def read(self, key):
            return self.values.get(key)

        def write(self, key, value):
            if len(self.values) == 2 and self.interrupt:
                raise OSError("simulated disconnection")
            self.values[key] = value

    store = Store()
    query = RetrievalQuery("q", "d", "needle")
    pages = tuple(PageText(i, "needle" if i == 6 else "other") for i in range(1, 7))
    logger = RuntimeEventLogger("test.research")
    with pytest.raises(OSError, match="disconnection"):
        build_rankings([query], {"d": pages}, store, {"version": 1}, logger)
    assert len(store.values) == 2
    saved = dict(store.values)
    store.interrupt = False
    result, recovery = build_rankings([query], {"d": pages}, store, {"version": 1}, logger)
    assert recovery["reused_families"] == 2
    assert all(store.values[key] == value for key, value in saved.items())
    assert len(result) == 1582
    repeated, recovery = build_rankings([query], {"d": pages}, store, {"version": 1}, logger)
    assert repeated == result and recovery["reused_families"] == 12
    changed, recovery = build_rankings(
        [replace(query, question="other")], {"d": pages}, store, {"version": 1}, logger
    )
    assert recovery["reused_families"] == 0 and changed != result
