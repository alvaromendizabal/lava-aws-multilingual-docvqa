"""Document-domain features preserve source coverage and label-isolated selection."""

from dataclasses import replace

import pymupdf
import pytest

from lava.evaluation.schemas import AnswerFormat, ReferenceRecord
from lava.retrieval.domain_features import (
    DocumentPage,
    evaluate_ablation,
    extract_document,
    numeric_tokens,
    page_features,
)
from lava.retrieval.lexical import BM25Index, PageText, RetrievalQuery
from lava.retrieval.research import BASELINE
from scripts.research_document_features import LocalArchive


def pages():
    return tuple(
        DocumentPage(
            i, "東京 ２０２６" if i == 6 else "Hà Nội", "body", "heading", ("block",), "", 0, "none"
        )
        for i in range(1, 8)
    )


def test_complete_catalog_retains_blank_pages_and_matches_frozen_baseline():
    doc = (*pages(), DocumentPage(8, "", "", "", (), "", 0, "none"))
    result = page_features(RetrievalQuery("q", "d", "東京 2026"), doc)
    assert len(result["rankings"]) == 17
    assert all(sorted(order) == list(range(1, 9)) for order in result["rankings"].values())
    expected = [
        p
        for p, _ in BM25Index(
            tuple(PageText(p.number, p.native, "ok" if p.native else "textless") for p in doc)
        ).rank("東京 2026")
    ]
    assert result["rankings"][BASELINE] == expected


def test_exact_numeric_matching_preserves_locale_separators():
    assert numeric_tokens("２０２６年 １２．５％ -2 +4 1,000") == {
        "2026",
        "12.5%",
        "-2",
        "+4",
        "1,000",
    }
    assert numeric_tokens("1,5") != numeric_tokens("1.5")


def test_no_signal_features_do_not_invent_a_ranking():
    result = page_features(RetrievalQuery("q", "d", "unmatchedzzz"), pages())
    assert all(order == result["rankings"][BASELINE] for order in result["rankings"].values())
    assert not any(result["active_features"].values())


def test_invalid_page_inventory_is_rejected():
    with pytest.raises(ValueError, match="physical page"):
        page_features(RetrievalQuery("q", "d", "question"), pages()[1:])


def test_complement_finds_evidence_for_uncovered_query_terms():
    doc = tuple(
        DocumentPage(i, text, text, "", (), "", 0, "none")
        for i, text in enumerate(["alpha " * 20] * 5 + ["beta gamma"], 1)
    )
    result = page_features(RetrievalQuery("q", "d", "alpha beta gamma"), doc)
    assert 6 in result["rankings"]["query_complement_top4_plus1"][:5]


def test_held_out_evidence_cannot_change_its_selected_policy():
    refs = [
        ReferenceRecord(
            question_id=str(i),
            document_id=str(i),
            question="東京 2026",
            answer="a",
            evidence_pages=[6],
            language="ja",
            answer_format=AnswerFormat.STRING,
        )
        for i in range(3)
    ]
    candidate = page_features(RetrievalQuery("q", "d", "東京 2026"), pages())["rankings"]
    rankings = {name: [order] * 3 for name, order in candidate.items()}
    original = evaluate_ablation(rankings, refs)
    changed = evaluate_ablation(
        rankings, [refs[0].model_copy(update={"evidence_pages": (1,)}), *refs[1:]]
    )
    assert (
        original["document_folds"][0]["selected_candidate"]
        == changed["document_folds"][0]["selected_candidate"]
    )
    assert (
        original["document_folds"][0]["selection_audit"]
        == changed["document_folds"][0]["selection_audit"]
    )
    assert original["document_folds"][0]["metrics"] != changed["document_folds"][0]["metrics"]


def test_pdf_layout_and_table_extraction_uses_actual_geometry():
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((50, 30), "Report title", fontsize=20)
        page.insert_text((50, 200), "Body evidence", fontsize=11)
        page.insert_text((50, 820), "Footer", fontsize=8)
        for x in (50, 150, 250):
            page.draw_line((x, 300), (x, 360))
        for y in (300, 330, 360):
            page.draw_line((50, y), (250, y))
        for x, y, text in (
            (60, 320, "Region"),
            (160, 320, "Count"),
            (60, 350, "West"),
            (160, 350, "42"),
        ):
            page.insert_text((x, y), text, fontsize=10)
        pdf.new_page()
        payload = pdf.tobytes()
    extracted = extract_document(payload)
    assert len(extracted) == 2
    assert "Report title" in extracted[0].headings
    assert "Body evidence" in extracted[0].body
    assert "Footer" not in extracted[0].body
    assert extracted[0].table_count == 1
    assert extracted[0].table_strategy == "lines_strict"
    assert "West | 42" in extracted[0].tables
    assert not extracted[1].native


def test_local_checkpoint_rejects_corruption_and_conflicting_overwrite(tmp_path):
    store = LocalArchive(tmp_path)
    store.write("x.json", {"a": 1})
    store.write("x.json", {"a": 1})
    with pytest.raises(ValueError, match="Conflicting"):
        store.write("x.json", {"a": 2})
    path = tmp_path / "x.json"
    path.write_text(path.read_text().replace('"a":1', '"a":2'))
    with pytest.raises(ValueError, match="corrupt"):
        store.read("x.json")


def test_features_do_not_depend_on_unrelated_document_identity():
    query = RetrievalQuery("q", "original", "東京 2026")
    assert page_features(query, pages()) == page_features(
        replace(query, document_id="unrelated"), pages()
    )


def test_complete_research_roundtrip_and_resume_keep_real_json_checkpoints(tmp_path, monkeypatch):
    from lava.evaluation.semantic import digest
    from lava.readers.runtime_logging import RuntimeEventLogger
    from scripts import research_document_features as runner

    refs = [
        ReferenceRecord(
            question_id=f"q{i:02}",
            document_id=f"d{i % 5}",
            question="needle 2026",
            answer="a",
            evidence_pages=[1],
            language="ja",
            answer_format=AnswerFormat.STRING,
        )
        for i in range(16)
    ]
    monkeypatch.setattr(runner, "parse_training_csv", lambda _: tuple(refs))
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((50, 100), "needle 2026")
        pdf.new_page()
        payload = pdf.tobytes()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    sources = {}
    for name in ("train.csv", *(f"d{i}.pdf" for i in range(5))):
        value = b"synthetic fixture" if name == "train.csv" else payload
        (inputs / name).write_bytes(value)
        sources[name] = {"key": name, "sha256": digest(value), "version_id": "fixture"}
    store = LocalArchive(tmp_path / "archive")
    first, reuse = runner.run(
        tmp_path, inputs, store, {"sources": sources}, RuntimeEventLogger("test")
    )
    assert reuse == {"reused_documents": 0, "reused_queries": 0}
    before = {p: p.read_bytes() for p in store.path.rglob("*.json")}
    second, reuse = runner.run(
        tmp_path, inputs, store, {"sources": sources}, RuntimeEventLogger("test")
    )
    assert second == first
    assert reuse == {"reused_documents": 5, "reused_queries": 16}
    assert all(p.read_bytes() == value for p, value in before.items())
