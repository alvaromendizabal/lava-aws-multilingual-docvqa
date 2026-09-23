from __future__ import annotations

import pytest

from lava.reproduction.frontier import (
    ARMS, adaptive_page_cap, generation_semantics, normalize_evidence_pages,
    reciprocal_rank_fusion, set_f1,
)


def test_fixed_six_arm_contract():
    assert [arm.key for arm in ARMS] == list("ABCDEF")

def test_digit_string_pages_are_losslessly_normalized():
    assert normalize_evidence_pages([" 11 ", "12"], [11, 12]) == [11, 12]

def test_ambiguous_page_representations_are_rejected():
    with pytest.raises(ValueError): normalize_evidence_pages([11.0], [11])
    with pytest.raises(ValueError): normalize_evidence_pages(["11.0"], [11])
    with pytest.raises(ValueError): normalize_evidence_pages([True], [1])

def test_generation_identity_ignores_parser_only_changes():
    old = {"prompt": "same", "implementation": {"parser": "old"}, "model": "m"}
    new = {"prompt": "same", "parser_contract_version": "v2", "model_runtime_sha256": "x", "model": "m"}
    assert generation_semantics(old) == generation_semantics(new)

def test_rrf_and_adaptive_caps():
    fused = reciprocal_rank_fusion([1, 2, 3], [3, 2, 1])
    assert {page for page, _ in fused} == {1, 2, 3}
    assert adaptive_page_cap(7) == 7
    assert adaptive_page_cap(15) == 8
    assert adaptive_page_cap(30) == 10

def test_grounding_precision_penalty():
    assert set_f1([1, 2], range(1, 11)) == pytest.approx(1 / 3)
