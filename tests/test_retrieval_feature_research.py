from lava.retrieval.feature_research import (
    BM25FeatureSpec,
    bm25_feature_grid,
    bm25_scores,
    deduplicate_rankings,
    evidence_metrics,
    exploration_grid,
    exploration_order,
    feature_tokens,
    fuse_ranks,
    fusion_grid,
)


def test_candidate_catalog_counts_are_frozen() -> None:
    features = bm25_feature_grid()
    assert len(features) == 1089
    assert len({feature.name for feature in features}) == len(features)
    assert len(fusion_grid()) == 472
    assert len(exploration_grid()) == 21


def test_baseline_representation_matches_word_char_2_3_contract() -> None:
    tokens = feature_tokens("A BC", "word_char23")
    assert tokens["w:a"] == 1
    assert tokens["w:bc"] == 1
    assert tokens["c2:bc"] == 1
    assert not any(token.startswith("c4:") for token in tokens)


def test_flat_trigrams_cross_whitespace_without_labels() -> None:
    tokens = feature_tokens("ab cd", "flat3")
    assert tokens["f3:abc"] == 1
    assert tokens["f3:bcd"] == 1


def test_bm25_scores_cover_every_page_and_break_ties_by_page_number() -> None:
    ranking = bm25_scores(
        "needle",
        ((1, "no match"), (2, "needle"), (3, "no match")),
        BM25FeatureSpec("word_char23", 1.2, 0.75),
    )
    assert tuple(page for page, _ in ranking) == (2, 1, 3)


def test_deduplication_is_label_blind() -> None:
    retained, rejected = deduplicate_rankings(
        (
            ("a", ((1, 2), (2, 1))),
            ("b", ((1, 2), (2, 1))),
            ("c", ((2, 1), (1, 2))),
        )
    )
    assert tuple(retained) == ("a", "c")
    assert rejected == 1


def test_rrf_and_borda_validate_full_page_rankings() -> None:
    rankings = ((1, 2, 3), (3, 2, 1))
    assert set(fuse_ranks(rankings, (2.0, 1.0), method="rrf", rrf_k=60)) == {1, 2, 3}
    assert set(fuse_ranks(rankings, (2.0, 1.0), method="borda")) == {1, 2, 3}


def test_exploration_preserves_baseline_budget_and_full_order() -> None:
    order = exploration_order(
        (1, 2, 3, 4, 5),
        ((5, 4, 3, 2, 1), (4, 5, 3, 2, 1)),
        baseline_keep=3,
    )
    assert order[:3] == (1, 2, 3)
    assert len(order) == 5
    assert set(order) == {1, 2, 3, 4, 5}


def test_evidence_metrics_match_grounding_upper_bound() -> None:
    metrics = evidence_metrics((1, 2, 3, 4, 5), (1, 3), k=2)
    assert metrics.oracle_grounding_f1_at_k == 2 / 3
    assert metrics.all_evidence_at_k == 0.0
    assert metrics.recall_at_k == 0.5
