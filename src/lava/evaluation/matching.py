"""Exact matching algorithms used by LAVA list-answer scoring."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

from lava.evaluation.judges import SemanticJudge


def maximum_bipartite_matches(edges: Sequence[Sequence[bool]]) -> int:
    """Return the maximum cardinality one-to-one matching via Hopcroft-Karp."""
    left_count = len(edges)
    if left_count == 0:
        return 0
    right_count = max((len(row) for row in edges), default=0)
    if right_count == 0:
        return 0
    if any(len(row) != right_count for row in edges):
        message = "All bipartite edge rows must have equal length"
        raise ValueError(message)

    adjacency = [tuple(index for index, allowed in enumerate(row) if allowed) for row in edges]
    unmatched = -1
    pair_left = [unmatched] * left_count
    pair_right = [unmatched] * right_count
    distance = [0] * left_count

    def breadth_first_search() -> bool:
        queue: deque[int] = deque()
        augmenting_path_exists = False
        for left in range(left_count):
            if pair_left[left] == unmatched:
                distance[left] = 0
                queue.append(left)
            else:
                distance[left] = -1
        while queue:
            left = queue.popleft()
            for right in adjacency[left]:
                paired_left = pair_right[right]
                if paired_left == unmatched:
                    augmenting_path_exists = True
                elif distance[paired_left] < 0:
                    distance[paired_left] = distance[left] + 1
                    queue.append(paired_left)
        return augmenting_path_exists

    def depth_first_search(left: int) -> bool:
        for right in adjacency[left]:
            paired_left = pair_right[right]
            if paired_left == unmatched or (
                distance[paired_left] == distance[left] + 1 and depth_first_search(paired_left)
            ):
                pair_left[left] = right
                pair_right[right] = left
                return True
        distance[left] = -1
        return False

    matches = 0
    while breadth_first_search():
        for left in range(left_count):
            if pair_left[left] == unmatched and depth_first_search(left):
                matches += 1
    return matches


def semantic_match_matrix(
    references: Sequence[str],
    predictions: Sequence[str],
    *,
    judge: SemanticJudge,
    language: str,
) -> tuple[tuple[bool, ...], ...]:
    """Evaluate every reference-prediction item pair exactly once."""
    return tuple(
        tuple(
            judge.equivalent(reference, prediction, language=language) for prediction in predictions
        )
        for reference in references
    )


def semantic_lcs_length(
    references: Sequence[str],
    predictions: Sequence[str],
    *,
    judge: SemanticJudge,
    language: str,
) -> int:
    """Compute semantic longest-common-subsequence length using dynamic programming."""
    matrix = semantic_match_matrix(
        references,
        predictions,
        judge=judge,
        language=language,
    )
    previous = [0] * (len(predictions) + 1)
    for reference_index in range(1, len(references) + 1):
        current = [0]
        for prediction_index in range(1, len(predictions) + 1):
            if matrix[reference_index - 1][prediction_index - 1]:
                current.append(previous[prediction_index - 1] + 1)
            else:
                current.append(max(previous[prediction_index], current[-1]))
        previous = current
    return previous[-1]
