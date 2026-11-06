from __future__ import annotations

import numpy as np

from topobpe.compatibility import (
    auc,
    boundary_jaccard,
    interval_overlap_beta1,
    is_crossing,
    merge_jaccard,
    union_beta1,
    vocab_jaccard,
)


def test_nested_partitions_are_not_crossing() -> None:
    a = [(0, 1), (1, 3), (3, 4), (4, 6)]
    b = [(0, 3), (3, 6)]
    assert not is_crossing(a, b, 6)
    assert boundary_jaccard(a, b, 6) < 1.0


def test_crossing_partition_has_zero_interval_beta1() -> None:
    a = [(0, 2), (2, 4)]
    b = [(0, 1), (1, 3), (3, 4)]
    assert is_crossing(a, b, 4)
    assert interval_overlap_beta1(a, b) == 0


def test_identical_partitions_have_perfect_boundary_jaccard() -> None:
    a = [(0, 2), (2, 5)]
    assert boundary_jaccard(a, a, 5) == 1.0
    assert not is_crossing(a, a, 5)


def test_union_beta1_zero_for_identical_trees() -> None:
    parents = {"ab": ("a", "b"), "abc": ("ab", "c")}
    assert union_beta1("abc", parents, parents) == 0


def test_union_beta1_positive_for_alternative_decompositions() -> None:
    left = {"ab": ("a", "b"), "abc": ("ab", "c")}
    right = {"bc": ("b", "c"), "abc": ("a", "bc")}
    assert union_beta1("abc", left, right) >= 1


def test_jaccard_helpers() -> None:
    assert vocab_jaccard({"a", "b"}, {"b", "c"}) == 1 / 3
    assert merge_jaccard([("a", "b")], [("a", "b"), ("b", "c")]) == 1 / 2


def test_auc_handles_separable_scores() -> None:
    scores = np.array([0.9, 0.8, 0.1, 0.2])
    labels = np.array([True, True, False, False])
    assert auc(scores, labels) == 1.0
