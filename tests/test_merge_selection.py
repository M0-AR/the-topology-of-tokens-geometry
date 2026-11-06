from __future__ import annotations

import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers

from topobpe.merge_selection import (
    alternate_decompositions,
    ancestor_closure,
    annotated_merges,
    build_subset_tokenizer,
    compression_statistics,
    creator_map,
    select_merges,
    simulate_bpe,
    strict_reachability,
    structural_persistence,
)


def _toy():
    vocab = {"a": 0, "b": 1, "c": 2, "ab": 3, "bc": 4, "abc": 5}
    merges = [("a", "b"), ("b", "c"), ("ab", "c")]
    entries = annotated_merges(merges)
    return vocab, merges, entries


def test_simulate_bpe_reproduces_parent():
    _, _, entries = _toy()
    ranks = {(entry.left, entry.right): entry.rank for entry in entries}
    assert simulate_bpe("abc", ranks) == ["abc"]
    assert simulate_bpe("abd", ranks) == ["ab", "d"]


def test_structural_persistence_and_alternatives():
    vocab, _, entries = _toy()
    persistence = structural_persistence(entries)
    assert persistence.shape == (3,)
    assert persistence[0] == 2 / 3
    assert persistence[2] == 1 / 3
    alternate = alternate_decompositions(entries, vocab)
    assert alternate.tolist() == [0.0, 0.0, 1.0]


def test_ancestor_closure_for_high_merge():
    _, _, entries = _toy()
    creators = creator_map(entries)
    by_rank = {entry.rank: entry for entry in entries}
    assert ancestor_closure(3, by_rank, creators) == {1, 3}


def test_selection_fills_budget_and_is_closed():
    _, _, entries = _toy()
    scores = np.array([0.0, 0.0, 10.0])
    selected = select_merges(entries, scores, budget=2, closure=True)
    assert selected == {1, 3}
    vocabulary = {"a": 0, "b": 1, "c": 2, "ab": 3, "bc": 4, "abc": 5}
    assert strict_reachability(entries, selected, vocabulary) == 1.0


def test_build_subset_tokenizer_roundtrip():
    vocab, merges, entries = _toy()
    base = Tokenizer(models.BPE(vocab=vocab, merges=merges))
    base.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer = build_subset_tokenizer(base, vocab, entries, {1, 3})
    encoded = tokenizer.encode("abc", add_special_tokens=False).ids
    assert len(encoded) == 1
    assert tokenizer.decode(encoded) == "abc"


def test_compression_statistics_detects_shift():
    left = [5.0, 5.1, 5.2, 5.3, 5.4]
    right = [4.0, 4.1, 4.2, 4.3, 4.4]
    stats = compression_statistics(left, right, iterations=500)
    assert stats["delta"] > 0.9
    assert stats["ci_low"] > 0.0
    assert stats["p_value"] < 0.05
