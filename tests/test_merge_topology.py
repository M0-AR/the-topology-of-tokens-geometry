from __future__ import annotations

import numpy as np

from topobpe.merge_topology import (
    _tree_stats,
    load_bpe_components,
    merge_feature_row,
    merge_gain_features,
    merge_graph_features,
    merge_index,
    tree_shape_features,
    truncate_bpe,
)
from topobpe.tokenizers import SPECIAL_TOKENS, build_tokenizer


def _fit(vocab_size: int = 300):
    texts = [
        "the quick brown fox jumps over the lazy dog",
        "tokenizer topology merges bytes into larger pieces",
        "水を飲むと元気になる",
        "café naïve résumé",
    ]
    tokenizer, trainer = build_tokenizer("bpe", vocab_size, SPECIAL_TOKENS, 1)
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer


def test_merge_index_and_tree_stats() -> None:
    parents, ranks = merge_index([("a", "b"), ("ab", "c")])
    assert parents["ab"] == ("a", "b")
    assert parents["abc"] == ("ab", "c")
    assert ranks["abc"] == 2
    depth, size, strahler, balance = _tree_stats("abc", parents, {})
    assert depth == 2
    assert size == 3
    assert strahler == 2
    assert balance == 0.5


def test_truncate_bpe_preserves_roundtrip(tmp_path) -> None:
    base = _fit()
    path = tmp_path / "base.json"
    base.save(str(path))
    vocab, merges = load_bpe_components(path)
    for target in (264, 280, 300):
        truncated, kept = truncate_bpe(base, vocab, merges, target)
        assert truncated.get_vocab_size() <= target
        assert len(kept) == len(merges[: len(kept)])
        for text in ("hello world", "café", "水を飲む"):
            ids = truncated.encode(text, add_special_tokens=False).ids
            assert truncated.decode(ids) == text


def test_merge_features_are_finite(tmp_path) -> None:
    tokenizer = _fit()
    path = tmp_path / "base.json"
    tokenizer.save(str(path))
    vocab, merges = load_bpe_components(path)
    ids = np.asarray(
        tokenizer.encode(
            "the quick brown fox jumps over the lazy dog " * 20, add_special_tokens=False
        ).ids,
        dtype=np.int64,
    )
    parents, ranks = merge_index(merges)
    tree = tree_shape_features(tokenizer, parents, set())
    assert tree["merge_tree_tokens"] > 0
    assert tree["merge_tree_mean_depth"] >= 0
    gain = merge_gain_features(tokenizer, parents, ranks, ids)
    assert gain["merge_gain_total"] > 0
    assert 0.0 <= gain["merge_gain_gini"] <= 1.0
    graph = merge_graph_features(
        tokenizer, merges, ids, 64, 7, set(), [0.05, 0.1, 0.2, 0.4]
    )
    assert graph["merge_graph_num_nodes"] > 1
    assert "merge_graph_h0_total_persistence" in graph
    assert "merge_graph_h1_total_persistence" in graph
    row = merge_feature_row(tokenizer, merges, ids, set(), 7, [0.05, 0.1, 0.2, 0.4])
    assert all(np.isfinite(value) for value in row.values() if isinstance(value, float))
