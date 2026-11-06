from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, models

from topobpe.merge_selection import annotated_merges

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

invention_c5 = import_module("invention_c5")



def test_select_texts_stops_at_exact_byte_budget():
    pool = ["abcdef", "ghij", "klmnop"]
    selected = invention_c5.select_texts(pool, [0, 1, 2], 12)
    assert selected == ["abcdef", "ghij", "kl"]


def test_strategy_order_ranks_high_and_low_fertility():
    features = {
        "bytes": np.ones(4),
        "fertility": np.array([0.1, 0.4, 0.2, 0.3]),
        "mismatch": np.array([0.0, 0.0, 0.0, 0.0]),
        "rare": np.array([0.0, 0.0, 0.0, 0.0]),
        "ttr": np.array([0.0, 0.0, 0.0, 0.0]),
    }
    assert invention_c5.strategy_order("high_fertility", features, 0) == [1, 3, 2, 0]
    assert invention_c5.strategy_order("low_fertility", features, 0) == [0, 2, 3, 1]
    assert invention_c5.strategy_order("first", features, 0) == [0, 1, 2, 3]
    assert invention_c5.strategy_order("full_16mb", features, 0) == [0, 1, 2, 3]
    first = invention_c5.strategy_order("random", features, 13)
    second = invention_c5.strategy_order("random", features, 13)
    assert first == second
    assert sorted(first) == [0, 1, 2, 3]


def test_document_features_compute_fertility_and_mismatch():
    donor = Tokenizer(
        models.BPE(
            vocab={"a": 0, "b": 1, "ab": 2},
            merges=[("a", "b")],
            unk_token="<unk>",
        )
    )
    native = Tokenizer(
        models.BPE(vocab={"a": 0, "b": 1}, merges=[], unk_token="<unk>")
    )
    features = invention_c5.document_features(["ab", "a"], donor, native)
    assert features["bytes"].tolist() == [2.0, 1.0]
    assert features["donor_tokens"].tolist() == [1.0, 1.0]
    assert features["fertility"].tolist() == [0.5, 1.0]
    assert features["mismatch"].tolist() == [-0.5, 0.0]


def test_strategy_order_rejects_unknown_strategy():
    features = {
        "bytes": np.ones(1),
        "fertility": np.ones(1),
        "mismatch": np.ones(1),
        "rare": np.ones(1),
        "ttr": np.ones(1),
    }
    try:
        invention_c5.strategy_order("nope", features, 0)
    except ValueError:
        return
    raise AssertionError("unknown strategy should raise")


def test_build_arm_scores_held_out_texts():
    tokenizer = Tokenizer(
        models.BPE(
            vocab={"a": 0, "b": 1, "ab": 2},
            merges=[("a", "b")],
            unk_token="<unk>",
        )
    )
    vocab = tokenizer.get_vocab()
    entries = annotated_merges([("a", "b")])
    _, row = invention_c5.build_arm(
        tokenizer, vocab, entries, ["ab"], ["a"], 1
    )
    assert row["test_compression"] == 1.0
    assert row["test_bytes"] == 1
