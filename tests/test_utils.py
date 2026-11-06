from __future__ import annotations

import numpy as np

from topobpe.utils import byte_entropy, stable_hash_split, token_entropy, utf8_length


def test_hash_split_is_deterministic_and_balanced() -> None:
    keys = [f"document-{index}" for index in range(4000)]
    first = [stable_hash_split(key, 0.8, 0.1) for key in keys]
    second = [stable_hash_split(key, 0.8, 0.1) for key in keys]
    assert first == second
    counts = {name: first.count(name) for name in ("train", "val", "test")}
    assert counts["train"] > 3000
    assert counts["val"] > 250
    assert counts["test"] > 250
    assert sum(counts.values()) == len(keys)


def test_entropy_bounds_and_empty_input() -> None:
    assert byte_entropy("") == 0.0
    assert 0.0 <= byte_entropy("aaaa") < 1e-12
    assert abs(byte_entropy(bytes(range(256))) - 8.0) < 1e-12
    assert token_entropy([]) == 0.0
    assert token_entropy(np.array([1, 1, 1])) == 0.0
    assert abs(token_entropy(np.array([0, 1])) - 1.0) < 1e-12


def test_utf8_length_counts_bytes_not_characters() -> None:
    assert utf8_length("abc") == 3
    assert utf8_length("é") == 2
    assert utf8_length("水") == 3
