from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from tokenizers import Tokenizer, models


@dataclass(frozen=True)
class MergeEntry:
    rank: int
    left: str
    right: str
    parent: str


def annotated_merges(merges: list[tuple[str, str]]) -> list[MergeEntry]:
    return [
        MergeEntry(rank=rank, left=left, right=right, parent=left + right)
        for rank, (left, right) in enumerate(merges, start=1)
    ]


def initial_tokens(vocab: dict[str, int], entries: list[MergeEntry]) -> list[str]:
    parents = {entry.parent for entry in entries}
    return [token for token in vocab if token not in parents]


def creator_map(entries: list[MergeEntry]) -> dict[str, int]:
    return {entry.parent: entry.rank for entry in entries}


def consumer_min_rank(entries: list[MergeEntry], max_rank: int) -> dict[str, int]:
    consumed: dict[str, int] = {}
    for entry in entries:
        for child in (entry.left, entry.right):
            current = consumed.get(child)
            if current is None or entry.rank < current:
                consumed[child] = entry.rank
    return consumed


def structural_persistence(entries: list[MergeEntry]) -> np.ndarray:
    max_rank = max((entry.rank for entry in entries), default=1)
    consumed = consumer_min_rank(entries, max_rank)
    values = np.zeros(len(entries), dtype=np.float64)
    for position, entry in enumerate(entries):
        absorbed = consumed.get(entry.parent, max_rank + 1)
        values[position] = max(0.0, float(absorbed - entry.rank)) / float(max_rank)
    return values


def alternate_decompositions(entries: list[MergeEntry], vocab: dict[str, int]) -> np.ndarray:
    values = np.zeros(len(entries), dtype=np.float64)
    for position, entry in enumerate(entries):
        parent = entry.parent
        splits = 0
        for cut in range(1, len(parent)):
            if parent[:cut] in vocab and parent[cut:] in vocab:
                splits += 1
        values[position] = float(max(0, splits - 1))
    return values


def simulate_bpe(surface: str, ranks: dict[tuple[str, str], int]) -> list[str]:
    symbols = list(surface)
    while len(symbols) > 1:
        best_rank: int | None = None
        for position in range(len(symbols) - 1):
            rank = ranks.get((symbols[position], symbols[position + 1]))
            if rank is not None and (best_rank is None or rank < best_rank):
                best_rank = rank
        if best_rank is None:
            break
        merged: list[str] = []
        position = 0
        while position < len(symbols):
            if (
                position + 1 < len(symbols)
                and ranks.get((symbols[position], symbols[position + 1])) == best_rank
            ):
                merged.append(symbols[position] + symbols[position + 1])
                position += 2
            else:
                merged.append(symbols[position])
                position += 1
        symbols = merged
    return symbols


def ancestor_closure(
    rank: int,
    entries_by_rank: dict[int, MergeEntry],
    creators: dict[str, int],
) -> set[int]:
    needed: set[int] = set()
    stack = [rank]
    while stack:
        current = stack.pop()
        if current in needed:
            continue
        needed.add(current)
        entry = entries_by_rank[current]
        for child in (entry.left, entry.right):
            child_rank = creators.get(child)
            if child_rank is not None:
                stack.append(child_rank)
    return needed


def select_merges(
    entries: list[MergeEntry],
    scores: np.ndarray,
    budget: int,
    closure: bool = True,
) -> set[int]:
    if budget <= 0:
        return set()
    entries_by_rank = {entry.rank: entry for entry in entries}
    creators = creator_map(entries)
    order = sorted(
        range(len(entries)),
        key=lambda position: (-float(scores[position]), entries[position].rank),
    )
    selected: set[int] = set()
    for position in order:
        if len(selected) >= budget:
            break
        rank = entries[position].rank
        if not closure:
            selected.add(rank)
            continue
        need = ancestor_closure(rank, entries_by_rank, creators)
        addition = need - selected
        if len(selected) + len(addition) <= budget:
            selected |= addition
    if closure and len(selected) < budget:
        for entry in entries:
            if len(selected) >= budget:
                break
            need = ancestor_closure(entry.rank, entries_by_rank, creators)
            addition = need - selected
            if addition and len(selected) + len(addition) <= budget:
                selected |= addition
    return selected


def build_subset_tokenizer(
    base: Tokenizer,
    vocab: dict[str, int],
    entries: list[MergeEntry],
    selected: set[int],
) -> Tokenizer:
    selected_entries = sorted(
        (entry for entry in entries if entry.rank in selected), key=lambda entry: entry.rank
    )
    initial = initial_tokens(vocab, entries)
    needed = set(initial)
    needed.update(entry.parent for entry in selected_entries)
    for entry in selected_entries:
        needed.add(entry.left)
        needed.add(entry.right)
    ordered = sorted(needed, key=lambda token: vocab.get(token, 10**9))
    new_vocab = {token: position for position, token in enumerate(ordered)}
    model = models.BPE(
        vocab=new_vocab,
        merges=[(entry.left, entry.right) for entry in selected_entries],
        unk_token="<unk>",
    )
    tokenizer = Tokenizer(model)
    tokenizer.normalizer = base.normalizer
    tokenizer.pre_tokenizer = base.pre_tokenizer
    tokenizer.post_processor = base.post_processor
    tokenizer.decoder = base.decoder
    return tokenizer


def compression(tokenizer: Tokenizer, texts: list[str]) -> tuple[float, int, int]:
    raw = 0
    tokens = 0
    for text in texts:
        raw += len(text.encode("utf-8"))
        tokens += len(tokenizer.encode(text, add_special_tokens=False).ids)
    return raw / max(1, tokens), raw, tokens


def strict_reachability(
    entries: list[MergeEntry],
    selected: set[int],
    vocab: dict[str, int],
) -> float:
    selected_entries = [entry for entry in entries if entry.rank in selected]
    if not selected_entries:
        return 0.0
    ranks = {(entry.left, entry.right): entry.rank for entry in selected_entries}
    reachable = 0
    for entry in selected_entries:
        if simulate_bpe(entry.parent, ranks) == [entry.parent]:
            reachable += 1
    return reachable / len(selected_entries)


def score_merges(
    method: str,
    entries: list[MergeEntry],
    vocab: dict[str, int],
    usage: np.ndarray,
    seed: int,
) -> np.ndarray:
    frequency = np.asarray(
        [usage[vocab[entry.parent]] if entry.parent in vocab else 0.0 for entry in entries],
        dtype=np.float64,
    )
    persistence = structural_persistence(entries)
    alternate = alternate_decompositions(entries, vocab)
    if method == "frequency":
        return frequency
    if method == "persistence":
        return persistence
    if method == "combined":
        return np.log1p(frequency) * (1.0 + persistence)
    if method == "redundancy":
        return np.log1p(frequency) * (1.0 + persistence) / (1.0 + np.log1p(alternate))
    if method == "rank_fusion":
        frequency_rank = _descending_ranks(frequency)
        persistence_rank = _descending_ranks(persistence)
        return -(1.0 / (60.0 + frequency_rank) + 1.0 / (60.0 + persistence_rank))
    if method == "random":
        return np.random.default_rng(seed).random(len(entries))
    if method == "prefix":
        return -np.asarray([entry.rank for entry in entries], dtype=np.float64)
    raise ValueError(f"unknown selection method: {method}")


def _descending_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    return ranks


def compression_statistics(
    left: list[float],
    right: list[float],
    iterations: int = 2000,
    seed: int = 20261106,
) -> dict[str, float]:
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if left_values.shape != right_values.shape or left_values.size == 0:
        return {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0}
    delta = float(left_values.mean() - right_values.mean())
    rng = np.random.default_rng(seed)
    count = left_values.size
    draws = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        index = rng.integers(0, count, size=count)
        draws[iteration] = float(
            left_values[index].mean() - right_values[index].mean()
        )
    ci_low, ci_high = np.percentile(draws, [2.5, 97.5])
    if delta >= 0:
        p_value = 2.0 * float(np.mean(draws <= 0.0))
    else:
        p_value = 2.0 * float(np.mean(draws >= 0.0))
    return {
        "delta": delta,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": min(1.0, p_value),
        "iterations": float(iterations),
    }


def document_compressions(tokenizer: Tokenizer, texts: list[str]) -> list[float]:
    values: list[float] = []
    for text in texts:
        raw = len(text.encode("utf-8"))
        tokens = len(tokenizer.encode(text, add_special_tokens=False).ids)
        values.append(raw / max(1, tokens))
    return values


def holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [1.0] * len(p_values)
    running = 0.0
    for position, index in enumerate(order):
        value = (len(p_values) - position) * p_values[index]
        running = max(running, value)
        adjusted[index] = min(1.0, running)
    return adjusted


def information_gain(values: np.ndarray) -> float:
    total = float(values.sum())
    if total <= 0:
        return 0.0
    probabilities = values[values > 0] / total
    return float(-(probabilities * np.log(probabilities)).sum())


def safe_log_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return math.nan
    return numerator / denominator
