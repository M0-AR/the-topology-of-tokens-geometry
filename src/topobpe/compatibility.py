from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata, spearmanr
from tokenizers import Tokenizer


@dataclass(frozen=True)
class CompatibilityRow:
    token: str
    surface: str
    demerge_loss: float
    union_beta1: float
    depth_gap: float
    length: float
    frequency: float


def token_spans(tokenizer: Tokenizer, text: str) -> list[tuple[int, int]]:
    encoding = tokenizer.encode(text, add_special_tokens=False)
    return [(int(start), int(end)) for start, end in encoding.offsets if end > start]


def boundary_set(spans: list[tuple[int, int]], length: int) -> set[int]:
    values: set[int] = set()
    for start, end in spans:
        if 0 < start < length:
            values.add(start)
        if 0 < end < length:
            values.add(end)
    return values


def boundary_jaccard(
    a_spans: list[tuple[int, int]], b_spans: list[tuple[int, int]], length: int
) -> float:
    left = boundary_set(a_spans, length)
    right = boundary_set(b_spans, length)
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def is_crossing(
    a_spans: list[tuple[int, int]], b_spans: list[tuple[int, int]], length: int
) -> bool:
    left = boundary_set(a_spans, length)
    right = boundary_set(b_spans, length)
    return not (left <= right or right <= left)


def interval_overlap_beta1(
    a_spans: list[tuple[int, int]], b_spans: list[tuple[int, int]]
) -> int:
    if not a_spans or not b_spans:
        return 0
    nodes = [("a", index) for index in range(len(a_spans))]
    nodes.extend(("b", index) for index in range(len(b_spans)))
    index = {node: position for position, node in enumerate(nodes)}
    parent = list(range(len(nodes)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    edges = 0
    for a_position, (a_start, a_end) in enumerate(a_spans):
        for b_position, (b_start, b_end) in enumerate(b_spans):
            if max(a_start, b_start) < min(a_end, b_end):
                edges += 1
                union(index[("a", a_position)], index[("b", b_position)])
    components = len({find(position) for position in range(len(nodes))})
    return max(0, edges - len(nodes) + components)


def merge_tree(
    token: str, parents: dict[str, tuple[str, str]]
) -> tuple[set[str], set[tuple[str, str]]]:
    nodes: set[str] = set()
    edges: set[tuple[str, str]] = set()
    stack = [token]
    while stack:
        current = stack.pop()
        if current in nodes:
            continue
        nodes.add(current)
        children = parents.get(current)
        if children is not None:
            for child in children:
                edges.add((current, child))
                stack.append(child)
    return nodes, edges


def merge_depth(token: str, parents: dict[str, tuple[str, str]]) -> int:
    if token not in parents:
        return 0
    left, right = parents[token]
    return 1 + max(merge_depth(left, parents), merge_depth(right, parents))


def union_beta1(
    token: str,
    parents_a: dict[str, tuple[str, str]],
    parents_b: dict[str, tuple[str, str]],
) -> int:
    nodes_a, edges_a = merge_tree(token, parents_a)
    nodes_b, edges_b = merge_tree(token, parents_b)
    nodes = nodes_a | nodes_b
    edges = edges_a | edges_b
    if not nodes:
        return 0
    index = {node: position for position, node in enumerate(nodes)}
    parent = list(range(len(nodes)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right in edges:
        union(index[left], index[right])
    components = len({find(position) for position in range(len(nodes))})
    return max(0, len(edges) - len(nodes) + components)


def demerge_loss(
    tokenizer_a: Tokenizer, tokenizer_b: Tokenizer, text: str
) -> float:
    spans_a = token_spans(tokenizer_a, text)
    spans_b = token_spans(tokenizer_b, text)
    return 1.0 - boundary_jaccard(spans_a, spans_b, len(text))


def vocab_jaccard(vocab_a: set[str], vocab_b: set[str]) -> float:
    union = vocab_a | vocab_b
    if not union:
        return 0.0
    return len(vocab_a & vocab_b) / len(union)


def merge_jaccard(
    merges_a: list[tuple[str, str]], merges_b: list[tuple[str, str]]
) -> float:
    left = set(merges_a)
    right = set(merges_b)
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(scores, dtype=np.float64)
    binary = np.asarray(labels, dtype=bool)
    positives = int(binary.sum())
    negatives = int(binary.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(values, method="average")
    return float(
        (ranks[binary].sum() - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3:
        return float("nan")
    result = spearmanr(left, right).statistic
    return float(result) if np.isfinite(result) else float("nan")


def shared_token_rows(
    tokenizer_a: Tokenizer,
    tokenizer_b: Tokenizer,
    parents_a: dict[str, tuple[str, str]],
    parents_b: dict[str, tuple[str, str]],
    vocab_a: dict[str, int],
    vocab_b: dict[str, int],
    frequencies: dict[str, int],
    max_tokens: int = 1000,
    contexts: dict[str, str] | None = None,
) -> list[CompatibilityRow]:
    shared = sorted(
        (token for token in vocab_a if token in vocab_b and frequencies.get(token, 0) > 0),
        key=lambda token: (-frequencies.get(token, 0), token),
    )[:max_tokens]
    rows: list[CompatibilityRow] = []
    for token in shared:
        if contexts is not None:
            surface = contexts.get(token)
            if surface is None:
                continue
        else:
            try:
                surface = tokenizer_a.decode([vocab_a[token]])
            except Exception:
                continue
        if not surface:
            continue
        loss = demerge_loss(tokenizer_a, tokenizer_b, surface)
        rows.append(
            CompatibilityRow(
                token=token,
                surface=surface,
                demerge_loss=float(loss),
                union_beta1=float(union_beta1(token, parents_a, parents_b)),
                depth_gap=float(abs(merge_depth(token, parents_a) - merge_depth(token, parents_b))),
                length=float(len(surface)),
                frequency=float(frequencies.get(token, 0)),
            )
        )
    return rows
