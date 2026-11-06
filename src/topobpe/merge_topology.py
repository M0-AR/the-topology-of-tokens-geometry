from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.stats import spearmanr
from tokenizers import Tokenizer, models

from .topology import persistence_summaries, rips_diagrams, validate_distance_matrix


def load_bpe_components(path: str | Path) -> tuple[dict[str, int], list[tuple[str, str]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    model = payload["model"]
    vocab = {str(token): int(index) for token, index in model["vocab"].items()}
    merges = [(str(pair[0]), str(pair[1])) for pair in model["merges"]]
    return vocab, merges


def truncate_bpe(
    base: Tokenizer,
    vocab: dict[str, int],
    merges: list[tuple[str, str]],
    target_vocab_size: int,
) -> tuple[Tokenizer, list[tuple[str, str]]]:
    initial = len(vocab) - len(merges)
    keep = max(0, min(len(merges), int(target_vocab_size) - initial))
    kept_merges = merges[:keep]
    later = {left + right for left, right in merges[keep:]}
    ordered = sorted(
        ((token, index) for token, index in vocab.items() if token not in later),
        key=lambda item: item[1],
    )
    new_vocab = {token: index for index, (token, _) in enumerate(ordered)}
    model = models.BPE(vocab=new_vocab, merges=kept_merges, unk_token="<unk>")
    tokenizer = Tokenizer(model)
    tokenizer.normalizer = base.normalizer
    tokenizer.pre_tokenizer = base.pre_tokenizer
    tokenizer.post_processor = base.post_processor
    tokenizer.decoder = base.decoder
    return tokenizer, kept_merges


def merge_index(
    merges: list[tuple[str, str]],
) -> tuple[dict[str, tuple[str, str]], dict[str, int]]:
    parents: dict[str, tuple[str, str]] = {}
    ranks: dict[str, int] = {}
    for rank, (left, right) in enumerate(merges, start=1):
        parent = left + right
        parents[parent] = (left, right)
        ranks[parent] = rank
    return parents, ranks


def _tree_stats(
    token: str,
    parents: dict[str, tuple[str, str]],
    memo: dict[str, tuple[int, int, int, float]],
) -> tuple[int, int, int, float]:
    if token in memo:
        return memo[token]
    if token not in parents:
        result = (0, 1, 1, 1.0)
    else:
        left, right = parents[token]
        left_depth, left_size, left_strahler, _ = _tree_stats(left, parents, memo)
        right_depth, right_size, right_strahler, _ = _tree_stats(right, parents, memo)
        depth = 1 + max(left_depth, right_depth)
        size = left_size + right_size
        strahler = (
            left_strahler + 1
            if left_strahler == right_strahler
            else max(left_strahler, right_strahler)
        )
        balance = min(left_size, right_size) / max(left_size, right_size)
        result = (depth, size, strahler, balance)
    memo[token] = result
    return result


def tree_shape_features(
    tokenizer: Tokenizer,
    parents: dict[str, tuple[str, str]],
    excluded_ids: set[int],
) -> dict[str, float]:
    depths: list[int] = []
    sizes: list[int] = []
    balances: list[float] = []
    strahlers: list[int] = []
    memo: dict[str, tuple[int, int, int, float]] = {}
    for token_id in range(tokenizer.get_vocab_size()):
        if token_id in excluded_ids:
            continue
        token = tokenizer.id_to_token(token_id)
        if token is None:
            continue
        depth, size, strahler, balance = _tree_stats(token, parents, memo)
        depths.append(depth)
        sizes.append(size)
        balances.append(balance)
        strahlers.append(strahler)
    if not depths:
        return {}
    depth_values = np.asarray(depths, dtype=np.float64)
    size_values = np.asarray(sizes, dtype=np.float64)
    balance_values = np.asarray(balances, dtype=np.float64)
    strahler_values = np.asarray(strahlers, dtype=np.float64)
    return {
        "merge_tree_tokens": float(len(depths)),
        "merge_tree_mean_depth": float(depth_values.mean()),
        "merge_tree_max_depth": float(depth_values.max()),
        "merge_tree_std_depth": float(depth_values.std()),
        "merge_tree_mean_size": float(size_values.mean()),
        "merge_tree_max_size": float(size_values.max()),
        "merge_tree_mean_log_size": float(np.log1p(size_values).mean()),
        "merge_tree_mean_balance": float(balance_values.mean()),
        "merge_tree_min_balance": float(balance_values.min()),
        "merge_tree_mean_strahler": float(strahler_values.mean()),
        "merge_tree_max_strahler": float(strahler_values.max()),
        "merge_tree_leaf_fraction": float(np.mean(depth_values == 0)),
    }


def _gini(values: np.ndarray) -> float:
    if len(values) == 0 or float(values.sum()) <= 0:
        return 0.0
    ordered = np.sort(values.astype(np.float64))
    count = len(ordered)
    index = np.arange(1, count + 1, dtype=np.float64)
    return float(
        2.0 * np.sum(index * ordered) / (count * np.sum(ordered)) - (count + 1.0) / count
    )


def merge_gain_features(
    tokenizer: Tokenizer,
    parents: dict[str, tuple[str, str]],
    ranks: dict[str, int],
    token_ids: np.ndarray,
) -> dict[str, float]:
    frequencies = np.bincount(token_ids, minlength=tokenizer.get_vocab_size())
    gains: dict[str, float] = {}
    for parent, (left, right) in parents.items():
        parent_id = tokenizer.token_to_id(parent)
        parent_frequency = float(frequencies[parent_id]) if parent_id is not None else 0.0
        gains[parent] = parent_frequency + gains.get(left, 0.0) + gains.get(right, 0.0)
    values = np.asarray([gains[parent] for parent in parents], dtype=np.float64)
    if len(values) == 0:
        return {}
    total = float(values.sum())
    probs = values[values > 0] / total if total > 0 else np.zeros(0)
    entropy = float(-(probs * np.log(probs)).sum()) if len(probs) else 0.0
    rank_values = np.asarray([ranks[parent] for parent in parents], dtype=np.float64)
    gain_values = values
    if len(rank_values) == len(gain_values) and len(rank_values) > 2:
        correlation, _ = spearmanr(rank_values, gain_values)
        rank_correlation = float(correlation) if np.isfinite(correlation) else 0.0
    else:
        rank_correlation = 0.0
    ordered = np.sort(values)[::-1]
    return {
        "merge_gain_total": total,
        "merge_gain_mean": float(values.mean()),
        "merge_gain_max": float(values.max()),
        "merge_gain_std": float(values.std()),
        "merge_gain_gini": _gini(values),
        "merge_gain_entropy": entropy,
        "merge_gain_effective_count": float(math.exp(entropy)),
        "merge_gain_top1_share": float(ordered[0] / total) if total > 0 else 0.0,
        "merge_gain_top10_share": float(ordered[:10].sum() / total) if total > 0 else 0.0,
        "merge_gain_zero_fraction": float(np.mean(values == 0)),
        "merge_gain_rank_correlation": rank_correlation,
    }


def _graph_stats(
    adjacency: dict[str, dict[str, float]],
    nodes: list[str],
    distance: np.ndarray,
) -> dict[str, float]:
    index = {node: position for position, node in enumerate(nodes)}
    rows: list[int] = []
    cols: list[int] = []
    weights: list[float] = []
    for node, neighbours in adjacency.items():
        for neighbour, weight in neighbours.items():
            rows.append(index[node])
            cols.append(index[neighbour])
            weights.append(weight)
    if not rows:
        return {
            "merge_graph_num_nodes": float(len(nodes)),
            "merge_graph_num_edges": 0.0,
            "merge_graph_components": float(len(nodes)),
            "merge_graph_cycle_rank": 0.0,
            "merge_graph_mean_degree": 0.0,
            "merge_graph_max_degree": 0.0,
            "merge_graph_mean_path": 0.0,
            "merge_graph_diameter": 0.0,
        }
    matrix = csr_matrix(
        (weights, (rows, cols)), shape=(len(nodes), len(nodes)), dtype=np.float64
    )
    components, _ = connected_components(matrix, directed=False)
    degrees = np.diff(matrix.indptr) if len(nodes) else np.zeros(0)
    off_diagonal = distance[~np.eye(len(nodes), dtype=bool)]
    finite_paths = off_diagonal[np.isfinite(off_diagonal)]
    return {
        "merge_graph_num_nodes": float(len(nodes)),
        "merge_graph_num_edges": float(len(rows) / 2),
        "merge_graph_components": float(components),
        "merge_graph_cycle_rank": float(len(rows) / 2 - len(nodes) + components),
        "merge_graph_mean_degree": float(degrees.mean()) if len(degrees) else 0.0,
        "merge_graph_max_degree": float(degrees.max()) if len(degrees) else 0.0,
        "merge_graph_mean_path": float(finite_paths.mean()) if len(finite_paths) else 0.0,
        "merge_graph_diameter": float(finite_paths.max()) if len(finite_paths) else 0.0,
    }


def merge_graph_features(
    tokenizer: Tokenizer,
    merges: list[tuple[str, str]],
    token_ids: np.ndarray,
    n_points: int,
    seed: int,
    excluded_ids: set[int],
    thresholds: list[float],
    maxdim: int = 1,
) -> dict[str, float]:
    frequencies = np.bincount(token_ids, minlength=tokenizer.get_vocab_size())
    candidates = [
        token_id
        for token_id in range(tokenizer.get_vocab_size())
        if token_id not in excluded_ids
        and token_id < len(frequencies)
        and frequencies[token_id] > 0
        and tokenizer.id_to_token(token_id) is not None
    ]
    candidates.sort(key=lambda token_id: (-int(frequencies[token_id]), token_id))
    chosen_ids = candidates[: int(n_points)]
    nodes = [str(tokenizer.id_to_token(token_id)) for token_id in chosen_ids]
    node_set = set(nodes)
    if len(nodes) < 2:
        return {}
    max_rank = max(1, len(merges))
    adjacency: dict[str, dict[str, float]] = {node: {} for node in nodes}

    def add_edge(left: str, right: str, weight: float) -> None:
        if left == right or left not in node_set or right not in node_set:
            return
        current = adjacency[left].get(right)
        if current is None or weight < current:
            adjacency[left][right] = weight
            adjacency[right][left] = weight

    for rank, (left, right) in enumerate(merges, start=1):
        add_edge(left, right, float(rank))
        parent = left + right
        add_edge(left, parent, float(rank))
        add_edge(right, parent, float(rank))

    index = {node: position for position, node in enumerate(nodes)}
    rows: list[int] = []
    cols: list[int] = []
    weights: list[float] = []
    for node, neighbours in adjacency.items():
        for neighbour, weight in neighbours.items():
            rows.append(index[node])
            cols.append(index[neighbour])
            weights.append(weight)
    if rows:
        matrix = csr_matrix(
            (weights, (rows, cols)), shape=(len(nodes), len(nodes)), dtype=np.float64
        )
        distances = shortest_path(matrix, directed=False)
    else:
        distances = np.full((len(nodes), len(nodes)), float(max_rank + 1))
    distances = np.asarray(distances, dtype=np.float64)
    distances[~np.isfinite(distances)] = float(max_rank + 1)
    np.fill_diagonal(distances, 0.0)
    distances = distances / float(max_rank)
    distances = validate_distance_matrix(distances)
    diagrams = rips_diagrams(distances, maxdim=maxdim)
    features: dict[str, float] = {}
    for dimension, diagram in enumerate(diagrams):
        features.update(
            persistence_summaries(diagram, thresholds, f"merge_graph_h{dimension}")
        )
    features.update(_graph_stats(adjacency, nodes, distances))
    nearest = distances.copy()
    np.fill_diagonal(nearest, np.inf)
    features["merge_graph_num_points"] = float(len(nodes))
    features["merge_graph_mean_nn_distance"] = float(np.mean(nearest.min(axis=1)))
    return features


def merge_feature_row(
    tokenizer: Tokenizer,
    merges: list[tuple[str, str]],
    token_ids: np.ndarray,
    excluded_ids: set[int],
    seed: int,
    thresholds: list[float],
    n_points: int = 512,
    maxdim: int = 1,
) -> dict[str, float]:
    parents, ranks = merge_index(merges)
    features: dict[str, float] = {
        "merge_num_merges": float(len(merges)),
        "merge_num_parents": float(len(parents)),
    }
    features.update(tree_shape_features(tokenizer, parents, excluded_ids))
    features.update(merge_gain_features(tokenizer, parents, ranks, token_ids))
    features.update(
        merge_graph_features(
            tokenizer,
            merges,
            token_ids,
            n_points,
            seed,
            excluded_ids,
            thresholds,
            maxdim=maxdim,
        )
    )
    return features
