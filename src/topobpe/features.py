from __future__ import annotations

import gzip
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import artifact_dir, results_dir
from .data import load_split_texts
from .tokenizers import (
    load_tokenizer,
    special_token_ids,
    token_byte_lengths,
    token_bytes,
)
from .topology import (
    betti_curve,
    mean_nearest_neighbor_distance,
    persistence_diagram_payload,
    persistence_summaries,
    rips_diagrams,
    validate_distance_matrix,
)
from .utils import (
    byte_entropy,
    conditional_entropy,
    read_json,
    sha256_file,
    token_entropy,
    write_json,
)

BASELINE_COLUMNS = [
    "actual_vocab_size",
    "byte_entropy",
    "token_entropy",
    "conditional_entropy",
    "type_token_ratio",
    "gzip_ratio",
    "mean_token_bytes",
    "train_compression",
]

TOPOLOGY_PREFIXES = ("vocab_h0_", "vocab_h1_", "cooc_h0_", "cooc_h1_")
TOPOLOGY_EXTRA = (
    "vocab_mean_nn_distance",
    "cooc_mean_nn_distance",
    "vocab_num_points",
    "cooc_num_points",
)


def _seed_for(*parts: str) -> int:
    return int(zlib.crc32("|".join(parts).encode("utf-8")) % (2**31 - 1))


def byte_ngram_profile(data: bytes, dim: int, max_n: int) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float64)
    for n in range(1, max_n + 1):
        if len(data) < n:
            continue
        weight = 1.0 / n
        for start in range(len(data) - n + 1):
            bucket = zlib.crc32(data[start : start + n]) % dim
            vector[bucket] += weight
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector


def cosine_distance_matrix(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or len(points) == 0:
        return np.zeros((0, 0), dtype=np.float64)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    normalized = points / np.clip(norms, 1e-12, None)
    similarity = normalized @ normalized.T
    return validate_distance_matrix(1.0 - similarity)


def sample_vocabulary_tokens(
    tokenizer: Any,
    frequencies: np.ndarray,
    n_points: int,
    seed: int,
    excluded_ids: set[int],
    dim: int,
    max_n: int,
) -> tuple[np.ndarray, np.ndarray]:
    vocab_size = tokenizer.get_vocab_size()
    candidates = np.asarray(
        [
            index
            for index in range(vocab_size)
            if index not in excluded_ids and frequencies[index] > 0
        ],
        dtype=np.int64,
    )
    if len(candidates) == 0:
        return candidates, np.zeros((0, dim), dtype=np.float64)
    weights = frequencies[candidates].astype(np.float64)
    total = float(weights.sum())
    if total <= 0:
        weights = np.ones_like(weights)
        total = float(weights.sum())
    if len(candidates) <= n_points:
        chosen = candidates
    else:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(candidates, size=n_points, replace=False, p=weights / total)
    chosen = np.sort(chosen)
    points = np.stack(
        [byte_ngram_profile(token_bytes(tokenizer, int(index)), dim, max_n) for index in chosen],
        axis=0,
    )
    return chosen, points


def cooccurrence_distance(
    tokenizer: Any,
    token_ids: np.ndarray,
    n_tokens: int,
    window: int,
    max_tokens: int,
    excluded_ids: set[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vocab_size = tokenizer.get_vocab_size()
    frequencies = np.bincount(token_ids, minlength=vocab_size)
    candidates = [
        index
        for index in range(vocab_size)
        if index not in excluded_ids and frequencies[index] > 0
    ]
    candidates.sort(key=lambda index: (-int(frequencies[index]), index))
    chosen = np.asarray(candidates[:n_tokens], dtype=np.int64)
    if len(chosen) == 0:
        return chosen, np.zeros((0, 0), dtype=np.float64), np.zeros((0, 0), dtype=np.float64)
    mapping = -np.ones(vocab_size, dtype=np.int64)
    mapping[chosen] = np.arange(len(chosen), dtype=np.int64)
    stream = np.asarray(token_ids[:max_tokens], dtype=np.int64)
    compact = mapping[stream]
    count = np.zeros((len(chosen), len(chosen)), dtype=np.float64)
    for offset in range(1, window + 1):
        if len(compact) <= offset:
            break
        left = compact[:-offset]
        right = compact[offset:]
        valid = (left >= 0) & (right >= 0)
        if np.any(valid):
            np.add.at(count, (left[valid], right[valid]), 1.0)
    count = count + count.T
    total = float(count.sum())
    if total <= 0:
        pmi = np.zeros_like(count)
    else:
        row = count.sum(axis=1, keepdims=True)
        col = count.sum(axis=0, keepdims=True)
        pmi = np.log((count + 1e-12) * total / (row @ col + 1e-12))
        pmi = np.clip(pmi, -10.0, 10.0)
        pmi = 0.5 * (pmi + pmi.T)
        np.fill_diagonal(pmi, 0.0)
    distance = float(pmi.max()) - pmi if pmi.size else pmi
    return chosen, validate_distance_matrix(distance), pmi


def _gzip_ratio(raw: bytes, max_bytes: int = 20_000_000) -> float:
    sample = raw[:max_bytes]
    if not sample:
        return 0.0
    compressed = gzip.compress(sample, compresslevel=6)
    return len(compressed) / len(sample)


def baseline_features(
    train_texts: list[str],
    test_texts: list[str],
    tokenizer: Any,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
) -> dict[str, float]:
    train_raw = b"".join(text.encode("utf-8") for text in train_texts)
    test_raw = b"".join(text.encode("utf-8") for text in test_texts)
    vocab_size = tokenizer.get_vocab_size()
    frequencies = np.bincount(train_ids, minlength=vocab_size)
    lengths = token_byte_lengths(tokenizer)
    usable = min(vocab_size, len(lengths))
    token_byte_total = float(
        (frequencies[:usable] * lengths[:usable]).sum() / max(1, len(train_ids))
    )
    return {
        "actual_vocab_size": float(vocab_size),
        "byte_entropy": byte_entropy(train_raw),
        "token_entropy": token_entropy(train_ids),
        "conditional_entropy": conditional_entropy(train_ids, max_pairs=1_000_000),
        "type_token_ratio": float(np.count_nonzero(frequencies) / max(1, len(train_ids))),
        "gzip_ratio": _gzip_ratio(train_raw),
        "mean_token_bytes": token_byte_total,
        "train_compression": float(len(train_raw) / max(1, len(train_ids))),
        "test_compression": float(len(test_raw) / max(1, len(test_ids))),
        "train_bytes": float(len(train_raw)),
        "test_bytes": float(len(test_raw)),
        "train_tokens": float(len(train_ids)),
        "test_tokens": float(len(test_ids)),
    }


def _feature_config_hash(config: dict[str, Any]) -> str:
    import hashlib
    import json

    payload = json.dumps(
        {
            "features": config["features"],
            "special_tokens": config["tokenizers"]["special_tokens"],
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _feature_cache_path(config: dict[str, Any], corpus: str, family: str, vocab_size: int) -> Path:
    tag = _feature_config_hash(config)
    return artifact_dir(config) / "features" / corpus / f"{family}_{vocab_size}_{tag}.json"


def _topology_cache_dir(config: dict[str, Any], corpus: str) -> Path:
    path = artifact_dir(config) / "topology" / corpus
    path.mkdir(parents=True, exist_ok=True)
    return path


def _encode_documents(tokenizer: Any, texts: list[str]) -> np.ndarray:
    pieces = [tokenizer.encode(text, add_special_tokens=False).ids for text in texts]
    if not pieces:
        return np.zeros(0, dtype=np.int64)
    return np.asarray([token for piece in pieces for token in piece], dtype=np.int64)


def _encode_documents_with_separator(
    tokenizer: Any, texts: list[str], separator_id: int | None
) -> np.ndarray:
    pieces: list[int] = []
    for text in texts:
        pieces.extend(tokenizer.encode(text, add_special_tokens=False).ids)
        if separator_id is not None:
            pieces.append(separator_id)
    return np.asarray(pieces, dtype=np.int64)


def extract_features(
    config: dict[str, Any],
    corpus: str,
    family: str,
    vocab_size: int,
    force: bool = False,
) -> dict[str, Any]:
    cache_path = _feature_cache_path(config, corpus, family, vocab_size)
    if cache_path.exists() and not force:
        return read_json(cache_path)
    tokenizer = load_tokenizer(config, corpus, family, vocab_size)
    train_texts = load_split_texts(corpus, "train", config=config)
    test_texts = load_split_texts(corpus, "test", config=config)
    if not train_texts or not test_texts:
        raise RuntimeError(f"missing train/test text for {corpus}")
    train_ids = _encode_documents(tokenizer, train_texts)
    test_ids = _encode_documents(tokenizer, test_texts)
    excluded = special_token_ids(tokenizer, list(config["tokenizers"]["special_tokens"]))
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is not None:
        excluded.add(int(eos_id))
    cooc_ids = _encode_documents_with_separator(tokenizer, train_texts, eos_id)
    frequencies = np.bincount(train_ids, minlength=tokenizer.get_vocab_size())
    seed = _seed_for(corpus, family, str(vocab_size), str(config["project"]["seed"]))
    feature_cfg = config["features"]
    dim = int(feature_cfg["byte_ngram_dim"])
    max_n = int(feature_cfg["byte_ngram_max"])
    chosen_vocab, vocab_points = sample_vocabulary_tokens(
        tokenizer,
        frequencies,
        int(feature_cfg["vocab_points"]),
        seed,
        excluded,
        dim,
        max_n,
    )
    vocab_distance = cosine_distance_matrix(vocab_points)
    vocab_diagrams = rips_diagrams(vocab_distance, maxdim=int(feature_cfg["maxdim"]))
    chosen_cooc, cooc_distance, pmi = cooccurrence_distance(
        tokenizer,
        cooc_ids,
        int(feature_cfg["cooccurrence_tokens"]),
        int(feature_cfg["cooccurrence_window"]),
        max_tokens=2_000_000,
        excluded_ids=excluded,
    )
    cooc_diagrams = rips_diagrams(cooc_distance, maxdim=int(feature_cfg["maxdim"]))
    thresholds = [float(value) for value in feature_cfg["persistence_thresholds"]]
    features: dict[str, Any] = {
        "corpus": corpus,
        "family": family,
        "requested_vocab_size": int(vocab_size),
        "feature_seed": seed,
    }
    features.update(baseline_features(train_texts, test_texts, tokenizer, train_ids, test_ids))
    for dimension, diagram in enumerate(vocab_diagrams):
        features.update(persistence_summaries(diagram, thresholds, f"vocab_h{dimension}"))
    for dimension, diagram in enumerate(cooc_diagrams):
        features.update(persistence_summaries(diagram, thresholds, f"cooc_h{dimension}"))
    features["vocab_mean_nn_distance"] = mean_nearest_neighbor_distance(vocab_distance)
    features["cooc_mean_nn_distance"] = mean_nearest_neighbor_distance(cooc_distance)
    features["vocab_num_points"] = float(len(chosen_vocab))
    features["cooc_num_points"] = float(len(chosen_cooc))
    features["tokenizer_sha256"] = sha256_file(
        artifact_dir(config) / "tokenizers" / corpus / f"{family}_{vocab_size}.json"
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(cache_path, features)
    topology_dir = _topology_cache_dir(config, corpus)
    stem = f"{family}_{vocab_size}"
    np.save(topology_dir / f"{stem}_vocab_ids.npy", chosen_vocab)
    np.save(topology_dir / f"{stem}_vocab_points.npy", vocab_points)
    np.save(topology_dir / f"{stem}_vocab_distance.npy", vocab_distance)
    np.savez(
        topology_dir / f"{stem}_vocab_diagrams.npz",
        **{f"h{index}": diagram for index, diagram in enumerate(vocab_diagrams)},
    )
    np.save(topology_dir / f"{stem}_cooc_ids.npy", chosen_cooc)
    np.save(topology_dir / f"{stem}_cooc_pmi.npy", pmi)
    np.save(topology_dir / f"{stem}_cooc_distance.npy", cooc_distance)
    np.savez(
        topology_dir / f"{stem}_cooc_diagrams.npz",
        **{f"h{index}": diagram for index, diagram in enumerate(cooc_diagrams)},
    )
    diagnostics = {
        "vocab_points": len(chosen_vocab),
        "cooc_points": len(chosen_cooc),
        "vocab_betti_grid": [0.0, *thresholds],
        "vocab_betti_curve": betti_curve(
            vocab_diagrams[1] if len(vocab_diagrams) > 1 else vocab_diagrams[0],
            np.asarray([0.0, *thresholds], dtype=np.float64),
        ).tolist(),
        "cooc_betti_curve": betti_curve(
            cooc_diagrams[1] if len(cooc_diagrams) > 1 else cooc_diagrams[0],
            np.asarray([0.0, *thresholds], dtype=np.float64),
        ).tolist(),
        "vocab_diagrams": [persistence_diagram_payload(d) for d in vocab_diagrams],
        "cooc_diagrams": [persistence_diagram_payload(d) for d in cooc_diagrams],
    }
    write_json(topology_dir / f"{stem}_diagnostics.json", diagnostics)
    return features


def build_feature_table(config: dict[str, Any], force: bool = False) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    families = list(config["tokenizers"]["families"])
    vocab_sizes = [int(value) for value in config["tokenizers"]["vocab_sizes"]]
    for spec in config["data"]["corpora"]:
        corpus = spec["name"]
        for family in families:
            for vocab_size in vocab_sizes:
                rows.append(extract_features(config, corpus, family, vocab_size, force=force))
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["corpus", "family", "requested_vocab_size"]).reset_index(drop=True)
    path = results_dir(config) / "features.csv"
    frame.to_csv(path, index=False)
    return frame
