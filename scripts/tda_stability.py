from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from topobpe.config import load_config, results_dir
from topobpe.data import load_split_texts
from topobpe.features import (
    _encode_documents,
    _encode_documents_with_separator,
    _seed_for,
    cooccurrence_distance,
    cosine_distance_matrix,
    sample_vocabulary_tokens,
)
from topobpe.tokenizers import load_tokenizer, special_token_ids
from topobpe.topology import (
    compare_diagrams,
    persistence_summaries,
    rips_diagrams,
)


def _summaries(diagram: np.ndarray, prefix: str, thresholds: list[float]) -> dict[str, float]:
    return persistence_summaries(diagram, thresholds, prefix)


def run_corpus(
    config: dict,
    corpus: str,
    family: str,
    vocab_size: int,
    sizes: list[int],
) -> dict:
    tokenizer = load_tokenizer(config, corpus, family, vocab_size)
    texts = load_split_texts(corpus, "train", config=config)
    train_ids = _encode_documents(tokenizer, texts)
    eos_id = tokenizer.token_to_id("<eos>")
    excluded = special_token_ids(tokenizer, list(config["tokenizers"]["special_tokens"]))
    if eos_id is not None:
        excluded.add(int(eos_id))
    cooc_ids = _encode_documents_with_separator(tokenizer, texts, eos_id)
    frequencies = np.bincount(train_ids, minlength=tokenizer.get_vocab_size())
    seed = _seed_for(corpus, family, str(vocab_size), str(config["project"]["seed"]))
    dim = int(config["features"]["byte_ngram_dim"])
    max_n = int(config["features"]["byte_ngram_max"])
    thresholds = [float(value) for value in config["features"]["persistence_thresholds"]]
    max_size = max(sizes)

    _, points_all = sample_vocabulary_tokens(
        tokenizer, frequencies, max_size, seed, excluded, dim, max_n
    )
    order = np.random.default_rng(seed + max_size).permutation(len(points_all))
    points_all = points_all[order]
    vocab_rows: dict[int, dict[str, float]] = {}
    vocab_agreement: dict[str, object] = {}
    vocab_distance_full: dict[int, np.ndarray] = {}
    for size in sizes:
        if len(points_all) < size:
            continue
        points = points_all[:size]
        distance = cosine_distance_matrix(points)
        vocab_distance_full[size] = distance
        diagrams = rips_diagrams(distance, maxdim=int(config["features"]["maxdim"]))
        row = {"corpus": corpus, "cloud": "vocab", "points": size}
        row.update(_summaries(diagrams[1], "h1", thresholds))
        vocab_rows[size] = row
    if 256 in vocab_distance_full:
        vocab_agreement = compare_diagrams(vocab_distance_full[256], maxdim=1, tolerance=1e-6)

    cooc_rows: dict[int, dict[str, float]] = {}
    cooc_agreement: dict[str, object] = {}
    cooc_distance_full: dict[int, np.ndarray] = {}
    for size in sizes:
        chosen, distance, _ = cooccurrence_distance(
            tokenizer,
            cooc_ids,
            size,
            int(config["features"]["cooccurrence_window"]),
            max_tokens=2_000_000,
            excluded_ids=excluded,
        )
        if len(chosen) < 2:
            continue
        cooc_distance_full[size] = distance
        diagrams = rips_diagrams(distance, maxdim=int(config["features"]["maxdim"]))
        row = {"corpus": corpus, "cloud": "cooc", "points": int(len(chosen))}
        diagram_h1 = diagrams[1] if len(diagrams) > 1 else diagrams[0]
        row.update(_summaries(diagram_h1, "h1", thresholds))
        cooc_rows[size] = row
    if 256 in cooc_distance_full:
        cooc_agreement = compare_diagrams(cooc_distance_full[256], maxdim=1, tolerance=1e-6)

    metrics = ["h1_total_persistence", "h1_persistence_entropy", "h1_num_bars"]
    stability: dict[str, dict[str, float | None]] = {}
    for cloud, rows in (("vocab", vocab_rows), ("cooc", cooc_rows)):
        for metric in metrics:
            available = [size for size in sizes if size in rows]
            for left_index, left in enumerate(available):
                for right in available[left_index + 1 :]:
                    if metric not in rows[left] or metric not in rows[right]:
                        continue
                    value = rows[left][metric] - rows[right][metric]
                    stability[f"{cloud}:{metric}:{left}-{right}"] = {
                        "difference": float(value),
                        "relative": float(value / max(abs(rows[right][metric]), 1e-12)),
                    }
    return {
        "corpus": corpus,
        "family": family,
        "vocab_size": vocab_size,
        "vocab_rows": [vocab_rows[size] for size in sorted(vocab_rows)],
        "cooc_rows": [cooc_rows[size] for size in sorted(cooc_rows)],
        "vocab_ripser_gudhi_256": vocab_agreement,
        "cooc_ripser_gudhi_256": cooc_agreement,
        "stability": stability,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="TDA stability and cross-library agreement")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--family", default="bpe")
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--corpora", nargs="*", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    corpora = args.corpora or [spec["name"] for spec in config["data"]["corpora"]]
    records = []
    for corpus in corpora:
        print(f"stability {corpus} {args.family} {args.vocab_size}", flush=True)
        records.append(run_corpus(config, corpus, args.family, args.vocab_size, args.sizes))
    tables = results_dir(config) / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in records:
        for row in record["vocab_rows"] + record["cooc_rows"]:
            row["family"] = record["family"]
            row["vocab_size"] = record["vocab_size"]
            rows.append(row)
    pd.DataFrame(rows).to_csv(tables / "tda_stability.csv", index=False)
    payload = {"records": records}
    output = results_dir(config) / "tda_stability.json"
    output.write_text(json.dumps(payload, indent=2, default=str))
    agreements = []
    for record in records:
        for cloud in ("vocab", "cooc"):
            block = record[f"{cloud}_ripser_gudhi_256"]
            for dimension, values in block.get("dimensions", {}).items():
                agreements.append(
                    {
                        "corpus": record["corpus"],
                        "cloud": cloud,
                        "dimension": dimension,
                        "agree": values["agree"],
                        "bottleneck_distance": values["bottleneck_distance"],
                    }
                )
    pd.DataFrame(agreements).to_csv(tables / "tda_library_agreement.csv", index=False)
    print(pd.DataFrame(agreements).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
