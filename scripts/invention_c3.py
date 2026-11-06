from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from topobpe.config import load_config
from topobpe.data import iter_split_texts
from topobpe.merge_selection import (
    annotated_merges,
    build_subset_tokenizer,
    compression,
    score_merges,
    select_merges,
)
from topobpe.merge_topology import load_bpe_components
from topobpe.tokenizers import load_tokenizer, tokenizer_path
from topobpe.utils import write_json

BASE_SIZE = 32768


def sample_texts(corpus: str, split: str, config: dict, max_bytes: int) -> list[str]:
    chosen: list[str] = []
    total = 0
    for text in iter_split_texts(corpus, split, config=config):
        encoded = text.encode("utf-8")
        if total + len(encoded) > max_bytes:
            remaining = max_bytes - total
            if remaining > 0:
                chosen.append(encoded[:remaining].decode("utf-8", errors="ignore"))
            break
        chosen.append(text)
        total += len(encoded)
    return chosen


def encode_usage(tokenizer, texts: list[str], vocab_size: int) -> np.ndarray:
    usage = np.zeros(vocab_size, dtype=np.float64)
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        if ids:
            counts = np.bincount(np.asarray(ids, dtype=np.int64), minlength=vocab_size)
            usage[: len(counts)] += counts[:vocab_size]
    return usage


def icc_2_1(matrix: np.ndarray) -> float:
    values = np.asarray(matrix, dtype=np.float64)
    n, k = values.shape
    if n < 2 or k < 2:
        return float("nan")
    grand = values.mean()
    row_means = values.mean(axis=1)
    column_means = values.mean(axis=0)
    ss_rows = k * float(((row_means - grand) ** 2).sum())
    ss_columns = n * float(((column_means - grand) ** 2).sum())
    residuals = values - row_means[:, None] - column_means[None, :] + grand
    ss_error = float((residuals**2).sum())
    ms_rows = ss_rows / (n - 1)
    ms_columns = ss_columns / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_columns - ms_error) / n
    if denominator == 0:
        return float("nan")
    return float((ms_rows - ms_error) / denominator)


def stability_analysis(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["records"]
    features = {
        "entropy_per_log_bar": lambda row: row["h1_persistence_entropy"]
        / math.log(max(2.0, row["h1_num_bars"])),
        "total_per_bar": lambda row: row["h1_total_persistence"]
        / max(1.0, row["h1_num_bars"]),
        "mean_persistence": lambda row: row["h1_mean_persistence"],
        "raw_total_persistence": lambda row: row["h1_total_persistence"],
    }
    groups: dict[tuple[str, int], list[dict]] = {}
    for record in records:
        groups.setdefault((record["family"], int(record["vocab_size"])), []).append(record)
    output: dict[str, dict[str, float]] = {}
    for (family, vocab_size), members in sorted(groups.items()):
        members = sorted(members, key=lambda record: record["corpus"])
        output[f"{family}:{vocab_size}"] = {}
        for feature, extractor in features.items():
            matrix = np.array(
                [
                    [extractor(row) for row in member["vocab_rows"]]
                    for member in members
                ],
                dtype=np.float64,
            )
            output[f"{family}:{vocab_size}"][feature] = icc_2_1(matrix)
    return output


def ablation_analysis(
    config: dict,
    donor: str,
    target: str,
    budget: int,
    train_bytes: int,
    test_bytes: int,
    count: int,
    seed: int,
) -> dict:
    donor_tokenizer = load_tokenizer(config, donor, "bpe", BASE_SIZE)
    vocab, merges = load_bpe_components(tokenizer_path(config, donor, "bpe", BASE_SIZE))
    entries = annotated_merges(merges)
    initial = len(vocab) - len(merges)
    merge_budget = max(0, budget - initial)
    train_texts = sample_texts(target, "train", config, train_bytes)
    test_texts = sample_texts(target, "test", config, test_bytes)
    usage = encode_usage(donor_tokenizer, train_texts, donor_tokenizer.get_vocab_size())
    frequency = score_merges("frequency", entries, vocab, usage, seed=0)
    selected = select_merges(entries, frequency, merge_budget, closure=True)
    full_tokenizer = build_subset_tokenizer(donor_tokenizer, vocab, entries, selected)
    full_compression = compression(full_tokenizer, test_texts)[0]

    selected_entries = sorted(
        (entry for entry in entries if entry.rank in selected),
        key=lambda entry: -frequency[entry.rank - 1],
    )
    targets = selected_entries[:count]
    rng = np.random.default_rng(seed)
    random_ranks = rng.choice(
        [entry.rank for entry in selected_entries],
        size=min(count, len(selected_entries)),
        replace=False,
    )
    random_entries = [entry for entry in entries if entry.rank in set(random_ranks)]

    def measure(chosen: list) -> dict[str, object]:
        rows = []
        for entry in chosen:
            reduced = set(selected)
            reduced.discard(entry.rank)
            tokenizer = build_subset_tokenizer(donor_tokenizer, vocab, entries, reduced)
            value = compression(tokenizer, test_texts)[0]
            rows.append(
                {
                    "rank": entry.rank,
                    "parent": entry.parent,
                    "loss": float(full_compression - value),
                }
            )
        return {"full_compression": float(full_compression), "rows": rows}

    frequency_ablation = measure(targets)
    random_ablation = measure(random_entries)

    scores = {
        method: score_merges(method, entries, vocab, usage, seed=seed)
        for method in ("frequency", "persistence", "combined", "redundancy", "rank_fusion")
    }
    correlations: dict[str, dict[str, float]] = {}
    for label, block in (("top_frequency", frequency_ablation), ("random", random_ablation)):
        losses = np.array([row["loss"] for row in block["rows"]], dtype=np.float64)
        correlations[label] = {}
        for method, values in scores.items():
            predictor = np.array(
                [values[row["rank"] - 1] for row in block["rows"]], dtype=np.float64
            )
            if losses.size >= 3 and np.std(predictor) > 0 and np.std(losses) > 0:
                value = spearmanr(predictor, losses).statistic
                correlations[label][method] = float(value) if np.isfinite(value) else float("nan")
            else:
                correlations[label][method] = float("nan")
        correlations[label]["mean_loss"] = float(losses.mean())
        correlations[label]["positive_loss_fraction"] = float((losses > 0).mean())
    gain = correlations["top_frequency"].get("combined", float("nan")) - correlations[
        "top_frequency"
    ].get("frequency", float("nan"))
    return {
        "donor": donor,
        "target": target,
        "budget": budget,
        "merge_budget": merge_budget,
        "selected_merges": len(selected),
        "full_compression": float(full_compression),
        "ablation": {"top_frequency": frequency_ablation, "random": random_ablation},
        "correlations": correlations,
        "combined_minus_frequency_spearman": float(gain),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--stability", default="results/tda_stability.json")
    parser.add_argument("--out", default="results/invention_c3.json")
    parser.add_argument("--train-bytes", type=int, default=4_000_000)
    parser.add_argument("--test-bytes", type=int, default=1_000_000)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    config = load_config(args.config)
    stability = stability_analysis(Path(args.stability))
    ablation = ablation_analysis(
        config,
        "wikitext_en",
        "wikipedia_sw",
        8192,
        args.train_bytes,
        args.test_bytes,
        args.count,
        args.seed,
    )
    payload = {"stability_icc": stability, "ablation": ablation}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, payload)
    print(f"wrote {out}")
    for key, values in stability.items():
        print(key, {name: round(value, 3) for name, value in values.items()})
    print("ablation correlations:", ablation["correlations"])
    print("combined - frequency Spearman:", round(ablation["combined_minus_frequency_spearman"], 3))


if __name__ == "__main__":
    main()
