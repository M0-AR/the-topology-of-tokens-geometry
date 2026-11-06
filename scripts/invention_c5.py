from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from topobpe.config import load_config, project_root, results_dir
from topobpe.data import iter_split_texts
from topobpe.merge_selection import (
    annotated_merges,
    build_subset_tokenizer,
    compression,
    compression_statistics,
    document_compressions,
    holm_adjust,
    score_merges,
    select_merges,
    strict_reachability,
)
from topobpe.merge_topology import load_bpe_components
from topobpe.tokenizers import load_tokenizer, tokenizer_path
from topobpe.utils import sha256_file, utc_now, write_json

BASE_SIZE = 32768
POOL_BYTES = 16_000_000
SELECTION_BYTES = 4_000_000
STRATEGIES = (
    "high_fertility",
    "low_fertility",
    "high_mismatch",
    "high_rare",
    "high_ttr",
    "first",
    "random",
    "full_16mb",
)
PRIMARY = "high_fertility"


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


def pool_documents(corpus: str, config: dict, max_bytes: int) -> list[str]:
    return sample_texts(corpus, "train", config, max_bytes)


def encode_usage(tokenizer, texts: list[str], vocab_size: int) -> np.ndarray:
    usage = np.zeros(vocab_size, dtype=np.float64)
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        if ids:
            counts = np.bincount(np.asarray(ids, dtype=np.int64), minlength=vocab_size)
            usage[: len(counts)] += counts[:vocab_size]
    return usage


def evaluate_constructed(
    tokenizer,
    texts: list[str],
    entries,
    selected: set[int],
    vocab: dict[str, int],
) -> dict:
    mean, raw, tokens = compression(tokenizer, texts)
    return {
        "test_compression": float(mean),
        "test_bytes": int(raw),
        "test_tokens": int(tokens),
        "actual_vocab_size": int(tokenizer.get_vocab_size()),
        "selected_merges": int(len(selected)),
        "strict_reachability": float(strict_reachability(entries, selected, vocab)),
    }


def document_features(
    docs: list[str], donor, native
) -> dict[str, np.ndarray | float | list[np.ndarray]]:
    donor_ids: list[np.ndarray] = []
    native_counts = np.zeros(len(docs), dtype=np.float64)
    byte_counts = np.zeros(len(docs), dtype=np.float64)
    for index, text in enumerate(docs):
        encoded = text.encode("utf-8")
        byte_counts[index] = len(encoded)
        donor_ids.append(
            np.asarray(
                donor.encode(text, add_special_tokens=False).ids, dtype=np.int64
            )
        )
        native_counts[index] = len(
            native.encode(text, add_special_tokens=False).ids
        )
    if donor_ids:
        combined = np.concatenate(donor_ids)
        type_frequencies = np.bincount(
            combined, minlength=donor.get_vocab_size()
        ).astype(np.float64)
    else:
        type_frequencies = np.zeros(donor.get_vocab_size(), dtype=np.float64)
    nonzero = type_frequencies[type_frequencies > 0]
    rare_threshold = float(np.percentile(nonzero, 25)) if nonzero.size else 0.0
    donor_counts = np.asarray([len(ids) for ids in donor_ids], dtype=np.float64)
    denominator = np.maximum(1.0, byte_counts)
    fertility = donor_counts / denominator
    native_fertility = native_counts / denominator
    rare = np.asarray(
        [
            float(np.sum(type_frequencies[ids] <= rare_threshold)) / max(1, len(ids))
            for ids in donor_ids
        ],
        dtype=np.float64,
    )
    ttr = np.asarray(
        [float(np.unique(ids).size) / max(1, len(ids)) for ids in donor_ids],
        dtype=np.float64,
    )
    return {
        "donor_ids": donor_ids,
        "bytes": byte_counts,
        "donor_tokens": donor_counts,
        "fertility": fertility,
        "mismatch": fertility - native_fertility,
        "rare": rare,
        "ttr": ttr,
        "rare_threshold": rare_threshold,
    }


def strategy_order(
    name: str,
    features: dict,
    seed: int,
) -> list[int]:
    count = len(features["bytes"])
    if name == "first":
        return list(range(count))
    if name == "random":
        return [int(index) for index in np.random.default_rng(seed).permutation(count)]
    if name == "full_16mb":
        return list(range(count))
    if name == "high_fertility":
        values = features["fertility"]
    elif name == "low_fertility":
        values = -np.asarray(features["fertility"], dtype=np.float64)
    elif name == "high_mismatch":
        values = features["mismatch"]
    elif name == "high_rare":
        values = features["rare"]
    elif name == "high_ttr":
        values = features["ttr"]
    else:
        raise ValueError(f"unknown strategy: {name}")
    return sorted(
        range(count),
        key=lambda index: (-float(values[index]), index),
    )


def select_texts(pool: list[str], order: list[int], budget: int) -> list[str]:
    chosen: list[str] = []
    total = 0
    for index in order:
        encoding = pool[index].encode("utf-8")
        if total + len(encoding) > budget:
            remaining = budget - total
            if remaining > 0:
                chosen.append(
                    encoding[:remaining].decode("utf-8", errors="ignore")
                )
            break
        chosen.append(pool[index])
        total += len(encoding)
    return chosen


def build_arm(
    donor_tokenizer,
    vocab: dict[str, int],
    entries,
    texts: list[str],
    eval_texts: list[str],
    merge_budget: int,
) -> tuple[object, dict]:
    usage = encode_usage(
        donor_tokenizer, texts, donor_tokenizer.get_vocab_size()
    )
    scores = score_merges("frequency", entries, vocab, usage, 0)
    selected = select_merges(entries, scores, merge_budget, closure=True)
    tokenizer = build_subset_tokenizer(
        donor_tokenizer, vocab, entries, selected
    )
    row = evaluate_constructed(tokenizer, eval_texts, entries, selected, vocab)
    return tokenizer, row


def run_pair(
    config: dict,
    donor: str,
    target: str,
    budgets: list[int],
    pool_bytes: int,
    selection_bytes: int,
    test_bytes: int,
    random_seeds: list[int],
) -> dict:
    donor_path = tokenizer_path(config, donor, "bpe", BASE_SIZE)
    if not donor_path.exists():
        raise FileNotFoundError(f"missing donor tokenizer: {donor_path}")
    donor_tokenizer = load_tokenizer(config, donor, "bpe", BASE_SIZE)
    target_native = load_tokenizer(config, target, "bpe", BASE_SIZE)
    vocab, merges = load_bpe_components(donor_path)
    entries = annotated_merges(merges)
    initial = len(vocab) - len(merges)

    pool = pool_documents(target, config, pool_bytes)
    pool_bytes_actual = sum(len(text.encode("utf-8")) for text in pool)
    features = document_features(pool, donor_tokenizer, target_native)
    target_test = sample_texts(target, "test", config, test_bytes)
    english_test = (
        sample_texts("wikitext_en", "test", config, test_bytes)
        if target != "wikitext_en"
        else None
    )

    payload: dict = {
        "donor": donor,
        "target": target,
        "donor_sha256": sha256_file(donor_path),
        "pool_bytes_requested": pool_bytes,
        "pool_bytes_actual": int(pool_bytes_actual),
        "pool_documents": len(pool),
        "selection_bytes": selection_bytes,
        "test_bytes": test_bytes,
        "test_documents": len(target_test),
        "initial_vocab_size": initial,
        "rare_threshold": float(features["rare_threshold"]),
        "budgets": {},
    }

    for budget in budgets:
        merge_budget = max(0, budget - initial)
        native = load_tokenizer(config, target, "bpe", budget)
        native_mean, native_raw, native_tokens = compression(native, target_test)
        block: dict = {
            "requested_vocab_size": budget,
            "merge_budget": merge_budget,
            "native": {
                "test_compression": float(native_mean),
                "test_bytes": int(native_raw),
                "test_tokens": int(native_tokens),
                "actual_vocab_size": int(native.get_vocab_size()),
            },
            "methods": {},
            "comparisons": {},
            "document_values": {},
        }
        native_en_mean = None
        if english_test is not None:
            native_en = load_tokenizer(config, "wikitext_en", "bpe", budget)
            native_en_mean = compression(native_en, english_test)[0]

        for strategy in STRATEGIES:
            if strategy == "random":
                rows = []
                values_by_seed: dict[int, list[float]] = {}
                for seed in random_seeds:
                    order = strategy_order(strategy, features, seed)
                    texts = select_texts(pool, order, selection_bytes)
                    tokenizer, row = build_arm(
                        donor_tokenizer,
                        vocab,
                        entries,
                        texts,
                        target_test,
                        merge_budget,
                    )
                    row["seed"] = seed
                    row["selected_bytes"] = int(
                        sum(len(text.encode("utf-8")) for text in texts)
                    )
                    row["selected_documents"] = len(texts)
                    if native_en_mean is not None:
                        english_mean, _, _ = compression(tokenizer, english_test)
                        row["english_retention"] = float(
                            english_mean / max(1e-12, native_en_mean)
                        )
                    else:
                        row["english_retention"] = None
                    rows.append(row)
                    values = document_compressions(tokenizer, target_test)
                    values_by_seed[seed] = values
                    block["document_values"][f"random:{seed}"] = values
                block["methods"][strategy] = {
                    "rows": rows,
                    "mean_compression": float(
                        np.mean([row["test_compression"] for row in rows])
                    ),
                    "min_compression": float(
                        np.min([row["test_compression"] for row in rows])
                    ),
                    "max_compression": float(
                        np.max([row["test_compression"] for row in rows])
                    ),
                    "mean_reachability": float(
                        np.mean([row["strict_reachability"] for row in rows])
                    ),
                    "mean_vocab": float(
                        np.mean([row["actual_vocab_size"] for row in rows])
                    ),
                    "mean_selected_bytes": float(
                        np.mean([row["selected_bytes"] for row in rows])
                    ),
                }
                block["document_values"]["random_mean"] = (
                    np.mean(
                        [values_by_seed[seed] for seed in random_seeds], axis=0
                    ).tolist()
                )
            else:
                order = strategy_order(strategy, features, 0)
                budget_bytes = (
                    pool_bytes if strategy == "full_16mb" else selection_bytes
                )
                texts = select_texts(pool, order, budget_bytes)
                tokenizer, row = build_arm(
                    donor_tokenizer,
                    vocab,
                    entries,
                    texts,
                    target_test,
                    merge_budget,
                )
                row["selected_bytes"] = int(
                    sum(len(text.encode("utf-8")) for text in texts)
                )
                row["selected_documents"] = len(texts)
                if native_en_mean is not None:
                    english_mean, _, _ = compression(tokenizer, english_test)
                    row["english_retention"] = float(
                        english_mean / max(1e-12, native_en_mean)
                    )
                else:
                    row["english_retention"] = None
                block["methods"][strategy] = row
                values = document_compressions(tokenizer, target_test)
                block["document_values"][strategy] = values

        high_values = block["document_values"][PRIMARY]
        random_mean_values = block["document_values"]["random_mean"]
        primary_stats = compression_statistics(
            high_values, random_mean_values
        )
        random_mean_level = float(np.mean(random_mean_values))
        primary_stats["random_mean_level"] = random_mean_level
        primary_stats["relative_gain"] = primary_stats["delta"] / max(
            1e-12, random_mean_level
        )
        block["comparisons"][f"{PRIMARY}_vs_random_mean"] = primary_stats
        for baseline in (
            "first",
            "full_16mb",
            "low_fertility",
            "high_mismatch",
            "high_rare",
            "high_ttr",
        ):
            block["comparisons"][f"{PRIMARY}_vs_{baseline}"] = (
                compression_statistics(high_values, block["document_values"][baseline])
            )
        block["comparisons"]["data_efficiency_vs_full_16mb"] = {
            "high_fertility": block["methods"][PRIMARY]["test_compression"],
            "full_16mb": block["methods"]["full_16mb"]["test_compression"],
            "ratio": block["methods"][PRIMARY]["test_compression"]
            / max(1e-12, block["methods"]["full_16mb"]["test_compression"]),
        }
        payload["budgets"][str(budget)] = block
        print(
            f"  {donor}->{target} {budget}: native "
            f"{block['native']['test_compression']:.4f} "
            f"first {block['methods']['first']['test_compression']:.4f} "
            f"high {block['methods']['high_fertility']['test_compression']:.4f} "
            f"random {block['methods']['random']['mean_compression']:.4f} "
            f"full16 {block['methods']['full_16mb']['test_compression']:.4f}",
            flush=True,
        )
    return payload


def pre_registration() -> dict:
    root = project_root()
    doc = root / "docs/PREREGISTRATION_C4_C5.md"
    sha_path = root / "docs/PREREGISTRATION_C4_C5.sha256"
    lines = sha_path.read_text(encoding="utf-8").splitlines()
    recorded = lines[0].split()[0]
    actual = hashlib.sha256(doc.read_bytes()).hexdigest()
    if actual != recorded:
        raise RuntimeError(f"pre-registration hash mismatch: {actual} != {recorded}")
    frozen_at = lines[1].split(":", 1)[1].strip()
    return {
        "path": str(doc.relative_to(root)),
        "sha256": actual,
        "frozen_at": frozen_at,
    }


def evaluate_primary(payload: dict, budgets: list[int]) -> dict:
    rows = []
    p_values = []
    for pair in payload["pairs"][:3]:
        for budget in map(str, budgets):
            block = pair["budgets"][budget]
            comparison = block["comparisons"][f"{PRIMARY}_vs_random_mean"]
            p_values.append(comparison["p_value"])
            rows.append(
                {
                    "label": f"{pair['donor']}->{pair['target']}:{budget}",
                    "delta": comparison["delta"],
                    "ci_low": comparison["ci_low"],
                    "ci_high": comparison["ci_high"],
                    "p_value": comparison["p_value"],
                    "relative_gain": comparison["relative_gain"],
                    "strict_reachability": block["methods"][PRIMARY][
                        "strict_reachability"
                    ],
                    "actual_vocab_size": block["methods"][PRIMARY][
                        "actual_vocab_size"
                    ],
                    "requested_vocab_size": block["requested_vocab_size"],
                    "data_efficiency_vs_full_16mb": block["comparisons"][
                        "data_efficiency_vs_full_16mb"
                    ]["ratio"],
                }
            )
    adjusted = holm_adjust(p_values)
    for row, p_adjusted in zip(rows, adjusted, strict=True):
        row["p_adjusted"] = p_adjusted
    positive = sum(
        1 for row in rows if row["delta"] > 0 and row["p_adjusted"] < 0.05
    )
    negative = sum(
        1 for row in rows if row["delta"] < 0 and row["p_adjusted"] < 0.05
    )
    mean_relative_gain = float(np.mean([row["relative_gain"] for row in rows]))
    valid = True
    for pair in payload["pairs"][:3]:
        for budget in map(str, budgets):
            block = pair["budgets"][budget]
            requested = block["requested_vocab_size"]
            for label in STRATEGIES:
                if label == "random":
                    method_rows = block["methods"][label]["rows"]
                else:
                    method_rows = [block["methods"][label]]
                for row in method_rows:
                    valid = valid and (
                        row["strict_reachability"] == 1.0
                        and abs(row["actual_vocab_size"] - requested)
                        <= 0.01 * requested
                    )
    passed = (
        positive >= 4
        and negative == 0
        and mean_relative_gain >= 0.005
        and valid
    )
    return {
        "primary_arm": PRIMARY,
        "comparison": "high_fertility_vs_random_mean",
        "rows": rows,
        "positive_significant": positive,
        "negative_significant": negative,
        "mean_relative_gain": mean_relative_gain,
        "valid_construction": valid,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="C5 tokenizer-aware target-data selection"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--pairs",
        default=(
            "wikitext_en:wikipedia_sw,"
            "wikitext_en:wikipedia_ar,"
            "wikitext_en:wikipedia_hu,"
            "wikitext_en:wikipedia_zh,"
            "codeparrot_py:wikitext_en"
        ),
    )
    parser.add_argument("--budgets", type=int, nargs="+", default=[8192, 16384])
    parser.add_argument("--pool-bytes", type=int, default=POOL_BYTES)
    parser.add_argument("--selection-bytes", type=int, default=SELECTION_BYTES)
    parser.add_argument("--test-bytes", type=int, default=1_000_000)
    parser.add_argument(
        "--random-seeds", type=int, nargs="+", default=[13, 37, 73, 101, 149]
    )
    parser.add_argument("--out", default="results/invention_c5.json")
    args = parser.parse_args()

    config = load_config(args.config)
    pairs = [tuple(item.split(":")) for item in args.pairs.split(",") if item]
    payload = {
        "created_at": utc_now(),
        "base_size": BASE_SIZE,
        "pre_registration": pre_registration(),
        "strategies": list(STRATEGIES),
        "primary_arm": PRIMARY,
        "random_seeds": args.random_seeds,
        "pairs": [],
    }
    for donor, target in pairs:
        print(f"C5 {donor} -> {target}", flush=True)
        payload["pairs"].append(
            run_pair(
                config,
                donor,
                target,
                args.budgets,
                args.pool_bytes,
                args.selection_bytes,
                args.test_bytes,
                args.random_seeds,
            )
        )

    payload["primary"] = evaluate_primary(payload, args.budgets)
    write_json(results_dir(config) / Path(args.out).name, payload)
    print(
        f"wrote {results_dir(config) / Path(args.out).name} "
        f"pass={payload['primary']['passed']} "
        f"positive={payload['primary']['positive_significant']}/6 "
        f"mean_rel={payload['primary']['mean_relative_gain']:.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
