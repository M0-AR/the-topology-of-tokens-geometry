from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from topobpe.config import load_config, results_dir
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
METHODS = [
    "prefix",
    "frequency",
    "persistence",
    "combined",
    "redundancy",
    "rank_fusion",
    "random",
]
PRIMARY_ARM = "combined"


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


def evaluate_tokenizer(
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


def run_pair(
    config: dict,
    donor: str,
    target: str,
    budgets: list[int],
    train_bytes: int,
    test_bytes: int,
    random_seeds: list[int],
) -> dict:
    donor_path = tokenizer_path(config, donor, "bpe", BASE_SIZE)
    if not donor_path.exists():
        raise FileNotFoundError(f"missing donor tokenizer: {donor_path}")
    donor_tokenizer = load_tokenizer(config, donor, "bpe", BASE_SIZE)
    vocab, merges = load_bpe_components(donor_path)
    entries = annotated_merges(merges)
    initial = len(vocab) - len(merges)

    target_train = sample_texts(target, "train", config, train_bytes)
    target_test = sample_texts(target, "test", config, test_bytes)
    english_test = (
        sample_texts("wikitext_en", "test", config, test_bytes)
        if target != "wikitext_en"
        else None
    )
    usage = encode_usage(donor_tokenizer, target_train, donor_tokenizer.get_vocab_size())

    native_cache: dict[int, dict] = {}
    for budget in budgets:
        native = load_tokenizer(config, target, "bpe", budget)
        native_cache[budget] = evaluate_tokenizer(
            native, target_test, entries, set(), vocab
        )
        native_cache[budget]["strict_reachability"] = None

    payload: dict = {
        "donor": donor,
        "target": target,
        "donor_sha256": sha256_file(donor_path),
        "train_bytes": train_bytes,
        "test_bytes": test_bytes,
        "train_documents": len(target_train),
        "test_documents": len(target_test),
        "initial_vocab_size": initial,
        "budgets": {},
    }

    for budget in budgets:
        merge_budget = max(0, budget - initial)
        block: dict = {
            "requested_vocab_size": budget,
            "merge_budget": merge_budget,
            "native": native_cache[budget],
            "methods": {},
            "comparisons": {},
        }
        doc_values: dict[str, list[float]] = {}
        random_rows: dict[str, list[float]] = {}
        native_en = (
            load_tokenizer(config, "wikitext_en", "bpe", budget)
            if english_test is not None
            else None
        )
        native_en_mean = (
            compression(native_en, english_test)[0] if native_en is not None else None
        )
        for method in METHODS:
            seeds = random_seeds if method == "random" else [0]
            rows = []
            for seed in seeds:
                scores = score_merges(method, entries, vocab, usage, seed)
                selected = select_merges(entries, scores, merge_budget, closure=True)
                tokenizer = build_subset_tokenizer(
                    donor_tokenizer, vocab, entries, selected
                )
                row = evaluate_tokenizer(
                    tokenizer, target_test, entries, selected, vocab
                )
                row["seed"] = seed
                values = document_compressions(tokenizer, target_test)
                doc_values[f"{method}:{seed}"] = values
                if method == "random":
                    random_rows.setdefault("compression", []).append(
                        row["test_compression"]
                    )
                    random_rows.setdefault("reachability", []).append(
                        row["strict_reachability"]
                    )
                    random_rows.setdefault("vocab", []).append(row["actual_vocab_size"])
                if english_test is not None:
                    english_mean, _, _ = compression(tokenizer, english_test)
                    row["english_retention"] = float(
                        english_mean / max(1e-12, native_en_mean)
                    )
                else:
                    row["english_retention"] = None
                rows.append(row)
            if method == "random":
                block["methods"][method] = {
                    "rows": rows,
                    "mean_compression": float(np.mean(random_rows["compression"])),
                    "min_compression": float(np.min(random_rows["compression"])),
                    "max_compression": float(np.max(random_rows["compression"])),
                    "mean_reachability": float(np.mean(random_rows["reachability"])),
                    "mean_vocab": float(np.mean(random_rows["vocab"])),
                }
            else:
                block["methods"][method] = rows[0]

        baseline = doc_values["frequency:0"]
        for arm in ("prefix", "persistence", "combined", "redundancy", "rank_fusion"):
            stats = compression_statistics(doc_values[f"{arm}:0"], baseline)
            block["comparisons"][f"{arm}_vs_frequency"] = stats
        random_deltas = []
        for seed in random_seeds:
            stats = compression_statistics(doc_values[f"random:{seed}"], baseline)
            random_deltas.append(stats["delta"])
        block["comparisons"]["random_vs_frequency"] = {
            "delta": float(np.mean(random_deltas)),
            "min_delta": float(np.min(random_deltas)),
            "max_delta": float(np.max(random_deltas)),
            "seeds": random_seeds,
        }
        block["comparisons"]["frequency_vs_native"] = {
            "delta": float(
                block["methods"]["frequency"]["test_compression"]
                - block["native"]["test_compression"]
            )
        }
        payload["budgets"][str(budget)] = block
        print(
            f"  {donor}->{target} {budget}: native {block['native']['test_compression']:.4f} "
            f"prefix {block['methods']['prefix']['test_compression']:.4f} "
            f"freq {block['methods']['frequency']['test_compression']:.4f} "
            f"combined {block['methods']['combined']['test_compression']:.4f} "
            f"persistence {block['methods']['persistence']['test_compression']:.4f} "
            f"redundancy {block['methods']['redundancy']['test_compression']:.4f} "
            f"rank_fusion {block['methods']['rank_fusion']['test_compression']:.4f} "
            f"random {block['methods']['random']['mean_compression']:.4f}",
            flush=True,
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="C1 persistence-selected merge cores")
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
    parser.add_argument("--train-bytes", type=int, default=4_000_000)
    parser.add_argument("--test-bytes", type=int, default=1_000_000)
    parser.add_argument("--random-seeds", type=int, nargs="+", default=[13, 37, 73])
    parser.add_argument("--out", default="results/invention_c1.json")
    args = parser.parse_args()

    config = load_config(args.config)
    pairs = [tuple(item.split(":")) for item in args.pairs.split(",") if item]
    payload = {
        "created_at": utc_now(),
        "base_size": BASE_SIZE,
        "methods": METHODS,
        "primary_arm": PRIMARY_ARM,
        "random_seeds": args.random_seeds,
        "pairs": [],
    }
    for donor, target in pairs:
        print(f"C1 {donor} -> {target}", flush=True)
        payload["pairs"].append(
            run_pair(
                config,
                donor,
                target,
                args.budgets,
                args.train_bytes,
                args.test_bytes,
                args.random_seeds,
            )
        )

    primary_p_values = []
    primary_labels = []
    for pair in payload["pairs"][:3]:
        for budget in map(str, args.budgets):
            comparison = pair["budgets"][budget]["comparisons"][
                f"{PRIMARY_ARM}_vs_frequency"
            ]
            primary_p_values.append(comparison["p_value"])
            primary_labels.append(f"{pair['donor']}->{pair['target']}:{budget}")
    adjusted = holm_adjust(primary_p_values)
    payload["primary_holm"] = [
        {"label": label, "p_value": p_value, "p_adjusted": adjusted_value}
        for label, p_value, adjusted_value in zip(
            primary_labels, primary_p_values, adjusted, strict=True
        )
    ]
    write_json(results_dir(config) / Path(args.out).name, payload)
    print(f"wrote {results_dir(config) / Path(args.out).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
