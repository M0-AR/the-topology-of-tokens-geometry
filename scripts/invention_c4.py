from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from topobpe.config import load_config, project_root, results_dir
from topobpe.data import iter_split_texts
from topobpe.hybrid import (
    build_hybrid_tokenizer,
    build_tokenizer_from_entries,
    filter_new_merges,
    learn_token_merges,
    prune_unreachable,
)
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
SHARES = (0.25, 0.50, 0.75)
PRIMARY_SHARE = 0.50
PRIMARY_LABEL = "continuation_50"


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


def evaluate_plain(tokenizer, texts: list[str]) -> dict:
    mean, raw, tokens = compression(tokenizer, texts)
    return {
        "test_compression": float(mean),
        "test_bytes": int(raw),
        "test_tokens": int(tokens),
        "actual_vocab_size": int(tokenizer.get_vocab_size()),
    }


def evaluate_constructed(
    tokenizer,
    texts: list[str],
    entries,
    selected: set[int],
    vocab: dict[str, int],
) -> dict:
    row = evaluate_plain(tokenizer, texts)
    row["selected_merges"] = int(len(selected))
    row["strict_reachability"] = float(strict_reachability(entries, selected, vocab))
    return row


def fallback_merges(
    entries,
    frequency_scores: np.ndarray,
    selected_donor: set[int],
    needed: int,
    accepted_new: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    if needed <= 0:
        return []
    expanded = select_merges(
        entries, frequency_scores, len(selected_donor) + needed, closure=True
    )
    extras = sorted(
        (entry for entry in entries if entry.rank in (expanded - selected_donor)),
        key=lambda entry: entry.rank,
    )
    forbidden = {entry.parent for entry in entries if entry.rank in selected_donor}
    forbidden.update(left + right for left, right in accepted_new)
    chosen: list[tuple[str, str]] = []
    for entry in extras:
        if len(chosen) >= needed:
            break
        if entry.parent in forbidden:
            continue
        chosen.append((entry.left, entry.right))
        forbidden.add(entry.parent)
    return chosen


def continuation_arm(
    base,
    vocab: dict[str, int],
    entries,
    selected_donor: set[int],
    frequency_scores: np.ndarray,
    target_train: list[str],
    merge_budget: int,
) -> tuple[object, list, dict]:
    partial = build_subset_tokenizer(base, vocab, entries, selected_donor)
    remaining = max(0, merge_budget - len(selected_donor))
    learned, diagnostics = learn_token_merges(partial, target_train, remaining)
    donor_parents = {
        entry.parent for entry in entries if entry.rank in selected_donor
    }
    filtered = filter_new_merges(learned, donor_parents, remaining)
    fallback = fallback_merges(
        entries,
        frequency_scores,
        selected_donor,
        remaining - len(filtered.new),
        filtered.new,
    )
    _, hybrid_entries = build_hybrid_tokenizer(
        base,
        vocab,
        entries,
        selected_donor,
        filtered.new,
        fallback,
    )
    kept, removed = prune_unreachable(hybrid_entries)
    tokenizer = build_tokenizer_from_entries(base, vocab, entries, kept)
    diagnostics.update(
        {
            "donor_merges": len(selected_donor),
            "requested_new_merges": remaining,
            "accepted_new_merges": len(filtered.new),
            "skipped_duplicate_parents": filtered.skipped_duplicates,
            "fallback_donor_merges": len(fallback),
            "pruned_unreachable_merges": len(removed),
            "partial_vocab_size": partial.get_vocab_size(),
            "actual_merges": len(kept),
        }
    )
    return tokenizer, kept, diagnostics


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
    frequency_scores = score_merges("frequency", entries, vocab, usage, 0)

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
            "native": evaluate_plain(
                load_tokenizer(config, target, "bpe", budget), target_test
            ),
            "methods": {},
            "comparisons": {},
            "document_values": {},
        }
        native_en_mean = None
        if english_test is not None:
            native_en = load_tokenizer(config, "wikitext_en", "bpe", budget)
            native_en_mean = compression(native_en, english_test)[0]

        selected_frequency = select_merges(
            entries, frequency_scores, merge_budget, closure=True
        )
        frequency_tokenizer = build_subset_tokenizer(
            donor_tokenizer, vocab, entries, selected_frequency
        )
        frequency_row = evaluate_constructed(
            frequency_tokenizer, target_test, entries, selected_frequency, vocab
        )
        if native_en_mean is not None:
            english_mean, _, _ = compression(frequency_tokenizer, english_test)
            frequency_row["english_retention"] = float(
                english_mean / max(1e-12, native_en_mean)
            )
        else:
            frequency_row["english_retention"] = None
        block["methods"]["frequency"] = frequency_row
        block["document_values"]["frequency"] = document_compressions(
            frequency_tokenizer, target_test
        )

        for share in SHARES:
            donor_budget = int(share * merge_budget)
            selected_donor = select_merges(
                entries, frequency_scores, donor_budget, closure=True
            )
            tokenizer, hybrid_entries, diagnostics = continuation_arm(
                donor_tokenizer,
                vocab,
                entries,
                selected_donor,
                frequency_scores,
                target_train,
                merge_budget,
            )
            selected = {entry.rank for entry in hybrid_entries}
            label = f"continuation_{int(share * 100):02d}"
            row = evaluate_constructed(
                tokenizer, target_test, hybrid_entries, selected, vocab
            )
            row["share"] = share
            row["diagnostics"] = diagnostics
            if native_en_mean is not None:
                english_mean, _, _ = compression(tokenizer, english_test)
                row["english_retention"] = float(
                    english_mean / max(1e-12, native_en_mean)
                )
            else:
                row["english_retention"] = None
            block["methods"][label] = row
            block["document_values"][label] = document_compressions(
                tokenizer, target_test
            )

        random_rows = []
        random_doc_values = []
        donor_budget = int(PRIMARY_SHARE * merge_budget)
        for seed in random_seeds:
            scores = score_merges("random", entries, vocab, usage, seed)
            selected_donor = select_merges(
                entries, scores, donor_budget, closure=True
            )
            tokenizer, hybrid_entries, diagnostics = continuation_arm(
                donor_tokenizer,
                vocab,
                entries,
                selected_donor,
                frequency_scores,
                target_train,
                merge_budget,
            )
            selected = {entry.rank for entry in hybrid_entries}
            row = evaluate_constructed(
                tokenizer, target_test, hybrid_entries, selected, vocab
            )
            row["seed"] = seed
            row["share"] = PRIMARY_SHARE
            row["diagnostics"] = diagnostics
            if native_en_mean is not None:
                english_mean, _, _ = compression(tokenizer, english_test)
                row["english_retention"] = float(
                    english_mean / max(1e-12, native_en_mean)
                )
            else:
                row["english_retention"] = None
            random_rows.append(row)
            values = document_compressions(tokenizer, target_test)
            random_doc_values.append(values)
            block["document_values"][f"random_50:{seed}"] = values
        block["methods"]["random_50"] = {
            "rows": random_rows,
            "mean_compression": float(
                np.mean([row["test_compression"] for row in random_rows])
            ),
            "min_compression": float(
                np.min([row["test_compression"] for row in random_rows])
            ),
            "max_compression": float(
                np.max([row["test_compression"] for row in random_rows])
            ),
            "mean_reachability": float(
                np.mean([row["strict_reachability"] for row in random_rows])
            ),
            "mean_vocab": float(
                np.mean([row["actual_vocab_size"] for row in random_rows])
            ),
        }

        baseline = block["document_values"]["frequency"]
        for label in ("continuation_25", "continuation_50", "continuation_75"):
            block["comparisons"][f"{label}_vs_frequency"] = compression_statistics(
                block["document_values"][label], baseline
            )
        random_deltas = [
            compression_statistics(values, baseline)["delta"]
            for values in random_doc_values
        ]
        block["comparisons"]["random_50_vs_frequency"] = {
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
        primary = block["comparisons"]["continuation_50_vs_frequency"]
        native_gap = (
            block["native"]["test_compression"]
            - block["methods"]["frequency"]["test_compression"]
        )
        aggregate_delta = (
            block["methods"]["continuation_50"]["test_compression"]
            - block["methods"]["frequency"]["test_compression"]
        )
        frequency_level = float(np.mean(block["document_values"]["frequency"]))
        block["comparisons"]["continuation_50_vs_native"] = {
            "delta": float(
                block["methods"]["continuation_50"]["test_compression"]
                - block["native"]["test_compression"]
            ),
            "relative_gain": float(
                aggregate_delta
                / max(1e-12, block["methods"]["frequency"]["test_compression"])
            ),
            "native_gap_closure": float(aggregate_delta / native_gap)
            if abs(native_gap) > 1e-12
            else None,
            "per_document_delta": float(primary["delta"]),
            "per_document_relative_gain": float(
                primary["delta"] / max(1e-12, frequency_level)
            ),
        }
        payload["budgets"][str(budget)] = block
        print(
            f"  {donor}->{target} {budget}: native "
            f"{block['native']['test_compression']:.4f} freq "
            f"{block['methods']['frequency']['test_compression']:.4f} "
            f"c25 {block['methods']['continuation_25']['test_compression']:.4f} "
            f"c50 {block['methods']['continuation_50']['test_compression']:.4f} "
            f"c75 {block['methods']['continuation_75']['test_compression']:.4f} "
            f"rand50 {block['methods']['random_50']['mean_compression']:.4f}",
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
    labels = []
    p_values = []
    rows = []
    for pair in payload["pairs"][:3]:
        for budget in map(str, budgets):
            block = pair["budgets"][budget]
            comparison = block["comparisons"][f"{PRIMARY_LABEL}_vs_frequency"]
            frequency = block["methods"]["frequency"]["test_compression"]
            labels.append(f"{pair['donor']}->{pair['target']}:{budget}")
            p_values.append(comparison["p_value"])
            rows.append(
                {
                    "label": labels[-1],
                    "delta": comparison["delta"],
                    "ci_low": comparison["ci_low"],
                    "ci_high": comparison["ci_high"],
                    "p_value": comparison["p_value"],
                    "relative_gain": comparison["delta"] / max(1e-12, frequency),
                    "strict_reachability": block["methods"][PRIMARY_LABEL][
                        "strict_reachability"
                    ],
                    "actual_vocab_size": block["methods"][PRIMARY_LABEL][
                        "actual_vocab_size"
                    ],
                    "requested_vocab_size": block["requested_vocab_size"],
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
    mean_relative_gain = float(
        np.mean([row["relative_gain"] for row in rows])
    )
    valid = True
    for pair in payload["pairs"][:3]:
        for budget in map(str, budgets):
            block = pair["budgets"][budget]
            requested = block["requested_vocab_size"]
            for label in (
                "frequency",
                "continuation_25",
                "continuation_50",
                "continuation_75",
            ):
                row = block["methods"][label]
                valid = valid and (
                    row["strict_reachability"] == 1.0
                    and abs(row["actual_vocab_size"] - requested)
                    <= 0.01 * requested
                )
            for row in block["methods"]["random_50"]["rows"]:
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
        "primary_arm": PRIMARY_LABEL,
        "comparison": "continuation_50_vs_frequency",
        "rows": rows,
        "positive_significant": positive,
        "negative_significant": negative,
        "mean_relative_gain": mean_relative_gain,
        "valid_construction": valid,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="C4 hybrid donor+target merge inventory")
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
    parser.add_argument("--out", default="results/invention_c4.json")
    args = parser.parse_args()

    config = load_config(args.config)
    pairs = [tuple(item.split(":")) for item in args.pairs.split(",") if item]
    payload = {
        "created_at": utc_now(),
        "base_size": BASE_SIZE,
        "pre_registration": pre_registration(),
        "shares": list(SHARES),
        "primary_share": PRIMARY_SHARE,
        "primary_arm": PRIMARY_LABEL,
        "random_seeds": args.random_seeds,
        "pairs": [],
    }
    for donor, target in pairs:
        print(f"C4 {donor} -> {target}", flush=True)
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
