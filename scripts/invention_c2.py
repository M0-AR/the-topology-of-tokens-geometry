from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from topobpe.compatibility import (
    auc,
    demerge_loss,
    interval_overlap_beta1,
    is_crossing,
    merge_jaccard,
    shared_token_rows,
    spearman_correlation,
    token_spans,
    vocab_jaccard,
)
from topobpe.config import load_config
from topobpe.data import load_split_texts
from topobpe.merge_topology import load_bpe_components, merge_index
from topobpe.tokenizers import load_tokenizer, tokenizer_path


def sample_texts(texts: list[str], max_bytes: int) -> list[str]:
    chosen: list[str] = []
    total = 0
    for text in texts:
        encoded = text.encode("utf-8")
        if total + len(encoded) > max_bytes:
            remaining = max_bytes - total
            if remaining <= 0:
                break
            text = encoded[:remaining].decode("utf-8", errors="ignore")
            encoded = text.encode("utf-8")
        chosen.append(text)
        total += len(encoded)
        if total >= max_bytes:
            break
    return chosen


PAIRS = [
    ("wikitext_en", "wikipedia_sw"),
    ("wikitext_en", "wikipedia_ar"),
    ("wikitext_en", "wikipedia_hu"),
    ("wikitext_en", "wikipedia_zh"),
    ("codeparrot_py", "wikitext_en"),
]


def load_frequencies(
    tokenizer: Tokenizer, texts: list[str], vocab: dict[str, int]
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for text in texts:
        encoding = tokenizer.encode(text, add_special_tokens=False)
        for token_id in encoding.ids:
            token = tokenizer.id_to_token(token_id)
            if token in vocab:
                counts[token] = counts.get(token, 0) + 1
    return counts


def collect_contexts(
    tokenizer: Tokenizer,
    texts: list[str],
    target_vocab: dict[str, int],
    max_tokens: int,
) -> dict[str, str]:
    contexts: dict[str, str] = {}
    for text in texts:
        encoding = tokenizer.encode(text, add_special_tokens=False)
        for (start, end), token_id in zip(encoding.offsets, encoding.ids, strict=True):
            if end <= start:
                continue
            token = tokenizer.id_to_token(token_id)
            if token in target_vocab and token not in contexts:
                left = max(0, start - 40)
                right = min(len(text), end + 40)
                contexts[token] = text[left:right]
                if len(contexts) >= max_tokens:
                    return contexts
    return contexts


def token_level(rows) -> dict[str, float]:
    if not rows:
        return {}
    loss = np.array([row.demerge_loss for row in rows], dtype=np.float64)
    beta1 = np.array([row.union_beta1 for row in rows], dtype=np.float64)
    depth = np.array([row.depth_gap for row in rows], dtype=np.float64)
    length = np.array([row.length for row in rows], dtype=np.float64)
    frequency = np.array([row.frequency for row in rows], dtype=np.float64)
    labels = loss > 0
    return {
        "n_tokens": len(rows),
        "mean_demerge_loss": float(loss.mean()),
        "loss_positive_fraction": float(labels.mean()),
        "mean_union_beta1": float(beta1.mean()),
        "beta1_positive_fraction": float((beta1 > 0).mean()),
        "mean_depth_gap": float(depth.mean()),
        "auc_beta1": auc(beta1, labels),
        "auc_depth_gap": auc(depth, labels),
        "auc_length": auc(length, labels),
        "auc_frequency": auc(-frequency, labels),
        "spearman_beta1_loss": spearman_correlation(beta1, loss),
        "spearman_depth_gap_loss": spearman_correlation(depth, loss),
        "spearman_length_loss": spearman_correlation(length, loss),
        "spearman_frequency_loss": spearman_correlation(frequency, loss),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--c1", default="results/invention_c1.json")
    parser.add_argument("--out", default="results/invention_c2.json")
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--train-bytes", type=int, default=500_000)
    parser.add_argument("--test-bytes", type=int, default=200_000)
    args = parser.parse_args()

    config = load_config(args.config)
    c1 = json.loads(Path(args.c1).read_text(encoding="utf-8"))
    gaps: dict[str, dict[str, float]] = {}
    for pair in c1["pairs"]:
        key = f"{pair['donor']}->{pair['target']}"
        gaps[key] = {
            budget: block["native"]["test_compression"]
            - block["methods"]["frequency"]["test_compression"]
            for budget, block in pair["budgets"].items()
        }

    payload: dict[str, object] = {"pairs": [], "negative_control": {}}
    pair_rows: list[dict[str, object]] = []
    for donor, target in PAIRS:
        print(f"C2 {donor} -> {target}")
        donor_tokenizer = load_tokenizer(config, donor, "bpe", 32768)
        target_tokenizer = load_tokenizer(config, target, "bpe", 32768)
        donor_vocab, donor_merges = load_bpe_components(
            tokenizer_path(config, donor, "bpe", 32768)
        )
        target_vocab, target_merges = load_bpe_components(
            tokenizer_path(config, target, "bpe", 32768)
        )
        donor_parents, _ = merge_index(donor_merges)
        target_parents, _ = merge_index(target_merges)

        texts = sample_texts(load_split_texts(target, "train", config), args.train_bytes)
        frequencies = load_frequencies(target_tokenizer, texts, target_vocab)
        contexts = collect_contexts(donor_tokenizer, texts, target_vocab, args.max_tokens)
        rows = shared_token_rows(
            donor_tokenizer,
            target_tokenizer,
            donor_parents,
            target_parents,
            donor_vocab,
            target_vocab,
            frequencies,
            max_tokens=args.max_tokens,
            contexts=contexts,
        )
        metrics = token_level(rows)
        metrics.update(
            {
                "donor": donor,
                "target": target,
                "vocab_jaccard": vocab_jaccard(set(donor_vocab), set(target_vocab)),
                "merge_jaccard": merge_jaccard(donor_merges, target_merges),
                "transfer_gap_8192": gaps[f"{donor}->{target}"].get("8192", float("nan")),
                "transfer_gap_16384": gaps[f"{donor}->{target}"].get("16384", float("nan")),
            }
        )
        pair_rows.append(metrics)
        payload["pairs"].append(metrics)

        control_loss = 0.0
        control_beta1 = 0.0
        control_crossing = 0
        for row in rows[:200]:
            spans_donor = token_spans(donor_tokenizer, row.surface)
            spans_target = token_spans(target_tokenizer, row.surface)
            beta1 = interval_overlap_beta1(spans_donor, spans_target)
            control_beta1 += beta1
            control_loss += demerge_loss(donor_tokenizer, target_tokenizer, row.surface)
            control_crossing += int(is_crossing(spans_donor, spans_target, len(row.surface)))
        payload["negative_control"][f"{donor}->{target}"] = {
            "n": len(rows[:200]),
            "mean_interval_beta1": control_beta1 / max(1, len(rows[:200])),
            "crossing_fraction": control_crossing / max(1, len(rows[:200])),
        }

    scores = [
        "vocab_jaccard",
        "merge_jaccard",
        "mean_union_beta1",
        "beta1_positive_fraction",
        "mean_depth_gap",
        "mean_demerge_loss",
    ]
    correlations: dict[str, dict[str, float]] = {}
    for budget in ("transfer_gap_8192", "transfer_gap_16384"):
        gaps_array = np.array([row[budget] for row in pair_rows], dtype=np.float64)
        correlations[budget] = {}
        for score in scores:
            values = np.array([row[score] for row in pair_rows], dtype=np.float64)
            correlations[budget][score] = spearman_correlation(values, gaps_array)
    payload["pair_level_correlations"] = correlations
    payload["transfer_gaps"] = gaps

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    for row in pair_rows:
        print(
            f"{row['donor']}->{row['target']}: gap8={row['transfer_gap_8192']:.3f} "
            f"vocabJ={row['vocab_jaccard']:.3f} mergeJ={row['merge_jaccard']:.3f} "
            f"beta1={row['mean_union_beta1']:.3f} loss={row['mean_demerge_loss']:.3f} "
            f"auc_beta1={row['auc_beta1']:.3f} auc_depth={row['auc_depth_gap']:.3f} "
            f"auc_len={row['auc_length']:.3f}"
        )
    for budget, values in correlations.items():
        print(budget, {key: round(value, 3) for key, value in values.items()})


if __name__ == "__main__":
    main()
