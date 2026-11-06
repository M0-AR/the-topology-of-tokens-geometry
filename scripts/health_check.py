from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from topobpe.config import load_config, results_dir
from topobpe.data import load_split_texts
from topobpe.tokenizers import load_tokenizer, token_byte_lengths
from topobpe.utils import utc_now, write_json


def _byte_token_ids(tokenizer: Any, family: str) -> set[int]:
    ids: set[int] = set()
    if family == "bpe":
        ids.update(range(min(256, tokenizer.get_vocab_size())))
    for index in range(tokenizer.get_vocab_size()):
        token = tokenizer.id_to_token(index) or ""
        if token.startswith("<0x") and token.endswith(">"):
            ids.add(index)
    return ids


def _grade(flags: list[str]) -> tuple[int, str]:
    score = 100
    if "byte_fallback_high" in flags:
        score -= 25
    if "unknown_tokens" in flags:
        score -= 25
    if "compression_low" in flags:
        score -= 15
    if "vocab_underused" in flags:
        score -= 15
    if "byte_heavy_vocab" in flags:
        score -= 20
    score = max(0, min(100, score))
    grade = "A" if score >= 85 else "B" if score >= 70 else "C" if score >= 55 else "D"
    return score, grade


def analyze(
    config: dict,
    corpus: str,
    family: str,
    vocab_size: int,
    merge_gini: dict[tuple[str, int], float],
) -> dict:
    tokenizer = load_tokenizer(config, corpus, family, vocab_size)
    texts = load_split_texts(corpus, "test", config=config)
    step = max(1, len(texts) // 600)
    texts = texts[::step][:600]
    pieces = [tokenizer.encode(text, add_special_tokens=False).ids for text in texts]
    ids = np.asarray([token for piece in pieces for token in piece], dtype=np.int64)
    lengths = token_byte_lengths(tokenizer)
    byte_ids = _byte_token_ids(tokenizer, family)
    unk_id = tokenizer.token_to_id("<unk>")
    actual = int(tokenizer.get_vocab_size())
    raw_bytes = sum(len(text.encode("utf-8")) for text in texts)
    words = sum(max(1, len(text.split())) for text in texts)
    if len(ids) == 0:
        raise RuntimeError(f"tokenizer produced no ids for {corpus}/{family}/{vocab_size}")
    byte_fallback = float(np.mean(np.isin(ids, list(byte_ids))))
    unknown = float(np.mean(ids == unk_id)) if unk_id is not None else 0.0
    byte_share = len(byte_ids) / actual
    compression = raw_bytes / len(ids)
    utilization = len(np.unique(ids)) / actual
    flags: list[str] = []
    if byte_fallback > 0.10:
        flags.append("byte_fallback_high")
    if unknown > 0:
        flags.append("unknown_tokens")
    if compression < 2.0:
        flags.append("compression_low")
    if utilization < 0.35:
        flags.append("vocab_underused")
    if byte_share > 0.25:
        flags.append("byte_heavy_vocab")
    score, grade = _grade(flags)
    return {
        "corpus": corpus,
        "family": family,
        "requested_vocab_size": int(vocab_size),
        "actual_vocab_size": actual,
        "sample_documents": len(texts),
        "sample_tokens": int(len(ids)),
        "compression": compression,
        "byte_fallback_rate": byte_fallback,
        "unknown_rate": unknown,
        "byte_token_vocab_share": byte_share,
        "tokens_per_word": float(len(ids) / max(1, words)),
        "vocab_utilization": utilization,
        "mean_token_bytes": float(np.mean(lengths)),
        "merge_gain_gini": merge_gini.get((corpus, int(vocab_size)), float("nan")),
        "flags": ";".join(flags),
        "flag_count": len(flags),
        "health_score": score,
        "grade": grade,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Per-tokenizer health report")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    out = results_dir(config)
    features = pd.read_csv(out / "features.csv")
    merge = pd.read_csv(out / "merge_features.csv")
    merge = merge[merge["variant"] == "native"]
    merge_gini = {
        (row.corpus, int(row.requested_vocab_size)): float(row.merge_gain_gini)
        for row in merge.itertuples()
    }
    rows = []
    for row in features.itertuples():
        print(f"health {row.corpus}/{row.family}/{row.requested_vocab_size}", flush=True)
        rows.append(
            analyze(
                config,
                row.corpus,
                row.family,
                int(row.requested_vocab_size),
                merge_gini,
            )
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "tokenizer_health.csv", index=False)
    summary = {
        "created_at": utc_now(),
        "tokenizers": int(len(frame)),
        "grade_counts": frame["grade"].value_counts().to_dict(),
        "flag_counts": {
            flag: int(frame["flags"].str.contains(flag).sum())
            for flag in (
                "byte_fallback_high",
                "unknown_tokens",
                "compression_low",
                "vocab_underused",
                "byte_heavy_vocab",
            )
        },
        "worst": frame.sort_values("health_score").head(10).to_dict(orient="records"),
        "by_family": frame.groupby("family")[
            ["compression", "byte_fallback_rate", "vocab_utilization", "health_score"]
        ]
        .mean()
        .reset_index()
        .to_dict(orient="records"),
    }
    write_json(out / "tokenizer_health.json", summary)
    columns = ["corpus", "family", "requested_vocab_size", "compression", "grade"]
    print(frame[columns].to_string(index=False))
    print("\nsummary:", summary["grade_counts"], summary["flag_counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
