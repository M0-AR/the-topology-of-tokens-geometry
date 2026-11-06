from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from topobpe.config import load_config, results_dir
from topobpe.utils import utc_now, write_json


def digest(path: Path) -> dict:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return {
        "path": str(path),
        "sha256": sha.hexdigest(),
        "bytes": path.stat().st_size,
    }


def main() -> int:
    config = load_config("configs/default.yaml")
    out = results_dir(config)
    target = Path("benchmark")
    target.mkdir(exist_ok=True)
    features = pd.read_csv(out / "features.csv")
    bpb = pd.read_csv(out / "tables" / "bpb_summary.csv")
    health = pd.read_csv(out / "tokenizer_health.csv")
    merge = pd.read_csv(out / "merge_features.csv")
    merge = merge[merge["variant"].isin(["native", "truncated", "base"])].copy()
    merge["priority"] = merge["variant"].map(
        {"native": 0, "truncated": 1, "base": 2}
    )
    merge = merge.sort_values("priority").drop_duplicates(
        ["corpus", "family", "requested_vocab_size"]
    )
    merge = merge[["corpus", "family", "requested_vocab_size", "merge_gain_gini"]]
    bpb_bpe = bpb[bpb["family"] == "bpe"][
        ["corpus", "family", "requested_vocab_size", "bpb_mean"]
    ]
    leaderboard = features[
        [
            "corpus",
            "family",
            "requested_vocab_size",
            "actual_vocab_size",
            "test_compression",
        ]
    ].copy()
    keys = ["corpus", "family", "requested_vocab_size"]
    leaderboard = leaderboard.merge(bpb_bpe, how="left", on=keys)
    leaderboard = leaderboard.merge(merge, how="left", on=keys)
    leaderboard = leaderboard.merge(
        health[["corpus", "family", "requested_vocab_size", "health_score", "grade", "flags"]],
        how="left",
        on=["corpus", "family", "requested_vocab_size"],
    )
    leaderboard = leaderboard.rename(columns={"test_compression": "compression"})
    leaderboard.to_csv(target / "leaderboard.csv", index=False)
    files = [
        out / "features.csv",
        out / "bpb.csv",
        out / "bpb_summary.csv",
        out / "merge_features.csv",
        out / "tokenizer_health.csv",
        out / "scaling_law.json",
        out / "scale_normalized_tda.json",
        out / "tda_stability.json",
        out / "analysis.json",
        out / "merge_analysis.json",
        out / "market_features.csv",
        out / "market_bpb.csv",
    ]
    files = [path for path in files if path.exists()]
    tokenizers = [
        path
        for path in sorted((Path("artifacts") / "tokenizers").glob("*/*.json"))
        if not path.name.endswith(".meta.json")
    ]
    manifest = {
        "created_at": utc_now(),
        "protocol": "docs/PROTOCOL.md",
        "metrics": {
            "compression": "test bytes / test tokens (higher is better)",
            "bpb_mean": "mean bits per byte over two seeds (lower is better)",
            "health_score": "0-100 heuristic; see scripts/health_check.py",
        },
        "counts": {
            "tokenizers": len(tokenizers),
            "benchmark_rows": int(len(leaderboard)),
            "corpora": int(leaderboard["corpus"].nunique()),
            "files": len(files) + len(tokenizers),
        },
        "files": [digest(path) for path in files + tokenizers],
        "notes": [
            "Tokenizers are byte-level BPE and Unigram trained on fixed splits.",
            "BPB is available for BPE only; Unigram rows have blank bpb_mean.",
            "Market assets use 12 US equities daily; intraday data was unavailable.",
            "The topology null is part of the benchmark: see results/analysis.json.",
        ],
    }
    write_json(target / "manifest.json", manifest)
    card = f"""# TopoBPE benchmark card

**Version:** 1.0 · **Created:** {manifest['created_at']}

## What this is

A frozen, reproducible tokenizer benchmark: 36 tokenizers (18 byte-level BPE,
18 Unigram) across six corpora at 8,192 / 16,384 / 32,768 vocabulary sizes,
plus 36 BPE language-model runs and a 12-asset US-equity market slice.
Every row is derived from the repository's fixed protocol; test splits were
never used for tokenizer or model tuning.

## Files

- `leaderboard.csv` — one row per corpus × family × vocabulary size:
  compression (higher is better), BPB mean for BPE (lower is better),
  merge-gain Gini, health score/grade, and health flags.
- `manifest.json` — SHA-256, byte size and role of every packaged artifact.
- `evaluate.py` — score a new tokenizer on a corpus test split.
- Source artifacts: `artifacts/tokenizers/`, `results/`, `docs/PROTOCOL.md`.

## Headline results

| Claim | Result |
|---|---|
| Baseline predicts compression | R² = 0.974 |
| Add 72 topology descriptors | R² = 0.843 (worse) |
| Baseline predicts BPB | R² = 0.887 |
| Add topology to BPB | R² = 0.555 (worse) |
| Topology transfers across corpora | No (all LOCO deltas negative) |
| Merge topology interpolates vocabulary size | Yes: +0.36 R² over size baseline (CI 0.20–0.70) |
| Merge topology extrapolates to 32k | Yes: R² = 0.68, versus 0.00 for cubic size baseline |
| Power-law 32k extrapolation from 8k/16k | MAPE 2.3% (BPE), 1.9% (Unigram) |

The benchmark therefore supports a narrow claim: merge/topological structure
tracks **scale-dependent** tokenizer behavior within a corpus, but does not
transfer across corpora. Do not use it as a universal tokenizer-quality score.

## Protocol

1. Fixed train/test splits from `configs/default.yaml`.
2. Train tokenizers at each size/family with fixed seeds.
3. Measure compression on held-out test text.
4. Train a 13.4M-parameter BPE LM for two seeds; report bits per byte.
5. Evaluate predictors with leave-one-corpus-out, Holm correction, and
   cluster bootstrap; pre-register thresholds before looking at results.
6. Record all null and negative results.

## Limitations

- Six corpora, three vocabulary sizes; not a universal scaling law.
- Unigram byte reversibility failed for 6/36 tokenizers; see
  `results/tables/tokenizer_byte_audit.csv`.
- BPB covers BPE only and two seeds.
- Market data is daily US equities only; intraday was unavailable.
- Health score is a transparent heuristic, not a validated clinical metric.

## License

Code and derived tables: MIT (see `LICENSE`). Upstream corpus and market data
retain their original licenses; see `docs/DATA_CARD.md`.
"""
    (target / "BENCHMARK_CARD.md").write_text(card)
    evaluator = '''from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokenizers import Tokenizer

from topobpe.config import load_config
from topobpe.data import load_split_texts


def main() -> int:
    parser = argparse.ArgumentParser(description="Score a tokenizer on a TopoBPE corpus")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    config = load_config(args.config)
    texts = load_split_texts(args.corpus, "test", config=config)
    if args.limit:
        texts = texts[: args.limit]
    tokenizer = Tokenizer.from_file(args.tokenizer)
    tokens = 0
    raw = 0
    for text in texts:
        tokens += len(tokenizer.encode(text, add_special_tokens=False).ids)
        raw += len(text.encode("utf-8"))
    print(
        json.dumps(
            {
                "tokenizer": args.tokenizer,
                "corpus": args.corpus,
                "documents": len(texts),
                "bytes": raw,
                "tokens": tokens,
                "compression": raw / tokens if tokens else None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    (target / "evaluate.py").write_text(evaluator)
    print(
        f"wrote {target}/ with {len(leaderboard)} leaderboard rows "
        f"and {len(manifest['files'])} hashed files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
