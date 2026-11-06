# TopoBPE benchmark card

**Version:** 1.0 · **Created:** 2026-11-06T12:26:30.569569+00:00

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
