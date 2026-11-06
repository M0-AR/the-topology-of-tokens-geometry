# Extensions: from the topology null to scale-aware tokenizer geometry

This document reports the six extensions built after the original pre-registered
study. The original conclusion still stands: **topology descriptors do not
transfer across corpora**. The extensions add a narrower, reproducible positive
result and four practical tools. All numbers below come from the repository's
frozen artifacts; none were tuned on test data.

## A. Merge-truncation augmentation

**Question.** Can one trained BPE tokenizer yield a family of valid tokenizers
at smaller vocabulary sizes, without retraining?

**Method.** For a BPE tokenizer with an ordered merge list, keep only the first
`target - 260` merges (256 byte tokens + 4 specials). The resulting tokenizer is
a valid BPE model. We generated 36 truncated tokenizers (6 corpora × 6 sizes:
512, 1,024, 2,048, 4,096, 8,192, 16,384) from the 32k models and compared them
with the natively trained 8k/16k models.

**Result.** Truncation is exact. At 8,192 and 16,384, every truncated model
reproduces the native model's compression and merge features to machine
precision. BPE training is incremental and deterministic, so the 32k merge
list contains the smaller models' merge lists as prefixes.

**Artifact.** `results/merge_features.csv` (54 rows), `results/merge_experiment_manifest.json`,
`src/topobpe/merge_topology.py`.

## B. Merge-graph persistent homology

**Question.** Does the *construction* of the vocabulary — not just the
distribution of token vectors — carry information about tokenizer quality?

**Method.** Build a graph whose nodes are tokens. Each merge contributes two
child–parent edges and one child–child edge. Filter edges by merge rank. Compute
H0/H1 persistent homology, derivation-tree depth/balance/Strahler, and merge-gain
inequality (Gini, entropy, top-10 share). This yields 63 non-constant
`merge_*` features, including 69 before variance filtering.

**Result: the all-feature model fails.** With 63–69 features and 42 rows, LOCO
across corpora produces catastrophic negative R² (e.g. −8.0 for merge-only
compression, −475 for merge-only BPB). A 3-component PCA of the same features
also fails (LOCO R² −14.6). This is an overfitting/ill-conditioning result, not
evidence of signal.

**Result: a pre-registered 10-feature subset reveals a scale effect.** Using
only dimensionless/topological summaries (tree balance/Strahler, gain Gini and
entropy, H1 persistence, cycle density, path length), the picture changes:
- **Across corpora (LOCO):** still negative. Adding the selected features to a
  cubic size baseline drops R² from 0.575 to −20.3. No cross-corpus transfer.
- **Within-corpus, unseen vocabulary size:** adding the selected features to a
  cubic size baseline raises R² from 0.575 to 0.934 (Δ = +0.359, cluster
  bootstrap 95% CI [0.196, 0.705], permutation p = 0.0083, bootstrap p = 0.0100).
- **Extrapolation to 32k** (train on 512–16k): R² = 0.680 versus −0.004 for the
  cubic size baseline.
- **Extrapolation down to 512** (train on 1k–32k): R² = 0.379 versus ≈0.000.

**Honest interpretation.** Merge topology tracks how compression changes with
vocabulary size inside a corpus, and can extrapolate one octave beyond the
fitted range. It does not identify a universally better tokenizer. The practical
use is scale planning, not cross-domain ranking.

**Artifacts.** `results/merge_analysis.json`,
`results/tables/merge_loco_summary.csv`,
`results/tables/merge_partial_compression.csv`,
`results/tables/merge_partial_bpb.csv`.

## C. Scale-normalized TDA

**Question.** Are persistence numbers unstable because the ambient scale changes
with sample density, and does normalization recover a stable signal?

**Method.** Normalize H1 summaries by median nearest-neighbour distance, graph
diameter, edge count, and log bar count. Compare coefficient of variation (CV)
across vocabulary sizes, and test raw versus normalized feature sets in the
same within-size and holdout protocol. Separately, re-analyze the 128/256/512
TDA sampling stability records with bar-count and entropy normalization.

**Result.** Normalization is a partial fix, not a universal one.
- H1 persistence entropy divided by `log(bars)` is highly stable: median CV
  0.0305 across vocabulary sizes, versus 0.2036 raw; in the sampling stability
  data, CV 0.0069 versus 1.0637 for raw total persistence.
- H1 total persistence divided by mean nearest-neighbour distance still drifts
  (CV 0.7341), because the *shape* changes with vocabulary size, not only its
  scale.
- Normalized features slightly improve within-size interpolation (Δ R² +0.384
  vs +0.359 raw) but **destroy** the 32k holdout (R² −0.046 vs +0.684 raw).
  They help at 512 (R² 0.295 vs −6.212 raw).

**Recommendation.** Use normalized persistence entropy for stable descriptions;
use raw persistence/tree features when extrapolating upward in vocabulary size.
Report both, and never call a normalized number "scale-free" without a drift
check.

**Artifacts.** `results/scale_normalized_tda.json`,
`results/tables/scale_normalized_drift.csv`.

## D. Scaling law and calculator

**Question.** Can bytes/token be predicted from vocabulary size with a simple,
auditable law?

**Method.** Fit `log(bytes/token) = log(a) + b log(vocab)` per corpus and family
on 8,192–32,768. Evaluate (i) extrapolation from 8k/16k to 32k, (ii) pooled
cross-corpus prediction with intrinsic features, and (iii) a model that includes
train compression.

**Result.**
- Per-corpus power laws fit extremely well: R² 0.964–0.998, residual σ(log)
  0.0039–0.0116. Exponents range from 0.045 (CodeParrot Unigram) to 0.150
  (Hungarian BPE), reflecting script and byte-entropy differences.
- Extrapolating 32k from 8k/16k: MAPE 2.3% (BPE) and 1.9% (Unigram); maximum
  absolute relative error 4.1% and 2.4%.
- Pooled cross-corpus model with intrinsic features: LOCO R² −0.54. Adding
  train compression raises it to 0.69. A universal law is not supported.

**Artifact.** `results/scaling_law.json`, `results/tables/scaling_law.csv`,
and the standalone `calculator.html`. The calculator is corpus-specific by
design and warns outside 8,192–32,768.

## E. Tokenizer health check

**Question.** Can we flag fragile tokenizers with cheap, transparent checks?

**Method.** For each of 36 tokenizers, encode up to 600 held-out documents and
measure byte-token occurrence rate, unknown-token rate, compression, vocabulary
utilization, byte-token vocabulary share, tokens per word, and merge-gain Gini
(BPE). A 0–100 heuristic score subtracts for high byte fallback, unknown tokens,
low compression, underused vocabulary, and byte-heavy vocabulary.

**Result.** 20 tokenizers grade A, 16 grade B. Ten flag high byte-fallback use;
six Unigram tokenizers (English/Swahili) flag unknown tokens. No tokenizer
flagged low compression, vocabulary underuse, or byte-heavy vocabulary. The
score is a screening heuristic, not a validated quality metric.

**Artifacts.** `results/tokenizer_health.csv`, `results/tokenizer_health.json`,
`scripts/health_check.py`.

## F. TokenizerBench package

**Question.** Can the protocol and artifacts be consumed without rerunning the
whole study?

**Method.** Package the 36-row leaderboard, 71 hashed artifacts, benchmark card,
manifest, and a standalone evaluator that scores any tokenizer file on a corpus
test split.

**Artifacts.** `benchmark/leaderboard.csv`, `benchmark/manifest.json`,
`benchmark/BENCHMARK_CARD.md`, `benchmark/evaluate.py`.

## What did not work

- All 63–69 merge features under LOCO: catastrophic overfitting.
- PCA-3 of merge features under LOCO: still negative.
- Cross-corpus merge topology: no improvement in any comparison.
- Scale normalization as a universal fix: hurts 32k extrapolation.
- A pooled cross-corpus "universal" scaling law: LOCO R² < 0 without train
  compression.

## Prior art and novelty check

A quick literature check on 2026-11-06 found:

- BPE merge rules have been formalized as a directed graph before, notably by
  Delobelle et al. (2022) and in **BPE-knockout** (NAACL 2024), where the graph
  supports vocabulary pruning. Those works use the graph for pruning, not for
  persistent homology or for predicting held-out compression.
- Compression-versus-vocabulary curves are observed in practitioner scaling
  notes and in compute-optimal tokenization work; the power law itself is not
  novel. This extension contributes a per-corpus calibration, an explicit
  extrapolation error, and a calculator — not a claim to discover the law.
- We found no direct prior application of persistent homology to the BPE
  merge graph. That is the narrowest defensible novelty claim here.

The claim we do **not** make: that merge topology is a universal tokenizer
quality metric. The cross-corpus null contradicts that.

## Reproduction

```bash
.venv/bin/python scripts/merge_experiment.py
.venv/bin/python scripts/merge_analysis.py --bootstrap-iterations 200 \
  --permutation-iterations 150 --partial-iterations 150
.venv/bin/python scripts/scale_normalized_tda.py
.venv/bin/python scripts/scaling_law.py
.venv/bin/python scripts/build_calculator.py
.venv/bin/python scripts/health_check.py
.venv/bin/python scripts/build_benchmark.py
.venv/bin/pytest -q
.venv/bin/ruff check .
```

Runtime on the study machine: merge features ~79 s; merge analysis ~73 s;
scale-normalized checks ~4 s; scaling law ~2 s; health check ~51 s; benchmark
build <1 s.
