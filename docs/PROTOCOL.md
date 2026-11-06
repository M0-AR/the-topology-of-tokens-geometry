# TopoBPE pre-registered protocol

Protocol frozen: 2026-11-05, before confirmatory data collection or model training. Revised the same day, before confirmatory runs, to record the six-corpus sampling frame and the market-provider fallback.

## Research question

Do persistent-homology descriptors of a byte-pair-encoding (BPE) vocabulary and its induced token-transition structure predict (a) the tokenizer's held-out byte compression and (b) the bits-per-byte (BPB) of a language model trained with that tokenizer, beyond entropy, frequency, and training-compression baselines? Does the same relation transfer to live financial-market sequences?

The study is an out-of-sample predictive study, not a claim that topology causes tokenizer quality. The unit of analysis is a tokenizer fitted to a corpus. Corpora are sampled at the document or time-block level, and all predictive features are computed only from training material.

## Hypotheses

| ID | Hypothesis | Primary endpoint | Decision rule |
|---|---|---|---|
| H1 | Topology features improve out-of-corpus prediction of held-out byte compression over a baseline containing vocabulary size, byte entropy, token entropy, type-token ratio, gzip ratio, and train-split compression. | Leave-one-corpus-out cross-validated R-squared and MAE | Confirm only if the paired R-squared difference is positive with a bootstrap 95% interval excluding zero and the permutation p-value is below 0.05 after Holm correction. |
| H2 | Topology features improve out-of-corpus prediction of BPB. | Same as H1, target BPB | Same as H1. |
| H3 | H1-persistence summaries remain associated with BPB after controlling for held-out compression. | Partial Spearman correlation and ridge coefficient | Confirm only if the bootstrap 95% interval excludes zero and the Holm-adjusted permutation p-value is below 0.05. |
| H4 | A model trained only on public text predicts live-market compression and BPB better than the no-topology baseline. | Transfer R-squared and MAE on the live-market holdout | Confirm only if transfer R-squared is positive with a bootstrap 95% interval excluding zero. |
| H5 | Specific topological summaries, especially H1 total persistence and persistence entropy, are positively associated with compression across tokenizers. | Spearman correlation with bootstrap interval | Exploratory; reported regardless of direction or significance. |

The study is reported as negative if H1-H3 are not supported. No endpoint may be changed after inspecting the holdout results.

## Data

Public text corpora:

| Name | Source | Language/domain | Sampling |
|---|---|---|---|
| `wikitext_en` | `Salesforce/wikitext`, `wikitext-103-raw-v1` | English prose | First 12,000 eligible training documents in stream order |
| `codeparrot_py` | `codeparrot/codeparrot-clean` | Python code | First 12,000 eligible training files in stream order |
| `wikipedia_zh` | `wikimedia/wikipedia`, `20231101.zh` | Chinese prose | First 12,000 eligible training documents in stream order |
| `wikipedia_hu` | `wikimedia/wikipedia`, `20231101.hu` | Hungarian prose | First 12,000 eligible training documents in stream order |
| `wikipedia_ar` | `wikimedia/wikipedia`, `20231101.ar` | Arabic prose | First 12,000 eligible training documents in stream order |
| `wikipedia_sw` | `wikimedia/wikipedia`, `20231101.sw` | Swahili prose | First 12,000 eligible training documents in stream order |

Live market data:

- Source: Yahoo Finance chart endpoint, no key, user agent identifying research use. If Yahoo returns a rate-limit or transport error for the 10-year daily corpus, the daily series is fetched from the Nasdaq public chart endpoint and the provider is recorded in the manifest. Intraday windows are attempted through Yahoo only; provider unavailability is reported and those windows are excluded from analysis rather than imputed.
- Tickers: AAPL, MSFT, NVDA, JPM, XOM, JNJ, WMT, CAT, KO, DIS, GS, TSLA.
- Windows: 10 years daily, 2 years hourly, 1 month five-minute.
- Access date and raw-response SHA-256 hashes are written to the artifact manifest.
- Raw market responses are cached locally and excluded from version control because Yahoo's terms restrict redistribution.
- Derived strings encode each bar as a deterministic text record with ticker, timestamp, open, high, low, close, volume, and log return.
- Market train/validation/test splits are chronological. For 10-year daily data, train is through 2023-12-31, validation through 2024-12-31, and test is after 2024-12-31. Intraday windows use a chronological 60/20/20 split because their history is shorter than the fixed-date window. No future information enters features.

Public text splits are document-level, deterministic, and non-overlapping: 80% train, 10% validation, 10% test. A document hash assigns the split; no paragraph or file from a training document appears in validation or test.

## Tokenizers and features

Tokenizers: byte-level BPE and byte-level Unigram, with vocabulary sizes 8,192, 16,384, and 32,768, trained only on the training split. The tokenizer grid is fitted independently per corpus and per market split.

Topology feature families, computed only from the training split and the fitted tokenizer:

1. **Vocabulary point-cloud persistence.** Each token is represented by a normalized byte n-gram profile (n = 1..3, hashed to 64 dimensions). Up to 256 tokens are sampled, cosine distances are computed, and Vietoris-Rips persistent homology is calculated in dimensions 0 and 1.
2. **Token co-occurrence persistence.** Up to 256 high-frequency tokens are represented by a positive-pointwise-mutual-information distance derived from within-document co-occurrence windows. Vietoris-Rips persistent homology is calculated in dimensions 0 and 1.
3. **Persistence summaries.** Total persistence, maximum persistence, persistence entropy, Betti-curve area, and counts above four persistence thresholds, separately for H0 and H1.

Baseline features: vocabulary size, byte entropy, token entropy, conditional token entropy, type-token ratio, gzip compression ratio, mean token length in bytes, and train-split bytes-per-token. The target, held-out bytes per token, is never used as a feature.

All features are computed from the training split. Test-split compression and BPB are targets only.

## Language-model benchmark

- Architecture: decoder-only transformer, 6 layers, 8 heads, width 256, feed-forward width 1024, learned positional embeddings, tied input/output embeddings, context 1,024 tokens.
- Vocabulary capacity is fixed at 32,768 plus special tokens for every tokenizer. Unused embedding rows are present but never receive gradients from targets, keeping parameter count fixed across tokenizer conditions.
- Training examples are contiguous 2,048-byte text chunks aligned to Unicode character boundaries. Batch size is 16 chunks, so each optimizer step sees approximately the same number of bytes across tokenizers.
- Training: 1,500 steps, AdamW, learning rate 3e-4 with 100-step warmup, weight decay 0.01, gradient clipping 1.0, bfloat16 autocast on CUDA.
- Seeds: 13 and 37 for BPE conditions. Unigram conditions are evaluated intrinsically but not used in the BPB primary endpoint.
- BPB: sum of token cross-entropy in nats divided by `ln(2)` times the number of test bytes. Special tokens are excluded from the loss and from byte counts.
- Checkpoint selection: final step only; no test-set model selection.
- The full parameter count, byte budget, token count, wall-clock time, GPU model, and dependency versions are logged per run.

## Statistical analysis

- Primary predictive evaluation: leave-one-corpus-out cross-validation. On each fold, all preprocessing and ridge alpha selection use only training corpora; the held-out corpus is untouched.
- Models compared: baseline, topology-only, and baseline-plus-topology. Ridge alpha is selected by inner leave-one-corpus-out cross-validation over the training folds.
- Uncertainty: 2,000 bootstrap resamples at the corpus level; 2,000 label permutations for feature-family tests; Holm correction across the four confirmatory hypotheses H1-H4. H5 is exploratory and is not included in the confirmatory family.
- Effect sizes: R-squared, MAE, Spearman rho, and partial Spearman rho. Confidence intervals are percentile intervals.
- Market transfer: fit on public text only, then evaluate on the market holdout. A within-market model is reported separately as an exploratory analysis.
- Multiple comparisons: all primary p-values are Holm-adjusted. Exploratory results are labelled as exploratory.
- No causal language is permitted. "Predicts" means out-of-sample association.

## Robustness and ablations

1. Remove the train-compression baseline to test whether topology carries information beyond a trivially strong proxy.
2. Replace ripser with gudhi on a 128-point subsample and verify diagram agreement on H0 and H1.
3. Vary the number of sampled points (128, 256, 512) and report stability.
4. Evaluate BPB with one additional seed for the two highest-performing and two lowest-performing tokenizer conditions.
5. Report every corpus individually and a leave-one-language-family-out variant; the market corpora never enter cross-corpus folds.
6. Report results by language/domain, not only pooled.

## Artifacts and reproducibility

Required artifacts:

- `artifacts/manifest.json` with source URLs, revisions, SHA-256 hashes, access timestamps, environment, and git commit if available; `artifacts/data_manifest.json`, `artifacts/market_manifest.json`, and `artifacts/tokenizer_manifest.json` are nested inside it.
- `results/features.csv`, `results/bpb.csv`, `results/analysis.json`, `results/tables/*.csv`, and `results/figures/*.pdf`.
- A single command that regenerates every table and figure from cached data.
- A CPU Docker path for data checks, tokenizer fitting, feature extraction, and analysis; a GPU Docker path for LM training.
- Tests for split leakage, byte/BPB normalization, persistence stability, and statistical resampling.

## Limitations fixed in advance

- Six public text corpora are not a representative sample of all languages or domains.
- The language models are small and undertrained relative to frontier models; BPB conclusions are scale-local.
- Persistent homology of a 256-point sample is an approximation whose stability is reported, not assumed.
- Live market data are non-stationary, subject to survivorship bias, and not redistributed; market results are external validation, not a trading claim.
- Tokenizer quality is multi-dimensional; BPB and compression do not capture downstream task accuracy, fairness, or inference cost.

## Execution deviations

Recorded 2026-11-05 after tokenizer fitting, feature extraction, and market-data collection, and before any confirmed-text BPB result was inspected. None of these deviations changes a confirmatory decision rule.

1. **Intraday market windows are excluded.** Yahoo returned HTTP 429 for every intraday request. The Nasdaq daily fallback does not serve the requested intraday windows (HTTP 404). All 24 intraday corpora (12 tickers × 2 windows) are recorded as unavailable in `artifacts/market_manifest.json` and excluded from H4 and from the exploratory within-market analysis. Only the 12 ten-year daily corpora enter market analyses.
2. **Daily market provider.** Yahoo was rate-limited for daily data as well, so all 12 daily corpora were fetched from the Nasdaq public chart endpoint and are recorded with that provider and a SHA-256 hash.
3. **Byte-reversibility audit.** BPE is exactly byte-reversible in all 18 conditions. Unigram is not: `wikitext_en` has one unknown token in one of 1,218 test documents at each vocabulary size, and `wikipedia_sw` has four unknown tokens in one of 1,198 test documents at each size. The `wikitext_en` Unigram 32,768 tokenizer also stops at an actual vocabulary of 29,379. These rows are retained and flagged in `results/tables/tokenizer_byte_audit.csv`; H2 uses BPE only, and the Unigram limitation is repeated in the results.
4. **Leave-one-language-family-out was not run.** Each language/domain is represented by exactly one corpus, so holding out a language family is identical to LOCO. Grouping the four Wikipedia corpora would leave only English prose and Python code for training, which would not identify the intended contrast. Held-out error is now reported per corpus for H1 and H2 instead.
5. **Robustness execution.** The train-compression ablation and the extended ridge alpha grid run inside `topobpe analysis`. Point-count stability (128/256/512) and ripser-versus-gudhi agreement on 256-point clouds are produced by `scripts/tda_stability.py`. All BPE BPB conditions are trained with both seeds (13 and 37), which exceeds the protocol's additional-seed requirement.
