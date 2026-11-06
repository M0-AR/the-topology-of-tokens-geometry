# Methodology audit against CS197 and 2025-2026 standards

Audit date: 2026-11-05.

## Sources inspected

- Harvard CS197, "AI Research Experiences", Pranav Rajpurkar: course site and course content.
- CS197 Lecture 3, "Shoulders of Giants": literature search, gap finding, and idea iteration.
- CS197 Lectures 10-11, "I Dreamed a Dream": research questions, experimental design, and findings.
- CS197 Lectures 12-13, "Today Was a Fairytale": global and local paper structure.
- CS197 Lecture 21, "Model Showdown": McNemar, paired t-test, bootstrap, confidence intervals, superiority/non-inferiority/equivalence.
- NeurIPS paper checklist and AAAI-26 reproducibility checklist.
- REFORMS consensus checklist for machine-learning-based science.
- General reproducibility guidance for AI research (NTNU) and the NERVE-ML checklist.
- SuperBPE and 2025-2026 tokenizer/BPB evaluations, including cross-lingual BPB benchmarking.
- 2025 TDA-in-NLP work: topological analysis of text-encoder embeddings, text homology, first-order homology of language, persistent homology of topic networks, and permutation-based TDA inference.
- Yahoo Finance chart endpoint documentation and 2026 market-data availability reports.

## CS197 lessons applied

| CS197 lesson | Application in this repository |
|---|---|
| Read wide, then deep; find gaps in question, setup, or findings | Gap audit: tokenizer evaluation usually reports compression/BPB without testing whether vocabulary geometry predicts them; TDA-in-NLP usually analyzes embeddings or documents, not tokenizer artifacts. |
| Convert the gap into a falsifiable question | H1-H4 in `docs/PROTOCOL.md`; negative results are explicitly admissible. |
| Design experiments that isolate the claimed mechanism | Fixed parameter capacity, fixed byte budget, train-only feature extraction, document-level splits, public-text-to-market transfer. |
| Anticipate confounds and leakage | Vocabulary capacity fixed; train compression included as a baseline; market split is chronological; all features from train only. |
| Structure the paper like the target venue | README follows abstract, introduction, related work, method, experiments, results, discussion, limitations, reproducibility, references. |
| Compare models with appropriate statistical tests | Bootstrap intervals, permutation tests, paired comparisons on identical test chunks, Holm correction. |
| Report uncertainty, not only point estimates | Every primary endpoint has bootstrap intervals; BPB has seed variability; TDA stability is checked across sample sizes and libraries. |

## 2025-2026 reproducibility standards applied

| Standard | Requirement | Repository response |
|---|---|---|
| NeurIPS checklist | Reproducibility statement, code/data details, error bars, compute | `docs/REPRODUCIBILITY.md`, Docker paths, pinned environment, seed logs, artifact manifest. |
| AAAI-26 checklist | Separate opinions from facts; document datasets and code; state limitations | Protocol labels confirmatory vs exploratory; `docs/DATA_CARD.md`; limitations section. |
| REFORMS | Dataset provenance, code version, reproduction script, leakage checks, uncertainty, external validity | `scripts/download_data.py`, `scripts/reproduce.sh`, SHA-256 manifest, `tests/test_tokenizers.py`, `tests/test_topology.py`, `tests/test_train.py`, `tests/test_stats.py`, LOCO evaluation, market transfer. |
| NERVE-ML / general AI guidance | Train/test independence, hyperparameter selection without test data, reported variance | Hash-based document splits, inner-loop alpha selection, two seeds for BPB, no test-based checkpoint selection. |
| Journal checklist | Public benchmark datasets, baselines, ablations, independent test | WikiText-103, CodeParrot, Wikipedia zh/hu/ar/sw, live market holdout, baseline/topology-only/full models, train-compression ablation, extended alpha grid, TDA point-count and library-agreement stability. |

## TDA-specific safeguards

1. Persistent homology is computed on fixed-size subsamples; the protocol tests 128, 256, and 512 points and cross-checks ripser against gudhi. In execution, all 24 ripser-versus-gudhi comparisons (six corpora x two clouds x two dimensions) agreed at bottleneck distance <= 2.4e-7 under a 1e-6 tolerance (`results/tables/tda_library_agreement.csv`).
2. Diagrams are converted to scalar summaries before predictive modeling; no test-set threshold tuning.
3. Feature-family significance uses permutation tests, and individual feature claims are exploratory and Holm-corrected.
4. The study does not interpret H1 cycles as semantic meaning; it treats them as structural summaries whose predictive value is measured.

## Tokenizer/BPB-specific safeguards

1. BPB is computed from total token cross-entropy divided by `ln(2)` times the original byte count, not from token perplexity.
2. Parameter count is held fixed by padding the output vocabulary to a common capacity.
3. Training batches are defined in bytes, not tokens, so tokenizers with different compression see comparable data per step.
4. Tokenizer fitting uses only the training split; held-out text is never merged into the vocabulary.
5. The protocol distinguishes intrinsic compression from extrinsic BPB and reports both.
6. A byte-reversibility audit runs for every tokenizer; all 18 BPE conditions reconstruct held-out text exactly, and the six Unigram conditions with `<unk>` tokens are flagged (`results/tables/tokenizer_byte_audit.csv`).

## Market-data safeguards

1. Raw provider responses (Yahoo primary, Nasdaq daily fallback) are cached with access timestamps and hashes but not redistributed.
2. The market holdout is chronological; no post-2024 data are used for fitting or feature selection.
3. Market results are external validity checks, not financial advice or a trading strategy.
4. Survivorship bias, non-stationarity, and data-provider instability are reported as limitations.

## Audit conclusion

The design satisfies the CS197 structure and the 2025-2026 reproducibility checklists at the level of a small, self-contained research paper. The main residual risks are scale-local BPB conclusions, a six-corpus sampling frame that is not representative of all languages, and the inherent instability of live market data. Intraday market windows were unavailable during execution and are reported as excluded rather than imputed. These are stated in the protocol and will be repeated in the final paper.
