# C4/C5 results: continued-BPE hybrid works; tokenizer-aware data selection does not

Status: final, pre-registered, uncommitted.
Pre-registration: [`docs/PREREGISTRATION_C4_C5.md`](PREREGISTRATION_C4_C5.md), frozen at
2026-11-06T13:54:36Z.
Frozen SHA-256: `c6d530af8ee3bde6bf7ba86b53f3d34fa03930788988785c323764a3b918a55e`.
Both runners verify this hash before producing results. No threshold, baseline, metric, or primary
arm was changed after the freeze.

**Verdicts:** C4 **PASS** (replication/variant, not a new invention). C5 **KILL**.

## TL;DR

- **C4 PASS.** `continuation_50` beat donor-only frequency selection in **6/6** primary
  comparisons, with Holm-adjusted p < 0.0005 (0/2000 bootstrap resamples), a mean per-document
  relative gain of **123.0%** (median **76.0%**), and a mean native-gap closure of **79.1%**.
  All 70 constructed tokenizers had strict reachability 1.0 and actual vocabulary within 1% of the
  requested budget.
- **C4 is not novel.** It is a fixed-budget variant of continued BPE training, which is already
  published (Purason et al., Findings of EACL 2026). The paper extends a base vocabulary and
  continues BPE training (with an optional leaf-based pruning stage); C4 reallocates a fixed
  budget, dropping half the donor merges to make room for target merges. That difference produces
  a real English-retention cost (mean 73.5% vs 91.2% for donor-only frequency on the six
  non-English primary comparisons).
- **C4 mechanism check.** The exploratory random-donor control is close: `continuation_50` beats
  `random_50` on 8/10 pair-budgets, but loses on the Chinese stress pair by 0.024 bytes/token.
  The dominant ingredient is target-specific continuation; frequency-based donor selection adds a
  smaller, uneven increment (negligible on Arabic, large on code, negative on Chinese).
- **C5 KILL.** High-fertility document selection produced **0/6** positive and **6/6** negative
  primary comparisons (Holm p < 0.0005), mean relative gain **-0.174%**. It is slightly but
  significantly worse than random selection. All 120 constructed tokenizers were valid.
- **C5 post-hoc controls do not rescue it.** Low-fertility selection is approximately random
  (3/10 Holm-significant tiny positive comparisons, mean +0.0093%); length controls show fertility
  is partly a length proxy. The `full_16mb` ceiling is practically tied with 4 MB (at most 0.15% relative
  above `first`), so the ~0.996-1.0 "data efficiency" ratio is evidence that extra data does not
  help, not that selection is efficient.

## Shared protocol actually used

- Donor `bpe_32768`; target native `bpe_8192` and `bpe_16384`; initial vocabulary 260; merge
  budgets 7932 and 16124.
- Primary pairs: `wikitext_en -> wikipedia_sw`, `-> wikipedia_ar`, `-> wikipedia_hu`.
  Stress pairs: `wikitext_en -> wikipedia_zh`, `codeparrot_py -> wikitext_en`.
- Training text: first 4,000,000 bytes of the target `train` split. Test: first 1,000,000 bytes of
  the target `test` split, in file order.
- Primary metric: aggregate test compression in bytes per token (higher is better), with
  per-document values for the paired bootstrap (2,000 resamples, seed 20261106, two-sided).
- Holm correction within each experiment's six primary comparisons; C4 and C5 are separate
  families and are never pooled. Stress pairs are reported but are not in either Holm family.
- Every selection uses ancestor closure. Every reported constructed tokenizer is checked for
  strict reachability 1.0 and actual vocabulary within 1% of the requested budget.

## C4 results

### Primary comparison: `continuation_50` vs donor-only `frequency`

Aggregate levels are bytes/token. The primary test statistic is the per-document delta
(`continuation_50 - frequency`), mean over documents; the interval is the paired bootstrap 95% CI.
`p = 0.000` means no resample crossed zero (0/2000).

| Pair / budget | freq | c50 | native | per-doc delta [95% CI] | rel. gain | native-gap | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|
| en->sw 8192 | 2.2601 | 3.2847 | 3.7032 | 1.11917 [1.08993, 1.14783] | 0.4952 | 0.7100 | <0.0005 |
| en->sw 16384 | 2.2792 | 3.6160 | 4.0506 | 1.38997 [1.35917, 1.41983] | 0.6098 | 0.7547 | <0.0005 |
| en->ar 8192 | 1.5394 | 4.9990 | 5.8039 | 3.39798 [3.28093, 3.52054] | 2.2073 | 0.8112 | <0.0005 |
| en->ar 16384 | 1.5395 | 5.5260 | 6.4291 | 3.92467 [3.77987, 4.07620] | 2.5492 | 0.8153 | <0.0005 |
| en->hu 8192 | 2.0602 | 3.4370 | 3.7339 | 1.38431 [1.31748, 1.45210] | 0.6719 | 0.8226 | <0.0005 |
| en->hu 16384 | 2.0692 | 3.8291 | 4.1885 | 1.75462 [1.67489, 1.83530] | 0.8480 | 0.8304 | <0.0005 |

- Positive significant: **6/6**. Negative significant: **0/6**.
- Mean relative gain: **1.230245** (123.0%). Median: **0.759946** (76.0%). The mean is pulled up
  by Arabic; the median is the more robust summary.
- Mean native-gap closure: **0.7907**; median **0.8133**; range 0.7100-0.8304.
- Validity: all 70 constructed tokenizers (four methods plus three random seeds across ten
  pair-budget blocks) had strict reachability 1.0. Hungarian 8192/16384 came in at 8190/16382
  (0.024%/0.012% below request); every other arm was exact. All are within the 1% rule.
- Pre-registered pass criteria: satisfied (6/6 positive, 0/6 negative, mean rel >= 0.5%, valid).

### Aggregate context and the random-donor control

The primary comparison is against donor-only frequency selection. The more interesting
mechanism check is whether the donor half must be selected by frequency at all. It must not:
`random_50` uses the same 3,966 / 8,062 donor merges and the same total budget, but picks the donor
subset uniformly at random (seeds 13, 37, 73). Its new-merge count is the full 3,966 / 8,062;
`continuation_50` accepts 0-3 fewer new merges because duplicate parents are skipped, and fills the
freed slots with fallback donor merges. The control is therefore if anything slightly favored.
Per-document delta vs `frequency`:

| Pair / budget | c50 delta | random_50 delta | winner |
|---|---:|---:|---|
| en->sw 8192 | 1.11917 | 0.76749 | c50 |
| en->sw 16384 | 1.38997 | 1.16812 | c50 |
| en->ar 8192 | 3.39798 | 3.32293 | c50 |
| en->ar 16384 | 3.92467 | 3.86994 | c50 |
| en->hu 8192 | 1.38431 | 1.21261 | c50 |
| en->hu 16384 | 1.75462 | 1.61197 | c50 |
| en->zh 8192 (stress) | 1.99371 | 2.01799 | random |
| en->zh 16384 (stress) | 2.32998 | 2.35434 | random |
| code->en 8192 (stress) | 0.35749 | 0.10620 | c50 |
| code->en 16384 (stress) | 0.71561 | 0.49989 | c50 |

Frequency selection gives a consistent edge over a random donor subset on 8/10 pair-budgets, but
the size of the edge is uneven: negligible on Arabic, substantial on Swahili and code, and slightly
negative on Chinese. Most of the improvement over donor-only selection is shared by both arms, so
the dominant ingredient is adding target-specific continuation merges, not choosing donor merges by
frequency. This is an exploratory control and does not change the pre-registered pass, but it
changes the practical claim: "continue BPE on target text" is the lever; "select the donor half by
frequency" is a secondary choice with uneven value.

### English retention (secondary)

English retention is the constructed tokenizer's bytes/token on 1 MB of `wikitext_en` test divided
by the native English tokenizer's bytes/token at the same budget. It is a relative compression
measure, not a token-ID compatibility measure.

| Pair / budget | frequency | continuation_50 |
|---|---:|---:|
| en->sw 8192 | 0.7743 | 0.6529 |
| en->sw 16384 | 0.9611 | 0.6995 |
| en->ar 8192 | 0.9820 | 0.7930 |
| en->ar 16384 | 0.9978 | 0.8873 |
| en->hu 8192 | 0.7890 | 0.6635 |
| en->hu 16384 | 0.9701 | 0.7125 |

Mean over the six non-English primary comparisons: **0.9124** for frequency, **0.7348** for
`continuation_50`; the hybrid gives up **17.8 percentage points** of English compression on
average. The donor-only baseline keeps English because it spends the whole budget on donor merges.
C4 spends half the budget on target merges. This is the expected trade-off, and it is why C4 is
not the same intervention as the published continued-BPE extension, which continues from the donor
vocabulary and adds new tokens instead of reallocating a fixed budget.

### Exploratory shares and stress pairs

Aggregate bytes/token; numbers in parentheses are the relative difference from `frequency`.
These are exploratory and were not used for the pass/kill decision.

| Pair / budget | freq | c25 | c50 | c75 | random_50 |
|---|---:|---:|---:|---:|---:|
| en->sw 8192 | 2.2601 | 3.3044 (+46.2%) | 3.2847 (+45.3%) | 3.1660 (+40.1%) | 2.9541 (+30.7%) |
| en->sw 16384 | 2.2792 | 3.6168 (+58.7%) | 3.6160 (+58.7%) | 3.4415 (+51.0%) | 3.3861 (+48.6%) |
| en->ar 8192 | 1.5394 | 5.3071 (+244.7%) | 4.9990 (+224.7%) | 4.4777 (+190.9%) | 4.9342 (+220.5%) |
| en->ar 16384 | 1.5395 | 5.8106 (+277.4%) | 5.5260 (+258.9%) | 5.0115 (+225.5%) | 5.4866 (+256.4%) |
| en->hu 8192 | 2.0602 | 3.5068 (+70.2%) | 3.4370 (+66.8%) | 3.1781 (+54.3%) | 3.2556 (+58.0%) |
| en->hu 16384 | 2.0692 | 3.9163 (+89.3%) | 3.8291 (+85.1%) | 3.5343 (+70.8%) | 3.6764 (+77.7%) |
| en->zh 8192 (stress) | 1.0737 | 3.2995 (+207.3%) | 3.0981 (+188.6%) | 2.7275 (+154.0%) | 3.1230 (+190.9%) |
| en->zh 16384 (stress) | 1.0743 | 3.6170 (+236.7%) | 3.4516 (+221.3%) | 3.1226 (+190.7%) | 3.4832 (+224.2%) |
| code->en 8192 (stress) | 3.4950 | 3.8454 (+10.0%) | 3.8459 (+10.0%) | 3.8353 (+9.7%) | 3.5986 (+3.0%) |
| code->en 16384 (stress) | 3.5325 | 4.2285 (+19.7%) | 4.2393 (+20.0%) | 4.1110 (+16.4%) | 4.0247 (+13.9%) |

The 25% share is often slightly better than 50% at both budgets (4/5 pairs); the differences among
25/50/75 are small compared with the gain over frequency. The stress pair
`codeparrot_py -> wikitext_en` has the smallest relative gain (+10.0% at 8192, +20.0% at 16384),
which is expected: English is already well covered by the donor inventory, so there is less
target-specific merge headroom.

### C4 interpretation

C4 confirms the C1 diagnosis: donor-only selection is limited by the donor merge inventory.
Continuing BPE on target text adds merges the donor does not have, and that is worth roughly a
0.5x-2.5x relative compression gain on the primary non-English targets, closing about 79% of the
gap to a native target tokenizer. This is a strong replication of published continued-BPE
behavior, under a fixed-budget variant.

It is **not** a new method. It is also not a free lunch: reallocating half the fixed budget away
from donor merges costs English compression, and the random-donor control shows that frequency
selection of the donor half is a secondary ingredient, not the driver.

## C5 results

### Primary comparison: `high_fertility` vs the mean of five random selections

Aggregate levels are bytes/token. The primary test statistic is the per-document delta
(`high_fertility - random mean`); `p = 0.000` means 0/2000 resamples crossed zero.
`deff` is the pre-registered diagnostic: high-fertility aggregate compression divided by the
`full_16mb` aggregate compression.

| Pair / budget | high | random mean | first | full_16mb | per-doc delta [95% CI] | rel. gain | Holm p | deff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| en->sw 8192 | 2.2523 | 2.2590 | 2.2601 | 2.2622 | -0.007305 [-0.008959, -0.005824] | -0.003250 | <0.0005 | 0.9956 |
| en->sw 16384 | 2.2739 | 2.2804 | 2.2792 | 2.2825 | -0.006972 [-0.008281, -0.005864] | -0.003075 | <0.0005 | 0.9962 |
| en->ar 8192 | 1.5393 | 1.5395 | 1.5394 | 1.5397 | -0.000343 [-0.000535, -0.000160] | -0.000220 | <0.0005 | 0.9998 |
| en->ar 16384 | 1.5395 | 1.5396 | 1.5395 | 1.5397 | -0.000177 [-0.000271, -0.000086] | -0.000114 | <0.0005 | 0.9998 |
| en->hu 8192 | 2.0562 | 2.0609 | 2.0602 | 2.0628 | -0.004704 [-0.007284, -0.002574] | -0.002309 | <0.0005 | 0.9968 |
| en->hu 16384 | 2.0668 | 2.0693 | 2.0692 | 2.0716 | -0.003065 [-0.005062, -0.001595] | -0.001497 | <0.0005 | 0.9977 |

- Positive significant: **0/6**. Negative significant: **6/6**.
- Mean relative gain: **-0.001744** (-0.174%). Median: **-0.001903** (-0.190%).
- All six primary comparisons are negative with Holm p < 0.0005. The effect is statistically
  clear but practically tiny in absolute terms (0.011%-0.325% relative; Swahili is the largest at
  -0.31%/-0.33%).
- Validity: all 120 constructed tokenizers (seven non-random strategies plus five random seeds
  across ten pair-budget blocks) had strict reachability 1.0 and exact requested vocabulary.
- Pre-registered pass criteria: failed on criteria 1-3; criterion 4 is satisfied. **H5 is killed.**

### All strategies (aggregate bytes/token, primary pairs)

| Pair / budget | high | low | mismatch | rare | ttr | first | random | full16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| en->sw 8192 | 2.2523 | 2.2588 | 2.2523 | 2.2542 | 2.2602 | 2.2601 | 2.2590 | 2.2622 |
| en->sw 16384 | 2.2739 | 2.2810 | 2.2709 | 2.2803 | 2.2810 | 2.2792 | 2.2804 | 2.2825 |
| en->ar 8192 | 1.5393 | 1.5395 | 1.5393 | 1.5395 | 1.5396 | 1.5394 | 1.5395 | 1.5397 |
| en->ar 16384 | 1.5395 | 1.5396 | 1.5395 | 1.5396 | 1.5396 | 1.5395 | 1.5396 | 1.5397 |
| en->hu 8192 | 2.0562 | 2.0610 | 2.0562 | 2.0608 | 2.0613 | 2.0602 | 2.0609 | 2.0628 |
| en->hu 16384 | 2.0668 | 2.0704 | 2.0654 | 2.0707 | 2.0702 | 2.0692 | 2.0693 | 2.0716 |

No strategy is meaningfully better than `first` or random. `high_mismatch` is nearly identical to
`high_fertility`; `low_fertility`, `high_rare`, and `high_ttr` are similar to random. The largest
strategy gap is still under 0.5% relative. `full_16mb` is also essentially tied with the 4 MB
arms, so adding 12 MB of candidate pool does not improve the donor-frequency inventory.

### The data-efficiency diagnostic is not evidence of selection value

The pre-registered diagnostic is `high_fertility / full_16mb`; values are 0.9956-0.9998, i.e.
4 MB of selected data matches 16 MB on aggregate compression. It is tempting to call this "4x
data efficiency." It is not. `first` 4 MB and random 4 MB also match 16 MB, because the extra
12 MB does not change the donor-frequency merge inventory much. The correct reading is: at these
budgets and this donor, more target data does not help donor-only selection, and no selection
strategy changes that. C5 isolates data selection without continuation; it does not test whether
data selection matters for continuation-based adaptation.

### Post-hoc controls (exploratory, not pre-registered)

These analyses were run after the frozen result and cannot rescue the kill. They are recorded to
make the negative result more informative.

- Fertility is partly a document-length proxy. Correlation of donor fertility with log document
  bytes: `sw -0.3213`, `ar -0.1030`, `hu -0.1509`, `zh +0.1513`, `code -0.0915`. Shorter documents
  tend to have higher donor fertility.
- `low_fertility` vs random: 3/10 positive Holm-significant comparisons, 0 negative, mean relative
  gain **+0.0093%**. The three positives are `hu 16384` (+0.0767%), `zh 8192` (+0.0361%),
  `zh 16384` (+0.0287%). This is noise-level, not a strategy.
- Length controls (`longest`, `shortest`) are also tiny. `low_fertility` vs `longest` is positive
  in several rows (absolute bytes/token: `sw 8192` +0.0067, `sw 16384` +0.0051, `hu 8192` +0.0042),
  but these are post-hoc, small, and inconsistent across pairs. No length signal is strong enough
  to promote.
- English retention for C5 strategies varies with the selected documents but does not change the
  verdict; the selected data still spends the full budget on donor merges.

The post-hoc script is `scripts/explore_c5_posthoc.py`; its output is
`results/exploratory_c5.json` and is explicitly marked `posthoc: true` and
`not_in_preregistration: true`.

### C5 interpretation

Tokenizer-aware target-data selection, at least the donor-fertility and related signals tested
here, does not improve donor-frequency merge selection. High fertility is slightly worse than
random; low fertility and the other scores are indistinguishable from random within practical
effect sizes. The donor merge inventory is the binding constraint, which is consistent with C1 and
with C4: the useful intervention is to add new target merges, not to choose which target documents
to count.

## Validity, deviations, and threats

Recorded honestly; none changes a pre-registered decision rule.

1. **C4 unreachable-merge pruning (implementation repair).** The final hybrid tokenizer is pruned
   of merges that cannot be reached under the copied donor pre-tokenizer, then rebuilt. This was
   added to enforce the pre-registered strict-reachability requirement. Up to three merges were
   pruned in 14 of 70 arms; actual vocabularies range 8190-8192 and 16381-16384, at most 0.024%
   below request. All arms remain within 1%. This is a validity repair, not a threshold change.
2. **C4 learned merges can cross pre-tokenized boundaries.** The learned character-level BPE is
   trained on the concatenated token sequence for each document with no pre-tokenizer, as frozen.
   Some learned merges span many tokens or whole document fragments (hundreds of characters).
   Under the copied ByteLevel regex pre-tokenizer, those merges are effectively never applied; the
   longest token observed in 200 Swahili test documents was 16 characters. Giant merges therefore
   waste merge slots but do not inflate test compression. If anything, C4 is conservative. The
   pre-registration did not require per-word continuation; this is a protocol limitation to fix in
   a follow-up, not a result change.
3. **C5 evaluation bug found and fixed before the final run.** The first C5 implementation scored
   each arm on the selected training texts, not the held-out target test set. The pre-registered
   primary significance tests already used per-document values on the held-out test set, so the
   pass/kill decision was unaffected; the printed diagnostic levels were affected. The final
   `results/invention_c5.json` is from the fixed code and scores every arm on the held-out test
   set. The bug and fix are recorded here rather than hidden.
4. **C4 native-gap reporting fix before the final run.** The first implementation divided a
   per-document delta by an aggregate native gap. The final version uses the aggregate delta for
   native-gap closure and reports the per-document delta separately. The primary pass/fail test
   was always the per-document bootstrap and was unaffected.
5. **Strict-reachability validation broadened before the final run.** C4 and C5 now check every
   reported arm, not only the primary arm. Both final runs pass.
6. **Single run, compression proxy.** There is no language-model BPB arm. Compression is a
   tokenizer-quality proxy; downstream BPB and task accuracy are not measured. The C4 English
   retention result is a warning that fixed-budget reallocation can hurt source-language
   compression even when target compression improves.
7. **BPE only, one donor family, one data scale.** The conclusions are local to byte-level BPE,
   these six corpora, 4 MB of target training text, and 8k/16k vocabularies. The donor is English
   `bpe_32768`; code-to-English is the only reversed direction.
8. **C5 candidate pool is a 16 MB prefix.** It is not the full target train split. The
   pre-registration required this limitation to be reported with every result.

## Prior art and novelty

**C4 is a replication/variant, not a new invention.** Continued BPE training for tokenizer
adaptation is established:

- Taido Purason, Pavel Chizhov, Ivan P. Yamshchikov, Mark Fishel. "Teaching Old Tokenizers New
  Words: Efficient Tokenizer Adaptation for Pre-trained Models." Findings of EACL 2026,
  pp. 6492-6516. DOI 10.18653/v1/2026.findings-eacl.341.
  https://aclanthology.org/2026.findings-eacl.341/
- Reference implementation: https://github.com/taidopurason/tokenizer-extension
- `bpe-continue`: https://pypi.org/project/bpe-continue/ (continued BPE training with existing
  token IDs and merge ranks preserved).

The EACL method extends a base vocabulary and continues BPE learning (with an optional leaf-based
pruning stage), so it adds tokens rather than reallocating an existing budget; it reports little
English degradation. C4 differs in the budget rule: it holds total vocabulary fixed, drops half the
donor merges, and appends target merges. That is the source of both the target gain and the English
retention cost. The random-donor control is the part of C4 that is not covered by the published
method; it shows the donor subset can be random without losing most of the gain. We do not claim
this as a new invention.

**C5 is incremental and failed.** Related work includes multilingual tokenizer data design and
entropy-based tokenizer-data curation:

- Souvik Rana et al. "MUTANT: A Recipe for Multilingual Tokenizer Design." ACL 2026.
  https://arxiv.org/abs/2511.03237
- Andrei-Valentin Tanase, Elena Pelican. "SupraTok: Cross-Boundary Tokenization for Enhanced
  Language Model Performance." https://arxiv.org/abs/2508.11857 (entropy-driven data curation as
  one component of tokenizer training).

We did not find an exact prior for selecting target documents by donor fertility for
donor-frequency merge selection. The experiment is still an incremental test, and it produced a
negative result. No novelty claim is made.

## Reproduction

```bash
.venv/bin/python scripts/invention_c4.py
.venv/bin/python scripts/invention_c5.py
.venv/bin/python scripts/explore_c5_posthoc.py
.venv/bin/python -m pytest -q
.venv/bin/ruff check src scripts tests
```

Outputs:

- `results/invention_c4.json` - pre-registration hash, per-pair/budget methods, per-document
  values for the bootstrap, primary Holm family, diagnostics.
- `results/invention_c5.json` - same structure, plus all selection strategies and data-efficiency
  diagnostic.
- `results/exploratory_c5.json` - explicitly post-hoc length and low-fertility controls.
- `logs/invention_c4_rerun.log`, `logs/invention_c5_rerun.log` - final run logs.

## What to do next

C4 says the useful lever is target-specific continuation, not donor-merge scoring. C5 says
document selection for donor-frequency counting is not the lever. The next honest experiments are:

1. **Pre-tokenized continuation.** Learn new merges per pre-tokenized word rather than on the
   concatenated document token stream. This avoids giant, unused merges and tests the same
   mechanism without wasting half the merge slots.
2. **BPB validation.** Repeat the best C4 arm with a small language model and measure bits per
   byte, not only compression. The English-retention cost makes this necessary before any
   practical recommendation.
3. **Budget allocation sweep.** 25/50/75 shares suggest a trade-off; a finer sweep with English
   retention as a constraint would turn C4 into a usable adaptation recipe.
4. **Donor subset ablation with more seeds.** The random control is close; determine whether
   frequency selection is worth any complexity at all.
5. **Data selection under continuation, not donor-only counting.** C5 tested the wrong stage for
   the C4 mechanism. Selecting documents for *learning new merges* is a different, still-open
   question; do not cite C5 as evidence against it.

## Final verdict

| Experiment | Pre-registered result | Evidence | Interpretation |
|---|---|---|---|
| C4 hybrid continuation | **PASS** | 6/6 positive, 0/6 negative, mean rel +123.0% (median +76.0%), gap closure 79.1%, all valid | Continued BPE works; replication of published method under a fixed-budget variant |
| C5 tokenizer-aware selection | **KILL** | 0/6 positive, 6/6 negative, mean rel -0.174%, all valid | Fertility-based document selection does not beat random; no practical signal |

C4 is a useful confirmation and a caution about English retention. C5 is a clean negative result.
Neither is a Nobel-level invention, and the write-up does not pretend otherwise.
