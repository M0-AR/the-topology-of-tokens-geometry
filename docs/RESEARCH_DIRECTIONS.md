# Research directions: pre-registered invention experiments

Status: pre-registration written before implementation and before any result was produced.
Date: 2026-11-06. Frozen data: `results/merge_features.csv`, `results/analysis.json`,
`results/scale_normalized_tda.json`, `results/scaling_law.json`, `results/tda_stability.json`.

The literature check that motivates these experiments is summarised in the chat report and in
`docs/EXTENSIONS.md`. The short version: tokenizer fingerprints/passports, merge pruning, and
cross-tokenizer alignment are crowded; persistence-weighted merge selection, sheaf-theoretic
compatibility, and a merge-filtration diagnostic are the three gaps that survived the search.

No experiment below may be reported as confirming a hypothesis unless it passes its pre-registered
kill criteria. Exploratory results are labelled exploratory.

## Shared rules

- Donor tokenizers are the existing 32768 BPE artifacts. Target tokenizers are the existing native
  BPE artifacts at the same requested size.
- Target train sample: first 4,000,000 bytes; target test sample: first 1,000,000 bytes; same
  sampling function as `scripts/merge_experiment.py`.
- All selection arms are ancestor-closed: if a merge is selected, every merge needed to produce its
  two children is selected too. This keeps vocabulary size matched and makes strict reachability a
  construction property, not a confound.
- Primary pairs: `wikitext_en -> wikipedia_sw`, `wikitext_en -> wikipedia_ar`,
  `wikitext_en -> wikipedia_hu`. Stress pair (reported, not in the primary decision):
  `wikitext_en -> wikipedia_zh`. Domain pair (exploratory): `codeparrot_py -> wikitext_en`.
- Budgets: 8192 and 16384 requested vocabulary size, i.e. 7932 and 16124 merges.
- Statistics: paired bootstrap over test documents, 2000 resamples, 95% percentile CI, two-sided
  bootstrap p-value. Holm correction over the six primary comparisons (3 pairs x 2 budgets).
- Every method is deterministic except `random`, which uses seeds 13, 37, 73 and is reported as a
  mean with a range.

## C1 - Persistence-selected merge cores

### Hypotheses

- H1a (primary): at matched vocabulary, persistence-weighted selection beats frequency-only
  selection on target `test_compression` (bytes per token, higher is better) for at least 2 of the
  3 primary pairs at 8192 or 16384, with a paired-bootstrap 95% CI excluding zero after Holm.
- H1b (secondary): the gain is larger for lower-resource targets (`sw`, `hu`) than for `ar`.
- H1c (mechanical): ancestor closure gives strict merge reachability at or above 99% for every arm;
  unconstrained selection is expected to fail and is reported only as a diagnostic.

### Scores compared

1. `prefix`: donor merge prefix (exact truncation baseline).
2. `native`: native target BPE at the same vocabulary (upper baseline, not a selection method).
3. `frequency`: effective parent-token frequency on target train (AdaptBPE-style proxy).
4. `persistence`: merge-tree lifetime, `next_absorption_rank - rank`, normalised by the merge count.
5. `combined`: `log1p(frequency) * (1 + persistence)`.
6. `redundancy`: `log1p(frequency) * (1 + persistence) / (1 + log1p(alternate_decompositions))`.
7. `random`: random ancestor-closed selection, three seeds.

`persistence` is the H0 lifetime of a merge in the merge DAG filtered by merge rank: a merge that is
absorbed quickly by a higher merge has a short lifetime; a merge that remains a terminal token for
many ranks has a long lifetime. `alternate_decompositions` counts splits `(a, b)` of the parent
surface with both `a` and `b` in the donor vocabulary, minus the canonical split; this is the local
H1 cycle count of the merge graph (alternative merge paths).

### Metrics

- `test_compression` on target test (bytes / tokens).
- `strict_reachability`: fraction of selected parent surfaces that the selected merge list reduces
  to a single token, simulated directly on the byte-level surface (pretokenizer bypassed).
- `english_retention`: compression on `wikitext_en` test divided by native en BPE compression at the
  same requested size.
- `actual_vocab_size`, `selected_merges`.

### Kill criteria

- Kill C1 if `combined` or `persistence` loses to `frequency` on 2 of 3 primary pairs at 8192 or
  16384, or if the best persistence arm gains less than 0.5% compression over `frequency`, or if the
  paired-bootstrap CI for the gain includes zero after Holm.
- Kill the claim that persistence is the active ingredient if `combined` does not beat
  `frequency` but `persistence` alone does; in that case the result is a frequency effect and is
  reported as such.

## C2 - Sheaf-theoretic compatibility certificate

### Correction before the C2 run (2026-11-06)

The originally pre-registered H2a was mathematically wrong. The interval-overlap graph of two
partitions of a line is a forest: a crossing partition such as A = [0,2],[2,4] and
B = [0,1],[1,3],[3,4] produces the path A0-B0-A1-B1-A2 with `beta1 = 0`. Therefore the interval
complex cannot certify crossing, and the original claim is withdrawn before any C2 result was
produced. The interval complex is retained only as a negative control.

### Corrected hypotheses

- H2a (primary, exploratory because n is small): the cycle rank of the union of the two
  tokenizers' merge trees for a shared token predicts that token's demerge loss better than token
  length or frequency. This is the first computable proxy for the sheaf obstruction on the merge
  DAG; it is not claimed to be the full sheaf.
- H2b (exploratory): a pair-level merge-DAG compatibility score (vocabulary Jaccard, merge-pair
  Jaccard, mean union cycle rank) predicts the transfer gap `native - adapted` across the five
  donor/target pairs better than vocabulary Jaccard alone.

### Construction

For a shared token surface `t`, build the union of the two tokenizers' binary merge decomposition
trees: nodes are substrings of `t` that are tokens in either tokenizer, edges are parent-to-child
merge relations. The cycle rank `E - V + C` is the H1 proxy of the union complex. Base tokens
contribute a single node. A token that is decomposed one way in A and another way in B can create a
cycle; a token that is a base token in one tokenizer and a merged token in the other does not, which
is a known limitation of this proxy. The pair-level score aggregates the cycle rank over sampled
shared tokens and adds vocabulary Jaccard and merge-pair Jaccard.

### Metrics and baselines

- Per-token: `demerge_loss = 1 - boundary_jaccard` on the token's decoded surface; `union_beta1`,
  `depth_gap`, `length`, corpus frequency.
- AUC of `union_beta1` for `demerge_loss > 0`; Spearman correlation with continuous loss; the same
  for length and frequency as baselines.
- Pair-level: Spearman correlation of each compatibility score with the C1 transfer gap
  `native - frequency` across the five pairs (n = 5, exploratory).
- Negative control: interval-overlap `beta1`, expected to be zero even for crossing partitions.

### Kill criteria

- Kill the per-token claim if `union_beta1` does not beat the best of length and frequency on AUC by
  at least 0.05, or has the wrong sign.
- Kill the pair-level claim if the best compatibility score's Spearman correlation is below 0.5 in
  absolute value or has the wrong sign.
- If both fail, report the merge-DAG topology as non-predictive for compatibility and do not claim
  a sheaf certificate.

## C3 - Merge-filtration persistence descriptor

### Hypotheses

- H3a: normalised H1 persistence entropy and log-bar summary have ICC(2,1) >= 0.8 across point-count
  subsamples (128, 256, 512) in the frozen `tda_stability.json`.
- H3b (decision value): the combined C1 score predicts the marginal compression loss of removing a
  merge better than raw frequency, with a Spearman gain of at least 0.05 on the en -> sw pair.

### Protocol

- ICC is computed on the existing stability records, no new topology runs.
- Marginal loss ablation: take the 64 highest-frequency selected merges on en -> sw at 8192; rebuild
  the tokenizer without each merge (and without its descendants, so the vocabulary stays valid);
  measure test-compression loss. Correlate loss with frequency, persistence, alternate
  decompositions, and the combined score.

### Kill criteria

- Kill H3a if ICC < 0.8.
- Kill H3b if the combined score's Spearman correlation is not at least 0.05 above frequency, or if
  the sign is negative.

## Reporting

Every run writes a frozen JSON with the config, input hashes, per-row metrics, and the bootstrap
results. Results are appended to this file only after the run, with an explicit pass/kill label.
No post-hoc threshold changes.

## Results (2026-11-06)

All three experiments were run exactly as pre-registered, with the C2 correction above made before
any C2 result existed. Artifacts: `results/invention_c1.json`, `results/invention_c2.json`,
`results/invention_c3.json`. Reproduction: `.venv/bin/python scripts/invention_c1.py`,
`.venv/bin/python scripts/invention_c2.py`, `.venv/bin/python scripts/invention_c3.py`.

### C1 - KILL H1a

Frequency selection is the only useful arm. The pre-registered `combined` score is numerically
identical to frequency in 4 of 6 primary comparisons, slightly worse in one (en -> sw 8192, -0.0014,
CI -0.0023 to -0.0006, p < 0.0005) and statistically indistinguishable in the other (en -> hu 8192,
+0.0001, CI -0.0002 to +0.0005, p = 0.494). It is also slightly worse on the code -> en stress pair
at 8192 (-0.0032). Persistence alone is worse than the rank prefix in nine of ten rows (equal on
en -> zh at 16384); equal-weight rank fusion is worse still. Native target BPE remains far ahead.

| pair | budget | native | prefix | frequency | combined | persistence | rank fusion | random |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| en -> sw | 8192 | 3.7032 | 1.9387 | 2.2601 | 2.2567 | 1.8323 | 1.8237 | 1.8748 |
| en -> sw | 16384 | 4.0506 | 2.1404 | 2.2792 | 2.2792 | 2.0740 | 1.9398 | 2.0406 |
| en -> ar | 8192 | 5.8039 | 1.0114 | 1.5394 | 1.5394 | 1.0110 | 1.0050 | 1.0596 |
| en -> ar | 16384 | 6.4291 | 1.1278 | 1.5395 | 1.5395 | 1.0119 | 1.0056 | 1.3175 |
| en -> hu | 8192 | 3.7339 | 1.8213 | 2.0602 | 2.0604 | 1.7004 | 1.7341 | 1.7738 |
| en -> hu | 16384 | 4.1885 | 1.9180 | 2.0692 | 2.0692 | 1.8956 | 1.8371 | 1.8987 |
| en -> zh | 8192 | 3.5846 | 1.0186 | 1.0737 | 1.0737 | 1.0173 | 1.0170 | 1.0286 |
| en -> zh | 16384 | 3.9606 | 1.0220 | 1.0743 | 1.0743 | 1.0220 | 1.0182 | 1.0482 |
| code -> en | 8192 | 4.0777 | 2.9299 | 3.4950 | 3.4917 | 2.6799 | 2.2574 | 2.4823 |
| code -> en | 16384 | 4.5291 | 3.2389 | 3.5325 | 3.5325 | 3.1688 | 2.5818 | 2.9673 |

The only Holm-significant primary comparison is en -> sw at 8192, and its delta is negative
(combined minus frequency = -0.0014, CI -0.0023 to -0.0006, p < 0.0005). The other five primary
comparisons have p = 1.0 or p = 0.494. All arms have strict reachability 1.0 and exact requested
vocabulary size. English retention at 8192 is 0.774 (sw), 0.982 (ar), 0.789 (hu), 0.950 (zh); at
16384 it is 0.961, 0.998, 0.970, 0.996. The mechanism is clear: `log1p(frequency)` spans orders of
magnitude while the persistence factor spans only 1 to 2, so `combined` is frequency.

### C2 - KILL H2a and H2b

Per-token Spearman correlations of union cycle rank with demerge loss are -0.028, +0.154, -0.074,
-0.077, and -0.227 across the five pairs: near zero or the wrong sign. Four of the five pairs have
positive loss for every sampled context (the code -> en pair is 98.0% positive), so the
pre-registered AUC is degenerate and cannot support the claim. Token length and frequency are equally
unstable. Pair-level correlations with the transfer gap are -1.0 for
vocabulary Jaccard, -0.9 for merge-pair Jaccard, -0.9 for mean union cycle rank, -0.7 for depth gap,
and +0.7 for mean demerge loss; this is consistent with language/script distance but the topology
adds nothing beyond vocabulary Jaccard, and n = 5 with no partial-correlation control.

The interval-overlap negative control is also informative: crossing fractions are 0.405 to 0.735, yet
mean interval cycle rank is 0 to 0.135 rather than zero. The theorem holds for true partitions; the
ByteLevel offsets overlap at whitespace boundaries, so the data are not a partition. The original
certificate claim is withdrawn, and the corrected merge-DAG proxy fails.

### C3 - KILL H3a and H3b

ICC(2,1) for the pre-registered normalized feature `entropy_per_log_bar` is 0.391, below 0.8. The
secondary `total_per_bar` and `mean_persistence` reach 0.931, while raw total persistence is 0.162;
the pre-registered feature does not pass, and the secondary result is reported as secondary.

In the decision-value ablation (en -> sw, 8192, 64 merges), full compression is 2.2601; mean marginal
loss is 0.0148 and 95.3% of removals hurt. Spearman correlation of the score with marginal loss is
0.485 for frequency, 0.114 for combined, -0.371 for persistence, 0.356 for redundancy, and -0.460 for
rank fusion. Combined minus frequency is -0.371, not the required +0.05. On the random-ablation
control frequency is 0.568 and combined is 0.503. Frequency is already the best decision value.

### Overall verdict

All three pre-registered claims are killed. The robust findings are: (1) frequency-based merge
selection is a strong and simple adaptation baseline; (2) persistence, alternate-decomposition, and
rank-fusion weights add no decision value; (3) the donor merge inventory, not the selection
criterion, is the binding constraint, because no selector over donor merges can create the
target-specific merges that native training creates. No supported invention remains among C1, C2, or
C3. Any further direction must be pre-registered afresh, and the honest default is to write up the
negative result rather than to promote a survivor.

## Follow-up: C4/C5 (2026-11-06)

Two follow-up experiments were pre-registered separately, after C1-C3, in
[`docs/PREREGISTRATION_C4_C5.md`](PREREGISTRATION_C4_C5.md) (frozen SHA-256
`c6d530af8ee3bde6bf7ba86b53f3d34fa03930788988785c323764a3b918a55e`). They are not amendments to the
C1-C3 protocol.

- **C4 (fixed-budget hybrid continuation): PASS, but a replication/variant.** Continuing BPE on
  target text and appending the new merges to a donor half beat donor-only frequency selection in
  6/6 primary comparisons (Holm p < 0.0005), with mean per-document relative gain 123.0% (median
  76.0%) and mean native-gap closure 79.1%. All 70 constructed tokenizers were valid. This
  confirms the C1-C3 diagnosis that the donor inventory is the constraint. It is not a new
  invention: continued BPE is published (Purason et al., Findings of EACL 2026), and C4 differs by
  reallocating a fixed budget, which costs English compression (mean retention 73.5% vs 91.2% for
  donor-only frequency). The random-donor control shows most of the gain comes from target
  continuation, not frequency-based donor selection.
- **C5 (tokenizer-aware target-document selection): KILL.** Selecting target documents by donor
  token fertility did not improve donor-frequency merge selection. High fertility was negative in
  6/6 primary comparisons (Holm p < 0.0005; mean relative gain -0.174%) and slightly worse than
  random; low fertility and other scores were indistinguishable from random. All 120 constructed
  tokenizers were valid. The post-hoc length and low-fertility controls are exploratory and do not
  rescue the kill.

Full results, validity checks, deviations, prior art, and reproduction commands:
[`docs/C4_C5_RESULTS.md`](C4_C5_RESULTS.md). The honest summary is: continuation works, donor-only
data selection does not, and neither result is a novel invention claim.
