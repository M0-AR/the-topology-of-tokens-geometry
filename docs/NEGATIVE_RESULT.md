# A pre-registered negative result on persistence-selected merge cores and merge-graph compatibility for cross-corpus BPE adaptation

## Abstract

We tested three pre-registered hypotheses about whether topological structure in a BPE merge graph
can improve cross-corpus tokenizer adaptation. C1 asked whether a persistence-weighted merge score
beats frequency selection when a donor tokenizer is truncated to a smaller vocabulary. C2 asked
whether the cycle rank of the union of two merge trees predicts per-token segmentation disagreement,
and whether pair-level merge-graph compatibility predicts transfer success. C3 asked whether a
merge-filtration persistence descriptor is stable across subsampling and whether it identifies merges
with high marginal compression value. All three hypotheses were killed by their pre-registered
criteria. Frequency selection is the only arm that works, and it is the arm with no topology in it.
Native target training remains far ahead of every donor-selection arm. The pattern is consistent with
a simple explanation: the binding constraint is the donor merge inventory, not the selection
criterion, because no selector over donor merges can create the target-specific merges that native
training creates. We report the full protocol, the exact frozen numbers, the pre-run correction of a
mathematical claim, and the limitations that keep this from being a universal negative result.

## 1. Motivation

Cross-corpus tokenizer adaptation asks whether a tokenizer trained on a high-resource donor corpus
can be truncated or reweighted for a lower-resource target instead of training a target tokenizer
from scratch. The practical appeal is obvious: reuse a large merge inventory, choose a subset, avoid
a full BPE training run. The scientific question is whether the choice of subset can be made better
by looking at the structure of the merge graph rather than only at token frequencies.

The repository behind this report previously found that persistent-homology features of token
co-occurrence clouds are not predictive of bits-per-byte beyond simple baselines, and that the
topology is stable under normalization but not under raw scale. That negative result motivated a
more targeted question: perhaps topology is useful not as a corpus-level predictor but as a
*merge-level* selection signal. A merge that creates a long-lived token, or that sits in a cyclic
part of the merge DAG, might be more valuable than a merge that is merely frequent. This report tests
that idea in the most direct form we could pre-register.

The prior-art review found no direct test of persistence-selected merge cores for cross-corpus BPE
adaptation, and no tokenizer-specific sheaf application. It did find a crowded space around merge
pruning and cross-tokenizer alignment, which is why the hypotheses below were designed to be killed
quickly rather than to be promoted.

## 2. Experimental setup

### 2.1 Data and base tokenizers

Six corpora were used: `wikitext_en` (11 MB), `wikipedia_sw` (21 MB), `wikipedia_hu` (89 MB),
`wikipedia_ar` (161 MB), `wikipedia_zh` (119 MB), and `codeparrot_py` (113 MB). The donor tokenizer
is a 32,768-token BPE model. It contains 260 initial tokens and 32,508 merges, so a requested
vocabulary of 8,192 corresponds to a merge budget of 7,932 and a requested vocabulary of 16,384
corresponds to 16,124.

The five donor/target pairs are `wikitext_en -> wikipedia_sw`, `-> wikipedia_ar`, `-> wikipedia_hu`,
`-> wikipedia_zh`, and `codeparrot_py -> wikitext_en`. The first three are the primary pairs; the
Chinese pair is a script stress test; the code-to-English pair is a domain shift.

C1 used 4 MB of target training text and 1 MB of target test text. A 500 KB / 200 KB split was used
only for pipeline smoke testing and is not reported as evidence. C2 used 500 KB of target training
text to collect token contexts. C3 used the same 4 MB / 1 MB split as C1.

### 2.2 C1: persistence-selected merge cores

Every arm selects a subset of the donor's 32,508 merges, then rebuilds a BPE tokenizer from the
donor's initial tokens, the selected merges, and the tokens those merges need. Selection is
ancestor-closed: if a merge is selected, every merge that creates one of its children is selected
too. This keeps the vocabulary valid, makes the requested vocabulary size exact, and makes strict
reachability a construction property rather than a post-hoc filter.

The arms are:

- `prefix`: the first `B` merges in donor rank order.
- `frequency`: merges sorted by donor-token usage on the target training text.
- `persistence`: the H0 lifetime of the merge in the donor merge DAG, filtered by rank:
  `(next_absorption_rank - rank) / max_rank`.
- `combined`: `log1p(frequency) * (1 + persistence)`.
- `redundancy`: `log1p(frequency) * (1 + persistence) / (1 + log1p(alternate))`, where `alternate`
  counts non-canonical decompositions of the merge's parent with both parts in the donor vocabulary.
- `rank_fusion`: reciprocal-rank fusion of frequency and persistence, with `k = 60`. This arm was
  added after the smoke run and is reported as exploratory; the pre-registered primary arm remained
  `combined`.
- `random`: a random subset, averaged over seeds 13, 37, and 73.

Metrics are target test compression in bytes per token, strict reachability (re-simulate BPE on the
parent surface with the pretokenizer bypassed), actual vocabulary size, and English retention
(compression on an English test set divided by a native English tokenizer at the same vocabulary
size). Comparisons use a paired bootstrap over test documents with 2,000 resamples. The primary
family is `combined - frequency` on the three primary pairs at the two budgets, with Holm
correction over six comparisons.

The kill criterion, fixed before the run: kill C1 if `combined` or `persistence` loses to
`frequency` on two of the three primary pairs at 8,192 or 16,384, or if the best persistence arm
gains less than 0.5% compression, or if the paired-bootstrap interval includes zero after Holm.

### 2.3 C2: merge-graph compatibility

The original C2 claim was that the first Betti number of the token-span overlap complex is one
exactly when two tokenizers' boundary sets cross. That claim is false. For two disjoint interval
partitions of a line, the overlap graph is a forest. A crossing example such as
A = [0,2], [2,4] and B = [0,1], [1,3], [3,4] produces the path A0-B0-A1-B1-A2 with cycle rank zero.
An exhaustive check over all partition pairs up to eight boundary points found no positive cycle
rank. The claim was withdrawn and the correction was written into the pre-registration before any C2
result existed. The interval complex is retained only as a negative control.

The corrected per-token hypothesis is that the cycle rank of the union of the two tokenizers' binary
merge trees for a shared token predicts that token's demerge loss. For a token surface, the union
complex has nodes equal to substrings that are tokens in either tokenizer and edges equal to
parent-to-child merge relations. A token decomposed one way in A and another way in B can create a
cycle. The outcome is `demerge_loss = 1 - boundary_jaccard` on a real corpus context around the
token, not on the token in isolation; an isolated shared token is encoded as one token by both
tokenizers and would trivially have zero loss.

The corrected pair-level hypothesis is exploratory: vocabulary Jaccard, merge-pair Jaccard, and mean
union cycle rank may predict the C1 transfer gap `native - frequency` across the five pairs. This is
reported with n = 5 and no causal claim.

Kill criteria: kill the per-token claim if union cycle rank does not beat the best of token length
and frequency on AUC by at least 0.05, or has the wrong sign. Kill the pair-level claim if the best
compatibility score's Spearman correlation is below 0.5 in absolute value or has the wrong sign.

### 2.4 C3: merge-filtration persistence descriptor

C3 has two parts. The stability part computes ICC(2,1) for normalized H1 persistence summaries
across point-count subsamples of 128, 256, and 512, with the six corpora as targets and the three
subsamples as raters. The pre-registered primary feature is persistence entropy per log bar. The
decision-value part removes each of the 64 highest-frequency selected merges from the en -> sw
8,192 tokenizer, rebuilds the tokenizer, and measures the marginal compression loss. It correlates
that loss with frequency, persistence, alternate decompositions, combined, and rank fusion.

Kill criteria: kill H3a if ICC < 0.8. Kill H3b if the combined score's Spearman correlation is not at
least 0.05 above frequency, or if the sign is negative.

## 3. Results

### 3.1 C1: frequency wins, persistence adds nothing

| pair | budget | native | prefix | frequency | combined | persistence | rank fusion | random |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
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

`combined` is numerically identical to `frequency` in four of the six primary comparisons (en -> sw
16,384; en -> ar at both budgets; en -> hu 16,384). In the other two it is statistically
indistinguishable: en -> sw 8,192 is slightly worse (bootstrap delta -0.0014, 95% interval -0.0023
to -0.0006; p < 0.0005) and en -> hu 8,192 is slightly better (+0.0001, interval -0.0002 to
+0.0005; p = 0.494). It is also slightly worse on the code -> en stress pair at 8,192 (-0.0032).
The only Holm-significant primary comparison is en -> sw 8,192, and its effect is negative. The
other five primary comparisons have p = 1.0 or p = 0.494. Persistence alone is worse than the rank
prefix in nine of ten rows (equal on en -> zh at 16,384). Rank fusion is worse than frequency in
every pair.

The mechanism is visible in the score scales. Frequency spans orders of magnitude; the persistence
factor spans roughly 1 to 2. A multiplicative combination therefore selects the same merges as
frequency, and an equal-weight rank fusion overweights the low-information persistence rank.

All arms have strict reachability 1.0 and the exact requested vocabulary size. English retention at
8,192 is 0.774 (sw), 0.982 (ar), 0.789 (hu), and 0.950 (zh); at 16,384 it is 0.961, 0.998, 0.970,
and 0.996.

The native gap `native - frequency` is:

| pair | 8192 | 16384 |
|---|---:|---:|
| en -> sw | 1.4431 | 1.7713 |
| en -> ar | 4.2645 | 4.8895 |
| en -> hu | 1.6737 | 2.1193 |
| en -> zh | 2.5110 | 2.8863 |
| code -> en | 0.5827 | 0.9966 |

C1 is killed.

### 3.2 C2: merge-tree topology does not predict demerge loss

Per-token results, using real corpus contexts:

| pair | n | mean loss | mean union beta1 | rho(beta1, loss) | rho(depth, loss) | rho(length, loss) |
|---|---:|---:|---:|---:|---:|---:|
| en -> sw | 475 | 0.504 | 1.272 | -0.028 | -0.051 | -0.042 |
| en -> ar | 591 | 0.633 | 0.338 | +0.154 | +0.150 | +0.351 |
| en -> hu | 569 | 0.570 | 0.448 | -0.074 | -0.006 | -0.001 |
| en -> zh | 761 | 0.317 | 0.652 | -0.077 | -0.057 | -0.261 |
| code -> en | 453 | 0.271 | 2.448 | -0.227 | -0.179 | -0.186 |

The correlations are near zero or the wrong sign. Four of the five pairs have positive loss for
every sampled context, so the pre-registered AUC is undefined; on the one pair with variation
(code -> en), the beta1 AUC is 0.331, worse than chance and worse than frequency (0.586) and depth
gap (0.583). Length is the only feature that reaches a moderate correlation, and only for Arabic
(+0.351), with the opposite sign for Chinese (-0.261).

Pair-level correlations with the transfer gap are -1.0 for vocabulary Jaccard, -0.9 for merge-pair
Jaccard, -0.9 for mean union cycle rank, -0.7 for depth gap, and +0.7 for mean demerge loss. The
sign is the expected one: more compatible pairs have smaller gaps. But vocabulary Jaccard is a proxy
for language and script distance, and the topology does not improve on it. With n = 5 and no partial
correlation control, this is a confound, not a certificate.

The interval-overlap negative control confirms the mathematical correction. Crossing fractions are
0.405 to 0.735, yet mean interval cycle rank is 0 to 0.135 rather than zero. The theorem holds for
true partitions; ByteLevel offsets overlap at whitespace boundaries, so the data are not a partition
and the control is not a clean test. The original certificate claim is withdrawn, and the corrected
proxy fails.

C2 is killed.

### 3.3 C3: the descriptor is not stable enough, and frequency is the decision value

ICC(2,1) for the BPE 16,384 group:

| feature | ICC(2,1) |
|---|---:|
| entropy per log bar (pre-registered) | 0.391 |
| total persistence per bar | 0.931 |
| mean persistence | 0.931 |
| raw total persistence | 0.162 |

The pre-registered feature fails at 0.391. The secondary normalized features pass, but they are
reported as secondary; moving the primary after seeing the result would be goalpost movement.

Decision-value ablation (en -> sw, 8,192, 64 merges): full compression is 2.2601 bytes/token, mean
marginal loss is 0.0148, and 95.3% of removals hurt compression. Spearman correlation with marginal
loss:

| score | top-64 frequency ablation | random ablation |
|---|---:|---:|
| frequency | 0.485 | 0.568 |
| combined | 0.114 | 0.503 |
| persistence | -0.371 | -0.085 |
| redundancy | 0.356 | 0.407 |
| rank fusion | -0.460 | -0.437 |

Combined minus frequency is -0.371, not the required +0.05. Frequency is already the best or
near-best decision value. Persistence is anti-correlated with marginal value.

C3 is killed.

## 4. Interpretation

Three things are now clear.

First, persistence is not a useful weight for merge selection in this setting. The H0 lifetime of a
merge in the donor DAG measures how long the merge survives as the vocabulary grows. It does not
measure how much compression the merge buys on a different corpus. Frequency does. The scale
mismatch is not a tuning problem: any monotone transformation of a quantity that spans orders of
magnitude will dominate a factor that spans 1 to 2, and the equal-weight rank fusion confirms that
the persistence rank is the weaker signal.

Second, the native gap is not a selection failure. Every C1 arm selects from the same donor
inventory. The native target tokenizer has merges that the donor inventory does not contain, and no
selector over donor merges can create them. The gap grows with script and language distance
(en -> ar and en -> zh are the largest), which is what an inventory explanation predicts. This is an
interpretation, not a direct causal test; the natural follow-up is a hybrid inventory experiment
that adds target-trained merges to the donor subset.

Third, topology at the merge level does not predict segmentation disagreement. The union cycle rank
is a measure of alternative merge paths, but demerge loss is driven by tokens that one tokenizer
represents as a single unit and the other splits into pieces, including base tokens that have no
merge tree at all. The proxy is structurally blind to that case. The pair-level compatibility signal
is real but trivial: it is language distance wearing a topological costume.

## 5. What would change the conclusion

The pre-registered thresholds were not met. A positive C1 result would require `combined` to beat
`frequency` by at least 0.5% with a paired-bootstrap interval excluding zero after Holm. A positive
C2 result would require union cycle rank to beat the best simple baseline by at least 0.05 AUC and a
pair-level correlation of at least 0.5 in absolute value with the correct sign, ideally with a
partial correlation controlling for vocabulary Jaccard. A positive C3 result would require the
pre-registered ICC to reach 0.8 and the combined score to beat frequency by at least 0.05 Spearman
on marginal loss.

A different operationalization could still succeed. The negative result is about these scores, this
donor, these targets, and this evaluation. It is not a proof that no topological method can ever
help.

## 6. Limitations

- Only BPE is tested. Unigram, WordPiece, and hybrid tokenizers have different merge structures, and
  the union-complex construction does not apply unchanged.
- The donor is English (and code). The five pairs are not a representative sample of language
  families, and n = 5 is too small for a pair-level law.
- The target training budget is 4 MB. Native tokenizers may behave differently at 100 KB or at 1 GB.
- Compression in bytes per token is not bits-per-byte. A language-model BPB evaluation was not run
  because the available GPU was shared; the compression result is a proxy.
- The non-random arms use a single deterministic selection. The bootstrap captures test-document
  variation, not selection variance across corpora or seeds.
- The C2 per-token sample is capped at roughly 450 to 760 contexts per pair, and four pairs have no
  negative labels, so the AUC analysis is degenerate.
- The C2 interval control is contaminated by overlapping ByteLevel offsets; the theorem applies to
  true partitions, not to these spans.
- The C3 stability analysis has one family and one vocabulary size in the frozen file, and the
  top-64 ablation is correlated with frequency by construction.

## 7. Reproducibility

Pre-registration: `docs/RESEARCH_DIRECTIONS.md`. C2 correction: the same file, dated before the C2
run. Frozen outputs: `results/invention_c1.json`, `results/invention_c2.json`,
`results/invention_c3.json`, and the smoke output `results/invention_c1_smoke.json`.

Commands:

```bash
.venv/bin/python scripts/invention_c1.py --out results/invention_c1.json
.venv/bin/python scripts/invention_c2.py --out results/invention_c2.json
.venv/bin/python scripts/invention_c3.py --out results/invention_c3.json
.venv/bin/python -m pytest -q
.venv/bin/ruff check src scripts tests
```

The test suite passes 37 tests, including synthetic checks for ancestor closure, strict
reachability, tokenizer round-trips, interval-overlap cycle rank, merge-tree union cycles, AUC, and
ICC. Lint is clean.

## 8. Conclusion

No supported invention survives C1, C2, or C3. Frequency-based merge selection is a strong and
simple adaptation baseline; persistence, alternate-decomposition, and rank-fusion weights add no
decision value; the donor merge inventory, not the selection criterion, is the binding constraint.
The honest deliverable is the negative result and the protocol that produced it. The next direction,
if any, must be pre-registered afresh. The two candidates worth considering are a hybrid inventory
that adds target-trained merges to the donor subset, and a tokenizer-aware data-selection rule that
uses compatibility to choose target training text. Neither is supported by the present evidence, and
neither should be described as an invention until it passes its own pre-registered test.
