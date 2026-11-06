# Pre-registration: C4 hybrid merge inventory and C5 tokenizer-aware data selection

Status: written before implementation of the experiment runners and before any C4/C5 result exists.
Date: 2026-11-06 (UTC). This document is frozen when its SHA-256 is recorded in
`docs/PREREGISTRATION_C4_C5.sha256`. No threshold, baseline, or metric below may be changed after
that point.

C4 and C5 are independent experiments. Each has its own six-comparison Holm family. p-values are
never pooled across C4 and C5. A pass in one experiment does not rescue a kill in the other.

## Shared rules

- Inputs are the existing frozen corpora and tokenizers: donor `bpe_32768`, target native `bpe_8192`
  and `bpe_16384`; the same train/test splits used by C1.
- Primary pairs: `wikitext_en -> wikipedia_sw`, `wikitext_en -> wikipedia_ar`,
  `wikitext_en -> wikipedia_hu`. Stress pairs: `wikitext_en -> wikipedia_zh`,
  `codeparrot_py -> wikitext_en`.
- Budgets: vocabulary sizes 8192 and 16384. Initial vocabulary is 260 tokens, so the merge budget is
  `vocab_size - 260` (7932 and 16124).
- Primary comparisons are the three primary pairs at both budgets, six in total. Stress pairs are
  reported but are not in the Holm family.
- Test set: the first 1,000,000 bytes of the target `test` split, sampled in file order, exactly as
  C1. Primary metric: aggregate test compression in bytes per token (`raw_bytes / tokens`), with
  per-document compression values used for paired bootstrap.
- Paired bootstrap: 2000 document resamples, seed 20261106, two-sided p-value as implemented by
  `compression_statistics`. Holm correction within each experiment's six primary comparisons.
  Significance is Holm-adjusted p < 0.05.
- Merge selection always uses ancestor closure. Every tokenizer is checked with
  `strict_reachability`; a run with strict reachability below 1.0 is invalid and is not reported as a
  result.
- Actual vocabulary size and actual donor/new merge counts are recorded. A run whose actual
  vocabulary differs from the requested budget by more than 1 percent is flagged; it is not silently
  treated as an exact-budget result.
- Results are appended to the research record only after the full run, with an explicit PASS or KILL
  label for every hypothesis. Exploratory arms are labelled exploratory and cannot rescue a kill.
- No threshold changes, no dropping pairs, no switching the primary arm after seeing results.

## C4 - Hybrid donor+target merge inventory (token-level continuation)

### Hypothesis

H4: continuing BPE training on target text after selecting half of the merge budget from the donor
tokenizer by target-corpus frequency produces a better target tokenizer than selecting the entire
merge budget from the donor inventory by frequency.

The mechanism under test is the C1 diagnosis: donor-only selection cannot create target-specific
merges. C4 creates target-specific merges on top of the donor token inventory and asks whether they
add value at the same total vocabulary budget.

### Construction (primary arm: continuation_50)

1. Load donor `bpe_32768`, its vocabulary, and its annotated merges. Count donor-token usage on the
   first 4,000,000 bytes of the target `train` split, exactly as C1.
2. Select `floor(0.5 * merge_budget)` donor merges by frequency, with ancestor closure, using
   `score_merges("frequency", ...)` and `select_merges(...)`.
3. Build a partial tokenizer from the selected donor merges and tokenize the same 4,000,000 target
   train bytes with it.
4. Map every distinct token string produced by the partial tokenizer to a distinct Unicode private
   use code point. Train a character-level BPE on the mapped token sequences for the remaining
   `merge_budget - selected_donor_count` merges: `BpeTrainer`, no pre-tokenizer, `min_frequency=1`,
   `initial_alphabet` equal to the used code points, deterministic input order.
5. Parse the learned merges. Map each learned symbol back to its sequence of donor tokens. A learned
   merge whose parent string is already a selected donor parent is skipped and not counted; later
   learned merges may still use the existing donor token. Remaining budget is filled with the next
   donor merges in frequency order only if the learned merge list is exhausted; this fallback is
   recorded.
6. Build the final tokenizer with the selected donor merges followed by the accepted new merges in
   learned order. The ordering is a valid topological order: donor merges keep their original ranks;
   new merges receive ranks after the donor block; each new merge's children are donor tokens or
   earlier new parents.
7. Copy the donor normalizer, pre-tokenizer, post-processor, and decoder.

Exploratory donor shares: 0.25 and 0.75. The primary arm is fixed at 0.50.

### Baselines

- `frequency`: C1 donor-only frequency selection for the full merge budget.
- `continuation_50`: primary arm above.
- `continuation_25`, `continuation_75`: exploratory, same construction with the other shares.
- `random_50`: select 50 percent of the donor block uniformly at random with seeds 13, 37, 73, then
  continue exactly as the primary arm. Report mean, minimum, and maximum over seeds.
- `native`: target native BPE at the requested budget. This is a ceiling/reference, not a competitor
  in the primary comparison.

### Metrics and comparisons

- Primary: `continuation_50` minus `frequency` in per-document test compression, on the six primary
  comparisons (three pairs x two budgets), paired bootstrap, Holm over six.
- Effect size: relative gain `(continuation_50 - frequency) / frequency` and native-gap closure
  `(continuation_50 - frequency) / (native - frequency)`, both averaged over the six primary
  comparisons and reported per comparison.
- Secondary: English retention on 1,000,000 bytes of `wikitext_en` test for non-English targets;
  actual vocabulary and merge counts; strict reachability; stress-pair results; exploratory shares
  and random control.

### Kill criteria

Kill H4 unless all of the following hold:

1. At least four of the six primary comparisons have a positive delta with Holm-adjusted p < 0.05.
2. No primary comparison has a negative delta with Holm-adjusted p < 0.05.
3. The mean relative gain over the six primary comparisons is at least 0.5 percent.
4. The construction is valid: all reported tokenizers have strict reachability 1.0 and actual
   vocabulary within 1 percent of the requested budget.

If H4 is killed, report token-level continuation as not improving on donor frequency selection. Do
not describe exploratory shares or the random control as a positive result.

## C5 - Tokenizer-aware target-data selection

### Hypothesis

H5: selecting high-donor-fertility documents from a larger target candidate pool, at the same 4 MB
selection budget, produces a better donor-frequency merge inventory than random selection of the same
size.

The mechanism under test is data efficiency: if the donor tokenizer struggles most on certain target
documents, those documents may provide more useful merge evidence per byte for adaptation.

### Candidate pool and selection

- Candidate pool: the first 16,000,000 bytes of the target `train` split, sampled in file order.
- Selection budget: 4,000,000 bytes, the same byte budget as C1's target training text.
- A strategy assigns a score to every document in the pool. Documents are sorted by descending score,
  ties broken by ascending pool index. Documents are appended until the budget is reached; the last
  document is truncated to the exact remaining byte budget. `first` uses the pool in file order
  without scoring; `random` shuffles pool indices with a fixed seed.
- The candidate pool is a prefix of the train split, not the entire train split. This limitation is
  reported with every result.

### Strategies

- `high_fertility` (primary): donor tokens per byte, descending.
- `low_fertility`: donor tokens per byte, ascending.
- `high_mismatch`: `donor_tokens/byte - target_native_tokens/byte`, descending, where the target
  native tokenizer is `bpe_32768`.
- `high_rare`: fraction of donor-token occurrences whose donor token type occurs at or below the 25th
  percentile of nonzero donor type frequencies on the 16 MB pool, descending.
- `high_ttr`: distinct donor token types divided by donor tokens in the document, descending.
- `first`: first 4 MB of the pool, the C1-style baseline.
- `random`: shuffle with seeds 13, 37, 73, 101, 149, then take 4 MB.
- `full_16mb`: all 16 MB, a ceiling reference for data efficiency.

The primary arm is fixed at `high_fertility`. All other scored strategies and `full_16mb` are
exploratory/secondary.

### Merge inventory and evaluation

For each selected text and each target/budget, count donor-token usage on the selected text and
select donor merges by frequency with ancestor closure, exactly as C1. Build the tokenizer and
evaluate the same 1,000,000-byte target test set used by C1 and C4. No hybrid continuation is used in
C5; the experiment isolates data selection.

### Metrics and comparisons

- Primary: `high_fertility` minus the mean of the five `random` runs in per-document test
  compression, on the six primary comparisons, paired bootstrap, Holm over six. The random baseline
  per document is the mean of that document's compression across the five random seeds.
- Secondary: `high_fertility` vs `first` (the C1 baseline), `high_fertility` vs `full_16mb`, the other
  scored strategies, English retention for non-English targets, actual vocabulary and merge counts,
  strict reachability, and stress-pair results.
- Data-efficiency diagnostic: `high_fertility` aggregate compression divided by `full_16mb` aggregate
  compression on the six primary comparisons. This is reported but is not a pass criterion.

### Kill criteria

Kill H5 unless all of the following hold:

1. At least four of the six primary comparisons have a positive delta with Holm-adjusted p < 0.05.
2. No primary comparison has a negative delta with Holm-adjusted p < 0.05.
3. The mean relative gain over the six primary comparisons is at least 0.5 percent.
4. The construction is valid: all reported tokenizers have strict reachability 1.0 and actual
   vocabulary within 1 percent of the requested budget.

If H5 is killed, report tokenizer-aware selection as not improving on random selection at the same
byte budget. Do not promote a secondary strategy as a positive result after a primary kill.

## Reporting

Both runners must write a frozen JSON containing config, input hashes, pre-registration hash, per-pair
and per-budget rows, per-document values needed to reproduce the bootstrap, and the primary Holm
table. The final research note must state PASS or KILL for H4 and H5 separately, report all primary
comparisons, and list the exact commands used. Negative results are published with the same care as
positive ones.
