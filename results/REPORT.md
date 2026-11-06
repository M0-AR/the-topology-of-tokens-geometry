# TopoBPE results report

Generated from `results/analysis.json`.

## Confirmatory models

| target | model | r2 | mae | rmse | bootstrap_ci_low | bootstrap_ci_high |
|---|---|---|---|---|---|---|
| compression | baseline | 0.9740 | 0.1350 | 0.1562 | 0.7867 | 0.9901 |
| compression | full | 0.8430 | 0.3066 | 0.3834 | 0.0033 | 0.8994 |
| compression | topology | -0.7669 | 0.9247 | 1.2863 | -6.1478 | -0.2138 |
| bpb | baseline | 0.8866 | 0.0905 | 0.1040 | -0.3911 | 0.9416 |
| bpb | full | 0.5547 | 0.1664 | 0.2062 | -6.0105 | 0.8657 |
| bpb | topology | 0.1176 | 0.2274 | 0.2902 | -14.8111 | 0.7871 |

### Held-out error by corpus

| target | corpus | rows | model | mae | rmse |
|---|---|---|---|---|---|
| compression | codeparrot_py | 6 | baseline | 0.2728 | 0.2734 |
| compression | codeparrot_py | 6 | full | 0.1919 | 0.2348 |
| compression | codeparrot_py | 6 | topology | 0.3247 | 0.3805 |
| compression | wikipedia_ar | 6 | baseline | 0.1435 | 0.1456 |
| compression | wikipedia_ar | 6 | full | 0.6331 | 0.6442 |
| compression | wikipedia_ar | 6 | topology | 2.5006 | 2.5599 |
| compression | wikipedia_hu | 6 | baseline | 0.0545 | 0.0576 |
| compression | wikipedia_hu | 6 | full | 0.1077 | 0.1277 |
| compression | wikipedia_hu | 6 | topology | 0.3750 | 0.4357 |
| compression | wikipedia_sw | 6 | baseline | 0.0868 | 0.0876 |
| compression | wikipedia_sw | 6 | full | 0.5065 | 0.5424 |
| compression | wikipedia_sw | 6 | topology | 1.2415 | 1.4560 |
| compression | wikipedia_zh | 6 | baseline | 0.1231 | 0.1233 |
| compression | wikipedia_zh | 6 | full | 0.2656 | 0.2811 |
| compression | wikipedia_zh | 6 | topology | 0.5282 | 0.6872 |
| compression | wikitext_en | 6 | baseline | 0.1290 | 0.1556 |
| compression | wikitext_en | 6 | full | 0.1348 | 0.1496 |
| compression | wikitext_en | 6 | topology | 0.5781 | 0.6690 |
| bpb | codeparrot_py | 3 | baseline | 0.1401 | 0.1415 |
| bpb | codeparrot_py | 3 | full | 0.3318 | 0.3339 |
| bpb | codeparrot_py | 3 | topology | 0.5225 | 0.5257 |
| bpb | wikipedia_ar | 3 | baseline | 0.0672 | 0.0771 |
| bpb | wikipedia_ar | 3 | full | 0.0387 | 0.0469 |
| bpb | wikipedia_ar | 3 | topology | 0.0635 | 0.0706 |
| bpb | wikipedia_hu | 3 | baseline | 0.0428 | 0.0439 |
| bpb | wikipedia_hu | 3 | full | 0.0376 | 0.0506 |
| bpb | wikipedia_hu | 3 | topology | 0.0471 | 0.0608 |
| bpb | wikipedia_sw | 3 | baseline | 0.0586 | 0.0594 |
| bpb | wikipedia_sw | 3 | full | 0.1108 | 0.1132 |
| bpb | wikipedia_sw | 3 | topology | 0.1308 | 0.1321 |
| bpb | wikipedia_zh | 3 | baseline | 0.1559 | 0.1576 |
| bpb | wikipedia_zh | 3 | full | 0.2837 | 0.2938 |
| bpb | wikipedia_zh | 3 | topology | 0.3546 | 0.3714 |
| bpb | wikitext_en | 3 | baseline | 0.0785 | 0.0932 |
| bpb | wikitext_en | 3 | full | 0.1956 | 0.1990 |
| bpb | wikitext_en | 3 | topology | 0.2458 | 0.2545 |

### Permutation tests

| hypothesis | observed_delta_r2 | p_value | iterations |
|---|---|---|---|
| H1 | -0.1310 | 0.9575 | 2000 |
| H2 | -0.3319 | 0.5932 | 2000 |

### Holm correction

| adjusted_p | hypotheses | raw_p |
|---|---|---|
| 1.0000 | H1 | 0.9575 |
| 1.0000 | H2 | 0.5932 |
| 0.0240 | H3 | 0.0060 |
| 1.0000 | H4 | 1.0000 |

## H3 partial Spearman

| ci_high | ci_low | iterations | p_value | rho | variable |
|---|---|---|---|---|---|
| 0.9240 | -0.4806 | 1962 | 0.0905 | 0.4362 | vocab_h1_total_persistence |
| 0.8376 | -0.4656 | 1962 | 0.2534 | 0.2921 | vocab_h1_persistence_entropy |
| 0.1126 | -0.8766 | 1962 | 0.0015 | -0.7176 | cooc_h1_total_persistence |
| 0.6971 | -0.6650 | 1962 | 0.4813 | 0.1820 | cooc_h1_persistence_entropy |

## H4 compression

| model | r2 | mae | ci_low | ci_high |
|---|---|---|---|---|
| baseline | -26.0178 | 0.5713 | -67.4468 | -14.9597 |
| full | -70.3013 | 0.8774 | -139.7740 | -45.4112 |
| topology | -803.4092 | 3.2081 | -1826.5477 | -528.7142 |

Delta R2 (full - baseline): -44.2835 [-92.9654, -20.3707], one-sided bootstrap p=0.9990

## H4 BPB

| model | r2 | mae | ci_low | ci_high |
|---|---|---|---|---|
| baseline | -9.8888 | 1.1378 | -86.5266 | -5.2201 |
| full | -18.3090 | 1.5268 | -151.3961 | -10.1961 |
| topology | -16.9227 | 1.4631 | -138.7531 | -9.5035 |

Delta R2 (full - baseline): -8.4203 [-66.4587, -4.9602], one-sided bootstrap p=1.0000

## Exploratory within-market model

| target | model | r2 | mae |
|---|---|---|---|
| bpb | baseline | -0.5840 | 0.3150 |
| bpb | full | -0.5455 | 0.3209 |
| bpb | topology | -0.8175 | 0.3422 |
| compression | baseline | -0.3333 | 0.1020 |
| compression | full | -0.2137 | 0.0996 |
| compression | topology | -0.1020 | 0.0967 |

## H5 exploratory feature correlations

| ci_high | ci_low | feature | p_value | spearman_rho |
|---|---|---|---|---|
| 0.9133 | 0.3899 | vocab_h0_num_bars | 0.0000 | 0.7452 |
| 0.9132 | 0.3834 | vocab_h0_num_finite_bars | 0.0000 | 0.7452 |
| -0.2708 | -0.8516 | vocab_h1_std_persistence | 0.0000 | -0.6965 |
| -0.2419 | -0.8778 | vocab_h1_mean_persistence | 0.0000 | -0.6381 |
| -0.2213 | -0.8125 | vocab_h1_count_0p2 | 0.0001 | -0.6227 |
| 0.0109 | -0.8342 | vocab_h1_betti_area | 0.0003 | -0.5647 |
| -0.0449 | -0.7927 | vocab_h1_betti_at_0p4 | 0.0005 | -0.5487 |
| -0.0514 | -0.7931 | vocab_h1_count_0p1 | 0.0006 | -0.5416 |
| 0.7749 | 0.5364 | vocab_h1_betti_at_0p1 | 0.0007 | 0.5364 |
| -0.2072 | -0.7129 | vocab_h1_max_persistence | 0.0008 | -0.5332 |

## Robustness: drop train-compression baseline

| target | model | r2 | mae |
|---|---|---|---|
| bpb | baseline | 0.4625 | 0.1690 |
| bpb | full | 0.3781 | 0.1962 |
| bpb | topology | 0.1176 | 0.2274 |
| compression | baseline | -0.3390 | 0.9187 |
| compression | full | -0.3143 | 0.8003 |
| compression | topology | -0.7669 | 0.9247 |

## Robustness: extended ridge alpha grid

| target | model | r2 | mae |
|---|---|---|---|
| bpb | baseline | 0.8866 | 0.0905 |
| bpb | full | 0.5547 | 0.1664 |
| bpb | topology | -0.2446 | 0.2657 |
| compression | baseline | 0.9740 | 0.1350 |
| compression | full | 0.8430 | 0.3066 |
| compression | topology | -0.7619 | 0.9295 |

## Tables on disk

- `results/tables/bpb_summary.csv`
- `results/tables/feature_importance_bpb.csv`
- `results/tables/feature_importance_compression.csv`
- `results/tables/h1_predictions.csv`
- `results/tables/h2_predictions.csv`
- `results/tables/h3_partial_spearman.csv`
- `results/tables/h5_feature_correlations.csv`
- `results/tables/tda_library_agreement.csv`
- `results/tables/tda_stability.csv`
- `results/tables/tokenizer_byte_audit.csv`

## Figures on disk

- `results/figures/fig1_persistence_examples.pdf`
- `results/figures/fig2_compression_scatter.pdf`
- `results/figures/fig3_loco_r2.pdf`
- `results/figures/fig4_bpb.pdf`
- `results/figures/fig5_market.pdf`
- `results/figures/fig6_feature_importance.pdf`
