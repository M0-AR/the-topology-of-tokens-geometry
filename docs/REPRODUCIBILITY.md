# Reproducibility

## Environment used for the reported runs

- Host: Linux, 28 CPU cores, 29 GB RAM, NVIDIA RTX 3090 (24 GB).
- Python 3.12.13, CUDA 13.0, PyTorch 2.13.0+cu130.
- Key packages: datasets 5.0.0, tokenizers 0.22.2, transformers 5.15.1,
  numpy 2.2.6 (host) / 2.5.4 (venv), scipy 1.18.1, scikit-learn 1.9.1,
  pandas 2.2.3, matplotlib 3.11.2, statsmodels 0.15.0, ripser 0.6.15,
  gudhi 3.13.0, persim 0.4.1.
- Exact direct-package pins: `requirements.lock`. Full resolved environment and
  GPU/CUDA versions are written to `artifacts/manifest.json` by the `manifest`
  command.

## One-command reproduction

```bash
make install          # create .venv and install pinned dependencies
make test             # unit tests, including byte/BPB and TDA checks
bash scripts/reproduce.sh configs/default.yaml
```

The pipeline stages are also available individually:

```bash
.venv/bin/topobpe download      --config configs/default.yaml
.venv/bin/topobpe tokenizers    --config configs/default.yaml
.venv/bin/topobpe features      --config configs/default.yaml
.venv/bin/topobpe lm            --config configs/default.yaml
.venv/bin/topobpe market-data   --config configs/default.yaml
.venv/bin/topobpe market-features --config configs/default.yaml
.venv/bin/topobpe market-lm     --config configs/default.yaml
.venv/bin/topobpe analysis      --config configs/default.yaml
.venv/bin/topobpe figures       --config configs/default.yaml
.venv/bin/topobpe manifest      --config configs/default.yaml
```

Every stage caches outputs keyed by corpus, tokenizer family, vocabulary size,
seed, and a hash of the relevant config block. Use `--force` to recompute.

## Docker

```bash
docker compose config --quiet
docker compose run --rm topobpe-cpu --help
docker compose run --rm topobpe-cpu all --config configs/smoke.yaml --skip-market
docker compose --profile gpu run --rm topobpe-gpu lm --config configs/default.yaml
```

The CPU image installs CPU PyTorch. The GPU image installs the CUDA 13.0 wheel
and requests all host GPUs. Both mount the repository at `/app` so artifacts and
results are written back to the host.

## Seeds and determinism

- Global seed: 20261105, used for document hashing and feature sampling.
- BPB runs use seeds 13 and 37; CUDA deterministic algorithms and deterministic
  cuDNN are requested, but bf16 GPU reductions are not guaranteed bitwise
  reproducible across driver versions. Seed variance is reported.
- Tokenizer training, feature sampling, ridge alpha selection, bootstrap
  resampling, and permutation tests all use explicit seeds.
- No test-set checkpoint selection or test-set hyperparameter tuning is used.

## Docker verification (2026-11-05)

The CPU image was built and the container entrypoint was verified on the
reported host:

```bash
docker compose config --quiet
docker compose build topobpe-cpu      # image topobpe:cpu, 2.17 GB
docker compose run --rm topobpe-cpu --help
```

The GPU profile is defined (`docker compose --profile gpu ...`) but was not
built or exercised in this run; the reported BPB runs used the host virtual
environment with CUDA 13.0, as recorded in the execution table.

## Execution record

Run completed 2026-11-05 on the host above with `configs/default.yaml`.

| Stage | Result |
|---|---|
| Tokenizers | 36/36 fitted (~18 min wall) |
| Features | 36/36 conditions, 80 predictors each (~36 min wall) |
| Main BPB grid | 36/36 runs, 1500 steps, seeds 13 and 37, 0.69 GPU-hours total, median 70 s/run |
| Market data | 12/12 daily corpora from Nasdaq; 24 intraday windows unavailable and excluded |
| Market tokenizers / BPB | 24/24 tokenizers; 24/24 runs on CPU (~35 min) |
| TDA audit | 6 corpora × 2 clouds × 3 point counts; Ripser/GUDHI agreement ≤ 2.4e-7 |
| Byte audit | 36/36 conditions audited; 18/18 BPE and 12/18 Unigram byte-reversible |
| Analysis | 2,000 cluster-bootstrap and 2,000 permutation iterations per hypothesis (~9 min wall) |

The only substantive software defect found in the final audit was a
permutation-test column slice that used the row count instead of the column
count. It was fixed before the reported analysis, a regression test was added
(`tests/test_stats.py::test_permutation_uses_only_topology_columns`), and all
confirmatory numbers come from the fixed run.

Storage can be reduced by lowering `data.docs_per_corpus` and
`tokenizers.sample_bytes_for_training`, but those changes are outside the
pre-registered configuration and must be reported.

## Checks before publication

1. `make test` passes.
2. `artifacts/manifest.json` contains data, market, and tokenizer manifests with
   hashes and environment versions.
3. `results/analysis.json` reports all confirmatory hypotheses, including
   negative or null results.
4. Every figure in `results/figures/` has a corresponding table in
   `results/tables/`.
5. The market manifest states which provider supplied each series and which
   intraday windows were unavailable.
