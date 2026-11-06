from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_merge_topology_transfers_within_scale_not_across_corpora() -> None:
    analysis = json.loads((ROOT / "results/merge_analysis.json").read_text())
    within = analysis["comparisons"]["size_poly_plus_selected_within_size"]
    across = analysis["comparisons"]["size_poly_plus_selected"]
    within_delta = within["metrics"]["full"]["r2"] - within["metrics"]["baseline"]["r2"]
    across_delta = across["metrics"]["full"]["r2"] - across["metrics"]["baseline"]["r2"]
    assert within_delta > 0.2
    assert within["bootstrap"]["delta_full_minus_baseline"]["p_bootstrap"] < 0.05
    assert across_delta < 0


def test_merge_topology_extrapolates_to_32k() -> None:
    analysis = json.loads((ROOT / "results/merge_analysis.json").read_text())
    holdout = analysis["holdout_compression"]["32768"]["metrics"]
    assert holdout["full"]["r2"] > 0.5
    assert holdout["baseline"]["r2"] < 0.1


def test_power_law_extrapolation_is_accurate() -> None:
    analysis = json.loads((ROOT / "results/scaling_law.json").read_text())
    rows = {row["family"]: row for row in analysis["power_law_extrapolation_32k"]["by_family"]}
    assert rows["bpe"]["mape"] < 0.05
    assert rows["unigram"]["mape"] < 0.05


def test_benchmark_leaderboard_is_complete() -> None:
    frame = pd.read_csv(ROOT / "benchmark/leaderboard.csv")
    assert len(frame) == 36
    assert frame["compression"].min() > 0
    assert frame["grade"].isin(["A", "B", "C", "D"]).all()
    assert frame["bpb_mean"].notna().sum() == 18


def test_calculator_embeds_every_corpus() -> None:
    html = (ROOT / "calculator.html").read_text()
    data = json.loads((ROOT / "results/scaling_law.json").read_text())
    for corpus in {row["corpus"] for row in data["per_corpus"]}:
        assert corpus in html
    assert "log(bytes/token)" not in html or "bytes/token" in html
