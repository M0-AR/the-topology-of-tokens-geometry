from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from topobpe.config import load_config, results_dir
from topobpe.stats import (
    _design,
    _regression_metrics,
    _select_alpha,
    loco_predictions,
)
from topobpe.utils import utc_now, write_json

RAW_SELECTED = [
    "merge_tree_mean_balance",
    "merge_tree_mean_strahler",
    "merge_gain_gini",
    "merge_gain_entropy",
    "merge_gain_top10_share",
    "merge_graph_h1_total_persistence",
    "merge_graph_h1_persistence_entropy",
    "merge_graph_cycle_rank",
    "merge_graph_mean_path",
    "merge_graph_mean_nn_distance",
]

NORM_SELECTED = [
    "merge_tree_mean_balance",
    "merge_tree_mean_strahler_per_log_merges",
    "merge_gain_gini",
    "merge_gain_entropy",
    "merge_gain_top10_share",
    "merge_graph_h1_total_per_nn",
    "merge_graph_h1_entropy_norm",
    "merge_graph_cycle_density",
    "merge_graph_mean_path",
]

ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]


def add_normalized(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    nn = frame["merge_graph_mean_nn_distance"].clip(lower=1e-12)
    diameter = frame["merge_graph_diameter"].clip(lower=1e-12)
    edges = frame["merge_graph_num_edges"].clip(lower=1)
    bars = frame["merge_graph_h1_num_bars"].clip(lower=2)
    merges = frame["merge_num_merges"].clip(lower=2)
    frame["merge_graph_h1_total_per_nn"] = (
        frame["merge_graph_h1_total_persistence"] / nn
    )
    frame["merge_graph_h1_total_per_diameter"] = (
        frame["merge_graph_h1_total_persistence"] / diameter
    )
    frame["merge_graph_h1_total_per_edge"] = (
        frame["merge_graph_h1_total_persistence"] / edges
    )
    frame["merge_graph_h1_mean_per_nn"] = (
        frame["merge_graph_h1_mean_persistence"] / nn
    )
    frame["merge_graph_h1_entropy_norm"] = (
        frame["merge_graph_h1_persistence_entropy"] / np.log(bars)
    )
    frame["merge_graph_h1_betti_area_per_nn2"] = (
        frame["merge_graph_h1_betti_area"] / (nn**2)
    )
    frame["merge_graph_cycle_density"] = (
        frame["merge_graph_cycle_rank"] / edges
    )
    frame["merge_tree_mean_depth_per_log_merges"] = (
        frame["merge_tree_mean_depth"] / np.log(merges)
    )
    frame["merge_tree_mean_strahler_per_log_merges"] = (
        frame["merge_tree_mean_strahler"] / np.log(merges)
    )
    return frame


def drift_table(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for corpus, group in frame.groupby("corpus", sort=True):
        ordered = group.sort_values("requested_vocab_size")
        for column in columns:
            values = ordered[column].to_numpy(dtype=np.float64)
            median = float(np.nanmedian(values))
            if len(values) < 3 or not np.isfinite(median) or abs(median) < 1e-12:
                continue
            rows.append(
                {
                    "corpus": corpus,
                    "feature": column,
                    "median": median,
                    "cv": float(np.nanstd(values) / abs(median)),
                    "range_fraction": float(
                        (np.nanmax(values) - np.nanmin(values)) / abs(median)
                    ),
                }
            )
    return pd.DataFrame(rows)


def compare(
    frame: pd.DataFrame,
    baseline_columns: list[str],
    merge_columns: list[str],
    group_column: str,
) -> dict:
    sets = {
        "baseline": baseline_columns,
        "topology": merge_columns,
        "full": baseline_columns + merge_columns,
    }
    matrices, y, groups, data = _design(frame, "test_compression", sets)
    if group_column != "corpus":
        groups = data[group_column].astype(str).to_numpy()
    metrics: dict[str, dict] = {}
    for name in ("baseline", "topology", "full"):
        prediction, selected = loco_predictions(
            matrices[name], y, groups, ALPHAS
        )
        metrics[name] = _regression_metrics(y, prediction)
        metrics[name]["selected_alphas"] = selected
    return {
        "grouping": group_column,
        "n": int(len(y)),
        "metrics": metrics,
        "delta_r2": metrics["full"]["r2"] - metrics["baseline"]["r2"],
    }


def holdout(
    frame: pd.DataFrame,
    baseline_columns: list[str],
    merge_columns: list[str],
    holdout_size: int,
) -> dict:
    sets = {
        "baseline": baseline_columns,
        "full": baseline_columns + merge_columns,
    }
    matrices, y, _, data = _design(frame, "test_compression", sets)
    test = data["requested_vocab_size"].to_numpy() == holdout_size
    train = ~test
    metrics: dict[str, dict] = {}
    for name in ("baseline", "full"):
        x = matrices[name]
        alpha = _select_alpha(
            x[train], y[train], data["corpus"].to_numpy()[train], ALPHAS
        )
        scaler = StandardScaler().fit(x[train])
        model = Ridge(alpha=alpha).fit(scaler.transform(x[train]), y[train])
        prediction = model.predict(scaler.transform(x[test]))
        metrics[name] = _regression_metrics(y[test], prediction)
        metrics[name]["alpha"] = alpha
    return {
        "holdout_size": int(holdout_size),
        "n_train": int(train.sum()),
        "n_test": int(test.sum()),
        "metrics": metrics,
        "delta_r2": metrics["full"]["r2"] - metrics["baseline"]["r2"],
    }


def stability_table(path: Path) -> dict:
    data = json.loads(path.read_text())
    rows: list[dict] = []
    for record in data["records"]:
        for cloud, key in (("vocab", "vocab_rows"), ("cooc", "cooc_rows")):
            values = pd.DataFrame(record[key]).sort_values("points")
            total = values["h1_total_persistence"].to_numpy()
            bars = values["h1_num_bars"].clip(lower=2).to_numpy()
            entropy = values["h1_persistence_entropy"].to_numpy()
            mean = total / bars
            norm_entropy = entropy / np.log(bars)
            for name, series in (
                ("raw_total_persistence", total),
                ("total_per_bar", mean),
                ("persistence_entropy", entropy),
                ("entropy_per_log_bar", norm_entropy),
            ):
                median = float(np.nanmedian(series))
                rows.append(
                    {
                        "corpus": record["corpus"],
                        "cloud": cloud,
                        "feature": name,
                        "cv": float(np.nanstd(series) / abs(median))
                        if abs(median) > 1e-12
                        else float("nan"),
                    }
                )
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby("feature")["cv"]
        .agg(["median", "max", "count"])
        .reset_index()
        .to_dict(orient="records")
    )
    return {"records": len(data["records"]), "summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description="Scale-normalized TDA checks")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    out = results_dir(config)
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    merge = pd.read_csv(out / "merge_features.csv")
    frame = merge[merge["variant"].isin(["truncated", "base"])].copy()
    frame["log_vocab"] = np.log(frame["actual_vocab_size"].astype(np.float64))
    frame = add_normalized(frame)
    raw_drift = drift_table(frame, RAW_SELECTED)
    norm_drift = drift_table(frame, NORM_SELECTED)
    raw_drift["kind"] = "raw"
    norm_drift["kind"] = "normalized"
    drift = pd.concat([raw_drift, norm_drift], ignore_index=True)
    drift.to_csv(tables / "scale_normalized_drift.csv", index=False)
    drift_summary = (
        drift.groupby(["kind", "feature"])["cv"]
        .median()
        .reset_index()
        .sort_values(["kind", "cv"])
        .to_dict(orient="records")
    )
    size_poly = ["log_vocab", "log_vocab_sq", "log_vocab_cube"]
    frame["log_vocab_sq"] = frame["log_vocab"] ** 2
    frame["log_vocab_cube"] = frame["log_vocab"] ** 3
    analysis: dict = {
        "created_at": utc_now(),
        "rows": int(len(frame)),
        "raw_selected": RAW_SELECTED,
        "normalized_selected": NORM_SELECTED,
        "drift_summary": drift_summary,
        "comparisons": {},
        "holdouts": {},
    }
    for name, columns in (("raw", RAW_SELECTED), ("normalized", NORM_SELECTED)):
        analysis["comparisons"][f"within_size_{name}"] = compare(
            frame, size_poly, columns, "requested_vocab_size"
        )
        analysis["comparisons"][f"across_corpus_{name}"] = compare(
            frame, size_poly, columns, "corpus"
        )
        for holdout_size in (32768, 512):
            analysis["holdouts"][f"{name}_{holdout_size}"] = holdout(
                frame, size_poly, columns, holdout_size
            )
    analysis["tda_stability"] = stability_table(out / "tda_stability.json")
    write_json(out / "scale_normalized_tda.json", analysis)
    print("drift median CV (lower is more stable):")
    for row in sorted(drift_summary, key=lambda item: item["cv"]):
        print(f"  {row['kind']:10s} {row['feature']:48s} {row['cv']:.4f}")
    print("\nwithin-size comparisons:")
    for name in ("within_size_raw", "within_size_normalized"):
        result = analysis["comparisons"][name]
        metrics = result["metrics"]
        print(
            f"  {name:24s} baseline={metrics['baseline']['r2']:.3f} "
            f"topology={metrics['topology']['r2']:.3f} "
            f"full={metrics['full']['r2']:.3f} delta={result['delta_r2']:+.3f}"
        )
    print("\nholdouts:")
    for name, result in analysis["holdouts"].items():
        print(
            f"  {name:18s} size={result['holdout_size']:5d} "
            f"baseline={result['metrics']['baseline']['r2']:.3f} "
            f"full={result['metrics']['full']['r2']:.3f} "
            f"delta={result['delta_r2']:+.3f}"
        )
    print("\ntda stability median CV:")
    for row in analysis["tda_stability"]["summary"]:
        print(f"  {row['feature']:24s} {row['median']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
