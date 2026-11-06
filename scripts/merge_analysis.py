from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from topobpe.config import load_config, results_dir
from topobpe.features import BASELINE_COLUMNS
from topobpe.stats import (
    _cluster_bootstrap,
    _design,
    _holm,
    _partial_spearman_inference,
    _permutation_test,
    _regression_metrics,
    _select_alpha,
    loco_predictions,
)
from topobpe.utils import utc_now, write_json

SELECTED_MERGE = [
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


def _sets(frame: pd.DataFrame) -> dict[str, list[str]]:
    baseline = [column for column in BASELINE_COLUMNS if column in frame.columns]
    lean = [
        column
        for column in baseline
        if column not in {"train_compression", "mean_token_bytes"}
    ]
    size = ["actual_vocab_size"]
    size_poly = ["log_vocab", "log_vocab_sq", "log_vocab_cube"]
    merge = [
        column
        for column in frame.columns
        if column.startswith("merge_")
        and np.std(frame[column].to_numpy(dtype=np.float64)) > 1e-12
    ]
    return {
        "baseline": baseline,
        "lean": lean,
        "size": size,
        "size_poly": size_poly,
        "merge": merge,
    }


def _comparison(
    frame: pd.DataFrame,
    target: str,
    baseline_columns: list[str],
    merge_columns: list[str],
    alphas: list[float],
    bootstrap_iterations: int,
    permutation_iterations: int,
    seed: int,
    group_column: str = "corpus",
) -> dict:
    sets = {
        "baseline": baseline_columns,
        "topology": merge_columns,
        "full": baseline_columns + merge_columns,
    }
    matrices, y, groups, data = _design(frame, target, sets)
    if group_column != "corpus":
        groups = data[group_column].astype(str).to_numpy()
    predictions: dict[str, np.ndarray] = {}
    metrics: dict[str, dict] = {}
    for name in ("baseline", "full", "topology"):
        prediction, _ = loco_predictions(matrices[name], y, groups, alphas)
        predictions[name] = prediction
        metrics[name] = _regression_metrics(y, prediction)
    bootstrap = _cluster_bootstrap(
        y, predictions, groups, bootstrap_iterations, seed + 100
    )
    permutation = _permutation_test(
        matrices["baseline"],
        matrices["topology"],
        y,
        groups,
        [10.0],
        permutation_iterations,
        seed + 200,
    )
    return {
        "target": target,
        "n": int(len(y)),
        "grouping": group_column,
        "metrics": metrics,
        "bootstrap": bootstrap,
        "permutation": permutation,
    }


def _summary_row(name: str, result: dict) -> dict:
    metrics = result["metrics"]
    delta = result["bootstrap"]["delta_full_minus_baseline"]
    delta_r2 = metrics["full"]["r2"] - metrics["baseline"]["r2"]
    permutation_p = result["permutation"]["p_value"]
    if delta_r2 <= 0 or not np.isfinite(permutation_p):
        permutation_p = 1.0
    return {
        "comparison": name,
        "grouping": result.get("grouping", "corpus"),
        "target": result["target"],
        "n": result["n"],
        "baseline_r2": metrics["baseline"]["r2"],
        "full_r2": metrics["full"]["r2"],
        "topology_r2": metrics.get("topology", {}).get("r2", float("nan")),
        "delta_r2": delta_r2,
        "delta_bootstrap_mean": delta["mean"],
        "delta_ci_low": delta["ci_low"],
        "delta_ci_high": delta["ci_high"],
        "permutation_p": permutation_p,
        "bootstrap_p": delta["p_bootstrap"],
    }


def _holdout_size_comparison(
    frame: pd.DataFrame,
    target: str,
    baseline_columns: list[str],
    merge_columns: list[str],
    alphas: list[float],
    holdout_size: int,
) -> dict:
    sets = {
        "baseline": baseline_columns,
        "full": baseline_columns + merge_columns,
    }
    matrices, y, _, data = _design(frame, target, sets)
    sizes = data["requested_vocab_size"].to_numpy()
    test = sizes == holdout_size
    train = ~test
    metrics: dict[str, dict] = {}
    for name in ("baseline", "full"):
        x = matrices[name]
        alpha = _select_alpha(
            x[train], y[train], data["corpus"].to_numpy()[train], alphas
        )
        scaler = StandardScaler().fit(x[train])
        model = Ridge(alpha=alpha).fit(scaler.transform(x[train]), y[train])
        prediction = model.predict(scaler.transform(x[test]))
        metrics[name] = _regression_metrics(y[test], prediction)
    metrics["delta_r2"] = {"value": metrics["full"]["r2"] - metrics["baseline"]["r2"]}
    return {
        "target": target,
        "holdout_size": int(holdout_size),
        "n_train": int(train.sum()),
        "n_test": int(test.sum()),
        "metrics": metrics,
    }


def _pca_comparison(
    frame: pd.DataFrame,
    target: str,
    baseline_columns: list[str],
    merge_columns: list[str],
    bootstrap_iterations: int,
    seed: int,
) -> dict:
    sets = {
        "baseline": baseline_columns,
        "topology": merge_columns,
        "full": baseline_columns + merge_columns,
    }
    matrices, y, groups, _ = _design(frame, target, sets)
    predictions: dict[str, np.ndarray] = {}
    for name in ("baseline", "full"):
        x = matrices[name]
        prediction = np.zeros_like(y)
        for group in np.unique(groups):
            train = groups != group
            test = groups == group
            scaler = StandardScaler().fit(x[train])
            train_x = scaler.transform(x[train])
            test_x = scaler.transform(x[test])
            if name == "full":
                components = min(3, train_x.shape[0] - 1, train_x.shape[1])
                pca = PCA(n_components=components).fit(train_x)
                train_x = pca.transform(train_x)
                test_x = pca.transform(test_x)
            model = Ridge(alpha=10.0).fit(train_x, y[train])
            prediction[test] = model.predict(test_x)
        predictions[name] = prediction
    metrics = {
        name: _regression_metrics(y, prediction)
        for name, prediction in predictions.items()
    }
    bootstrap = _cluster_bootstrap(
        y, predictions, groups, bootstrap_iterations, seed
    )
    return {
        "target": target,
        "n": int(len(y)),
        "grouping": "corpus",
        "metrics": metrics,
        "bootstrap": bootstrap,
        "permutation": {"p_value": float("nan")},
    }


def _merge_only(
    frame: pd.DataFrame,
    target: str,
    columns: list[str],
    alphas: list[float],
) -> dict:
    matrices, y, groups, _ = _design(frame, target, {"topology": columns})
    prediction, selected = loco_predictions(matrices["topology"], y, groups, alphas)
    return {
        "target": target,
        "n": int(len(y)),
        "r2": _regression_metrics(y, prediction)["r2"],
        "selected_alphas": selected,
    }


def _importance(
    frame: pd.DataFrame,
    target: str,
    columns: list[str],
) -> list[dict]:
    matrices, y, _, _ = _design(frame, target, {"full": columns})
    x = matrices["full"]
    scaler = StandardScaler().fit(x)
    model = Ridge(alpha=10.0).fit(scaler.transform(x), y)
    rows = [
        {"feature": column, "coefficient": float(coefficient)}
        for column, coefficient in zip(columns, model.coef_, strict=True)
    ]
    rows.sort(key=lambda row: abs(row["coefficient"]), reverse=True)
    return rows


def _partial_rows(
    frame: pd.DataFrame,
    target: str,
    control: str,
    variables: list[str],
    iterations: int,
    seed: int,
    top: int = 20,
) -> list[dict]:
    data = frame.dropna(subset=[target, control, *variables]).copy()
    values = data[target].to_numpy(dtype=np.float64)
    scored = []
    for variable in variables:
        feature_values = data[variable].to_numpy(dtype=np.float64)
        if np.std(feature_values) < 1e-12:
            continue
        rho, _ = spearmanr(feature_values, values)
        if not np.isfinite(rho):
            continue
        scored.append((abs(float(rho)), variable))
    scored.sort(reverse=True)
    selected = [variable for _, variable in scored[:top]]
    groups = data["corpus"].astype(str).to_numpy()
    rows = []
    for variable in selected:
        inference = _partial_spearman_inference(
            data[variable].to_numpy(dtype=np.float64),
            values,
            data[control].to_numpy(dtype=np.float64),
            groups,
            iterations,
            seed,
        )
        inference["variable"] = variable
        inference["control"] = control
        inference["target"] = target
        rows.append(inference)
    if rows:
        adjusted = _holm([row["p_value"] for row in rows])
        for row, value in zip(rows, adjusted, strict=True):
            row["holm_p"] = float(value)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze merge-topology features")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--bootstrap-iterations", type=int, default=500)
    parser.add_argument("--permutation-iterations", type=int, default=300)
    parser.add_argument("--partial-iterations", type=int, default=300)
    args = parser.parse_args()
    config = load_config(args.config)
    alphas = [float(value) for value in config["analysis"]["ridge_alphas"]]
    seed = int(config["project"]["seed"])
    out_dir = results_dir(config)
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    merge_all = pd.read_csv(out_dir / "merge_features.csv")
    compression = merge_all[merge_all["variant"].isin(["truncated", "base"])].copy()
    compression["log_vocab"] = np.log(compression["actual_vocab_size"].astype(np.float64))
    compression["log_vocab_sq"] = compression["log_vocab"] ** 2
    compression["log_vocab_cube"] = compression["log_vocab"] ** 3
    sets = _sets(compression)
    print(
        f"compression rows={len(compression)} corpora={compression['corpus'].nunique()} "
        f"merge_features={len(sets['merge'])}",
        flush=True,
    )
    analysis: dict = {
        "created_at": utc_now(),
        "rows": int(len(compression)),
        "merge_features": len(sets["merge"]),
        "comparisons": {},
    }
    summary_rows: list[dict] = []
    selected = [column for column in SELECTED_MERGE if column in compression.columns]
    analysis["selected_merge_features"] = selected
    comparisons = [
        ("baseline8_plus_merge", "test_compression", sets["baseline"], sets["merge"], "corpus"),
        ("lean_plus_merge", "test_compression", sets["lean"], sets["merge"], "corpus"),
        ("size_plus_merge", "test_compression", sets["size"], sets["merge"], "corpus"),
        ("baseline8_plus_selected", "test_compression", sets["baseline"], selected, "corpus"),
        ("lean_plus_selected", "test_compression", sets["lean"], selected, "corpus"),
        ("size_plus_selected", "test_compression", sets["size"], selected, "corpus"),
        (
            "size_plus_selected_within_size",
            "test_compression",
            sets["size"],
            selected,
            "requested_vocab_size",
        ),
        (
            "lean_plus_selected_within_size",
            "test_compression",
            sets["lean"],
            selected,
            "requested_vocab_size",
        ),
        (
            "size_poly_plus_selected",
            "test_compression",
            sets["size_poly"],
            selected,
            "corpus",
        ),
        (
            "size_poly_plus_selected_within_size",
            "test_compression",
            sets["size_poly"],
            selected,
            "requested_vocab_size",
        ),
        (
            "size_poly_plus_merge_within_size",
            "test_compression",
            sets["size_poly"],
            sets["merge"],
            "requested_vocab_size",
        ),
    ]
    for name, target, baseline_columns, merge_columns, grouping in comparisons:
        print(f"comparison {name}", flush=True)
        result = _comparison(
            compression,
            target,
            baseline_columns,
            merge_columns,
            alphas,
            args.bootstrap_iterations,
            args.permutation_iterations,
            seed,
            group_column=grouping,
        )
        analysis["comparisons"][name] = result
        summary_rows.append(_summary_row(name, result))
    print("comparison pca3_selected", flush=True)
    pca_result = _pca_comparison(
        compression,
        "test_compression",
        sets["size"],
        selected,
        args.bootstrap_iterations,
        seed + 4,
    )
    analysis["comparisons"]["pca3_selected"] = pca_result
    summary_rows.append(_summary_row("pca3_selected", pca_result))
    analysis["holdout_compression"] = {}
    for holdout_size in (32768, 512):
        print(f"holdout size {holdout_size}", flush=True)
        result = _holdout_size_comparison(
            compression,
            "test_compression",
            sets["size_poly"],
            selected,
            alphas,
            holdout_size,
        )
        analysis["holdout_compression"][str(holdout_size)] = result
        summary_rows.append(
            {
                "comparison": f"size_poly_plus_selected_holdout_{holdout_size}",
                "grouping": f"holdout_{holdout_size}",
                "target": result["target"],
                "n": result["n_test"],
                "baseline_r2": result["metrics"]["baseline"]["r2"],
                "full_r2": result["metrics"]["full"]["r2"],
                "topology_r2": float("nan"),
                "delta_r2": result["metrics"]["delta_r2"]["value"],
                "delta_bootstrap_mean": float("nan"),
                "delta_ci_low": float("nan"),
                "delta_ci_high": float("nan"),
                "permutation_p": float("nan"),
                "bootstrap_p": float("nan"),
            }
        )
    analysis["merge_only_compression"] = _merge_only(
        compression, "test_compression", sets["merge"], alphas
    )
    analysis["importance_compression"] = _importance(
        compression, "test_compression", sets["lean"] + sets["merge"]
    )[:15]
    partial_compression = _partial_rows(
        compression,
        "test_compression",
        "log_vocab",
        sets["merge"],
        args.partial_iterations,
        seed + 1,
    )
    partial_compression.sort(key=lambda row: abs(row["rho"]), reverse=True)
    analysis["partial_compression"] = partial_compression[:15]
    pd.DataFrame(partial_compression).to_csv(
        tables / "merge_partial_compression.csv", index=False
    )

    bpb_summary = pd.read_csv(tables / "bpb_summary.csv")
    bpb_merge = bpb_summary[bpb_summary["family"] == "bpe"].merge(
        merge_all[merge_all["variant"].isin(["native", "base"])],
        on=["corpus", "family", "requested_vocab_size"],
        how="inner",
    )
    bpb_merge["log_vocab"] = np.log(bpb_merge["actual_vocab_size"].astype(np.float64))
    bpb_merge["log_vocab_sq"] = bpb_merge["log_vocab"] ** 2
    bpb_merge["log_vocab_cube"] = bpb_merge["log_vocab"] ** 3
    print(f"bpb rows={len(bpb_merge)}", flush=True)
    analysis["bpb_rows"] = int(len(bpb_merge))
    bpb_rows: list[dict] = []
    if len(bpb_merge) >= 6:
        for name, baseline_columns in (
            ("bpb_baseline8_plus_merge", sets["baseline"]),
            ("bpb_lean_plus_merge", sets["lean"]),
            ("bpb_size_plus_merge", sets["size"]),
            ("bpb_size_plus_selected", sets["size"]),
            ("bpb_lean_plus_selected", sets["lean"]),
        ):
            merge_columns = selected if name.endswith("selected") else sets["merge"]
            result = _comparison(
                bpb_merge,
                "bpb_mean",
                baseline_columns,
                merge_columns,
                alphas,
                min(args.bootstrap_iterations, 300),
                min(args.permutation_iterations, 200),
                seed + 2,
            )
            analysis["comparisons"][name] = result
            bpb_rows.append(_summary_row(name, result))
        for name, baseline_columns, merge_columns, grouping in (
            (
                "bpb_size_plus_selected_within_size",
                sets["size"],
                selected,
                "requested_vocab_size",
            ),
            (
                "bpb_size_poly_plus_selected_within_size",
                sets["size_poly"],
                selected,
                "requested_vocab_size",
            ),
        ):
            result = _comparison(
                bpb_merge,
                "bpb_mean",
                baseline_columns,
                merge_columns,
                alphas,
                min(args.bootstrap_iterations, 300),
                min(args.permutation_iterations, 150),
                seed + 5,
                group_column=grouping,
            )
            analysis["comparisons"][name] = result
            bpb_rows.append(_summary_row(name, result))
        analysis["merge_only_bpb"] = _merge_only(
            bpb_merge, "bpb_mean", sets["merge"], alphas
        )
        analysis["importance_bpb"] = _importance(
            bpb_merge, "bpb_mean", sets["lean"] + sets["merge"]
        )[:15]
        partial_bpb = _partial_rows(
            bpb_merge,
            "bpb_mean",
            "test_compression",
            sets["merge"],
            args.partial_iterations,
            seed + 3,
            top=15,
        )
        partial_bpb.sort(key=lambda row: abs(row["rho"]), reverse=True)
        analysis["partial_bpb"] = partial_bpb[:10]
        pd.DataFrame(partial_bpb).to_csv(tables / "merge_partial_bpb.csv", index=False)
    summary_rows.extend(bpb_rows)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(tables / "merge_loco_summary.csv", index=False)
    write_json(out_dir / "merge_analysis.json", analysis)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
