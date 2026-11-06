from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from .config import results_dir
from .features import BASELINE_COLUMNS, TOPOLOGY_EXTRA, TOPOLOGY_PREFIXES
from .utils import utc_now, write_json


def _r2(y: np.ndarray, prediction: np.ndarray) -> float:
    if len(y) < 2 or float(np.std(y)) < 1e-12:
        return float("nan")
    return float(r2_score(y, prediction))


def _regression_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "r2": _r2(y, prediction),
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(math.sqrt(mean_squared_error(y, prediction))),
        "n": float(len(y)),
    }


def _feature_sets(frame: pd.DataFrame) -> dict[str, list[str]]:
    baseline = [column for column in BASELINE_COLUMNS if column in frame.columns]
    topology = [
        column
        for column in frame.columns
        if column.startswith(TOPOLOGY_PREFIXES) or column in TOPOLOGY_EXTRA
    ]
    if not baseline:
        raise ValueError("no baseline columns found")
    if not topology:
        raise ValueError("no topology columns found")
    return {
        "baseline": baseline,
        "topology": topology,
        "full": baseline + topology,
    }


def _design(
    frame: pd.DataFrame,
    target: str,
    feature_sets: dict[str, list[str]],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, pd.DataFrame]:
    all_columns = sorted({column for columns in feature_sets.values() for column in columns})
    data = frame.dropna(subset=[target, *all_columns]).copy()
    for column in all_columns:
        data = data[np.isfinite(data[column].to_numpy(dtype=np.float64))]
    if len(data) < 3:
        raise ValueError(f"not enough complete rows for target {target}: {len(data)}")
    y = data[target].to_numpy(dtype=np.float64)
    groups = data["corpus"].astype(str).to_numpy()
    matrices = {
        name: data[columns].to_numpy(dtype=np.float64)
        for name, columns in feature_sets.items()
    }
    return matrices, y, groups, data


def _select_alpha(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    alphas: list[float],
) -> float:
    unique = np.unique(groups)
    if len(unique) < 2:
        return float(alphas[len(alphas) // 2])
    best_alpha = float(alphas[0])
    best_score = float("inf")
    for alpha in alphas:
        prediction = np.zeros_like(y)
        for group in unique:
            train = groups != group
            test = groups == group
            scaler = StandardScaler().fit(x[train])
            model = Ridge(alpha=float(alpha)).fit(scaler.transform(x[train]), y[train])
            prediction[test] = model.predict(scaler.transform(x[test]))
        score = float(np.mean((prediction - y) ** 2))
        if score < best_score:
            best_score = score
            best_alpha = float(alpha)
    return best_alpha


def loco_predictions(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    alphas: list[float],
) -> tuple[np.ndarray, dict[str, float]]:
    unique = np.unique(groups)
    prediction = np.zeros_like(y)
    selected: dict[str, float] = {}
    for group in unique:
        train = groups != group
        test = groups == group
        alpha = _select_alpha(x[train], y[train], groups[train], alphas)
        selected[str(group)] = alpha
        scaler = StandardScaler().fit(x[train])
        model = Ridge(alpha=alpha).fit(scaler.transform(x[train]), y[train])
        prediction[test] = model.predict(scaler.transform(x[test]))
    return prediction, selected


def _cluster_indices(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    unique = np.unique(groups)
    sampled = rng.choice(unique, size=len(unique), replace=True)
    pieces = [np.flatnonzero(groups == group) for group in sampled if np.any(groups == group)]
    if not pieces:
        return np.arange(len(groups))
    return np.concatenate(pieces)


def _cluster_bootstrap(
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    groups: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    model_scores: dict[str, list[float]] = {name: [] for name in predictions}
    for _ in range(iterations):
        indices = _cluster_indices(groups, rng)
        if len(np.unique(y[indices])) < 2:
            continue
        for name, prediction in predictions.items():
            model_scores[name].append(_r2(y[indices], prediction[indices]))
        deltas.append(
            _r2(y[indices], predictions["full"][indices])
            - _r2(y[indices], predictions["baseline"][indices])
        )
    result: dict[str, Any] = {}
    for name, values in model_scores.items():
        arr = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
        result[name] = {
            "mean": float(arr.mean()) if len(arr) else float("nan"),
            "ci_low": float(np.percentile(arr, 2.5)) if len(arr) else float("nan"),
            "ci_high": float(np.percentile(arr, 97.5)) if len(arr) else float("nan"),
            "iterations": int(len(arr)),
        }
    delta_arr = np.asarray([value for value in deltas if np.isfinite(value)], dtype=np.float64)
    result["delta_full_minus_baseline"] = {
        "mean": float(delta_arr.mean()) if len(delta_arr) else float("nan"),
        "ci_low": float(np.percentile(delta_arr, 2.5)) if len(delta_arr) else float("nan"),
        "ci_high": float(np.percentile(delta_arr, 97.5)) if len(delta_arr) else float("nan"),
        "p_bootstrap": float(
            (1 + np.sum(delta_arr <= 0)) / (1 + len(delta_arr))
        )
        if len(delta_arr)
        else float("nan"),
        "iterations": int(len(delta_arr)),
    }
    return result


def _permutation_test(
    base: np.ndarray,
    topology: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    alphas: list[float],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    baseline_prediction, baseline_alphas = loco_predictions(base, y, groups, alphas)
    full_prediction, full_alphas = loco_predictions(
        np.hstack([base, topology]), y, groups, alphas
    )
    observed = _r2(y, full_prediction) - _r2(y, baseline_prediction)
    rng = np.random.default_rng(seed)
    null: list[float] = []
    for _ in range(iterations):
        permutation = rng.permutation(len(y))
        permuted = np.hstack([base, topology[permutation]])
        prediction, _ = loco_predictions(permuted, y, groups, alphas)
        null.append(_r2(y, prediction) - _r2(y, baseline_prediction))
    null_arr = np.asarray([value for value in null if np.isfinite(value)], dtype=np.float64)
    p_value = float((1 + np.sum(null_arr >= observed)) / (1 + len(null_arr)))
    return {
        "observed_delta_r2": float(observed),
        "p_value": p_value,
        "null_mean": float(null_arr.mean()) if len(null_arr) else float("nan"),
        "null_95": float(np.percentile(null_arr, 95)) if len(null_arr) else float("nan"),
        "iterations": int(len(null_arr)),
        "baseline_alphas": baseline_alphas,
        "full_alphas": full_alphas,
    }


def _partial_spearman(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> float:
    if min(np.std(x), np.std(y), np.std(control)) < 1e-12:
        return float("nan")
    rx = rankdata(x)
    ry = rankdata(y)
    rc = rankdata(control)

    def residual(values: np.ndarray, covariate: np.ndarray) -> np.ndarray:
        design = np.column_stack([np.ones_like(covariate), covariate])
        coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
        return values - design @ coefficients

    x_residual = residual(rx, rc)
    y_residual = residual(ry, rc)
    if np.std(x_residual) < 1e-12 or np.std(y_residual) < 1e-12:
        return float("nan")
    correlation, _ = pearsonr(x_residual, y_residual)
    return float(correlation)


def _partial_spearman_inference(
    x: np.ndarray,
    y: np.ndarray,
    control: np.ndarray,
    groups: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    observed = _partial_spearman(x, y, control)
    rng = np.random.default_rng(seed)
    null: list[float] = []
    for _ in range(iterations):
        null.append(_partial_spearman(rng.permutation(x), y, control))
    null_arr = np.asarray([value for value in null if np.isfinite(value)], dtype=np.float64)
    p_value = float(
        (1 + np.sum(np.abs(null_arr) >= abs(observed))) / (1 + len(null_arr))
    )
    rng = np.random.default_rng(seed + 1)
    bootstrap: list[float] = []
    for _ in range(iterations):
        indices = _cluster_indices(groups, rng)
        bootstrap.append(
            _partial_spearman(x[indices], y[indices], control[indices])
        )
    boot_arr = np.asarray([value for value in bootstrap if np.isfinite(value)], dtype=np.float64)
    return {
        "rho": observed,
        "p_value": p_value,
        "ci_low": float(np.percentile(boot_arr, 2.5)) if len(boot_arr) else float("nan"),
        "ci_high": float(np.percentile(boot_arr, 97.5)) if len(boot_arr) else float("nan"),
        "iterations": int(len(boot_arr)),
    }


def _holm(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(values) - rank) * values[index])
        value = max(value, running)
        adjusted[index] = value
        running = value
    return [float(value) for value in adjusted]


def _model_comparison(
    matrices: dict[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    alphas: list[float],
    bootstrap_iterations: int,
    permutation_iterations: int,
    seed: int,
) -> dict[str, Any]:
    predictions: dict[str, np.ndarray] = {}
    selected: dict[str, dict[str, float]] = {}
    metrics: dict[str, Any] = {}
    for name, matrix in matrices.items():
        prediction, chosen = loco_predictions(matrix, y, groups, alphas)
        predictions[name] = prediction
        selected[name] = chosen
        metrics[name] = _regression_metrics(y, prediction)
    bootstrap = _cluster_bootstrap(
        y, predictions, groups, bootstrap_iterations, seed + 100
    )
    permutation = _permutation_test(
        matrices["baseline"],
        matrices["full"][:, matrices["baseline"].shape[1] :],
        y,
        groups,
        alphas,
        permutation_iterations,
        seed + 200,
    )
    per_group = []
    for group in np.unique(groups):
        mask = groups == group
        per_group.append(
            {
                "group": str(group),
                "n": int(mask.sum()),
                "baseline": _regression_metrics(y[mask], predictions["baseline"][mask]),
                "full": _regression_metrics(y[mask], predictions["full"][mask]),
            }
        )
    return {
        "metrics": metrics,
        "selected_alphas": selected,
        "bootstrap": bootstrap,
        "permutation": permutation,
        "per_group": per_group,
        "predictions": predictions,
    }


def _predictions_frame(
    data: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    frame = data[["corpus", "family", "requested_vocab_size"]].copy()
    frame["target"] = y
    frame["group"] = groups
    for name, prediction in predictions.items():
        frame[f"prediction_{name}"] = prediction
    return frame


def _transfer(
    public_frame: pd.DataFrame,
    market_frame: pd.DataFrame,
    public_target: str,
    market_target: str,
    feature_sets: dict[str, list[str]],
    alphas: list[float],
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, Any]:
    public_all = sorted({column for columns in feature_sets.values() for column in columns})
    market_all = [column for column in public_all if column in market_frame.columns]
    missing = [column for column in public_all if column not in market_frame.columns]
    if missing:
        return {"status": "skipped", "reason": f"missing market features: {missing}"}
    public = public_frame.dropna(subset=[public_target, *public_all]).copy()
    market = market_frame.dropna(subset=[market_target, *market_all]).copy()
    for column in public_all:
        public = public[np.isfinite(public[column].to_numpy(dtype=np.float64))]
        market = market[np.isfinite(market[column].to_numpy(dtype=np.float64))]
    if len(public) < 3 or len(market) < 3:
        return {"status": "skipped", "reason": "insufficient complete rows"}
    y_public = public[public_target].to_numpy(dtype=np.float64)
    groups_public = public["corpus"].astype(str).to_numpy()
    y_market = market[market_target].to_numpy(dtype=np.float64)
    groups_market = market["corpus"].astype(str).to_numpy()
    result: dict[str, Any] = {
        "status": "ok",
        "public_rows": int(len(public)),
        "market_rows": int(len(market)),
        "models": {},
    }
    predictions: dict[str, np.ndarray] = {}
    for name, columns in feature_sets.items():
        x_public = public[columns].to_numpy(dtype=np.float64)
        x_market = market[columns].to_numpy(dtype=np.float64)
        alpha = _select_alpha(x_public, y_public, groups_public, alphas)
        scaler = StandardScaler().fit(x_public)
        model = Ridge(alpha=alpha).fit(scaler.transform(x_public), y_public)
        prediction = model.predict(scaler.transform(x_market))
        predictions[name] = prediction
        result["models"][name] = {
            "alpha": float(alpha),
            "metrics": _regression_metrics(y_market, prediction),
        }
    rng = np.random.default_rng(seed + 300)
    bootstrap: dict[str, list[float]] = {name: [] for name in predictions}
    delta_bootstrap: list[float] = []
    for _ in range(bootstrap_iterations):
        indices = _cluster_indices(groups_market, rng)
        if len(np.unique(y_market[indices])) < 2:
            continue
        scores = {}
        for name, prediction in predictions.items():
            score = _r2(y_market[indices], prediction[indices])
            bootstrap[name].append(score)
            scores[name] = score
        delta_bootstrap.append(scores["full"] - scores["baseline"])
    for name, values in bootstrap.items():
        arr = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
        result["models"][name]["bootstrap"] = {
            "mean": float(arr.mean()) if len(arr) else float("nan"),
            "ci_low": float(np.percentile(arr, 2.5)) if len(arr) else float("nan"),
            "ci_high": float(np.percentile(arr, 97.5)) if len(arr) else float("nan"),
            "p_positive": (
                float((1 + np.sum(arr <= 0)) / (1 + len(arr))) if len(arr) else float("nan")
            ),
        }
    delta_arr = np.asarray(
        [value for value in delta_bootstrap if np.isfinite(value)], dtype=np.float64
    )
    result["delta_full_minus_baseline"] = {
        "value": result["models"]["full"]["metrics"]["r2"]
        - result["models"]["baseline"]["metrics"]["r2"],
        "ci_low": float(np.percentile(delta_arr, 2.5)) if len(delta_arr) else float("nan"),
        "ci_high": float(np.percentile(delta_arr, 97.5)) if len(delta_arr) else float("nan"),
        "p_positive": float((1 + np.sum(delta_arr <= 0)) / (1 + len(delta_arr)))
        if len(delta_arr)
        else float("nan"),
    }
    return result


def _loco_metrics(
    matrices: dict[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    alphas: list[float],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    predictions: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, float]] = {}
    selected: dict[str, dict[str, float]] = {}
    for name, matrix in matrices.items():
        prediction, chosen = loco_predictions(matrix, y, groups, alphas)
        predictions[name] = prediction
        metrics[name] = _regression_metrics(y, prediction)
        selected[name] = chosen
    return predictions, metrics, selected


def _drop_compression_baseline(
    features: pd.DataFrame,
    bpb_features: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    alphas: list[float],
) -> dict[str, Any]:
    removed = {"train_compression", "mean_token_bytes"}
    baseline = [column for column in feature_sets["baseline"] if column not in removed]
    topology = list(feature_sets["topology"])
    reduced = {"baseline": baseline, "topology": topology, "full": baseline + topology}
    result: dict[str, Any] = {"removed_features": sorted(removed), "targets": {}}
    for label, frame, target in (
        ("compression", features, "test_compression"),
        ("bpb", bpb_features, "bpb_mean"),
    ):
        matrices, y, groups, _ = _design(frame, target, reduced)
        _, metrics, selected = _loco_metrics(matrices, y, groups, alphas)
        result["targets"][label] = {
            "target": target,
            "metrics": metrics,
            "selected_alphas": selected,
        }
    return result


def _extended_alpha_grid(
    features: pd.DataFrame,
    bpb_features: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    alphas: list[float],
) -> dict[str, Any]:
    extended = sorted({*alphas, 1000.0, 10000.0})
    result: dict[str, Any] = {"alphas": extended, "targets": {}}
    for label, frame, target in (
        ("compression", features, "test_compression"),
        ("bpb", bpb_features, "bpb_mean"),
    ):
        matrices, y, groups, _ = _design(frame, target, feature_sets)
        _, metrics, selected = _loco_metrics(matrices, y, groups, extended)
        result["targets"][label] = {
            "target": target,
            "metrics": metrics,
            "selected_alphas": selected,
        }
    return result


def _within_market(
    config: dict[str, Any],
    feature_sets: dict[str, list[str]],
    alphas: list[float],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    features_path = results_dir(config) / "market_features.csv"
    bpb_path = results_dir(config) / "market_bpb.csv"
    if not features_path.exists():
        return {"status": "skipped", "reason": "market_features.csv not found"}
    market = pd.read_csv(features_path)
    if market.empty:
        return {"status": "skipped", "reason": "empty market feature table"}
    if bpb_path.exists():
        bpb = pd.read_csv(bpb_path)
        market = market.merge(
            bpb[["corpus", "family", "requested_vocab_size", "bpb"]],
            on=["corpus", "family", "requested_vocab_size"],
            how="inner",
        )
    result: dict[str, Any] = {
        "status": "ok",
        "rows": int(len(market)),
        "corpora": int(market["corpus"].nunique()),
        "targets": {},
    }
    for label, target in (("compression", "test_compression"), ("bpb", "bpb")):
        if target not in market.columns:
            continue
        matrices, y, groups, _ = _design(market, target, feature_sets)
        predictions, metrics, selected = _loco_metrics(matrices, y, groups, alphas)
        bootstrap = _cluster_bootstrap(
            y, predictions, groups, iterations, seed + 400
        )
        result["targets"][label] = {
            "target": target,
            "metrics": metrics,
            "selected_alphas": selected,
            "bootstrap": bootstrap,
        }
    return result


def _per_corpus_breakdown(
    y: np.ndarray,
    groups: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for corpus in sorted({str(value) for value in groups}):
        mask = groups == corpus
        if int(np.sum(mask)) < 2:
            continue
        entry: dict[str, Any] = {"corpus": corpus, "rows": int(np.sum(mask)), "models": {}}
        for name, prediction in predictions.items():
            residual = y[mask] - prediction[mask]
            entry["models"][name] = {
                "mae": float(np.mean(np.abs(residual))),
                "rmse": float(np.sqrt(np.mean(residual**2))),
            }
        rows.append(entry)
    return rows


def _bpb_summary(config: dict[str, Any]) -> pd.DataFrame:
    path = results_dir(config) / "bpb.csv"
    frame = pd.read_csv(path)
    grouped = (
        frame.groupby(["corpus", "family", "requested_vocab_size"], as_index=False)
        .agg(
            bpb_mean=("bpb", "mean"),
            bpb_std=("bpb", "std"),
            bpb_seeds=("bpb", "count"),
            bpb_min=("bpb", "min"),
            bpb_max=("bpb", "max"),
        )
        .reset_index(drop=True)
    )
    grouped["bpb_std"] = grouped["bpb_std"].fillna(0.0)
    return grouped


def run_analysis(config: dict[str, Any]) -> dict[str, Any]:
    analysis_cfg = config["analysis"]
    alphas = [float(value) for value in analysis_cfg["ridge_alphas"]]
    iterations = int(analysis_cfg["bootstrap_iterations"])
    permutation_iterations = int(analysis_cfg["permutation_iterations"])
    seed = int(config["project"]["seed"])
    out_dir = results_dir(config)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(out_dir / "features.csv")
    feature_sets = _feature_sets(features)
    analysis: dict[str, Any] = {
        "created_at": utc_now(),
        "config_path": config.get("_config_path"),
        "feature_sets": feature_sets,
        "primary_targets": list(analysis_cfg["primary_targets"]),
    }

    h1_matrices, h1_y, h1_groups, h1_data = _design(
        features, "test_compression", feature_sets
    )
    h1 = _model_comparison(
        h1_matrices,
        h1_y,
        h1_groups,
        alphas,
        iterations,
        permutation_iterations,
        seed,
    )
    _predictions_frame(h1_data, h1_y, h1_groups, h1["predictions"]).to_csv(
        tables_dir / "h1_predictions.csv", index=False
    )
    h1["per_corpus"] = _per_corpus_breakdown(h1_y, h1_groups, h1["predictions"])
    h1.pop("predictions")
    analysis["h1"] = h1

    bpb = _bpb_summary(config)
    bpb.to_csv(tables_dir / "bpb_summary.csv", index=False)
    bpb_features = features[features["family"] == "bpe"].merge(
        bpb[bpb["family"] == "bpe"],
        on=["corpus", "family", "requested_vocab_size"],
        how="inner",
    )
    h2_matrices, h2_y, h2_groups, h2_data = _design(
        bpb_features, "bpb_mean", feature_sets
    )
    h2 = _model_comparison(
        h2_matrices,
        h2_y,
        h2_groups,
        alphas,
        iterations,
        permutation_iterations,
        seed + 1,
    )
    _predictions_frame(h2_data, h2_y, h2_groups, h2["predictions"]).to_csv(
        tables_dir / "h2_predictions.csv", index=False
    )
    h2["per_corpus"] = _per_corpus_breakdown(h2_y, h2_groups, h2["predictions"])
    h2.pop("predictions")
    analysis["h2"] = h2

    h3_rows: list[dict[str, Any]] = []
    h3_variables = [
        "vocab_h1_total_persistence",
        "vocab_h1_persistence_entropy",
        "cooc_h1_total_persistence",
        "cooc_h1_persistence_entropy",
    ]
    h3_data = bpb_features.dropna(
        subset=["bpb_mean", "test_compression", *h3_variables]
    ).copy()
    h3_groups = h3_data["corpus"].astype(str).to_numpy()
    for variable in h3_variables:
        inference = _partial_spearman_inference(
            h3_data[variable].to_numpy(dtype=np.float64),
            h3_data["bpb_mean"].to_numpy(dtype=np.float64),
            h3_data["test_compression"].to_numpy(dtype=np.float64),
            h3_groups,
            iterations,
            seed + 10,
        )
        inference["variable"] = variable
        h3_rows.append(inference)
    analysis["h3"] = {
        "control": "test_compression",
        "target": "bpb_mean",
        "variables": h3_rows,
    }
    pd.DataFrame(h3_rows).to_csv(tables_dir / "h3_partial_spearman.csv", index=False)

    public_bpe = features[features["family"] == "bpe"].merge(
        bpb[bpb["family"] == "bpe"],
        on=["corpus", "family", "requested_vocab_size"],
        how="inner",
    )
    market_features_path = out_dir / "market_features.csv"
    market_bpb_path = out_dir / "market_bpb.csv"
    if market_features_path.exists():
        market_features = pd.read_csv(market_features_path)
        h4_compression = _transfer(
            public_bpe,
            market_features,
            "test_compression",
            "test_compression",
            feature_sets,
            alphas,
            iterations,
            seed + 20,
        )
        analysis["h4_compression"] = h4_compression
        if market_bpb_path.exists():
            market_bpb = pd.read_csv(market_bpb_path)
            market_merged = market_features.merge(
                market_bpb[market_bpb["family"] == "bpe"][
                    ["corpus", "family", "requested_vocab_size", "bpb"]
                ],
                on=["corpus", "family", "requested_vocab_size"],
                how="inner",
            )
            h4_bpb = _transfer(
                public_bpe,
                market_merged,
                "bpb_mean",
                "bpb",
                feature_sets,
                alphas,
                iterations,
                seed + 21,
            )
            analysis["h4_bpb"] = h4_bpb
    else:
        analysis["h4_compression"] = {
            "status": "skipped",
            "reason": "market_features.csv not found",
        }
        analysis["h4_bpb"] = {
            "status": "skipped",
            "reason": "market_features.csv not found",
        }

    analysis["robustness"] = {
        "drop_train_compression": _drop_compression_baseline(
            features, bpb_features, feature_sets, alphas
        ),
        "extended_alpha_grid": _extended_alpha_grid(
            features, bpb_features, feature_sets, alphas
        ),
    }
    if market_features_path.exists():
        analysis["exploratory_within_market"] = _within_market(
            config, feature_sets, alphas, iterations, seed + 20
        )

    h5_rows: list[dict[str, Any]] = []
    compression = features["test_compression"].to_numpy(dtype=np.float64)
    groups = features["corpus"].astype(str).to_numpy()
    rng = np.random.default_rng(seed + 30)
    for column in feature_sets["topology"]:
        values = features[column].to_numpy(dtype=np.float64)
        if np.std(values) < 1e-12:
            continue
        correlation, p_value = spearmanr(values, compression)
        bootstrap_values: list[float] = []
        for _ in range(iterations):
            indices = _cluster_indices(groups, rng)
            if np.std(values[indices]) < 1e-12 or np.std(compression[indices]) < 1e-12:
                continue
            rho, _ = spearmanr(values[indices], compression[indices])
            if np.isfinite(rho):
                bootstrap_values.append(float(rho))
        boot = np.asarray(bootstrap_values, dtype=np.float64)
        h5_rows.append(
            {
                "feature": column,
                "spearman_rho": float(correlation),
                "p_value": float(p_value),
                "ci_low": float(np.percentile(boot, 2.5)) if len(boot) else float("nan"),
                "ci_high": float(np.percentile(boot, 97.5)) if len(boot) else float("nan"),
            }
        )
    h5_frame = pd.DataFrame(h5_rows).sort_values(
        "spearman_rho", key=lambda values: values.abs(), ascending=False
    )
    h5_frame.to_csv(tables_dir / "h5_feature_correlations.csv", index=False)
    analysis["h5"] = {
        "target": "test_compression",
        "exploratory": True,
        "top_features": json.loads(h5_frame.head(10).to_json(orient="records")),
    }

    h1_p = analysis["h1"]["permutation"]["p_value"]
    h2_p = analysis["h2"]["permutation"]["p_value"]
    h3_raw = [
        row["p_value"] for row in analysis["h3"]["variables"] if np.isfinite(row["p_value"])
    ] or [1.0]
    h3_p = min(_holm(h3_raw))
    h4_values = []
    for key in ("h4_compression", "h4_bpb"):
        block = analysis.get(key, {})
        if block.get("status") == "ok":
            full_p = block["models"]["full"]["bootstrap"]["p_positive"]
            delta_p = block["delta_full_minus_baseline"]["p_positive"]
            h4_values.append(max(full_p, delta_p))
    h4_p = max(h4_values) if h4_values else 1.0
    adjusted = _holm([h1_p, h2_p, h3_p, h4_p])
    analysis["holm"] = {
        "hypotheses": ["H1", "H2", "H3", "H4"],
        "raw_p": [float(h1_p), float(h2_p), float(h3_p), float(h4_p)],
        "adjusted_p": adjusted,
    }

    importance: dict[str, Any] = {}
    for name, target_frame, target in [
        ("compression", features, "test_compression"),
        ("bpb", bpb_features, "bpb_mean"),
    ]:
        matrices, y, group_values, data = _design(
            target_frame, target, feature_sets
        )
        columns = feature_sets["full"]
        x = matrices["full"]
        alpha = _select_alpha(x, y, group_values, alphas)
        scaler = StandardScaler().fit(x)
        model = Ridge(alpha=alpha).fit(scaler.transform(x), y)
        coefficients = [
            {"feature": column, "coefficient": float(coefficient)}
            for column, coefficient in zip(columns, model.coef_, strict=True)
        ]
        coefficients.sort(key=lambda row: abs(row["coefficient"]), reverse=True)
        importance[name] = {
            "alpha": float(alpha),
            "coefficients": coefficients,
        }
        pd.DataFrame(coefficients).to_csv(
            tables_dir / f"feature_importance_{name}.csv", index=False
        )
    analysis["feature_importance"] = importance

    write_json(out_dir / "analysis.json", analysis)
    return analysis
