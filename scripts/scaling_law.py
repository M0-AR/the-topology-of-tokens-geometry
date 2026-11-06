from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from topobpe.config import load_config, results_dir
from topobpe.stats import _design, _regression_metrics, loco_predictions
from topobpe.utils import utc_now, write_json

INTRINSIC = [
    "log_vocab",
    "byte_entropy",
    "conditional_entropy",
    "type_token_ratio",
    "gzip_ratio",
    "log_train_tokens",
]
WITH_TRAIN = INTRINSIC + ["train_compression"]
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def power_law(frame: pd.DataFrame, corpus: str, family: str) -> dict:
    subset = frame[
        (frame["corpus"] == corpus) & (frame["family"] == family)
    ].sort_values("actual_vocab_size")
    x = np.log(subset["actual_vocab_size"].to_numpy(dtype=np.float64))
    y = np.log(subset["test_compression"].to_numpy(dtype=np.float64))
    design = np.column_stack([np.ones_like(x), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    prediction = design @ coefficients
    residual = y - prediction
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "corpus": corpus,
        "family": family,
        "a": float(np.exp(coefficients[0])),
        "b": float(coefficients[1]),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "sigma_log": float(np.std(residual, ddof=1)) if len(y) > 2 else 0.0,
        "points": [
            {
                "vocab": int(row.actual_vocab_size),
                "compression": float(row.test_compression),
            }
            for row in subset.itertuples()
        ],
    }


def extrapolate_32k(per_corpus: list[dict]) -> dict:
    rows: list[dict] = []
    for entry in per_corpus:
        points = [point for point in entry["points"] if point["vocab"] <= 16384]
        target = [point for point in entry["points"] if point["vocab"] == 32768]
        if len(points) < 2 or not target:
            continue
        x = np.log([point["vocab"] for point in points], dtype=np.float64)
        y = np.log([point["compression"] for point in points], dtype=np.float64)
        design = np.column_stack([np.ones_like(x), x])
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        predicted = float(
            np.exp(coefficients[0] + coefficients[1] * np.log(target[0]["vocab"]))
        )
        rows.append(
            {
                "corpus": entry["corpus"],
                "family": entry["family"],
                "actual": target[0]["compression"],
                "predicted": predicted,
                "relative_error": predicted / target[0]["compression"] - 1.0,
            }
        )
    frame = pd.DataFrame(rows)
    by_family = []
    for family, group in frame.groupby("family"):
        by_family.append(
            {
                "family": family,
                "n": int(len(group)),
                "mape": float(np.mean(np.abs(group["relative_error"]))),
                "max_abs_relative_error": float(
                    np.max(np.abs(group["relative_error"]))
                ),
            }
        )
    return {"rows": rows, "by_family": by_family}


def pooled(frame: pd.DataFrame, columns: list[str]) -> dict:
    sets = {
        "baseline": ["log_vocab"],
        "full": columns,
        "topology": columns,
    }
    matrices, y, groups, _ = _design(frame, "log_compression", sets)
    prediction, selected = loco_predictions(matrices["full"], y, groups, ALPHAS)
    metrics = _regression_metrics(y, prediction)
    return {
        "n": int(len(y)),
        "features": columns,
        "r2_log": metrics["r2"],
        "rmse_log": metrics["rmse"],
        "selected_alphas": selected,
    }


def holdout_32k(frame: pd.DataFrame, columns: list[str]) -> dict:
    rows: list[dict] = []
    for family in ("bpe", "unigram"):
        subset = frame[frame["family"] == family]
        train = subset[subset["requested_vocab_size"] < 32768]
        test = subset[subset["requested_vocab_size"] == 32768]
        if train.empty or test.empty:
            continue
        sets = {"full": columns}
        x_train, y_train, _, _ = _design(train, "log_compression", sets)
        x_test, y_test, _, _ = _design(test, "log_compression", sets)
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler().fit(x_train["full"])
        model = Ridge(alpha=10.0).fit(
            scaler.transform(x_train["full"]), y_train
        )
        prediction = model.predict(scaler.transform(x_test["full"]))
        rows.append(
            {
                "family": family,
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                **_regression_metrics(y_test, prediction),
            }
        )
    return {"target": "log_compression", "by_family": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit tokenizer scaling laws")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    out = results_dir(config)
    frame = pd.read_csv(out / "features.csv")
    frame["log_vocab"] = np.log(frame["actual_vocab_size"].astype(np.float64))
    frame["log_train_tokens"] = np.log(frame["train_tokens"].astype(np.float64))
    frame["log_compression"] = np.log(frame["test_compression"].astype(np.float64))
    corpora = sorted(frame["corpus"].unique())
    per_corpus = [
        power_law(frame, corpus, family)
        for corpus in corpora
        for family in ("bpe", "unigram")
    ]
    analysis = {
        "created_at": utc_now(),
        "target": "test_compression (bytes per token)",
        "per_corpus": per_corpus,
        "pooled_intrinsic": pooled(frame, INTRINSIC),
        "pooled_with_train_compression": pooled(frame, WITH_TRAIN),
        "holdout_32k": holdout_32k(frame, INTRINSIC),
        "power_law_extrapolation_32k": extrapolate_32k(per_corpus),
    }
    write_json(out / "scaling_law.json", analysis)
    table = pd.DataFrame(
        [
            {
                "corpus": row["corpus"],
                "family": row["family"],
                "a": row["a"],
                "b": row["b"],
                "r2": row["r2"],
                "sigma_log": row["sigma_log"],
            }
            for row in per_corpus
        ]
    )
    table.to_csv(out / "tables" / "scaling_law.csv", index=False)
    print(table.to_string(index=False))
    print("\npooled intrinsic:", analysis["pooled_intrinsic"])
    print("pooled with train_compression:", analysis["pooled_with_train_compression"])
    print("holdout 32k:", analysis["holdout_32k"])
    print(
        "power-law extrapolation 32k:",
        analysis["power_law_extrapolation_32k"]["by_family"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
