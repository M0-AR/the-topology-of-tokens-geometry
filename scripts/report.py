from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from topobpe.config import load_config, results_dir


def _table(frame: pd.DataFrame, float_format: str = "{:.4f}") -> str:
    if frame.empty:
        return "_no rows_"
    columns = list(frame.columns)
    lines = ["| " + " | ".join(str(column) for column in columns) + " |"]
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                cells.append(float_format.format(value))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _metric_rows(block: dict[str, Any], label: str) -> pd.DataFrame:
    rows = []
    for model, metrics in block.get("metrics", {}).items():
        bootstrap = block.get("bootstrap", {}).get(model, {})
        rows.append(
            {
                "target": label,
                "model": model,
                "r2": metrics.get("r2"),
                "mae": metrics.get("mae"),
                "rmse": metrics.get("rmse"),
                "bootstrap_ci_low": bootstrap.get("ci_low"),
                "bootstrap_ci_high": bootstrap.get("ci_high"),
            }
        )
    return pd.DataFrame(rows)


def build_report(config: dict[str, Any]) -> str:
    out = results_dir(config)
    analysis = json.loads((out / "analysis.json").read_text())
    lines = ["# TopoBPE results report", ""]
    lines.append("Generated from `results/analysis.json`.")
    lines.append("")
    lines.append("## Confirmatory models")
    lines.append("")
    frames = [
        _metric_rows(analysis["h1"], "compression"),
        _metric_rows(analysis["h2"], "bpb"),
    ]
    lines.append(_table(pd.concat(frames, ignore_index=True)))
    lines.append("")
    per_corpus_rows = []
    for name, label in (("h1", "compression"), ("h2", "bpb")):
        for entry in analysis[name].get("per_corpus", []):
            for model, metrics in entry["models"].items():
                per_corpus_rows.append(
                    {
                        "target": label,
                        "corpus": entry["corpus"],
                        "rows": entry["rows"],
                        "model": model,
                        "mae": metrics["mae"],
                        "rmse": metrics["rmse"],
                    }
                )
    if per_corpus_rows:
        lines.append("### Held-out error by corpus")
        lines.append("")
        lines.append(_table(pd.DataFrame(per_corpus_rows)))
        lines.append("")
    lines.append("### Permutation tests")
    lines.append("")
    permutation = pd.DataFrame(
        [
            {
                "hypothesis": name.upper(),
                "observed_delta_r2": analysis[name]["permutation"]["observed_delta_r2"],
                "p_value": analysis[name]["permutation"]["p_value"],
                "iterations": analysis[name]["permutation"]["iterations"],
            }
            for name in ("h1", "h2")
        ]
    )
    lines.append(_table(permutation))
    lines.append("")
    lines.append("### Holm correction")
    lines.append("")
    lines.append(_table(pd.DataFrame(analysis["holm"])))
    lines.append("")
    lines.append("## H3 partial Spearman")
    lines.append("")
    lines.append(_table(pd.DataFrame(analysis["h3"]["variables"])))
    lines.append("")
    for key, label in (("h4_compression", "H4 compression"), ("h4_bpb", "H4 BPB")):
        block = analysis.get(key, {})
        lines.append(f"## {label}")
        lines.append("")
        if block.get("status") != "ok":
            lines.append(f"Status: {block.get('status')} — {block.get('reason', '')}")
        else:
            rows = []
            for model, entry in block["models"].items():
                rows.append(
                    {
                        "model": model,
                        "r2": entry["metrics"]["r2"],
                        "mae": entry["metrics"]["mae"],
                        "ci_low": entry.get("bootstrap", {}).get("ci_low"),
                        "ci_high": entry.get("bootstrap", {}).get("ci_high"),
                    }
                )
            lines.append(_table(pd.DataFrame(rows)))
            delta = block["delta_full_minus_baseline"]
            lines.append("")
            lines.append(
                f"Delta R2 (full - baseline): {delta['value']:.4f} "
                f"[{delta['ci_low']:.4f}, {delta['ci_high']:.4f}], "
                f"one-sided bootstrap p={delta['p_positive']:.4f}"
            )
        lines.append("")
    within = analysis.get("exploratory_within_market")
    if within and within.get("status") == "ok":
        lines.append("## Exploratory within-market model")
        lines.append("")
        rows = []
        for label, block in within["targets"].items():
            for model, metrics in block["metrics"].items():
                rows.append(
                    {
                        "target": label,
                        "model": model,
                        "r2": metrics["r2"],
                        "mae": metrics["mae"],
                    }
                )
        lines.append(_table(pd.DataFrame(rows)))
        lines.append("")
    lines.append("## H5 exploratory feature correlations")
    lines.append("")
    lines.append(_table(pd.DataFrame(analysis["h5"]["top_features"])))
    lines.append("")
    robustness = analysis.get("robustness", {}).get("drop_train_compression")
    if robustness:
        lines.append("## Robustness: drop train-compression baseline")
        lines.append("")
        rows = []
        for label, block in robustness["targets"].items():
            for model, metrics in block["metrics"].items():
                rows.append(
                    {
                        "target": label,
                        "model": model,
                        "r2": metrics["r2"],
                        "mae": metrics["mae"],
                    }
                )
        lines.append(_table(pd.DataFrame(rows)))
        lines.append("")
    extended = analysis.get("robustness", {}).get("extended_alpha_grid")
    if extended:
        lines.append("## Robustness: extended ridge alpha grid")
        lines.append("")
        rows = []
        for label, block in extended["targets"].items():
            for model, metrics in block["metrics"].items():
                rows.append(
                    {
                        "target": label,
                        "model": model,
                        "r2": metrics["r2"],
                        "mae": metrics["mae"],
                    }
                )
        lines.append(_table(pd.DataFrame(rows)))
        lines.append("")
    tables_dir = out / "tables"
    if tables_dir.exists():
        lines.append("## Tables on disk")
        lines.append("")
        for path in sorted(tables_dir.glob("*.csv")):
            lines.append(f"- `{path.relative_to(out.parent)}`")
        lines.append("")
    figures_dir = out / "figures"
    if figures_dir.exists():
        lines.append("## Figures on disk")
        lines.append("")
        for path in sorted(figures_dir.glob("*.pdf")):
            lines.append(f"- `{path.relative_to(out.parent)}`")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the results report from analysis.json")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    text = build_report(config)
    path = results_dir(config) / "REPORT.md"
    path.write_text(text)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
