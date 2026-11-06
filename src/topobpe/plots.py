from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import artifact_dir, results_dir
from .topology import betti_curve
from .utils import read_json


def _figures_dir(config: dict[str, Any]) -> Path:
    path = results_dir(config) / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def _diagram_arrays(diagrams: list[list[list[float | None]]]) -> list[np.ndarray]:
    output = []
    for diagram in diagrams:
        if not diagram:
            output.append(np.zeros((0, 2), dtype=np.float64))
            continue
        array = np.asarray(
            [[birth, death if death is not None else np.inf] for birth, death in diagram],
            dtype=np.float64,
        )
        output.append(array)
    return output


def plot_persistence_examples(
    config: dict[str, Any],
    corpus: str = "wikitext_en",
    family: str = "bpe",
    vocab_size: int = 16384,
) -> Path | None:
    path = (
        artifact_dir(config)
        / "topology"
        / corpus
        / f"{family}_{vocab_size}_diagnostics.json"
    )
    if not path.exists():
        return None
    diagnostics = read_json(path)
    vocab = _diagram_arrays(diagnostics["vocab_diagrams"])
    cooc = _diagram_arrays(diagnostics["cooc_diagrams"])
    figure, axes = plt.subplots(2, 2, figsize=(9, 7))
    for column, (name, diagrams) in enumerate([("vocabulary", vocab), ("co-occurrence", cooc)]):
        diagram = diagrams[1] if len(diagrams) > 1 else diagrams[0]
        axis = axes[0, column]
        if len(diagram):
            finite = diagram[np.isfinite(diagram[:, 1])]
            infinite = diagram[~np.isfinite(diagram[:, 1])]
            if len(finite):
                axis.scatter(finite[:, 0], finite[:, 1], s=10, alpha=0.7)
            if len(infinite):
                top = finite[:, 1].max() if len(finite) else 1.0
                axis.scatter(
                    infinite[:, 0],
                    np.full(len(infinite), top * 1.05),
                    s=14,
                    marker="^",
                    alpha=0.8,
                )
        axis.set_title(f"{name} H1")
        axis.set_xlabel("birth")
        axis.set_ylabel("death")
        axis.grid(alpha=0.25)
        axis = axes[1, column]
        finite_deaths = diagram[:, 1][np.isfinite(diagram[:, 1])]
        top = float(finite_deaths.max()) if len(finite_deaths) else 0.01
        grid = np.linspace(0.0, max(0.01, top), 80)
        axis.plot(grid, betti_curve(diagram, grid), linewidth=2)
        axis.set_title(f"{name} Betti-1 curve")
        axis.set_xlabel("filtration value")
        axis.set_ylabel(r"$\beta_1$")
        axis.grid(alpha=0.25)
    figure.suptitle(f"{corpus} · {family.upper()} · vocabulary {vocab_size:,}", fontsize=11)
    figure.tight_layout()
    destination = _figures_dir(config) / "fig1_persistence_examples"
    _save(figure, destination)
    return destination


def plot_compression_scatter(config: dict[str, Any]) -> Path | None:
    path = results_dir(config) / "features.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    for corpus, subset in frame.groupby("corpus"):
        axis.scatter(
            subset["vocab_h1_total_persistence"],
            subset["test_compression"],
            s=46,
            label=corpus,
            alpha=0.85,
        )
    axis.set_xlabel("H1 total persistence of vocabulary point cloud")
    axis.set_ylabel("held-out bytes per token")
    axis.set_title("Topological signature and tokenizer compression")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    destination = _figures_dir(config) / "fig2_compression_scatter"
    _save(figure, destination)
    return destination


def _bar_panel(
    axis: plt.Axes,
    block: dict[str, Any],
    title: str,
) -> None:
    models = ["baseline", "topology", "full"]
    values = []
    lower = []
    upper = []
    for model in models:
        metrics = block["metrics"][model]
        bootstrap = block["bootstrap"][model]
        values.append(metrics["r2"])
        lower.append(max(0.0, metrics["r2"] - bootstrap["ci_low"]))
        upper.append(max(0.0, bootstrap["ci_high"] - metrics["r2"]))
    axis.bar(
        models, values, yerr=[lower, upper], capsize=4, color=["#7f8c8d", "#3498db", "#2c3e50"]
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_title(title)
    axis.set_ylabel(r"leave-one-corpus-out $R^2$")
    axis.grid(axis="y", alpha=0.25)


def plot_loco_r2(config: dict[str, Any]) -> Path | None:
    path = results_dir(config) / "analysis.json"
    if not path.exists():
        return None
    analysis = read_json(path)
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    _bar_panel(axes[0], analysis["h1"], "Held-out compression")
    _bar_panel(axes[1], analysis["h2"], "Bits per byte")
    figure.tight_layout()
    destination = _figures_dir(config) / "fig3_loco_r2"
    _save(figure, destination)
    return destination


def plot_bpb(config: dict[str, Any]) -> Path | None:
    path = results_dir(config) / "bpb.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    grouped = frame.groupby(["corpus", "requested_vocab_size"], as_index=False).agg(
        mean=("bpb", "mean"), std=("bpb", "std")
    )
    grouped["std"] = grouped["std"].fillna(0.0)
    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    for corpus, subset in grouped.groupby("corpus"):
        subset = subset.sort_values("requested_vocab_size")
        axis.errorbar(
            subset["requested_vocab_size"],
            subset["mean"],
            yerr=subset["std"],
            marker="o",
            capsize=3,
            label=corpus,
        )
    axis.set_xscale("log", base=2)
    axis.set_xlabel("requested vocabulary size")
    axis.set_ylabel("test bits per byte")
    axis.set_title("BPB across tokenizer budgets (BPE, two seeds)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    destination = _figures_dir(config) / "fig4_bpb"
    _save(figure, destination)
    return destination


def plot_market(config: dict[str, Any]) -> Path | None:
    path = results_dir(config) / "analysis.json"
    if not path.exists():
        return None
    analysis = read_json(path)
    blocks = [
        ("compression", analysis.get("h4_compression", {})),
        ("BPB", analysis.get("h4_bpb", {})),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for axis, (title, block) in zip(axes, blocks, strict=True):
        if block.get("status") != "ok":
            axis.text(0.5, 0.5, "market data unavailable", ha="center", va="center")
            axis.set_axis_off()
            continue
        models = ["baseline", "full"]
        values = [block["models"][model]["metrics"]["r2"] for model in models]
        lower = [
            max(
                0.0,
                block["models"][model]["metrics"]["r2"]
                - block["models"][model]["bootstrap"]["ci_low"],
            )
            for model in models
        ]
        upper = [
            max(
                0.0,
                block["models"][model]["bootstrap"]["ci_high"]
                - block["models"][model]["metrics"]["r2"],
            )
            for model in models
        ]
        axis.bar(models, values, yerr=[lower, upper], capsize=4, color=["#7f8c8d", "#2c3e50"])
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(f"Market transfer: {title}")
        axis.set_ylabel(r"$R^2$")
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    destination = _figures_dir(config) / "fig5_market"
    _save(figure, destination)
    return destination


def plot_feature_importance(config: dict[str, Any]) -> Path | None:
    path = results_dir(config) / "analysis.json"
    if not path.exists():
        return None
    analysis = read_json(path)
    coefficients = analysis["feature_importance"]["compression"]["coefficients"][:15]
    if not coefficients:
        return None
    frame = pd.DataFrame(coefficients).iloc[::-1]
    figure, axis = plt.subplots(figsize=(7.6, 5.6))
    colors = ["#c0392b" if value < 0 else "#2c3e50" for value in frame["coefficient"]]
    axis.barh(frame["feature"], frame["coefficient"], color=colors)
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("standardized ridge coefficient")
    axis.set_title("Feature importance for held-out compression")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    destination = _figures_dir(config) / "fig6_feature_importance"
    _save(figure, destination)
    return destination


def make_all_figures(config: dict[str, Any]) -> list[str]:
    outputs = []
    for function in [
        plot_persistence_examples,
        plot_compression_scatter,
        plot_loco_r2,
        plot_bpb,
        plot_market,
        plot_feature_importance,
    ]:
        result = function(config)
        if result is not None:
            outputs.append(str(result))
    return outputs
