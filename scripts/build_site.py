"""Build the static showcase site (preview.html) from verified result files.

Reads only artefacts produced by the pipeline, computes a 3D PCA embedding of the
80 measured tokenizer features, and injects the payload into site/template.html.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SITE = ROOT / "site"

LABELS = {
    "wikitext_en": ("English Wikipedia", "English", "Latin", "WikiText-103"),
    "codeparrot_py": ("Python code", "Python", "Code", "CodeParrot"),
    "wikipedia_zh": ("Chinese Wikipedia", "Chinese", "Han", "Wikimedia"),
    "wikipedia_hu": ("Hungarian Wikipedia", "Hungarian", "Latin", "Wikimedia"),
    "wikipedia_ar": ("Arabic Wikipedia", "Arabic", "Arabic", "Wikimedia"),
    "wikipedia_sw": ("Swahili Wikipedia", "Swahili", "Latin", "Wikimedia"),
}

TICKERS = {
    "market_aapl_10y": "AAPL",
    "market_msft_10y": "MSFT",
    "market_nvda_10y": "NVDA",
    "market_tsla_10y": "TSLA",
    "market_jpm_10y": "JPM",
    "market_gs_10y": "GS",
    "market_jnj_10y": "JNJ",
    "market_xom_10y": "XOM",
    "market_ko_10y": "KO",
    "market_cat_10y": "CAT",
    "market_dis_10y": "DIS",
    "market_wmt_10y": "WMT",
}


def _f(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return float(x)


def _metrics(m):
    return {k: _f(m.get(k)) for k in ("r2", "mae", "rmse", "n")}


def main() -> None:
    analysis = json.loads((RESULTS / "analysis.json").read_text())
    feats = pd.read_csv(RESULTS / "features.csv")
    bpb = pd.read_csv(RESULTS / "bpb.csv")
    bpb_summary = pd.read_csv(RESULTS / "tables" / "bpb_summary.csv")
    audit = pd.read_csv(RESULTS / "tables" / "tokenizer_byte_audit.csv")
    agreement = pd.read_csv(RESULTS / "tables" / "tda_library_agreement.csv")
    stability = json.loads((RESULTS / "tda_stability.json").read_text())
    manifest = json.loads((ROOT / "artifacts" / "manifest.json").read_text())
    market_feats = pd.read_csv(RESULTS / "market_features.csv")

    # ---- 3D embedding of the measured feature space -------------------------
    full_features = analysis["feature_sets"]["full"]
    missing = [c for c in full_features if c not in feats.columns]
    if missing:
        raise SystemExit(f"missing feature columns: {missing}")
    X = feats[full_features].to_numpy(dtype=float)
    Xz = StandardScaler().fit_transform(X)
    pca = PCA(n_components=3, random_state=0)
    coords = pca.fit_transform(Xz)

    bpb_mean = (
        bpb.groupby(["corpus", "requested_vocab_size"])["bpb"]
        .mean()
        .rename("bpb_mean")
        .reset_index()
    )
    points = []
    for i, row in feats.iterrows():
        bpb_val = None
        if row["family"] == "bpe":
            match = (bpb_mean["corpus"] == row["corpus"]) & (
                bpb_mean["requested_vocab_size"] == row["requested_vocab_size"]
            )
            hit = bpb_mean[match]
            if not hit.empty:
                bpb_val = float(hit.iloc[0]["bpb_mean"])
        points.append(
            {
                "x": _f(coords[i, 0]),
                "y": _f(coords[i, 1]),
                "z": _f(coords[i, 2]),
                "corpus": row["corpus"],
                "label": LABELS.get(row["corpus"], (row["corpus"],))[0],
                "family": row["family"],
                "vocab": int(row["requested_vocab_size"]),
                "compression": _f(row["test_compression"]),
                "train_compression": _f(row["train_compression"]),
                "bpb": bpb_val,
                "byte_entropy": _f(row["byte_entropy"]),
                "bytes_per_token": _f(row["mean_token_bytes"]),
                "h1_total": _f(row["vocab_h1_total_persistence"]),
                "h0_bars": _f(row["vocab_h0_num_bars"]),
                "cooc_h1_total": _f(row["cooc_h1_total_persistence"]),
            }
        )

    corpora = []
    for corpus in LABELS:
        sub = feats[feats["corpus"] == corpus]
        if sub.empty:
            continue
        label, language, script, source = LABELS[corpus]
        corpora.append(
            {
                "id": corpus,
                "label": label,
                "language": language,
                "script": script,
                "source": source,
                "train_bytes": int(sub["train_bytes"].iloc[0]),
                "test_bytes": int(sub["test_bytes"].iloc[0]),
                "train_tokens": int(sub["train_tokens"].iloc[0]),
                "families": sorted(sub["family"].unique().tolist()),
            }
        )

    # ---- tokenizer rows -----------------------------------------------------
    rows = []
    for _, r in feats.iterrows():
        bpb_val = None
        bpb_std = None
        if r["family"] == "bpe":
            hit = bpb_summary[
                (bpb_summary["corpus"] == r["corpus"])
                & (bpb_summary["family"].eq("bpe"))
                & (bpb_summary["requested_vocab_size"] == r["requested_vocab_size"])
            ]
            if not hit.empty:
                bpb_val = _f(hit.iloc[0]["bpb_mean"])
                bpb_std = _f(hit.iloc[0]["bpb_std"])
        rows.append(
            {
                "corpus": r["corpus"],
                "label": LABELS.get(r["corpus"], (r["corpus"],))[0],
                "family": r["family"],
                "vocab": int(r["requested_vocab_size"]),
                "actual_vocab": _f(r["actual_vocab_size"]),
                "compression": _f(r["test_compression"]),
                "train_compression": _f(r["train_compression"]),
                "bytes_per_token": _f(r["mean_token_bytes"]),
                "bpb": bpb_val,
                "bpb_std": bpb_std,
                "byte_entropy": _f(r["byte_entropy"]),
                "token_entropy": _f(r["token_entropy"]),
                "h1_total": _f(r["vocab_h1_total_persistence"]),
                "cooc_h1_total": _f(r["cooc_h1_total_persistence"]),
            }
        )

    # ---- market rows --------------------------------------------------------
    mkt_bpb = pd.read_csv(RESULTS / "market_bpb.csv").groupby("corpus")["bpb"].mean()
    market = []
    for _, r in market_feats.iterrows():
        corpus = r["corpus"]
        market.append(
            {
                "id": corpus,
                "ticker": TICKERS.get(corpus, corpus),
                "compression": _f(r["test_compression"]),
                "bpb": _f(mkt_bpb.get(corpus)),
                "byte_entropy": _f(r["byte_entropy"]),
                "bytes_per_token": _f(r["mean_token_bytes"]),
            }
        )

    stability_rows = len(stability.get("records", []))
    agreement_all = bool(agreement["agree"].all()) if "agree" in agreement else None
    max_bottleneck = (
        _f(agreement["bottleneck_distance"].max())
        if "bottleneck_distance" in agreement
        else None
    )

    payload = {
        "meta": {
            "created_at": analysis.get("created_at"),
            "git_commit": manifest.get("git_commit"),
            "files": len(manifest.get("files") or {}),
            "tokenizers": int(len(feats)),
            "lm_runs": int(len(bpb)),
            "market_series": int(market_feats["corpus"].nunique()),
            "permutations": int(analysis["h1"]["permutation"]["iterations"]),
            "bootstrap": int(analysis["h1"]["bootstrap"]["baseline"]["iterations"]),
            "languages": 6,
            "features": len(full_features),
            "baseline_features": len(analysis["feature_sets"]["baseline"]),
            "topology_features": len(full_features) - len(analysis["feature_sets"]["baseline"]),
            "pca_explained": [round(float(v) * 100, 1) for v in pca.explained_variance_ratio_],
        },
        "corpora": corpora,
        "tokenizers": rows,
        "pca": {"points": points},
        "h1": {
            "metrics": {k: _metrics(v) for k, v in analysis["h1"]["metrics"].items()},
            "bootstrap": analysis["h1"]["bootstrap"],
            "permutation": analysis["h1"]["permutation"],
            "per_group": analysis["h1"]["per_group"],
        },
        "h2": {
            "metrics": {k: _metrics(v) for k, v in analysis["h2"]["metrics"].items()},
            "bootstrap": analysis["h2"]["bootstrap"],
            "permutation": analysis["h2"]["permutation"],
            "per_group": analysis["h2"]["per_group"],
        },
        "h3": analysis["h3"],
        "holm": analysis["holm"],
        "h4": {
            "compression": analysis["h4_compression"],
            "bpb": analysis["h4_bpb"],
            "within": analysis["exploratory_within_market"],
        },
        "importance": analysis["feature_importance"],
        "h5": analysis["h5"],
        "robustness": {
            "drop_train_compression": analysis["robustness"]["drop_train_compression"]["targets"],
            "target_permutation": analysis["robustness"].get("target_permutation"),
        },
        "market": market,
        "tda": {
            "stability_records": stability_rows,
            "all_agree": agreement_all,
            "max_bottleneck": max_bottleneck,
            "comparisons": int(len(agreement)),
        },
        "audit": {
            "total": int(len(audit)),
            "reversible": int(audit["byte_reversible"].sum()),
            "bpe_reversible": int(audit[audit["family"] == "bpe"]["byte_reversible"].sum()),
            "bpe_total": int((audit["family"] == "bpe").sum()),
            "unigram_reversible": int(audit[audit["family"] == "unigram"]["byte_reversible"].sum()),
            "unigram_total": int((audit["family"] == "unigram").sum()),
        },
        "bpb_summary": bpb_summary.to_dict(orient="records"),
    }

    template = (SITE / "template.html").read_text()
    marker = "/*__DATA__*/"
    if marker not in template:
        raise SystemExit("template marker missing")
    html = template.replace(marker, json.dumps(payload, allow_nan=False, separators=(",", ":")))
    (ROOT / "preview.html").write_text(html)
    (ROOT / "index.html").write_text(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta http-equiv=\"refresh\" content=\"0; url=preview.html\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        "<title>TopoBPE — tokenizer intelligence</title>"
        "<link rel=\"canonical\" href=\"preview.html\"></head>"
        "<body style=\"background:#06090f;color:#e8eef7;font-family:system-ui;"
        "display:grid;place-items:center;height:100vh;margin:0\">"
        "<p>Opening <a style=\"color:#5eead4\" href=\"preview.html\">the TopoBPE showcase</a>…</p>"
        "</body></html>"
    )
    print(f"wrote preview.html ({len(html):,} bytes) and index.html")


if __name__ == "__main__":
    main()
