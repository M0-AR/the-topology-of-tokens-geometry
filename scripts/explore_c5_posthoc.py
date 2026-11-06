from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import invention_c5 as c5

from topobpe.config import load_config, project_root, results_dir
from topobpe.merge_selection import (
    annotated_merges,
    compression_statistics,
    document_compressions,
    holm_adjust,
)
from topobpe.merge_topology import load_bpe_components
from topobpe.tokenizers import load_tokenizer, tokenizer_path
from topobpe.utils import utc_now, write_json

PAIRS = (
    "wikitext_en:wikipedia_sw,"
    "wikitext_en:wikipedia_ar,"
    "wikitext_en:wikipedia_hu,"
    "wikitext_en:wikipedia_zh,"
    "codeparrot_py:wikitext_en"
)


def length_order(features: dict, longest: bool) -> list[int]:
    sign = -1.0 if longest else 1.0
    return sorted(
        range(len(features["bytes"])),
        key=lambda index: (sign * float(features["bytes"][index]), index),
    )


def load_json(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Post-hoc length controls for C5 (not pre-registered)"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--pairs", default=PAIRS)
    parser.add_argument("--budgets", type=int, nargs="+", default=[8192, 16384])
    parser.add_argument(
        "--results", default="results/invention_c5.json"
    )
    parser.add_argument("--out", default="results/exploratory_c5.json")
    args = parser.parse_args()

    config = load_config(args.config)
    c5_results = load_json(project_root() / args.results)
    pairs = [tuple(item.split(":")) for item in args.pairs.split(",") if item]

    payload = {
        "created_at": utc_now(),
        "posthoc": True,
        "not_in_preregistration": True,
        "note": (
            "Exploratory length-control analysis. Do not treat as confirmatory. "
            "The frozen C5 primary was high_fertility vs random_mean and it failed."
        ),
        "pairs": [],
    }

    low_random_rows = []
    low_random_p = []
    for donor, target in pairs:
        print(f"posthoc {donor} -> {target}", flush=True)
        donor_path = tokenizer_path(config, donor, "bpe", c5.BASE_SIZE)
        donor_tokenizer = load_tokenizer(config, donor, "bpe", c5.BASE_SIZE)
        native = load_tokenizer(config, target, "bpe", c5.BASE_SIZE)
        vocab, merges = load_bpe_components(donor_path)
        entries = annotated_merges(merges)
        pool = c5.pool_documents(target, config, c5.POOL_BYTES)
        features = c5.document_features(pool, donor_tokenizer, native)
        target_test = c5.sample_texts(target, "test", config, 1_000_000)

        ok = features["bytes"] > 0
        corr = float(
            np.corrcoef(
                features["fertility"][ok], np.log(features["bytes"][ok])
            )[0, 1]
        )
        pair = {
            "donor": donor,
            "target": target,
            "pool_documents": len(pool),
            "fertility_log_length_correlation": corr,
            "budgets": {},
        }
        frozen_pair = next(
            item
            for item in c5_results["pairs"]
            if item["donor"] == donor and item["target"] == target
        )
        for budget in args.budgets:
            block = {"requested_vocab_size": budget, "methods": {}, "comparisons": {}}
            frozen_block = frozen_pair["budgets"][str(budget)]
            document_values = {}
            for name, order in (
                ("longest", length_order(features, True)),
                ("shortest", length_order(features, False)),
            ):
                texts = c5.select_texts(pool, order, c5.SELECTION_BYTES)
                tokenizer, row = c5.build_arm(
                    donor_tokenizer,
                    vocab,
                    entries,
                    texts,
                    target_test,
                    max(0, budget - (len(vocab) - len(merges))),
                )
                row["selected_bytes"] = int(
                    sum(len(text.encode("utf-8")) for text in texts)
                )
                row["selected_documents"] = len(texts)
                block["methods"][name] = row
                document_values[name] = document_compressions(
                    tokenizer, target_test
                )

            low_values = frozen_block["document_values"]["low_fertility"]
            random_values = frozen_block["document_values"]["random_mean"]
            low_random = compression_statistics(low_values, random_values)
            low_random["random_mean_level"] = float(np.mean(random_values))
            low_random["relative_gain"] = low_random["delta"] / max(
                1e-12, low_random["random_mean_level"]
            )
            block["comparisons"]["low_fertility_vs_random_mean"] = low_random
            low_random_rows.append(
                {
                    "label": f"{donor}->{target}:{budget}",
                    "delta": low_random["delta"],
                    "relative_gain": low_random["relative_gain"],
                    "p_value": low_random["p_value"],
                }
            )
            low_random_p.append(low_random["p_value"])

            for baseline in ("first", "full_16mb", "longest", "shortest"):
                baseline_values = (
                    document_values[baseline]
                    if baseline in document_values
                    else frozen_block["document_values"][baseline]
                )
                block["comparisons"][f"low_fertility_vs_{baseline}"] = (
                    compression_statistics(low_values, baseline_values)
                )
            pair["budgets"][str(budget)] = block
            for name in ("longest", "shortest"):
                row = block["methods"][name]
                print(
                    f"  {budget} {name}: {row['test_compression']:.4f} "
                    f"reach={row['strict_reachability']:.3f} "
                    f"vocab={row['actual_vocab_size']}",
                    flush=True,
                )
            print(
                f"  {budget} low-vs-random: delta={low_random['delta']:.5f} "
                f"rel={low_random['relative_gain']:.4f} "
                f"p={low_random['p_value']}",
                flush=True,
            )
        payload["pairs"].append(pair)

    adjusted = holm_adjust(low_random_p)
    for row, p_adjusted in zip(low_random_rows, adjusted, strict=True):
        row["p_adjusted"] = p_adjusted
    payload["low_fertility_vs_random_mean"] = {
        "rows": low_random_rows,
        "positive": sum(
            1
            for row in low_random_rows
            if row["delta"] > 0 and row["p_adjusted"] < 0.05
        ),
        "negative": sum(
            1
            for row in low_random_rows
            if row["delta"] < 0 and row["p_adjusted"] < 0.05
        ),
        "mean_relative_gain": float(
            np.mean([row["relative_gain"] for row in low_random_rows])
        ),
    }
    out = results_dir(config) / Path(args.out).name
    write_json(out, payload)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
