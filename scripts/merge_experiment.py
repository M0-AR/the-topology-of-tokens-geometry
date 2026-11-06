from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from topobpe.config import load_config, results_dir
from topobpe.data import load_split_texts
from topobpe.features import _encode_documents, _seed_for, baseline_features
from topobpe.merge_topology import (
    load_bpe_components,
    merge_feature_row,
    truncate_bpe,
)
from topobpe.tokenizers import (
    load_tokenizer,
    special_token_ids,
    tokenizer_path,
)
from topobpe.utils import sha256_file, utc_now, write_json

DEFAULT_SIZES = [512, 1024, 2048, 4096, 8192, 16384, 32768]


def _sample_texts(texts: list[str], max_docs: int, max_bytes: int) -> list[str]:
    chosen: list[str] = []
    total = 0
    for text in texts[:max_docs]:
        encoded = text.encode("utf-8")
        if total + len(encoded) > max_bytes:
            remaining = max_bytes - total
            if remaining <= 0:
                break
            text = encoded[:remaining].decode("utf-8", errors="ignore")
            encoded = text.encode("utf-8")
        chosen.append(text)
        total += len(encoded)
        if total >= max_bytes:
            break
    return chosen


def _row(
    config: dict,
    corpus: str,
    target_vocab_size: int,
    variant: str,
    base_vocab_size: int,
    tokenizer,
    merges: list[tuple[str, str]],
    train_texts: list[str],
    test_texts: list[str],
    base_sha256: str,
) -> dict:
    train_ids = _encode_documents(tokenizer, train_texts)
    test_ids = _encode_documents(tokenizer, test_texts)
    excluded = special_token_ids(tokenizer, list(config["tokenizers"]["special_tokens"]))
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is not None:
        excluded.add(int(eos_id))
    seed = _seed_for(corpus, "bpe", str(target_vocab_size), variant, "merge")
    thresholds = [float(value) for value in config["features"]["persistence_thresholds"]]
    features = {
        "corpus": corpus,
        "family": "bpe",
        "requested_vocab_size": int(target_vocab_size),
        "variant": variant,
        "base_vocab_size": int(base_vocab_size),
        "base_tokenizer_sha256": base_sha256,
        "sample_train_documents": len(train_texts),
        "sample_test_documents": len(test_texts),
    }
    features.update(
        baseline_features(train_texts, test_texts, tokenizer, train_ids, test_ids)
    )
    features.update(
        merge_feature_row(
            tokenizer,
            merges,
            train_ids,
            excluded,
            seed,
            thresholds,
            n_points=int(config["features"]["vocab_points"]),
            maxdim=int(config["features"]["maxdim"]),
        )
    )
    return features


def run_corpus(
    config: dict,
    corpus: str,
    sizes: list[int],
    max_docs: int,
    train_bytes: int,
    test_bytes: int,
) -> pd.DataFrame:
    base_size = max(sizes)
    base_path = tokenizer_path(config, corpus, "bpe", base_size)
    if not base_path.exists():
        raise FileNotFoundError(f"missing base tokenizer: {base_path}")
    base = load_tokenizer(config, corpus, "bpe", base_size)
    vocab, merges = load_bpe_components(base_path)
    base_sha = sha256_file(base_path)
    train_texts = _sample_texts(
        load_split_texts(corpus, "train", config=config), max_docs, train_bytes
    )
    test_texts = _sample_texts(
        load_split_texts(corpus, "test", config=config), max_docs, test_bytes
    )
    rows: list[dict] = []
    for target in sizes:
        if target == base_size:
            rows.append(
                _row(
                    config,
                    corpus,
                    target,
                    "base",
                    base_size,
                    base,
                    merges,
                    train_texts,
                    test_texts,
                    base_sha,
                )
            )
            continue
        truncated, kept = truncate_bpe(base, vocab, merges, target)
        rows.append(
            _row(
                config,
                corpus,
                target,
                "truncated",
                base_size,
                truncated,
                kept,
                train_texts,
                test_texts,
                base_sha,
            )
        )
        native_path = tokenizer_path(config, corpus, "bpe", target)
        if native_path.exists():
            native = load_tokenizer(config, corpus, "bpe", target)
            native_vocab, native_merges = load_bpe_components(native_path)
            rows.append(
                _row(
                    config,
                    corpus,
                    target,
                    "native",
                    base_size,
                    native,
                    native_merges,
                    train_texts,
                    test_texts,
                    sha256_file(native_path),
                )
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="BPE merge truncation and topology experiment")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--corpora", nargs="*", default=None)
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--max-docs", type=int, default=2000)
    parser.add_argument("--train-bytes", type=int, default=4_000_000)
    parser.add_argument("--test-bytes", type=int, default=1_000_000)
    args = parser.parse_args()
    config = load_config(args.config)
    corpora = args.corpora or [spec["name"] for spec in config["data"]["corpora"]]
    out_dir = results_dir(config)
    manifest = {
        "created_at": utc_now(),
        "sizes": args.sizes,
        "max_docs": args.max_docs,
        "train_bytes": args.train_bytes,
        "test_bytes": args.test_bytes,
        "corpora": {},
    }
    for corpus in corpora:
        print(f"merge experiment {corpus}", flush=True)
        frame = run_corpus(
            config,
            corpus,
            sorted(args.sizes),
            args.max_docs,
            args.train_bytes,
            args.test_bytes,
        )
        path = out_dir / f"merge_features_{corpus}.csv"
        frame.to_csv(path, index=False)
        manifest["corpora"][corpus] = {
            "rows": int(len(frame)),
            "sizes": sorted(frame["requested_vocab_size"].unique().tolist()),
            "variants": sorted(frame["variant"].unique().tolist()),
            "path": str(path.relative_to(out_dir.parent)),
        }
        print(f"  wrote {path} ({len(frame)} rows)", flush=True)
    write_json(out_dir / "merge_experiment_manifest.json", manifest)
    frames = [pd.read_csv(out_dir / f"merge_features_{corpus}.csv") for corpus in corpora]
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out_dir / "merge_features.csv", index=False)
    print(f"wrote {out_dir / 'merge_features.csv'} ({len(combined)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
