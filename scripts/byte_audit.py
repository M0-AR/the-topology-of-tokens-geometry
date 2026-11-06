from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from topobpe.config import load_config, results_dir
from topobpe.data import load_split_texts
from topobpe.tokenizers import load_tokenizer, token_bytes


def main() -> int:
    parser = argparse.ArgumentParser(description="Byte-reversibility audit for every tokenizer")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    config = load_config(args.config)
    rows = []
    for spec in config["data"]["corpora"]:
        corpus = spec["name"]
        texts = load_split_texts(corpus, args.split, config=config)
        raw = [text.encode("utf-8") for text in texts]
        for family in config["tokenizers"]["families"]:
            for vocab_size in config["tokenizers"]["vocab_sizes"]:
                tokenizer = load_tokenizer(config, corpus, family, int(vocab_size))
                unk_id = tokenizer.token_to_id("<unk>")
                mismatches = 0
                unknown = 0
                tokens = 0
                for text, expected in zip(texts, raw, strict=True):
                    ids = tokenizer.encode(text, add_special_tokens=False).ids
                    tokens += len(ids)
                    if unk_id is not None:
                        unknown += sum(1 for token_id in ids if token_id == unk_id)
                    reconstructed = b"".join(token_bytes(tokenizer, token_id) for token_id in ids)
                    if reconstructed != expected:
                        mismatches += 1
                rows.append(
                    {
                        "corpus": corpus,
                        "family": family,
                        "requested_vocab_size": int(vocab_size),
                        "actual_vocab_size": tokenizer.get_vocab_size(),
                        "documents": len(texts),
                        "tokens": tokens,
                        "unknown_tokens": unknown,
                        "reconstruction_mismatches": mismatches,
                        "byte_reversible": mismatches == 0,
                    }
                )
    frame = pd.DataFrame(rows)
    tables = results_dir(config) / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    frame.to_csv(tables / "tokenizer_byte_audit.csv", index=False)
    print(frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
