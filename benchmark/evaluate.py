from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokenizers import Tokenizer

from topobpe.config import load_config
from topobpe.data import load_split_texts


def main() -> int:
    parser = argparse.ArgumentParser(description="Score a tokenizer on a TopoBPE corpus")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    config = load_config(args.config)
    texts = load_split_texts(args.corpus, "test", config=config)
    if args.limit:
        texts = texts[: args.limit]
    tokenizer = Tokenizer.from_file(args.tokenizer)
    tokens = 0
    raw = 0
    for text in texts:
        tokens += len(tokenizer.encode(text, add_special_tokens=False).ids)
        raw += len(text.encode("utf-8"))
    print(
        json.dumps(
            {
                "tokenizer": args.tokenizer,
                "corpus": args.corpus,
                "documents": len(texts),
                "bytes": raw,
                "tokens": tokens,
                "compression": raw / tokens if tokens else None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
