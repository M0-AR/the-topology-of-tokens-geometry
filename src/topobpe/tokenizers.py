from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .config import artifact_dir
from .data import load_split_texts
from .utils import sha256_file, sha256_text, utc_now, write_json

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def tokenizer_dir(config: dict[str, Any]) -> Path:
    path = artifact_dir(config) / "tokenizers"
    path.mkdir(parents=True, exist_ok=True)
    return path


def tokenizer_path(config: dict[str, Any], corpus: str, family: str, vocab_size: int) -> Path:
    return tokenizer_dir(config) / corpus / f"{family}_{vocab_size}.json"


def _sample_documents(texts: list[str], max_bytes: int) -> tuple[list[str], int]:
    chosen: list[str] = []
    total = 0
    for text in texts:
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
    return chosen, total


def build_tokenizer(
    family: str,
    vocab_size: int,
    special_tokens: list[str],
    min_frequency: int,
) -> tuple[Tokenizer, Any]:
    if family == "bpe":
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        return tokenizer, trainer
    if family == "unigram":
        tokenizer = Tokenizer(models.Unigram())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            unk_token="<unk>",
            show_progress=False,
        )
        return tokenizer, trainer
    raise ValueError(f"unknown tokenizer family: {family}")


def train_tokenizer(
    config: dict[str, Any],
    corpus: str,
    family: str,
    vocab_size: int,
    force: bool = False,
) -> tuple[Tokenizer, dict[str, Any]]:
    path = tokenizer_path(config, corpus, family, vocab_size)
    meta_path = path.with_suffix(".meta.json")
    if path.exists() and meta_path.exists() and not force:
        return Tokenizer.from_file(str(path)), json.loads(meta_path.read_text(encoding="utf-8"))
    texts = load_split_texts(corpus, "train", config=config)
    if not texts:
        raise RuntimeError(f"no training documents for corpus {corpus}")
    sample, sample_bytes = _sample_documents(
        texts, int(config["tokenizers"]["sample_bytes_for_training"])
    )
    tokenizer, trainer = build_tokenizer(
        family,
        vocab_size,
        list(config["tokenizers"]["special_tokens"]),
        int(config["tokenizers"]["min_frequency"]),
    )
    tokenizer.train_from_iterator(sample, trainer=trainer)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))
    meta = {
        "corpus": corpus,
        "family": family,
        "requested_vocab_size": vocab_size,
        "actual_vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": list(config["tokenizers"]["special_tokens"]),
        "min_frequency": int(config["tokenizers"]["min_frequency"]),
        "sample_documents": len(sample),
        "sample_bytes": sample_bytes,
        "created_at": utc_now(),
        "tokenizer_sha256": sha256_file(path),
        "format": "tokenizers.Tokenizer",
    }
    write_json(meta_path, meta)
    return tokenizer, meta


def load_tokenizer(config: dict[str, Any], corpus: str, family: str, vocab_size: int) -> Tokenizer:
    path = tokenizer_path(config, corpus, family, vocab_size)
    if not path.exists():
        return train_tokenizer(config, corpus, family, vocab_size)[0]
    return Tokenizer.from_file(str(path))


def encode_text(tokenizer: Tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False).ids


def encode_batch(tokenizer: Tokenizer, texts: list[str]) -> list[list[int]]:
    return [encoding.ids for encoding in tokenizer.encode_batch(texts, add_special_tokens=False)]


def _bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for byte in range(256):
        if byte not in bs:
            bs.append(byte)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs), strict=True))


_BYTE_TO_UNICODE = _bytes_to_unicode()
_UNICODE_TO_BYTE = {char: byte for byte, char in _BYTE_TO_UNICODE.items()}


def token_bytes(tokenizer: Tokenizer, token_id: int) -> bytes:
    token = tokenizer.id_to_token(token_id)
    if token is None:
        return b""
    out = bytearray()
    for char in token:
        byte = _UNICODE_TO_BYTE.get(char)
        if byte is None:
            out.extend(char.encode("utf-8"))
        else:
            out.append(byte)
    return bytes(out)


def token_byte_lengths(tokenizer: Tokenizer) -> np.ndarray:
    size = tokenizer.get_vocab_size()
    return np.asarray([len(token_bytes(tokenizer, index)) for index in range(size)], dtype=np.int32)


def special_token_ids(tokenizer: Tokenizer, special_tokens: list[str]) -> set[int]:
    ids: set[int] = set()
    for token in special_tokens:
        token_id = tokenizer.token_to_id(token)
        if token_id is not None:
            ids.add(int(token_id))
    return ids


def tokenizer_fingerprint(tokenizer: Tokenizer) -> str:
    pieces = [tokenizer.id_to_token(index) or "" for index in range(tokenizer.get_vocab_size())]
    return sha256_text("\n".join(pieces))
