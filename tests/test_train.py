from __future__ import annotations

import math

import torch
import torch.nn as nn

from topobpe.tokenizers import SPECIAL_TOKENS, build_tokenizer, token_bytes
from topobpe.train import evaluate_bpb


class UniformModel(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch, length = token_ids.shape
        return torch.zeros((batch, length, self.vocab_size), dtype=torch.float32)


def test_bpb_uses_nats_per_byte_not_per_token() -> None:
    texts = ["the quick brown fox jumps over the lazy dog"]
    tokenizer, trainer = build_tokenizer("bpe", 512, SPECIAL_TOKENS, 1)
    tokenizer.train_from_iterator(texts, trainer=trainer)
    text = texts[0]
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    vocab_size = tokenizer.get_vocab_size()
    model = UniformModel(vocab_size)
    result = evaluate_bpb(
        model,
        tokenizer,
        text,
        context_length=8,
        vocab_capacity=vocab_size,
        pad_id=0,
        device="cpu",
        bf16=False,
    )
    expected = (len(ids) - 1) * math.log(vocab_size) / (math.log(2.0) * len(text.encode("utf-8")))
    assert abs(result["bpb"] - expected) < 1e-6
    assert result["bytes"] == len(text.encode("utf-8"))


def test_token_bytes_cover_every_utf8_byte() -> None:
    text = "café 水"
    tokenizer, trainer = build_tokenizer("bpe", 512, SPECIAL_TOKENS, 1)
    tokenizer.train_from_iterator([text], trainer=trainer)
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    assert b"".join(token_bytes(tokenizer, token_id) for token_id in ids) == text.encode("utf-8")
