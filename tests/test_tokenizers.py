from __future__ import annotations

from topobpe.tokenizers import SPECIAL_TOKENS, build_tokenizer, token_bytes


def _fit(family: str, texts: list[str]):
    tokenizer, trainer = build_tokenizer(family, 512, SPECIAL_TOKENS, 1)
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer


def test_byte_accounting_round_trips_for_bpe_and_unigram() -> None:
    texts = [
        "the quick brown fox jumps over the lazy dog",
        "unigram tokenizers should preserve every byte",
        "水を飲むと元気になる",
        "café naïve résumé",
    ]
    for family in ("bpe", "unigram"):
        tokenizer = _fit(family, texts)
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            reconstructed = b"".join(token_bytes(tokenizer, token_id) for token_id in ids)
            assert reconstructed == text.encode("utf-8")
