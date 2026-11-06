from __future__ import annotations

import json
from dataclasses import dataclass

from tokenizers import Tokenizer, models, trainers

from .merge_selection import MergeEntry, initial_tokens, simulate_bpe

PUA_START = 0xF0000
PUA_END = 0xFFFFD


def pua_char(index: int) -> str:
    code = PUA_START + index
    if code > PUA_END:
        raise ValueError(f"token index {index} exceeds the supplementary private use area")
    return chr(code)


def learn_token_merges(
    tokenizer: Tokenizer,
    texts: list[str],
    n_merges: int,
    min_frequency: int = 1,
) -> tuple[list[tuple[str, str]], dict[str, int]]:
    diagnostics = {
        "unique_tokens": 0,
        "requested_new_merges": n_merges,
        "learned_raw_merges": 0,
        "mapped_merges": 0,
        "unmapped_merges": 0,
        "train_documents": len(texts),
        "train_symbols": 0,
    }
    if n_merges <= 0 or not texts:
        return [], diagnostics

    token_to_char: dict[str, str] = {}
    char_to_tokens: dict[str, tuple[str, ...]] = {}
    token_strings: list[str] = []
    for text in texts:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        symbols: list[str] = []
        for token_id in encoded.ids:
            token = tokenizer.id_to_token(token_id)
            if token is None:
                raise ValueError(f"missing token string for id {token_id}")
            char = token_to_char.get(token)
            if char is None:
                char = pua_char(len(token_to_char))
                token_to_char[token] = char
                char_to_tokens[char] = (token,)
            symbols.append(char)
        token_strings.append("".join(symbols))

    diagnostics["unique_tokens"] = len(token_to_char)
    diagnostics["train_symbols"] = sum(len(value) for value in token_strings)
    trainer = trainers.BpeTrainer(
        vocab_size=len(token_to_char) + n_merges,
        min_frequency=min_frequency,
        show_progress=False,
        special_tokens=[],
        initial_alphabet=list(token_to_char.values()),
        max_token_length=None,
    )
    learned = Tokenizer(models.BPE(unk_token=None))
    learned.pre_tokenizer = None
    learned.decoder = None
    learned.train_from_iterator(token_strings, trainer=trainer)
    payload = json.loads(learned.to_str())
    raw_merges = payload["model"]["merges"]
    diagnostics["learned_raw_merges"] = len(raw_merges)

    expansion = dict(char_to_tokens)
    merges: list[tuple[str, str]] = []
    for left_symbol, right_symbol in raw_merges:
        left_tokens = expansion.get(left_symbol)
        right_tokens = expansion.get(right_symbol)
        if left_tokens is None or right_tokens is None:
            diagnostics["unmapped_merges"] += 1
            continue
        left_token = "".join(left_tokens)
        right_token = "".join(right_tokens)
        merges.append((left_token, right_token))
        expansion[left_symbol + right_symbol] = left_tokens + right_tokens
    diagnostics["mapped_merges"] = len(merges)
    return merges, diagnostics


@dataclass(frozen=True)
class HybridSelection:
    new: list[tuple[str, str]]
    skipped_duplicates: int
    skipped_unmapped: int


def filter_new_merges(
    merges: list[tuple[str, str]],
    existing_tokens: set[str],
    max_new: int,
) -> HybridSelection:
    seen = set(existing_tokens)
    accepted: list[tuple[str, str]] = []
    duplicates = 0
    for left, right in merges:
        if len(accepted) >= max_new:
            break
        parent = left + right
        if parent in seen:
            duplicates += 1
            continue
        seen.add(parent)
        accepted.append((left, right))
    return HybridSelection(
        new=accepted,
        skipped_duplicates=duplicates,
        skipped_unmapped=0,
    )


def build_hybrid_tokenizer(
    base: Tokenizer,
    vocab: dict[str, int],
    donor_entries: list[MergeEntry],
    selected_donor: set[int],
    new_merges: list[tuple[str, str]],
    fallback_merges: list[tuple[str, str]] | None = None,
) -> tuple[Tokenizer, list[MergeEntry]]:
    selected = sorted(
        (entry for entry in donor_entries if entry.rank in selected_donor),
        key=lambda entry: entry.rank,
    )
    next_rank = max((entry.rank for entry in selected), default=0) + 1
    hybrid_entries = list(selected)
    parents = {entry.parent for entry in hybrid_entries}
    for left, right in list(new_merges) + list(fallback_merges or []):
        parent = left + right
        if parent in parents:
            raise ValueError(f"duplicate merge parent: {parent}")
        hybrid_entries.append(
            MergeEntry(rank=next_rank, left=left, right=right, parent=parent)
        )
        parents.add(parent)
        next_rank += 1

    tokenizer = build_tokenizer_from_entries(base, vocab, donor_entries, hybrid_entries)
    return tokenizer, hybrid_entries


def build_tokenizer_from_entries(
    base: Tokenizer,
    vocab: dict[str, int],
    donor_entries: list[MergeEntry],
    entries: list[MergeEntry],
) -> Tokenizer:
    initial = initial_tokens(vocab, donor_entries)
    needed = set(initial)
    for entry in entries:
        needed.add(entry.parent)
        needed.add(entry.left)
        needed.add(entry.right)
    ordered = sorted(needed, key=lambda token: (vocab.get(token, 10**9), token))
    new_vocab = {token: position for position, token in enumerate(ordered)}
    model = models.BPE(
        vocab=new_vocab,
        merges=[(entry.left, entry.right) for entry in entries],
        unk_token="<unk>",
    )
    tokenizer = Tokenizer(model)
    tokenizer.normalizer = base.normalizer
    tokenizer.pre_tokenizer = base.pre_tokenizer
    tokenizer.post_processor = base.post_processor
    tokenizer.decoder = base.decoder
    return tokenizer


def prune_unreachable(
    entries: list[MergeEntry],
) -> tuple[list[MergeEntry], list[MergeEntry]]:
    kept = list(entries)
    removed: list[MergeEntry] = []
    while kept:
        ranks = {(entry.left, entry.right): entry.rank for entry in kept}
        bad = [
            entry
            for entry in kept
            if simulate_bpe(entry.parent, ranks) != [entry.parent]
        ]
        if not bad:
            break
        bad_ranks = {entry.rank for entry in bad}
        removed.extend(bad)
        kept = [entry for entry in kept if entry.rank not in bad_ranks]
    return kept, removed
