from __future__ import annotations

from tokenizers import Tokenizer, models

from topobpe.hybrid import (
    build_hybrid_tokenizer,
    filter_new_merges,
    learn_token_merges,
    prune_unreachable,
    pua_char,
)
from topobpe.merge_selection import (
    annotated_merges,
    build_subset_tokenizer,
    strict_reachability,
)


def _toy():
    vocab = {"a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4}
    merges = [("a", "b"), ("ab", "c")]
    entries = annotated_merges(merges)
    base = Tokenizer(models.BPE(vocab=vocab, merges=merges, unk_token="<unk>"))
    return vocab, merges, entries, base


def test_pua_char_is_distinct_and_in_private_use_area():
    first = pua_char(0)
    second = pua_char(1)
    assert first != second
    assert 0xF0000 <= ord(first) < 0xFFFFE
    assert 0xF0000 <= ord(second) < 0xFFFFE


def test_learn_token_merges_maps_back_to_donor_tokens():
    vocab, _, entries, base = _toy()
    partial = build_subset_tokenizer(base, vocab, entries, {1})
    learned, diagnostics = learn_token_merges(partial, ["abc"], 1)
    assert learned == [("ab", "c")]
    assert diagnostics["mapped_merges"] == 1
    assert diagnostics["unmapped_merges"] == 0


def test_filter_new_merges_skips_existing_and_duplicate_parents():
    selection = filter_new_merges([("ab", "c"), ("ab", "c")], {"ab"}, 2)
    assert selection.new == [("ab", "c")]
    assert selection.skipped_duplicates == 1
    blocked = filter_new_merges([("ab", "c")], {"abc"}, 1)
    assert blocked.new == []
    assert blocked.skipped_duplicates == 1


def test_build_hybrid_tokenizer_is_reachable_and_roundtrips():
    vocab, _, entries, base = _toy()
    tokenizer, hybrid_entries = build_hybrid_tokenizer(
        base, vocab, entries, {1}, [("ab", "c")]
    )
    encoded = tokenizer.encode("abc", add_special_tokens=False)
    assert len(encoded.ids) == 1
    assert tokenizer.decode(encoded.ids) == "abc"
    selected = {entry.rank for entry in hybrid_entries}
    assert strict_reachability(hybrid_entries, selected, vocab) == 1.0


def test_prune_unreachable_removes_invalid_merge():
    entries = annotated_merges([("a", "b"), ("a", "bc"), ("b", "c")])
    kept, removed = prune_unreachable(entries)
    assert [entry.rank for entry in removed] == [2]
    assert [entry.rank for entry in kept] == [1, 3]
