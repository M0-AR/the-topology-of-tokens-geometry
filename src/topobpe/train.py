from __future__ import annotations

import contextlib
import copy
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .config import artifact_dir, results_dir, set_seed
from .data import corpus_dir, load_split_texts
from .model import TinyGPT
from .tokenizers import load_tokenizer
from .utils import read_json, utc_now, write_json


def _autocast(device: str, enabled: bool):
    if enabled and device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _sample_texts(raw: bytes, count: int, chunk_bytes: int, rng: np.random.Generator) -> list[str]:
    if len(raw) <= chunk_bytes:
        text = raw.decode("utf-8", errors="ignore")
        return [text] * count
    starts = rng.integers(0, len(raw) - chunk_bytes, size=count)
    texts: list[str] = []
    for start in starts:
        start = int(start)
        while start < len(raw) and (raw[start] & 0xC0) == 0x80:
            start += 1
        end = min(len(raw), start + chunk_bytes)
        text = raw[start:end].decode("utf-8", errors="ignore")
        if not text:
            text = raw[:chunk_bytes].decode("utf-8", errors="ignore")
        texts.append(text)
    return texts


def _sample_batch(
    tokenizer: Any,
    raw: bytes,
    batch_size: int,
    byte_chunk: int,
    context_length: int,
    pad_id: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int] | None:
    texts = _sample_texts(raw, batch_size, byte_chunk, rng)
    encodings = tokenizer.encode_batch(texts, add_special_tokens=False)
    all_sequences = [encoding.ids for encoding in encodings]
    pairs = zip(texts, all_sequences, strict=False)
    kept = [(text, sequence) for text, sequence in pairs if len(sequence) >= 2]
    if not kept:
        return None
    bytes_seen = sum(len(text.encode("utf-8")) for text, _ in kept)
    truncated = sum(len(sequence) > context_length for _, sequence in kept)
    sequences = [sequence[:context_length] for _, sequence in kept]
    max_len = max(len(sequence) for sequence in sequences)
    input_ids = np.full((len(sequences), max_len - 1), pad_id, dtype=np.int64)
    target_ids = np.full((len(sequences), max_len - 1), pad_id, dtype=np.int64)
    mask = np.zeros((len(sequences), max_len - 1), dtype=np.float32)
    for index, sequence in enumerate(sequences):
        input_ids[index, : len(sequence) - 1] = sequence[:-1]
        target_ids[index, : len(sequence) - 1] = sequence[1:]
        mask[index, : len(sequence) - 1] = 1.0
    return (
        torch.from_numpy(input_ids),
        torch.from_numpy(target_ids),
        torch.from_numpy(mask),
        bytes_seen,
        truncated,
    )


def _lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return float(step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def evaluate_bpb(
    model: TinyGPT,
    tokenizer: Any,
    test_text: str,
    context_length: int,
    vocab_capacity: int,
    pad_id: int,
    device: str,
    bf16: bool,
    eval_batch_size: int = 16,
) -> dict[str, float]:
    ids = tokenizer.encode(test_text, add_special_tokens=False).ids
    raw = test_text.encode("utf-8")
    if len(ids) < 2:
        return {"bpb": float("nan"), "tokens": float(len(ids)), "bytes": float(len(raw))}
    chunks = []
    for start in range(0, len(ids) - 1, context_length):
        chunk = ids[start : start + context_length + 1]
        if len(chunk) >= 2:
            chunks.append(chunk)
    total_nats = 0.0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for batch_start in range(0, len(chunks), eval_batch_size):
            batch = chunks[batch_start : batch_start + eval_batch_size]
            max_len = max(len(chunk) for chunk in batch)
            input_ids = np.full((len(batch), max_len - 1), pad_id, dtype=np.int64)
            target_ids = np.full((len(batch), max_len - 1), pad_id, dtype=np.int64)
            mask = np.zeros((len(batch), max_len - 1), dtype=np.float32)
            for index, chunk in enumerate(batch):
                input_ids[index, : len(chunk) - 1] = chunk[:-1]
                target_ids[index, : len(chunk) - 1] = chunk[1:]
                mask[index, : len(chunk) - 1] = 1.0
                total_tokens += len(chunk) - 1
            x = torch.from_numpy(input_ids).to(device)
            y = torch.from_numpy(target_ids).to(device)
            m = torch.from_numpy(mask).to(device)
            with _autocast(device, bf16):
                logits = model(x)
            logits = logits.float()
            loss = F.cross_entropy(
                logits.view(-1, vocab_capacity),
                y.view(-1),
                reduction="none",
            )
            total_nats += float((loss * m.view(-1)).sum().item())
    bpb = total_nats / (math.log(2.0) * max(1, len(raw)))
    return {"bpb": bpb, "tokens": float(total_tokens), "bytes": float(len(raw))}


def _model_config_hash(config: dict[str, Any]) -> str:
    import hashlib
    import json

    payload = json.dumps(config["model"], sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _lm_cache_path(
    config: dict[str, Any],
    corpus: str,
    family: str,
    vocab_size: int,
    seed: int,
    tag: str = "main",
) -> Path:
    return (
        artifact_dir(config)
        / "lm"
        / corpus
        / f"{family}_{vocab_size}_seed{seed}_{tag}_{_model_config_hash(config)}.json"
    )


def train_condition(
    config: dict[str, Any],
    corpus: str,
    family: str,
    vocab_size: int,
    seed: int,
    force: bool = False,
    tag: str = "main",
) -> dict[str, Any]:
    cache_path = _lm_cache_path(config, corpus, family, vocab_size, seed, tag=tag)
    if cache_path.exists() and not force:
        return read_json(cache_path)
    set_seed(seed)
    model_cfg = config["model"]
    tokenizer = load_tokenizer(config, corpus, family, vocab_size)
    special_tokens = list(config["tokenizers"]["special_tokens"])
    vocab_capacity = int(model_cfg["vocab_capacity"]) + len(special_tokens)
    if tokenizer.get_vocab_size() > vocab_capacity:
        raise ValueError(
            f"tokenizer vocab {tokenizer.get_vocab_size()} exceeds capacity {vocab_capacity}"
        )
    pad_id = tokenizer.token_to_id("<pad>")
    if pad_id is None:
        pad_id = 0
    train_texts = load_split_texts(corpus, "train", config=config)
    test_texts = load_split_texts(corpus, "test", config=config)
    train_text = "\n\n".join(train_texts)
    test_text = "\n\n".join(test_texts)
    train_raw = train_text.encode("utf-8")
    test_ids = [tokenizer.encode(text, add_special_tokens=False).ids for text in test_texts]
    test_tokens = sum(len(ids) for ids in test_ids)
    test_bytes = sum(len(text.encode("utf-8")) for text in test_texts)
    test_compression = test_bytes / max(1, test_tokens)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    model = TinyGPT(
        vocab_capacity=vocab_capacity,
        context_length=int(model_cfg["context_length"]),
        d_model=int(model_cfg["d_model"]),
        n_layers=int(model_cfg["n_layers"]),
        n_heads=int(model_cfg["n_heads"]),
        d_ff=int(model_cfg["d_ff"]),
        dropout=float(model_cfg["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_cfg["learning_rate"]),
        weight_decay=float(model_cfg["weight_decay"]),
        betas=(0.9, 0.95),
    )
    steps = int(model_cfg["steps"])
    warmup = int(model_cfg["warmup_steps"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _lr_lambda(step, warmup, steps)
    )
    rng = np.random.default_rng(seed)
    batch_size = int(model_cfg["batch_size"])
    byte_chunk = int(model_cfg["byte_chunk"])
    context_length = int(model_cfg["context_length"])
    bf16 = bool(model_cfg["bf16"])
    grad_clip = float(model_cfg["grad_clip"])
    bytes_seen = 0
    tokens_seen = 0
    truncated_chunks = 0
    loss_curve: list[dict[str, float]] = []
    start_time = time.time()
    model.train()
    for step in range(steps):
        batch = _sample_batch(
            tokenizer,
            train_raw,
            batch_size,
            byte_chunk,
            context_length,
            int(pad_id),
            rng,
        )
        if batch is None:
            continue
        x, y, mask, batch_bytes, truncated = batch
        bytes_seen += batch_bytes
        truncated_chunks += truncated
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)
        with _autocast(device, bf16):
            logits = model(x)
        logits = logits.float()
        loss_values = F.cross_entropy(
            logits.view(-1, vocab_capacity),
            y.view(-1),
            reduction="none",
        )
        loss = (loss_values * mask.view(-1)).sum() / mask.sum().clamp_min(1.0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        tokens_seen += int(mask.sum().item())
        if step % 50 == 0 or step == steps - 1:
            loss_curve.append({"step": float(step), "loss": float(loss.item())})
    if device == "cuda":
        torch.cuda.synchronize()
    wall_time = time.time() - start_time
    evaluation = evaluate_bpb(
        model,
        tokenizer,
        test_text,
        context_length,
        vocab_capacity,
        int(pad_id),
        device,
        bf16,
    )
    metrics: dict[str, Any] = {
        "corpus": corpus,
        "family": family,
        "requested_vocab_size": int(vocab_size),
        "actual_vocab_size": int(tokenizer.get_vocab_size()),
        "seed": int(seed),
        "bpb": evaluation["bpb"],
        "test_compression": float(test_compression),
        "test_bytes": float(test_bytes),
        "test_tokens": float(test_tokens),
        "eval_tokens": evaluation["tokens"],
        "eval_bytes": evaluation["bytes"],
        "steps": int(steps),
        "batch_size": int(batch_size),
        "byte_chunk": int(byte_chunk),
        "context_length": int(context_length),
        "parameter_count": int(model.parameter_count()),
        "vocab_capacity": int(vocab_capacity),
        "bytes_seen": int(bytes_seen),
        "tokens_seen": int(tokens_seen),
        "truncated_chunks": int(truncated_chunks),
        "wall_time_seconds": float(wall_time),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if device == "cuda" else None,
        "loss_curve": loss_curve,
        "created_at": utc_now(),
    }
    write_json(cache_path, metrics)
    return metrics


def run_lm_grid(config: dict[str, Any], force: bool = False) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    families = list(config["model"]["tokenizer_families"])
    vocab_sizes = [int(value) for value in config["tokenizers"]["vocab_sizes"]]
    seeds = [int(value) for value in config["model"]["seeds"]]
    for spec in config["data"]["corpora"]:
        corpus = spec["name"]
        for family in families:
            for vocab_size in vocab_sizes:
                for seed in seeds:
                    rows.append(
                        train_condition(
                            config, corpus, family, vocab_size, seed, force=force
                        )
                    )
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["corpus", "family", "requested_vocab_size", "seed"])
    frame = frame.reset_index(drop=True)
    path = results_dir(config) / "bpb.csv"
    frame.to_csv(path, index=False)
    return frame


def run_market_lm(
    config: dict[str, Any],
    corpora: list[str],
    force: bool = False,
) -> pd.DataFrame:
    market_config = copy.deepcopy(config)
    market_config["model"].update(
        {
            "d_model": 128,
            "n_layers": 4,
            "n_heads": 4,
            "d_ff": 512,
            "context_length": 256,
            "batch_size": 16,
            "byte_chunk": 512,
            "steps": 300,
            "warmup_steps": 20,
            "vocab_capacity": 4096,
            "seeds": [13],
        }
    )
    rows: list[dict[str, Any]] = []
    for corpus in corpora:
        if not (corpus_dir(corpus, config) / "train.jsonl").exists():
            continue
        for vocab_size in [1024, 2048]:
            rows.append(
                train_condition(
                    market_config,
                    corpus,
                    "bpe",
                    vocab_size,
                    13,
                    force=force,
                    tag="market",
                )
            )
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["corpus", "requested_vocab_size"])
    frame = frame.reset_index(drop=True)
    path = results_dir(market_config) / "market_bpb.csv"
    frame.to_csv(path, index=False)
    return frame
