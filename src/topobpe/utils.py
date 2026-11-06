from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def environment_record() -> dict[str, Any]:
    record: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    try:
        import torch

        record["torch"] = torch.__version__
        record["cuda_available"] = torch.cuda.is_available()
        record["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            record["gpu"] = torch.cuda.get_device_name(0)
            record["gpu_total_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 3
            )
    except ImportError:
        record["torch"] = None
    for name in [
        "datasets",
        "tokenizers",
        "transformers",
        "numpy",
        "scipy",
        "sklearn",
        "pandas",
        "matplotlib",
        "statsmodels",
        "ripser",
        "gudhi",
        "persim",
    ]:
        try:
            module = __import__(name)
            record[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            record[name] = None
    return record


def stable_hash_split(key: str, train_fraction: float, val_fraction: float) -> str:
    value = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if value < train_fraction:
        return "train"
    if value < train_fraction + val_fraction:
        return "val"
    return "test"


def utf8_length(text: str) -> int:
    return len(text.encode("utf-8"))


def byte_entropy(text: str | bytes) -> float:
    raw = text.encode("utf-8") if isinstance(text, str) else text
    values = np.frombuffer(raw, dtype=np.uint8)
    counts = np.bincount(values, minlength=256).astype(np.float64)
    probs = counts[counts > 0] / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def token_entropy(ids: Iterable[int] | np.ndarray) -> float:
    values = np.asarray(ids, dtype=np.int64)
    if values.size == 0:
        return 0.0
    counts = np.bincount(values).astype(np.float64)
    probs = counts[counts > 0] / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def conditional_entropy(
    ids: Iterable[int] | np.ndarray,
    max_pairs: int = 1_000_000,
    seed: int = 0,
) -> float:
    values = np.asarray(ids, dtype=np.int64)
    if values.size < 2:
        return 0.0
    if values.size - 1 > max_pairs:
        rng = np.random.default_rng(seed)
        starts = rng.integers(0, values.size - 1, size=max_pairs)
        contexts = values[starts]
        nexts = values[starts + 1]
    else:
        contexts = values[:-1]
        nexts = values[1:]
    vocab_size = int(values.max()) + 1
    joint_ids = contexts * vocab_size + nexts
    unique_joint, joint_counts = np.unique(joint_ids, return_counts=True)
    total = float(joint_counts.sum())
    context_counts = np.bincount(contexts, minlength=vocab_size).astype(np.float64)
    joint_prob = joint_counts.astype(np.float64) / total
    joint_context = unique_joint // vocab_size
    conditional_prob = joint_counts.astype(np.float64) / context_counts[joint_context]
    return float(-(joint_prob * np.log2(conditional_prob)).sum())
