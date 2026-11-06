from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import requests
from datasets import load_dataset
from tqdm import tqdm

from .config import artifact_dir, project_root
from .utils import (
    read_json,
    read_jsonl,
    sha256_text,
    stable_hash_split,
    utc_now,
    write_json,
    write_jsonl,
)


def data_root(config: dict[str, Any] | None = None) -> Path:
    relative = "data/processed"
    if config is not None:
        relative = str(config["data"].get("data_dir", relative))
    return project_root() / relative


def corpus_dir(name: str, config: dict[str, Any] | None = None) -> Path:
    return data_root(config) / name


def _dataset_revision(hf_id: str) -> str | None:
    try:
        response = requests.get(f"https://huggingface.co/api/datasets/{hf_id}", timeout=30)
        response.raise_for_status()
        payload = response.json()
        return payload.get("sha")
    except Exception:
        return None


def _doc_id(example: dict[str, Any], index: int, field: str) -> str:
    text = str(example.get(field, ""))
    return f"{index}:{sha256_text(text)[:16]}"


def _eligible(text: str, min_chars: int, max_chars: int) -> bool:
    if not isinstance(text, str):
        return False
    size = len(text)
    return min_chars <= size <= max_chars


def download_corpora(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "created_at": utc_now(),
        "corpora": {},
    }
    for spec in config["data"]["corpora"]:
        name = spec["name"]
        target = corpus_dir(name, config)
        manifest_path = target / "manifest.json"
        if manifest_path.exists() and not force:
            manifest["corpora"][name] = read_json(manifest_path)
            continue
        target.mkdir(parents=True, exist_ok=True)
        config_name = spec.get("hf_config")
        revision = _dataset_revision(spec["hf_id"])
        load_kwargs: dict[str, Any] = {"split": spec["split"], "streaming": True}
        if revision:
            load_kwargs["revision"] = revision
        dataset = load_dataset(
            spec["hf_id"],
            config_name,
            **load_kwargs,
        )
        rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
        total_chars = 0
        seen: set[str] = set()
        wanted = int(config["data"]["docs_per_corpus"])
        progress = tqdm(total=wanted, desc=f"download {name}")
        for index, example in enumerate(dataset):
            text = example.get(spec["field"])
            min_chars = int(config["data"]["min_chars"])
            max_chars = int(config["data"]["max_chars"])
            if not _eligible(str(text), min_chars, max_chars):
                continue
            dedupe_key = sha256_text(str(text))
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            doc_id = _doc_id(example, index, spec["field"])
            split = stable_hash_split(
                doc_id,
                float(config["data"]["train_fraction"]),
                float(config["data"]["val_fraction"]),
            )
            rows_by_split[split].append(
                {
                    "id": doc_id,
                    "text": str(text),
                    "source": spec["hf_id"],
                    "config": config_name,
                    "language": spec.get("language"),
                    "domain": spec.get("domain"),
                    "split": split,
                }
            )
            total_chars += len(str(text))
            progress.update(1)
            if sum(len(rows) for rows in rows_by_split.values()) >= wanted:
                break
        progress.close()
        file_hashes = {}
        for split, rows in rows_by_split.items():
            path = target / f"{split}.jsonl"
            write_jsonl(path, rows)
            file_hashes[split] = {
                "path": str(path.relative_to(project_root())),
                "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
                "documents": len(rows),
                "characters": sum(len(row["text"]) for row in rows),
            }
        manifest["corpora"][name] = {
            "hf_id": spec["hf_id"],
            "hf_config": config_name,
            "hf_revision": revision,
            "download_timestamp": utc_now(),
            "files": file_hashes,
            "total_characters": total_chars,
        }
        write_json(manifest_path, manifest["corpora"][name])
    write_json(artifact_dir(config) / "data_manifest.json", manifest)
    return manifest


def load_split_texts(
    corpus: str, split: str, config: dict[str, Any] | None = None
) -> list[str]:
    path = corpus_dir(corpus, config) / f"{split}.jsonl"
    return [row["text"] for row in read_jsonl(path)]


def iter_split_texts(
    corpus: str, split: str, config: dict[str, Any] | None = None
) -> Iterator[str]:
    path = corpus_dir(corpus, config) / f"{split}.jsonl"
    for row in read_jsonl(path):
        yield row["text"]
