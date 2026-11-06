from __future__ import annotations

import hashlib
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .config import artifact_dir, project_root, results_dir
from .data import corpus_dir
from .features import extract_features
from .tokenizers import train_tokenizer
from .utils import read_json, sha256_file, utc_now, write_json, write_jsonl


def market_raw_dir(config: dict[str, Any]) -> Path:
    path = project_root() / config["market"]["cache_dir"] / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path


def market_corpus_name(ticker: str, range_: str) -> str:
    return f"market_{ticker.lower()}_{range_}"


def market_corpus_names(config: dict[str, Any], range_: str = "10y") -> list[str]:
    return [market_corpus_name(ticker, range_) for ticker in config["market"]["tickers"]]


def _request_chart(
    config: dict[str, Any],
    ticker: str,
    range_: str,
    interval: str,
) -> dict[str, Any]:
    endpoint = str(config["market"]["endpoint"])
    fallback = endpoint.replace("query1.", "query2.")
    params = {
        "range": range_,
        "interval": interval,
        "includePrePost": "false",
        "events": "div,splits",
    }
    headers = {"User-Agent": str(config["market"]["user_agent"]), "Accept": "application/json"}
    last_error: Exception | None = None
    for base in (endpoint, fallback):
        for attempt in range(int(config["market"]["max_retries"])):
            try:
                response = requests.get(base, params=params, headers=headers, timeout=30)
                if response.status_code == 200:
                    payload = response.json()
                    if payload.get("chart", {}).get("error"):
                        raise RuntimeError(str(payload["chart"]["error"]))
                    return payload
                if response.status_code == 429:
                    time.sleep(5 if attempt == 0 else 10)
                else:
                    time.sleep(float(config["market"]["sleep_seconds"]))
                last_error = RuntimeError(f"HTTP {response.status_code}")
            except Exception as error:
                last_error = error
                time.sleep(float(config["market"]["sleep_seconds"]))
    raise RuntimeError(f"failed to fetch {ticker} {range_} {interval}: {last_error}")


def fetch_raw(
    config: dict[str, Any],
    ticker: str,
    range_: str,
    interval: str,
    force: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = market_raw_dir(config) / f"{ticker}_{range_}_{interval}.json"
    if path.exists() and not force:
        payload = read_json(path)
        return payload, {
            "path": str(path.relative_to(project_root())),
            "sha256": sha256_file(path),
            "cached": True,
        }
    payload = _request_chart(config, ticker, range_, interval)
    write_json(path, payload)
    return payload, {
        "path": str(path.relative_to(project_root())),
        "sha256": sha256_file(path),
        "cached": False,
        "accessed_at": utc_now(),
        "url": str(config["market"]["endpoint"]),
        "ticker": ticker,
        "range": range_,
        "interval": interval,
    }


def _number(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace("$", "").replace(",", "").strip()
    return float(text) if text else 0.0


def fetch_nasdaq_daily(
    config: dict[str, Any],
    ticker: str,
    force: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = market_raw_dir(config) / f"nasdaq_{ticker}_10y_1d.json"
    if path.exists() and not force:
        return read_json(path), {
            "path": str(path.relative_to(project_root())),
            "sha256": sha256_file(path),
            "cached": True,
            "provider": "nasdaq",
        }
    base = str(config["market"]["nasdaq_endpoint"]).rstrip("/")
    url = f"{base}/{ticker}/chart"
    params = {"assetclass": "stocks", "fromdate": "2016-01-01", "todate": "2026-11-05"}
    headers = {"User-Agent": str(config["market"]["user_agent"]), "Accept": "application/json"}
    last_error: Exception | None = None
    for _ in range(int(config["market"]["max_retries"])):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            payload = response.json()
            if not payload.get("data", {}).get("chart"):
                raise RuntimeError(f"empty Nasdaq chart for {ticker}")
            write_json(path, payload)
            return payload, {
                "path": str(path.relative_to(project_root())),
                "sha256": sha256_file(path),
                "cached": False,
                "accessed_at": utc_now(),
                "url": url,
                "provider": "nasdaq",
                "ticker": ticker,
                "range": "10y",
                "interval": "1d",
            }
        except Exception as error:
            last_error = error
            time.sleep(float(config["market"]["sleep_seconds"]))
    raise RuntimeError(f"failed to fetch Nasdaq daily data for {ticker}: {last_error}")


def parse_nasdaq_chart(payload: dict[str, Any]) -> list[dict[str, Any]]:
    points = payload["data"]["chart"]
    rows: list[dict[str, Any]] = []
    for point in points:
        values = point.get("z", {})
        close = _number(values.get("close", 0))
        if close <= 0:
            continue
        rows.append(
            {
                "timestamp": int(point["x"]) // 1000,
                "open": _number(values.get("open", 0)),
                "high": _number(values.get("high", 0)),
                "low": _number(values.get("low", 0)),
                "close": close,
                "volume": _number(values.get("volume", 0)),
            }
        )
    rows.sort(key=lambda row: row["timestamp"])
    return rows


def parse_chart(payload: dict[str, Any]) -> list[dict[str, Any]]:
    result = payload["chart"]["result"][0]
    timestamps = result.get("timestamp") or []
    quote = result["indicators"]["quote"][0]
    opens = quote.get("open") or []
    highs = quote.get("high") or []
    lows = quote.get("low") or []
    closes = quote.get("close") or []
    volumes = quote.get("volume") or []
    rows: list[dict[str, Any]] = []
    for index, timestamp in enumerate(timestamps):
        if index >= min(len(opens), len(highs), len(lows), len(closes)):
            break
        open_, high, low, close = opens[index], highs[index], lows[index], closes[index]
        if None in (open_, high, low, close) or timestamp is None:
            continue
        volume = volumes[index] if index < len(volumes) and volumes[index] is not None else 0
        rows.append(
            {
                "timestamp": int(timestamp),
                "open": float(open_),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "volume": float(volume),
            }
        )
    rows.sort(key=lambda row: row["timestamp"])
    return rows


def _record_text(ticker: str, row: dict[str, Any], log_return: float, interval: str) -> str:
    moment = datetime.fromtimestamp(row["timestamp"], UTC)
    stamp = moment.strftime("%Y-%m-%d") if interval == "1d" else moment.isoformat()
    return (
        f"{ticker}|{stamp}|{row['open']:.6f}|{row['high']:.6f}|{row['low']:.6f}|"
        f"{row['close']:.6f}|{int(row['volume'])}|{log_return:.6f}"
    )


def _split_rows(
    rows: list[dict[str, Any]],
    range_: str,
    interval: str,
    train_end: str,
    val_end: str,
) -> dict[str, list[dict[str, Any]]]:
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    if range_ == "10y" and interval == "1d":
        for row in rows:
            date = datetime.fromtimestamp(row["timestamp"], UTC).strftime("%Y-%m-%d")
            if date <= train_end:
                splits["train"].append(row)
            elif date <= val_end:
                splits["val"].append(row)
            else:
                splits["test"].append(row)
        if all(splits.values()):
            return splits
    count = len(rows)
    train_stop = int(count * 0.6)
    val_stop = int(count * 0.8)
    splits = {
        "train": rows[:train_stop],
        "val": rows[train_stop:val_stop],
        "test": rows[val_stop:],
    }
    return splits


def build_market_corpus(
    config: dict[str, Any],
    ticker: str,
    range_: str,
    interval: str,
    force: bool = False,
) -> dict[str, Any]:
    name = market_corpus_name(ticker, range_)
    target = corpus_dir(name, config)
    manifest_path = target / "manifest.json"
    if target.exists() and manifest_path.exists() and not force:
        return read_json(manifest_path)
    try:
        payload, source = fetch_raw(config, ticker, range_, interval, force=force)
        bars = parse_chart(payload)
    except Exception:
        if range_ != "10y" or interval != "1d":
            raise
        payload, source = fetch_nasdaq_daily(config, ticker, force=force)
        bars = parse_nasdaq_chart(payload)
    if len(bars) < 30:
        raise RuntimeError(f"too few bars for {ticker} {range_}: {len(bars)}")
    previous_close = None
    records: list[dict[str, Any]] = []
    for row in bars:
        log_return = 0.0
        if previous_close is not None and previous_close > 0 and row["close"] > 0:
            log_return = math.log(row["close"] / previous_close)
        records.append(
            {
                "id": f"{ticker}:{row['timestamp']}",
                "text": _record_text(ticker, row, log_return, interval),
                "timestamp": row["timestamp"],
                "ticker": ticker,
                "range": range_,
                "interval": interval,
            }
        )
        previous_close = row["close"]
    splits = _split_rows(
        records,
        range_,
        interval,
        str(config["market"]["train_end"]),
        str(config["market"]["val_end"]),
    )
    target.mkdir(parents=True, exist_ok=True)
    file_manifest: dict[str, Any] = {}
    for split, split_rows in splits.items():
        path = target / f"{split}.jsonl"
        write_jsonl(path, split_rows)
        file_manifest[split] = {
            "path": str(path.relative_to(project_root())),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "records": len(split_rows),
            "characters": sum(len(row["text"]) for row in split_rows),
        }
    manifest = {
        "corpus": name,
        "ticker": ticker,
        "range": range_,
        "interval": interval,
        "source": source,
        "created_at": utc_now(),
        "bars": len(bars),
        "splits": file_manifest,
        "train_end": str(config["market"]["train_end"]),
        "val_end": str(config["market"]["val_end"]),
        "split_policy": (
            "fixed dates for 10y daily; chronological 60/20/20 otherwise"
        ),
    }
    write_json(manifest_path, manifest)
    return manifest


def build_market_data(config: dict[str, Any], force: bool = False) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    failures: dict[tuple[str, str], int] = {}
    for ticker in config["market"]["tickers"]:
        for spec in config["market"]["ranges"]:
            range_ = str(spec["range"])
            interval = str(spec["interval"])
            key = (range_, interval)
            if failures.get(key, 0) >= 2:
                manifests.append(
                    {
                        "corpus": market_corpus_name(ticker, range_),
                        "ticker": ticker,
                        "range": range_,
                        "interval": interval,
                        "error": (
                            "skipped after repeated provider unavailability for this "
                            "window; excluded from analysis"
                        ),
                        "created_at": utc_now(),
                    }
                )
                continue
            try:
                manifests.append(
                    build_market_corpus(config, ticker, range_, interval, force=force)
                )
            except Exception as error:
                failures[key] = failures.get(key, 0) + 1
                manifests.append(
                    {
                        "corpus": market_corpus_name(ticker, range_),
                        "ticker": ticker,
                        "range": range_,
                        "interval": interval,
                        "error": str(error),
                        "created_at": utc_now(),
                    }
                )
    write_json(artifact_dir(config) / "market_manifest.json", {"corpora": manifests})
    return manifests


def build_market_features(
    config: dict[str, Any],
    ranges: tuple[str, ...] | None = None,
    vocab_sizes: tuple[int, ...] = (1024, 2048),
    force: bool = False,
) -> pd.DataFrame:
    if ranges is None:
        ranges = tuple(str(spec["range"]) for spec in config["market"]["ranges"])
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for range_ in ranges:
        for corpus in market_corpus_names(config, range_):
            if not (corpus_dir(corpus, config) / "train.jsonl").exists():
                skipped.append(corpus)
                continue
            for vocab_size in vocab_sizes:
                train_tokenizer(config, corpus, "bpe", vocab_size, force=force)
                rows.append(extract_features(config, corpus, "bpe", vocab_size, force=force))
    write_json(
        artifact_dir(config) / "market_feature_manifest.json",
        {"rows": len(rows), "skipped": skipped, "ranges": list(ranges)},
    )
    frame = pd.DataFrame(rows).sort_values(["corpus", "requested_vocab_size"])
    frame.reset_index(drop=True).to_csv(results_dir(config) / "market_features.csv", index=False)
    return frame
