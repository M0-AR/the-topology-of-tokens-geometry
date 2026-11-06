from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Any

from .config import artifact_dir, load_config, project_root, results_dir
from .data import download_corpora
from .features import build_feature_table
from .market import build_market_data, build_market_features, market_corpus_names
from .plots import make_all_figures
from .stats import run_analysis
from .tokenizers import train_tokenizer
from .train import run_lm_grid, run_market_lm
from .utils import environment_record, read_json, sha256_file, utc_now, write_json


def _public_tokenizers(config: dict[str, Any], force: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in config["data"]["corpora"]:
        for family in config["tokenizers"]["families"]:
            for vocab_size in config["tokenizers"]["vocab_sizes"]:
                _, meta = train_tokenizer(
                    config, spec["name"], family, int(vocab_size), force=force
                )
                rows.append(meta)
    write_json(artifact_dir(config) / "tokenizer_manifest.json", {"tokenizers": rows})
    return rows


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root(), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _collect_hashes(config: dict[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    manifest_path = artifact_dir(config) / "manifest.json"
    for base in (artifact_dir(config), results_dir(config)):
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path == manifest_path:
                continue
            if path.suffix in {".json", ".csv", ".pdf", ".png"}:
                hashes[str(path.relative_to(project_root()))] = sha256_file(path)
    return hashes


def build_manifest(config: dict[str, Any]) -> dict[str, Any]:
    artifacts = artifact_dir(config)
    payload: dict[str, Any] = {
        "created_at": utc_now(),
        "environment": environment_record(),
        "config_path": config.get("_config_path"),
        "config_sha256": sha256_file(config["_config_path"]),
        "config": {key: value for key, value in config.items() if key != "_config_path"},
        "git_commit": _git_commit(),
        "files": {},
    }
    for name in ("data_manifest.json", "market_manifest.json", "tokenizer_manifest.json"):
        path = artifacts / name
        if path.exists():
            payload[name.removesuffix(".json")] = read_json(path)
    payload["files"] = _collect_hashes(config)
    write_json(artifacts / "manifest.json", payload)
    return payload


def _cmd_download(args: argparse.Namespace) -> None:
    download_corpora(args.config_data, force=args.force)


def _cmd_tokenizers(args: argparse.Namespace) -> None:
    _public_tokenizers(args.config_data, force=args.force)


def _cmd_features(args: argparse.Namespace) -> None:
    build_feature_table(args.config_data, force=args.force)


def _cmd_lm(args: argparse.Namespace) -> None:
    run_lm_grid(args.config_data, force=args.force)


def _cmd_market_data(args: argparse.Namespace) -> None:
    build_market_data(args.config_data, force=args.force)


def _cmd_market_features(args: argparse.Namespace) -> None:
    build_market_features(args.config_data, force=args.force)


def _cmd_market_lm(args: argparse.Namespace) -> None:
    run_market_lm(args.config_data, market_corpus_names(args.config_data, "10y"), force=args.force)


def _cmd_analysis(args: argparse.Namespace) -> None:
    run_analysis(args.config_data)


def _cmd_figures(args: argparse.Namespace) -> None:
    make_all_figures(args.config_data)


def _cmd_manifest(args: argparse.Namespace) -> None:
    build_manifest(args.config_data)


def _cmd_all(args: argparse.Namespace) -> None:
    config = args.config_data
    download_corpora(config, force=args.force)
    _public_tokenizers(config, force=args.force)
    build_feature_table(config, force=args.force)
    if not args.skip_lm:
        run_lm_grid(config, force=args.force)
    if config["market"]["enabled"] and not args.skip_market:
        build_market_data(config, force=args.force)
        build_market_features(config, force=args.force)
        run_market_lm(config, market_corpus_names(config, "10y"), force=args.force)
    run_analysis(config)
    make_all_figures(config)
    build_manifest(config)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", default="configs/default.yaml", help="path to a YAML experiment config"
    )
    parser.add_argument("--force", action="store_true", help="recompute cached artifacts")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="topobpe", description="TopoBPE experiment pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    commands = {
        "download": _cmd_download,
        "tokenizers": _cmd_tokenizers,
        "features": _cmd_features,
        "lm": _cmd_lm,
        "market-data": _cmd_market_data,
        "market-features": _cmd_market_features,
        "market-lm": _cmd_market_lm,
        "analysis": _cmd_analysis,
        "figures": _cmd_figures,
        "manifest": _cmd_manifest,
        "all": _cmd_all,
    }
    for name, function in commands.items():
        command = sub.add_parser(name)
        _add_common(command)
        if name == "all":
            command.add_argument("--skip-lm", action="store_true", help="skip public BPB training")
            command.add_argument(
                "--skip-market", action="store_true", help="skip live market-data steps"
            )
        command.set_defaults(func=function)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.config_data = load_config(args.config)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
