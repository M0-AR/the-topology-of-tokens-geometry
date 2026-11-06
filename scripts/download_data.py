from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from topobpe.cli import main as cli_main


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and cache the public corpora")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    return cli_main(["download", "--config", args.config, *(["--force"] if args.force else [])])


if __name__ == "__main__":
    raise SystemExit(main())
