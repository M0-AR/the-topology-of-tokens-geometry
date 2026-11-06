#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
mkdir -p logs
TOPOBPE="${TOPOBPE:-.venv/bin/topobpe}"

"${TOPOBPE}" all --config configs/smoke.yaml --skip-market 2>&1 | tee logs/smoke.log
