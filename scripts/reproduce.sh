#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
CONFIG="${1:-configs/default.yaml}"
mkdir -p logs
TOPOBPE="${TOPOBPE:-.venv/bin/topobpe}"

"${TOPOBPE}" download --config "${CONFIG}" 2>&1 | tee logs/01_download.log
"${TOPOBPE}" tokenizers --config "${CONFIG}" 2>&1 | tee logs/02_tokenizers.log
"${TOPOBPE}" features --config "${CONFIG}" 2>&1 | tee logs/03_features.log
"${TOPOBPE}" lm --config "${CONFIG}" 2>&1 | tee logs/04_lm.log
"${TOPOBPE}" market-data --config "${CONFIG}" 2>&1 | tee logs/05_market_data.log
"${TOPOBPE}" market-features --config "${CONFIG}" 2>&1 | tee logs/06_market_features.log
"${TOPOBPE}" market-lm --config "${CONFIG}" 2>&1 | tee logs/07_market_lm.log
"${TOPOBPE}" analysis --config "${CONFIG}" 2>&1 | tee logs/08_analysis.log
"${TOPOBPE}" figures --config "${CONFIG}" 2>&1 | tee logs/09_figures.log
"${TOPOBPE}" manifest --config "${CONFIG}" 2>&1 | tee logs/10_manifest.log
