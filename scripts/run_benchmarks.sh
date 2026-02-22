#!/usr/bin/env bash
# Run benchmarks using pregenerated fixtures (100k, 1M, 50M rows) and write logs.
#
# Generate fixtures first (one-time or when missing):
#   uv run python scripts/generate_benchmark_fixtures.py
#
# Usage (from project root):
#   ./scripts/run_benchmarks.sh
#   bash scripts/run_benchmarks.sh
#
# Optional: pass benchmark args (they are forwarded to benchmark_cross_match.py).
#   ./scripts/run_benchmarks.sh --rows 500000 --rust --verbose

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOGS_DIR="${PROJECT_ROOT}/logs"
FIXTURES_DIR="${PROJECT_ROOT}/data/benchmark_fixtures"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_STDOUT="${LOGS_DIR}/benchmark_${TIMESTAMP}.log"
LOG_STDERR="${LOGS_DIR}/benchmark_${TIMESTAMP}.stderr.log"

mkdir -p "$LOGS_DIR"

cd "$PROJECT_ROOT"
echo "Running benchmarks (pregenerated fixture)"
echo "  fixtures dir: $FIXTURES_DIR"
echo "  log: $LOG_STDOUT"
echo "  extra args: ${*:-none}"
echo "---"

ROWS=1000000
BATCH=$((ROWS / 4))
N_SHARDS=128
CATALOG_A="${FIXTURES_DIR}/catalog_a_${ROWS}.parquet"
CATALOG_B="${FIXTURES_DIR}/catalog_b_${ROWS}.parquet"
# PLEIADES_GPU=wgpu PLEIADES_GPU_MIN_PAIRS=0 uv run python scripts/benchmark_cross_match.py "$@" 2> "$LOG_STDOUT"
uv run python scripts/benchmark_cross_match.py \
    --catalog-a "$CATALOG_A" \
    --catalog-b "$CATALOG_B" \
    --rows "$ROWS" \
    --batch-size "$BATCH" \
    --n_shards "$N_SHARDS" \
    --verbose \
    "$@" > "$LOG_STDOUT"

echo "---"
echo "Done. Log: $LOG_STDOUT"
