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

# 1M-row fixtures; tool defaults (batch 250k, n_shards 16) match this scale
ROWS=1000000
CATALOG_A="${FIXTURES_DIR}/catalog_a_${ROWS}.parquet"
CATALOG_B="${FIXTURES_DIR}/catalog_b_${ROWS}.parquet"
uv run python scripts/benchmark_cross_match.py \
    --catalog-a "$CATALOG_A" \
    --catalog-b "$CATALOG_B" \
    --verbose \
    "$@" > "$LOG_STDOUT"

echo "---"
echo "Done. Log: $LOG_STDOUT"
