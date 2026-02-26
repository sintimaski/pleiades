#!/usr/bin/env bash
# Run benchmarks using pregenerated fixtures (100k, 1M, 50M rows) and write logs.
# Keeps only the 10 most recent runs in logs/ (older log files are removed).
#
# Generate fixtures first (one-time or when missing):
#   uv run python scripts/generate_benchmark_fixtures.py
#
# Usage (from project root):
#   ./scripts/run_benchmarks.sh
#   bash scripts/run_benchmarks.sh
#
# Optional: pass benchmark args (they are forwarded to benchmark_cross_match.py).
#   ./scripts/run_benchmarks.sh --rows 500000 --verbose

set -euo pipefail

KEEP_LAST_N_RUNS=10
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOGS_DIR="${PROJECT_ROOT}/logs"
FIXTURES_DIR="${PROJECT_ROOT}/data/benchmark_fixtures"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_STDOUT="${LOGS_DIR}/benchmark_${TIMESTAMP}.log"
LOG_STDERR="${LOGS_DIR}/benchmark_${TIMESTAMP}.stderr.log"

mkdir -p "$LOGS_DIR"

# Prune logs: keep only the $KEEP_LAST_N_RUNS most recent runs (by mtime)
# Each run has benchmark_<timestamp>.log and optionally .stderr.log
prune_logs() {
  (cd "$LOGS_DIR" && ls -t benchmark_*.log 2>/dev/null) | tail -n +$((KEEP_LAST_N_RUNS + 1)) | while read -r f; do
    rm -f "${LOGS_DIR}/${f}" "${LOGS_DIR}/${f%.log}.stderr.log"
  done
}

cd "$PROJECT_ROOT"
echo "Running benchmarks (pregenerated fixture)"
echo "  fixtures dir: $FIXTURES_DIR"
echo "  log: $LOG_STDOUT (logs/ keeps last $KEEP_LAST_N_RUNS runs)"
echo "  extra args: ${*:-none}"
echo "---"

# 10M-row fixtures; build with maturin develop --release for best perf.
# I/O optimizations: parquet_mmap + macos_readahead are default features.
# Optional: set RAYON_NUM_THREADS to cap or boost parallelism (e.g. 8).
RAYON_NUM_THREADS=10
ROWS=10000000
BATCH=$((ROWS / 4))
CATALOG_A="${FIXTURES_DIR}/catalog_a_${ROWS}.parquet"
CATALOG_B="${FIXTURES_DIR}/catalog_b_${ROWS}.parquet"
uv run python scripts/benchmark_cross_match.py \
    --catalog-a "$CATALOG_A" \
    --catalog-b "$CATALOG_B" \
    --verbose \
    --batch-size $BATCH \
    --n-shards 16 \
    "$@" > "$LOG_STDOUT"

prune_logs

echo "---"
echo "Done. Log: $LOG_STDOUT"
