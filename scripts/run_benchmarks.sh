#!/usr/bin/env bash
# Run benchmarks and write logs.
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
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_STDOUT="${LOGS_DIR}/benchmark_${TIMESTAMP}.log"
LOG_STDERR="${LOGS_DIR}/benchmark_${TIMESTAMP}.stderr.log"

mkdir -p "$LOGS_DIR"

cd "$PROJECT_ROOT"
echo "Running benchmarks (args: ${*:-default})"
echo "  stdout -> $LOG_STDOUT"
echo "---"

PLEIADES_GPU=wgpu PLEIADES_GPU_MIN_PAIRS=0 uv run python scripts/benchmark_cross_match.py "$@" 2> "$LOG_STDOUT"
# uv run python scripts/benchmark_cross_match.py --rows 1000000 --batch-size 500000 --verbose 2> "$LOG_STDOUT"

echo "---"
echo "Done. Logs: $LOG_STDOUT"
