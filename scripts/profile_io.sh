#!/usr/bin/env bash
# Profile I/O during the cross-match benchmark (macOS).
#
# 1) Runs the benchmark under /usr/bin/time -l and saves resource usage
#    (real/user/sys, disk input/output operations, page reclaims, etc.).
# 2) Optionally runs fs_usage to log which files are read (requires sudo).
#
# Usage (from project root):
#   ./scripts/profile_io.sh              # time -l only (no sudo)
#   ./scripts/profile_io.sh --fs-usage   # also run fs_usage (will prompt for sudo)
#
# Run in a normal terminal so /usr/bin/time -l can read kernel stats (disk I/O counts).
#
# Manual fs_usage (run in a second terminal while benchmark runs here):
#   sudo fs_usage -f filesystem -w -e 2>&1 | tee logs/fs_usage.log
# Then run: PLEIADES_VERBOSE=1 ./scripts/run_benchmarks.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOGS_DIR="${PROJECT_ROOT}/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
PROFILE_LOG="${LOGS_DIR}/profile_io_${TIMESTAMP}.txt"
FS_USAGE_LOG="${LOGS_DIR}/fs_usage_${TIMESTAMP}.log"
USE_FS_USAGE=false

for arg in "$@"; do
  case "$arg" in
    --fs-usage) USE_FS_USAGE=true ;;
  esac
done

mkdir -p "$LOGS_DIR"

echo "I/O profile run: $TIMESTAMP"
echo "  profile log: $PROFILE_LOG"
echo "  (benchmark stdout still goes to logs/benchmark_<timestamp>.log)"
echo "---"

# Run benchmark under time -l; time writes its resource block to stderr.
# We capture stderr (pleiades verbose + time block) and tee to terminal.
run_with_time() {
  # PLEIADES_VERBOSE=1 is set so we see chunk/load B timing in the log.
  PLEIADES_VERBOSE=1 /usr/bin/time -l "$SCRIPT_DIR/run_benchmarks.sh" 2>&1 | tee "$PROFILE_LOG"
}

run_fs_usage_background() {
  echo "Starting fs_usage (sudo) in background; output -> $FS_USAGE_LOG"
  sudo fs_usage -f filesystem -w -e 2>&1 | tee "$FS_USAGE_LOG" &
  local pid=$!
  echo "  fs_usage PID: $pid"
  # Give it a moment to attach
  sleep 2
  return 0
}

kill_fs_usage() {
  local p
  for p in $(pgrep -f "fs_usage -f filesystem" 2>/dev/null); do
    sudo kill "$p" 2>/dev/null || true
  done
}

if [[ "$USE_FS_USAGE" == true ]]; then
  run_fs_usage_background
  trap kill_fs_usage EXIT
fi

run_with_time

echo "---"
echo "Profile saved: $PROFILE_LOG"
echo ""
echo "Summary (from time -l):"
grep -E "^(real|user|sys|maximum resident|number of disk|involuntary)" "$PROFILE_LOG" 2>/dev/null || true
echo ""
if [[ "$USE_FS_USAGE" == true ]]; then
  echo "fs_usage log: $FS_USAGE_LOG (filter for 'read' and fixture/shard paths to see load B I/O)"
fi
