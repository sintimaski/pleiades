#!/usr/bin/env bash
# Profile cross_match benchmark to identify CPU hotspots.
#
# On macOS: uses `sample` (built-in) to capture a 10s CPU profile of the Python
# process running the Rust engine. Output: logs/profile_<timestamp>.txt
#
# For flamegraphs on macOS:
#   1. Install: cargo install inferno
#   2. Run with --dtrace to use DTrace (requires sudo) and generate flamegraph.svg
#
# Usage (from project root):
#   ./scripts/run_profile.sh
#   ./scripts/run_profile.sh --dtrace   # DTrace + flamegraph (macOS, sudo)
#   ./scripts/run_profile.sh --profile   # PLEIADES_PROFILE=1 for sub-phase timing
#
# Build release first for meaningful results:
#   uv run maturin develop --release --no-default-features --features simd

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOGS_DIR="${PROJECT_ROOT}/logs"
FIXTURES_DIR="${PROJECT_ROOT}/data/benchmark_fixtures"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
ROWS=10000000
SAMPLE_SECS=20

mkdir -p "$LOGS_DIR"
cd "$PROJECT_ROOT"

USE_DTRACE=false
USE_SUBPHASE=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dtrace)   USE_DTRACE=true; shift ;;
        --profile)  USE_SUBPHASE=true; shift ;;
        *)          break ;;
    esac
done

CATALOG_A="${FIXTURES_DIR}/catalog_a_${ROWS}.parquet"
CATALOG_B="${FIXTURES_DIR}/catalog_b_${ROWS}.parquet"
if [[ ! -f "$CATALOG_A" || ! -f "$CATALOG_B" ]]; then
    echo "Fixtures missing. Generate first: uv run python scripts/generate_benchmark_fixtures.py"
    exit 1
fi

OUTPUT_LOG="${LOGS_DIR}/profile_${TIMESTAMP}.txt"
FLAMEGRAPH_SVG="${LOGS_DIR}/flamegraph_${TIMESTAMP}.svg"
DTRACE_OUT="${LOGS_DIR}/profile_${TIMESTAMP}.dtrace"

export PLEIADES_VERBOSE=1
export RAYON_NUM_THREADS=8
[[ "$USE_SUBPHASE" == true ]] && export PLEIADES_PROFILE=1

echo "Profiling cross_match (${ROWS} rows)"
echo "  catalog_a: $CATALOG_A"
echo "  catalog_b: $CATALOG_B"
echo "  sample_secs: $SAMPLE_SECS"
echo "  sub-phase timing: $USE_SUBPHASE"
echo "  dtrace+flamegraph: $USE_DTRACE"
echo "---"

if [[ "$USE_DTRACE" == true ]]; then
    if [[ "$(uname)" != "Darwin" ]]; then
        echo "DTrace mode is macOS-only. Use sample mode instead."
        exit 1
    fi
    echo "Starting benchmark in background..."
    uv run python scripts/benchmark_cross_match.py \
        --catalog-a "$CATALOG_A" \
        --catalog-b "$CATALOG_B" \
        --verbose \
        --batch-size $((ROWS / 4)) \
        --n-shards 16 \
        > /dev/null 2>&1 &
    ORIG_PID=$!
    sleep 2
    SAMPLE_PID=$ORIG_PID
    PYTHON_PID=$(pgrep -P "$ORIG_PID" -f "python" 2>/dev/null | head -1)
    [[ -n "$PYTHON_PID" ]] && SAMPLE_PID=$PYTHON_PID
    echo "Sampling PID $SAMPLE_PID for ${SAMPLE_SECS}s with DTrace (may prompt for sudo)..."
    sudo dtrace -x ustackframes=100 \
        -n "profile-997 /pid == $SAMPLE_PID/ { @[ustack()] = count(); } tick-${SAMPLE_SECS}s { exit(0); }" \
        -o "$DTRACE_OUT" 2>/dev/null || true
    kill $ORIG_PID 2>/dev/null || true
    wait $ORIG_PID 2>/dev/null || true
    if command -v inferno-collapse-dtrace &>/dev/null && [[ -f "$DTRACE_OUT" ]]; then
        echo "Generating flamegraph..."
        inferno-collapse-dtrace < "$DTRACE_OUT" | inferno-flamegraph > "$FLAMEGRAPH_SVG"
        echo "Flamegraph: $FLAMEGRAPH_SVG"
    else
        echo "Install inferno for flamegraph: cargo install inferno"
        echo "Raw DTrace output: $DTRACE_OUT"
    fi
else
    echo "Starting benchmark in background..."
    # Run Python directly (no tee) so $! is the Python process, not tee
    uv run python scripts/benchmark_cross_match.py \
        --catalog-a "$CATALOG_A" \
        --catalog-b "$CATALOG_B" \
        --verbose \
        --batch-size $((ROWS / 4)) \
        --n-shards 16 \
        > "${LOGS_DIR}/profile_bench_${TIMESTAMP}.log" 2>&1 &
    ORIG_PID=$!
    sleep 2
    # If uv spawns python as child, sample the python process (where Rust runs)
    SAMPLE_PID=$ORIG_PID
    PYTHON_PID=$(pgrep -P "$ORIG_PID" -f "python" 2>/dev/null | head -1)
    [[ -n "$PYTHON_PID" ]] && SAMPLE_PID=$PYTHON_PID
    echo "Sampling PID $SAMPLE_PID for ${SAMPLE_SECS}s..."
    sample "$SAMPLE_PID" "$SAMPLE_SECS" -file "$OUTPUT_LOG" 2>/dev/null || true
    kill $ORIG_PID 2>/dev/null || true
    wait $ORIG_PID 2>/dev/null || true
    echo "Profile written: $OUTPUT_LOG"
    echo "View: open $OUTPUT_LOG (or cat, search for hot functions)"
fi

echo "---"
echo "Done."
