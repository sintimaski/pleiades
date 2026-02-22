#!/usr/bin/env python3
"""Single entry point to run all tests: Rust (cargo test), Python (pytest), benchmarks.

From repo root:
  uv run python run_tests.py                    # tests only
  uv run python run_tests.py --benchmark        # tests + benchmark (small catalogs)
  uv run python run_tests.py --benchmark-only   # only benchmark (no tests)
  uv run python run_tests.py --benchmark-only --benchmark-rows 50000 --rust
  uv run python scripts/benchmark_cross_match.py --rows 100000 --rust  # or run script directly
Rust tests are skipped if cargo is not available.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parent


def run_rust_tests() -> bool:
    """Run cargo test (engine only; --no-default-features to avoid PyO3 link).
    Return True if success or skipped (no cargo), False if failed."""
    root = project_root()
    if not (root / "Cargo.toml").is_file():
        return True
    cargo = os.environ.get("CARGO", "cargo")
    try:
        proc = subprocess.run(
            [cargo, "test", "--no-default-features"],
            cwd=root,
            capture_output=False,
            timeout=300,
        )
        return proc.returncode == 0
    except FileNotFoundError:
        print("cargo not found; skipping Rust tests")
        return True
    except subprocess.TimeoutExpired:
        print("cargo test timed out")
        return False


def run_python_tests() -> bool:
    """Run pytest on tests/. Return True if success, False otherwise."""
    root = project_root()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--strict-markers"],
            cwd=root,
            capture_output=False,
            timeout=600,
        )
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        print("pytest timed out")
        return False


def run_benchmark(
    rows: int = 5000,
    rust: bool = False,
    timeout: int = 120,
) -> bool:
    """Run benchmark_cross_match.py with small catalogs. Return True if success."""
    root = project_root()
    script = root / "scripts" / "benchmark_cross_match.py"
    if not script.is_file():
        print("Benchmark script not found; skipping.")
        return True
    cmd = [sys.executable, str(script), "--rows", str(rows)]
    if rust:
        cmd.append("--rust")
    try:
        proc = subprocess.run(
            cmd,
            cwd=root,
            capture_output=False,
            timeout=timeout,
        )
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        print("Benchmark timed out.")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Rust tests, Python tests, and optional benchmark."
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark (small catalogs) after tests.",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Run only the benchmark (skip Rust and Python tests).",
    )
    parser.add_argument(
        "--benchmark-rows",
        type=int,
        default=5000,
        metavar="N",
        help="Rows per catalog for benchmark (default: 5000).",
    )
    parser.add_argument(
        "--rust",
        action="store_true",
        help="Include Rust engine in benchmark (when --benchmark or --benchmark-only).",
    )
    args = parser.parse_args()

    if args.benchmark_only:
        print("=== Benchmark (cross_match) only ===\n")
        bench_ok = run_benchmark(rows=args.benchmark_rows, rust=args.rust)
        if not bench_ok:
            return 1
        print("\nBenchmark done.")
        return 0

    print("=== Rust tests (cargo test) ===\n")
    rust_ok = run_rust_tests()
    print("\n=== Python tests (pytest) ===\n")
    python_ok = run_python_tests()
    if not rust_ok:
        print("\nRust tests failed.", file=sys.stderr)
        return 1
    if not python_ok:
        print("\nPython tests failed.", file=sys.stderr)
        return 1

    if args.benchmark:
        print("\n=== Benchmark (cross_match) ===\n")
        bench_ok = run_benchmark(rows=args.benchmark_rows, rust=args.rust)
        if not bench_ok:
            print("\nBenchmark failed.", file=sys.stderr)
            return 1

    print("\nAll tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
