#!/usr/bin/env python3
"""Single entry point to run all tests: Rust (cargo test) and Python (pytest).

From repo root:
  uv run python run_tests.py   # or: python run_tests.py (with venv active)
Rust tests are skipped if cargo is not available.
"""

from __future__ import annotations

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


def main() -> int:
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
    print("\nAll tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
