#!/usr/bin/env python3
"""Benchmark cross_match: optional Python vs Rust, report time and match count.

Usage:
  uv run python scripts/benchmark_cross_match.py [--rows 100000] [--rust]
  uv run python scripts/benchmark_cross_match.py --rows 50000 --rows-b 50000
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def generate_catalog(path: Path, n: int, seed: int, id_col: str = "source_id") -> None:
    rng = np.random.default_rng(seed)
    ra = rng.uniform(0, 360, size=n)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1, size=n)))
    ids = np.arange(n, dtype=np.int64) if id_col == "source_id" else [f"B{i}" for i in range(n)]
    table = pa.table({
        id_col: ids,
        "ra": ra.astype(np.float64),
        "dec": dec.astype(np.float64),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark astrojoin.cross_match")
    parser.add_argument("--rows", type=int, default=100_000, help="Rows in catalog A")
    parser.add_argument("--rows-b", type=int, default=None, help="Rows in catalog B (default: same as A)")
    parser.add_argument("--radius", type=float, default=2.0, help="Match radius (arcsec)")
    parser.add_argument("--rust", action="store_true", help="Also run with Rust engine (use_rust=True)")
    args = parser.parse_args()
    n_b = args.rows_b if args.rows_b is not None else args.rows

    import astrojoin

    with tempfile.TemporaryDirectory(prefix="astrojoin_bench_") as tmp:
        tmp = Path(tmp)
        path_a = tmp / "catalog_a.parquet"
        path_b = tmp / "catalog_b.parquet"
        out = tmp / "matches.parquet"
        generate_catalog(path_a, args.rows, 42, "source_id")
        generate_catalog(path_b, n_b, 123, "object_id")

        # runs: list[tuple[str, bool]] = [("Python", False)]
        runs: list[tuple[str, bool]] = []
        if args.rust:
            try:
                import astrojoin_core  # noqa: F401
                runs.append(("Rust", True))
            except ImportError:
                print("Rust engine not available (maturin develop). Skipping Rust run.")

        for label, use_rust in runs:
            t0 = time.perf_counter()
            result = astrojoin.cross_match(
                catalog_a=path_a,
                catalog_b=path_b,
                radius_arcsec=args.radius,
                output_path=out,
                use_rust=use_rust,
            )
            elapsed = time.perf_counter() - t0
            print(
                f"{label}: {result.rows_a_read} x {result.rows_b_read} -> "
                f"{result.matches_count} matches in {elapsed:.2f}s "
                f"({result.rows_a_read * result.rows_b_read / 1e9:.4f} G pairs)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
