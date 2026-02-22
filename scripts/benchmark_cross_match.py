#!/usr/bin/env python3
"""Benchmark cross_match: optional Python vs Rust, report time and match count.

Uses Int64 IDs for both catalogs so the Rust engine and Parquet encoding behave
reliably at scale. For pre-generated files (e.g. to avoid OOM), use
--catalog-a / --catalog-b or run scripts/generate_benchmark_fixtures.py first.

Usage:
  uv run python scripts/benchmark_cross_match.py [--rows 100000] [--rust]
  uv run python scripts/benchmark_cross_match.py --rows 1000000 --rust
  uv run python scripts/benchmark_cross_match.py --catalog-a path/a.parquet --catalog-b path/b.parquet -o out.parquet
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def generate_catalog(
    path: Path,
    n: int,
    seed: int,
    id_col: str = "source_id",
) -> None:
    """Write a synthetic catalog with Int64 IDs (ra, dec) for reliable Rust/Parquet at scale."""
    rng = np.random.default_rng(seed)
    ra = rng.uniform(0, 360, size=n)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1, size=n)))
    ids = np.arange(n, dtype=np.int64)
    table = pa.table({
        id_col: ids,
        "ra": ra.astype(np.float64),
        "dec": dec.astype(np.float64),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, use_dictionary=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark astrojoin.cross_match")
    parser.add_argument("--rows", type=int, default=100_000, help="Rows in catalog A (ignored if --catalog-a)")
    parser.add_argument("--rows-b", type=int, default=None, help="Rows in catalog B (default: same as A)")
    parser.add_argument("--radius", type=float, default=2.0, help="Match radius (arcsec)")
    parser.add_argument("--rust", action="store_true", help="Use Rust engine (default: True)")
    parser.add_argument(
        "--catalog-a",
        type=Path,
        default=None,
        help="Use pre-generated catalog A (skip in-memory generation)",
    )
    parser.add_argument(
        "--catalog-b",
        type=Path,
        default=None,
        help="Use pre-generated catalog B (skip in-memory generation)",
    )
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output matches path (default: temp)")
    args = parser.parse_args()
    n_b = args.rows_b if args.rows_b is not None else args.rows

    import astrojoin

    if args.catalog_a is not None and args.catalog_b is not None:
        path_a = args.catalog_a
        path_b = args.catalog_b
        out = args.output or Path(tempfile.gettempdir()) / "astrojoin_bench_matches.parquet"
        if not path_a.is_file() or not path_b.is_file():
            raise FileNotFoundError(f"Pre-generated catalogs must exist: {path_a}, {path_b}")
    else:
        tmp = Path(tempfile.mkdtemp(prefix="astrojoin_bench_"))
        path_a = tmp / "catalog_a.parquet"
        path_b = tmp / "catalog_b.parquet"
        out = args.output or tmp / "matches.parquet"
        generate_catalog(path_a, args.rows, 42, "source_id")
        generate_catalog(path_b, n_b, 123, "object_id")

    print(f"Generated {args.rows} for catalogs a and b")

    runs: list[tuple[str, bool]] = [("Rust", True)]
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
