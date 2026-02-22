#!/usr/bin/env python3
"""Pre-generate Parquet catalogs for benchmarking. Avoids OOM and slow in-memory generation for large runs.

Write fixtures to data/benchmark_fixtures/ (or --out-dir). Default is to generate all standard sizes
(100k, 1M, 50M). Then run benchmarks with those files:

  uv run python scripts/generate_benchmark_fixtures.py
  ./scripts/run_benchmarks.sh

Or generate a single size and run manually:
  uv run python scripts/generate_benchmark_fixtures.py --sizes 1000000
  uv run python scripts/benchmark_cross_match.py --catalog-a data/benchmark_fixtures/catalog_a_1000000.parquet --catalog-b data/benchmark_fixtures/catalog_b_1000000.parquet --rust -o matches.parquet

Usage:
  uv run python scripts/generate_benchmark_fixtures.py [--sizes 100000 1000000 50000000] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def generate_catalog(path: Path, n: int, seed: int, id_col: str = "source_id") -> None:
    """Write a synthetic catalog with Int64 IDs (ra, dec). Streams in chunks to limit memory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    chunk = 200_000
    rng = np.random.default_rng(seed)
    writer = None
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        size = end - start
        ra = rng.uniform(0, 360, size=size)
        dec = np.degrees(np.arcsin(rng.uniform(-1, 1, size=size)))
        ids = np.arange(start, end, dtype=np.int64)
        table = pa.table({
            id_col: ids,
            "ra": ra.astype(np.float64),
            "dec": dec.astype(np.float64),
        })
        if writer is None:
            writer = pq.ParquetWriter(path, table.schema, use_dictionary=False)
        writer.write_table(table)
    if writer is not None:
        writer.close()


# Default benchmark sizes (rows): 100k, 1M, 50M.
DEFAULT_SIZES = [10_000_000]


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-generate benchmark catalog Parquet files")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="*",
        default=DEFAULT_SIZES,
        help=f"Row counts to generate (default: {DEFAULT_SIZES}). Produces catalog_a_{{n}}.parquet and catalog_b_{{n}}.parquet for each n.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "benchmark_fixtures",
        help="Output directory for Parquet files",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for n in args.sizes:
        path_a = args.out_dir / f"catalog_a_{n}.parquet"
        path_b = args.out_dir / f"catalog_b_{n}.parquet"
        print(f"Writing A: {path_a} ({n} rows)")
        generate_catalog(path_a, n, 42, "source_id")
        print(f"Writing B: {path_b} ({n} rows)")
        generate_catalog(path_b, n, 123, "object_id")
    print(f"Done. Fixtures in {args.out_dir}. Run: ./scripts/run_benchmarks.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
