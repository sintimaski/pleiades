#!/usr/bin/env python3
"""Pre-generate Parquet catalogs for benchmarking. Avoids OOM and slow in-memory generation for large runs.

Write fixtures to scripts/benchmark_fixtures/ (or --out-dir). Then run:
  uv run python scripts/benchmark_cross_match.py --catalog-a scripts/benchmark_fixtures/catalog_a_1M.parquet --catalog-b scripts/benchmark_fixtures/catalog_b_1M.parquet -o matches.parquet --rust

Usage:
  uv run python scripts/generate_benchmark_fixtures.py [--rows 100000] [--rows-b 100000] [--out-dir DIR]
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-generate benchmark catalog Parquet files")
    parser.add_argument("--rows", type=int, default=100_000, help="Rows in catalog A")
    parser.add_argument("--rows-b", type=int, default=None, help="Rows in catalog B (default: same as A)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "benchmark_fixtures",
        help="Output directory for Parquet files",
    )
    args = parser.parse_args()
    n_b = args.rows_b if args.rows_b is not None else args.rows
    args.out_dir.mkdir(parents=True, exist_ok=True)

    path_a = args.out_dir / f"catalog_a_{args.rows}.parquet"
    path_b = args.out_dir / f"catalog_b_{n_b}.parquet"
    print(f"Writing A: {path_a} ({args.rows} rows)")
    generate_catalog(path_a, args.rows, 42, "source_id")
    print(f"Writing B: {path_b} ({n_b} rows)")
    generate_catalog(path_b, n_b, 123, "object_id")
    print(f"Done. Run: uv run python scripts/benchmark_cross_match.py --catalog-a {path_a} --catalog-b {path_b} --rust -o matches.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
