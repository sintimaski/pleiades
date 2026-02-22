#!/usr/bin/env python3
"""Benchmark cross_match: optional Python vs Rust, report time and match count.

Uses Int64 IDs for both catalogs so the Rust engine and Parquet encoding behave
reliably at scale. For pre-generated files (e.g. to avoid OOM), use
--catalog-a / --catalog-b or run scripts/generate_benchmark_fixtures.py first.

With --verbose, sets PLEIADES_VERBOSE=1 so the Rust engine prints timing logs
(index, load B, join, write per chunk; partition B and total) to stderr, and
prints per-chunk memory (peak RSS on Unix, current on Windows) after each chunk.

For best throughput, build the Rust extension in release mode:
  uv run maturin develop --release

Usage:
  uv run python scripts/benchmark_cross_match.py [--rows 100000] [--rust]
  uv run python scripts/benchmark_cross_match.py --rows 1000000 --rust --verbose
  uv run python scripts/benchmark_cross_match.py --rows 500000 --batch-size 250000
  uv run python scripts/benchmark_cross_match.py --catalog-a path/a.parquet --catalog-b path/b.parquet -o out.parquet
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _get_max_rss_bytes() -> tuple[float | None, bool]:
    """Return (peak or current RSS in bytes, True if peak else current), or (None, False) if unavailable."""
    if sys.platform == "win32":
        try:
            import psutil
            return (float(psutil.Process().memory_info().rss), False)  # current
        except ImportError:
            return (None, False)
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF)
        # macOS: ru_maxrss in bytes; Linux: in kilobytes
        if sys.platform == "darwin":
            return (float(ru.ru_maxrss), True)
        return (float(ru.ru_maxrss) * 1024, True)
    except (ImportError, AttributeError):
        return (None, False)


def _format_memory_bytes(b: float) -> str:
    if b >= 1024 ** 3:
        return f"{b / 1024**3:.2f} GiB"
    return f"{b / 1024**2:.2f} MiB"


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
    parser = argparse.ArgumentParser(description="Benchmark pleiades.cross_match")
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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Set PLEIADES_VERBOSE=1 for Rust engine timing logs (stderr)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch_size_a and batch_size_b (default 100000); larger = fewer chunks, more RAM",
    )
    parser.add_argument(
        "--keep-b-in-memory",
        action="store_true",
        help="Partition B into RAM (no shard I/O). Use only when B is small; default is out-of-core (disk).",
    )
    args = parser.parse_args()
    n_b = args.rows_b if args.rows_b is not None else args.rows

    if args.verbose:
        os.environ["PLEIADES_VERBOSE"] = "1"
        print("Rust timing logs (stderr):", flush=True)

    import pleiades

    if args.catalog_a is not None and args.catalog_b is not None:
        path_a = args.catalog_a
        path_b = args.catalog_b
        out = args.output or Path(tempfile.gettempdir()) / "pleiades_bench_matches.parquet"
        if not path_a.is_file() or not path_b.is_file():
            raise FileNotFoundError(f"Pre-generated catalogs must exist: {path_a}, {path_b}")
    else:
        tmp = Path(tempfile.mkdtemp(prefix="pleiades_bench_"))
        path_a = tmp / "catalog_a.parquet"
        path_b = tmp / "catalog_b.parquet"
        out = args.output or tmp / "matches.parquet"
        generate_catalog(path_a, args.rows, 42, "source_id")
        generate_catalog(path_b, n_b, 123, "object_id")

    runs: list[tuple[str, bool]] = [("Rust", True)]
    batch_kw: dict = {}
    if args.batch_size is not None:
        batch_kw = {"batch_size_a": args.batch_size, "batch_size_b": args.batch_size}
    if args.keep_b_in_memory:
        batch_kw["keep_b_in_memory"] = True

    # Ctrl+C: set flag so progress_callback returns False next chunk (stops within ~1 chunk delay)
    cancel_requested: list[bool] = [False]

    def _on_sigint(_sig: int, _frame: object) -> None:
        cancel_requested[0] = True
        print("\nCancelling after current chunk...", flush=True)

    signal.signal(signal.SIGINT, _on_sigint)

    def _progress(chunk_ix: int, total: int | None, rows_a: int, matches_count: int) -> bool:
        if args.verbose:
            rss_bytes, is_peak = _get_max_rss_bytes()
            rss_str = _format_memory_bytes(rss_bytes) if rss_bytes is not None else "N/A"
            kind = "peak RSS" if is_peak else "current RSS"
            print(
                f"[chunk {chunk_ix}] rows_a={rows_a} matches={matches_count} memory={rss_str} ({kind})",
                flush=True,
            )
        return not cancel_requested[0]

    batch_kw["progress_callback"] = _progress

    for label, use_rust in runs:
        t0 = time.perf_counter()
        try:
            result = pleiades.cross_match(
                catalog_a=path_a,
                catalog_b=path_b,
                radius_arcsec=args.radius,
                output_path=out,
                use_rust=use_rust,
                **batch_kw,
            )
        except OSError as e:
            if "cancelled" in str(e).lower():
                print("Cancelled.", flush=True)
                return 130
            raise
        elapsed = time.perf_counter() - t0
        max_rss, is_peak = _get_max_rss_bytes()
        print(
            f"{label}: {result.rows_a_read} x {result.rows_b_read} -> "
            f"{result.matches_count} matches in {elapsed:.2f}s "
            f"({result.rows_a_read * result.rows_b_read / 1e9:.4f} G pairs)"
        )
        if max_rss is not None:
            kind = "peak RSS" if is_peak else "current RSS"
            print(f"Max memory: {_format_memory_bytes(max_rss)} ({kind})")
        else:
            print("Max memory: N/A (install psutil on Windows for current RSS)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
