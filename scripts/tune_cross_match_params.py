#!/usr/bin/env python3
"""Find good batch size and other params for cross_match on this machine.

Runs the Rust engine with different batch_size_a/b, n_shards, and keep_b_in_memory,
then reports timings and suggests the best config for both out-of-core (disk) and
in-memory B. Uses a single generated catalog set so runs are comparable.

Usage:
  uv run python scripts/tune_cross_match_params.py --rows 200000
  uv run python scripts/tune_cross_match_params.py --rows 500000 --batch-sizes 100000 200000 500000
  uv run python scripts/tune_cross_match_params.py --rows 200000 --sweep-shards
  uv run python scripts/tune_cross_match_params.py --rows 200000 --out-of-core-only   # skip in-memory runs
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
            return (float(psutil.Process().memory_info().rss), False)
        except ImportError:
            return (None, False)
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF)
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
    """Write a synthetic catalog with Int64 IDs (ra, dec)."""
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


def run_one(
    path_a: Path,
    path_b: Path,
    out_path: Path,
    radius_arcsec: float,
    batch_size: int,
    n_shards: int,
    keep_b_in_memory: bool,
    progress_callback: object | None = None,
) -> tuple[float, float | None, int] | None:
    """Run cross_match once; return (time_sec, max_rss_bytes or None, matches_count) or None if cancelled."""
    import astrojoin
    t0 = time.perf_counter()
    kwargs: dict = {
        "catalog_a": path_a,
        "catalog_b": path_b,
        "radius_arcsec": radius_arcsec,
        "output_path": out_path,
        "use_rust": True,
        "batch_size_a": batch_size,
        "batch_size_b": batch_size,
        "n_shards": n_shards,
        "keep_b_in_memory": keep_b_in_memory,
    }
    if progress_callback is not None:
        kwargs["progress_callback"] = progress_callback
    try:
        result = astrojoin.cross_match(**kwargs)
    except OSError as e:
        if "cancelled" in str(e).lower():
            return None
        raise
    elapsed = time.perf_counter() - t0
    rss_after, _ = _get_max_rss_bytes()
    max_rss = float(rss_after) if rss_after is not None else None
    return (elapsed, max_rss, result.matches_count)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tune cross_match batch size and params for this machine.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=200_000,
        help="Rows in catalog A and B (default: 200000)",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=2.0,
        help="Match radius in arcsec (default: 2.0)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[50_000, 100_000, 200_000, 400_000],
        metavar="N",
        help="Batch sizes to try (default: 50000 100000 200000 400000)",
    )
    parser.add_argument(
        "--sweep-shards",
        action="store_true",
        help="Also sweep n_shards (256, 512, 1024); increases run count",
    )
    parser.add_argument(
        "--out-of-core-only",
        action="store_true",
        help="Only sweep keep_b_in_memory=False (out-of-core); skip in-memory runs to reduce time.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs per config; use 2+ to average (default: 1)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the recommendation table and best config",
    )
    args = parser.parse_args()

    # Ensure we run from repo root so astrojoin is importable
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)

    n_shards_list: list[int] = [512]
    if args.sweep_shards:
        n_shards_list = [256, 512, 1024]

    memory_list: list[bool] = [False, True]  # Always compare out-of-core vs in-memory
    if args.out_of_core_only:
        memory_list = [False]

    tmp = Path(tempfile.mkdtemp(prefix="astrojoin_tune_"))
    path_a = tmp / "catalog_a.parquet"
    path_b = tmp / "catalog_b.parquet"

    if not args.quiet:
        print(f"Generating catalogs: {args.rows} x {args.rows} rows in {tmp}")
    generate_catalog(path_a, args.rows, 42, "source_id")
    generate_catalog(path_b, args.rows, 123, "object_id")

    # Suppress Rust verbose logs during tuning
    env_verbose = os.environ.pop("ASTROJOIN_VERBOSE", None)

    cancel_requested: list[bool] = [False]

    def _on_sigint(_signum: int, _frame: object) -> None:
        cancel_requested[0] = True
        print("\nCancelling after current chunk (Ctrl+C again to force)...", flush=True)

    signal.signal(signal.SIGINT, _on_sigint)

    def _progress(_chunk: int, _total: int | None, _rows: int, _matches: int) -> bool:
        return not cancel_requested[0]

    results: list[dict] = []
    total_runs = (
        len(args.batch_sizes)
        * len(n_shards_list)
        * len(memory_list)
        * args.runs
    )
    run_idx = 0

    for batch_size in args.batch_sizes:
        for n_shards in n_shards_list:
            for keep_b_in_memory in memory_list:
                times: list[float] = []
                rss_list: list[float] = []
                for r in range(args.runs):
                    run_idx += 1
                    out_path = tmp / f"matches_{run_idx}.parquet"
                    if not args.quiet:
                        mem_label = "mem" if keep_b_in_memory else "disk"
                        print(
                            f"  [{run_idx}/{total_runs}] batch={batch_size} n_shards={n_shards} {mem_label} ...",
                            end=" ",
                            flush=True,
                        )
                    try:
                        one = run_one(
                            path_a,
                            path_b,
                            out_path,
                            args.radius,
                            batch_size,
                            n_shards,
                            keep_b_in_memory,
                            progress_callback=_progress,
                        )
                        if one is None:
                            if not args.quiet:
                                print("cancelled")
                            return 130
                        elapsed, max_rss, matches = one
                        times.append(elapsed)
                        if max_rss is not None:
                            rss_list.append(max_rss)
                        if not args.quiet:
                            rss_str = _format_memory_bytes(max_rss) if max_rss else "N/A"
                            print(f"{elapsed:.2f}s  {rss_str}")
                    except Exception as e:
                        if not args.quiet:
                            print(f"FAILED: {e}")
                        results.append({
                            "batch_size": batch_size,
                            "n_shards": n_shards,
                            "keep_b_in_memory": keep_b_in_memory,
                            "time_sec": float("inf"),
                            "max_rss_mib": None,
                            "matches": -1,
                        })
                        continue
                if not times:
                    continue
                avg_time = sum(times) / len(times)
                avg_rss = sum(rss_list) / len(rss_list) if rss_list else None
                results.append({
                    "batch_size": batch_size,
                    "n_shards": n_shards,
                    "keep_b_in_memory": keep_b_in_memory,
                    "time_sec": avg_time,
                    "max_rss_mib": avg_rss / (1024 * 1024) if avg_rss else None,
                    "matches": matches if args.runs == 1 else 0,
                })

    if env_verbose is not None:
        os.environ["ASTROJOIN_VERBOSE"] = env_verbose

    valid = [r for r in results if r["time_sec"] != float("inf")]
    if not valid:
        print("No successful runs.", file=sys.stderr)
        return 1

    out_of_core = [r for r in valid if r["keep_b_in_memory"] is False]
    in_memory = [r for r in valid if r["keep_b_in_memory"] is True]
    best_out_of_core = min(out_of_core, key=lambda r: r["time_sec"]) if out_of_core else None
    best_in_memory = min(in_memory, key=lambda r: r["time_sec"]) if in_memory else None

    # Table
    print()
    print("Results (lower time is better)")
    print("-" * 72)
    header = f"{'batch_size':>12} {'n_shards':>10} {'B_in_mem':>10} {'time_sec':>10} {'max_rss':>10}"
    print(header)
    print("-" * 72)
    for r in sorted(valid, key=lambda x: (x["time_sec"], x["batch_size"])):
        mem_flag = "yes" if r["keep_b_in_memory"] else "no"
        rss_str = f"{r['max_rss_mib']:.1f} MiB" if r["max_rss_mib"] else "N/A"
        print(f"{r['batch_size']:>12} {r['n_shards']:>10} {mem_flag:>10} {r['time_sec']:>10.2f} {rss_str:>10}")
    print("-" * 72)
    print()

    def _recommend(label: str, r: dict) -> None:
        print(f"{label}:")
        print(
            f"  batch_size_a = batch_size_b = {r['batch_size']}, "
            f"n_shards = {r['n_shards']}, keep_b_in_memory = {r['keep_b_in_memory']}"
        )
        cmd = (
            f"  uv run python scripts/benchmark_cross_match.py --rows {args.rows} --rust "
            f"--batch-size {r['batch_size']}"
        )
        if r["keep_b_in_memory"]:
            cmd += " --keep-b-in-memory"
        print(f"  Example: {cmd}")
        print()

    print("Recommended for this machine:")
    if best_out_of_core:
        _recommend("Best out-of-core (keep_b_in_memory=False)", best_out_of_core)
    if best_in_memory:
        _recommend("Best with B in memory (keep_b_in_memory=True)", best_in_memory)
    if not best_out_of_core and not best_in_memory:
        best = min(valid, key=lambda r: r["time_sec"])
        _recommend("Best", best)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
