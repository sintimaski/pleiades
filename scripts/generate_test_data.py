"""Generate small synthetic Parquet catalogs for testing Pleiades.

Creates two catalogs (A and B) with ra, dec in degrees and optional id columns.
A few sources in A are placed within 2 arcsec of sources in B for match testing.
"""

from __future__ import annotations

import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def arcsec_to_deg(arcsec: float) -> float:
    """Convert arcseconds to degrees."""
    return arcsec / 3600.0


def make_catalog_a(n: int = 500, seed: int = 42) -> pa.Table:
    """Catalog A: random sky positions (ra, dec in degrees)."""
    # Deterministic "random" for reproducibility
    rng = _simple_rng(seed)
    ra = [360.0 * next(rng) for _ in range(n)]
    dec_rad = [math.asin(2.0 * next(rng) - 1.0) for _ in range(n)]
    dec = [math.degrees(d) for d in dec_rad]
    source_id = list(range(1, n + 1))
    return pa.table(
        {"source_id": source_id, "ra": ra, "dec": dec},
        schema=pa.schema([
            ("source_id", pa.int64()),
            ("ra", pa.float64()),
            ("dec", pa.float64()),
        ]),
    )


def make_catalog_b(n: int = 300, seed: int = 123) -> pa.Table:
    """Catalog B: random sky positions; a small region overlaps A for matches."""
    rng = _simple_rng(seed)
    ra = [360.0 * next(rng) for _ in range(n)]
    dec_rad = [math.asin(2.0 * next(rng) - 1.0) for _ in range(n)]
    dec = [math.degrees(d) for d in dec_rad]
    object_id = [f"B{i}" for i in range(1, n + 1)]
    return pa.table(
        {"object_id": object_id, "ra": ra, "dec": dec},
        schema=pa.schema([
            ("object_id", pa.string()),
            ("ra", pa.float64()),
            ("dec", pa.float64()),
        ]),
    )


def make_catalog_with_pairs(
    n_a: int = 100,
    n_b: int = 80,
    n_pairs: int = 10,
    separation_arcsec: float = 1.5,
    seed: int = 1,
) -> tuple[pa.Table, pa.Table]:
    """Two catalogs with n_pairs known pairs within separation_arcsec.

    Catalog A has n_a rows; catalog B has n_b rows. n_pairs of them are
    placed at the same (ra, dec) with small random offset so they are
    within separation_arcsec for cross-match tests.
    """
    rng = _simple_rng(seed)
    deg = separation_arcsec / 3600.0

    # Random base positions for the pairs
    pair_ra = [360.0 * next(rng) for _ in range(n_pairs)]
    pair_dec_rad = [math.asin(2.0 * next(rng) - 1.0) for _ in range(n_pairs)]
    pair_dec = [math.degrees(d) for d in pair_dec_rad]

    # Catalog A: pairs first, then random fill
    a_ra = list(pair_ra)
    a_dec = list(pair_dec)
    while len(a_ra) < n_a:
        a_ra.append(360.0 * next(rng))
        a_dec.append(math.degrees(math.asin(2.0 * next(rng) - 1.0)))
    a_ids = list(range(1, n_a + 1))

    # Catalog B: same pair positions + tiny offset (within separation_arcsec), then random
    b_ra = []
    b_dec = []
    for i in range(n_pairs):
        # Offset by ~1 arcsec in ra (cos(dec)-scaled)
        scale = max(1e-6, math.cos(math.radians(pair_dec[i])))
        b_ra.append(pair_ra[i] + (next(rng) - 0.5) * 2.0 * deg / scale)
        b_dec.append(pair_dec[i] + (next(rng) - 0.5) * 2.0 * deg)
    while len(b_ra) < n_b:
        b_ra.append(360.0 * next(rng))
        b_dec.append(math.degrees(math.asin(2.0 * next(rng) - 1.0)))
    b_ids = [f"B{i}" for i in range(1, n_b + 1)]

    table_a = pa.table(
        {"source_id": a_ids, "ra": a_ra, "dec": a_dec},
        schema=pa.schema([
            ("source_id", pa.int64()),
            ("ra", pa.float64()),
            ("dec", pa.float64()),
        ]),
    )
    table_b = pa.table(
        {"object_id": b_ids, "ra": b_ra, "dec": b_dec},
        schema=pa.schema([
            ("object_id", pa.string()),
            ("ra", pa.float64()),
            ("dec", pa.float64()),
        ]),
    )
    return table_a, table_b


def _simple_rng(seed: int):
    """Simple deterministic PRNG (LCG). Yields floats in (0, 1)."""
    state = seed & 0xFFFFFFFF
    while True:
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        yield state / 0x7FFFFFFF


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Larger random catalogs (no guaranteed pairs)
    table_a = make_catalog_a(500)
    table_b = make_catalog_b(300)
    pq.write_table(table_a, out_dir / "catalog_a.parquet")
    pq.write_table(table_b, out_dir / "catalog_b.parquet")
    print(f"Wrote {out_dir / 'catalog_a.parquet'} ({len(table_a)} rows)")
    print(f"Wrote {out_dir / 'catalog_b.parquet'} ({len(table_b)} rows)")

    # Small catalogs with known pairs for assertion tests
    table_a_small, table_b_small = make_catalog_with_pairs(
        n_a=100, n_b=80, n_pairs=10, separation_arcsec=1.5
    )
    pq.write_table(table_a_small, out_dir / "catalog_a_small.parquet")
    pq.write_table(table_b_small, out_dir / "catalog_b_small.parquet")
    print(f"Wrote {out_dir / 'catalog_a_small.parquet'} ({len(table_a_small)} rows)")
    print(f"Wrote {out_dir / 'catalog_b_small.parquet'} ({len(table_b_small)} rows)")
    print("  -> 10 known pairs within ~1.5 arcsec for radius_arcsec=2.0 tests")


if __name__ == "__main__":
    main()
