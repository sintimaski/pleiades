"""Helpers to generate Parquet catalogs with known cross-match outcomes for integration tests."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def _rng(seed: int):
    state = seed & 0xFFFFFFFF
    while True:
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        yield state / 0x7FFFFFFF


def _haversine_arcsec(
    ra1_deg: float, dec1_deg: float, ra2_deg: float, dec2_deg: float
) -> float:
    """Angular separation in arcsec between two points (degrees)."""
    r1 = math.radians(ra1_deg)
    d1 = math.radians(dec1_deg)
    r2 = math.radians(ra2_deg)
    d2 = math.radians(dec2_deg)
    x = (
        math.sin((d2 - d1) / 2) ** 2
        + math.cos(d1) * math.cos(d2) * math.sin((r2 - r1) / 2) ** 2
    )
    return math.degrees(2 * math.asin(min(math.sqrt(x), 1.0))) * 3600.0


def _offset_for_separation_arcsec(
    ra_deg: float,
    dec_deg: float,
    separation_arcsec: float,
) -> tuple[float, float]:
    """Return (ra_b, dec_b) such that separation from (ra_deg, dec_deg) is separation_arcsec.
    Uses dec offset only (exact for small angles).
    """
    deg = separation_arcsec / 3600.0
    return (ra_deg, dec_deg + deg)


ExpectedPair = tuple[Any, Any, float]  # (id_a, id_b, separation_arcsec)


def make_catalogs_exact_n_pairs(
    n_pairs: int,
    radius_arcsec: float,
    separation_arcsec: float | None = None,
    n_a_extra: int = 0,
    n_b_extra: int = 0,
    seed: int = 42,
) -> tuple[pa.Table, pa.Table, list[ExpectedPair]]:
    """Generate A and B with exactly n_pairs within radius_arcsec; return expected (id_a, id_b, sep).

    Pairs are placed at separation_arcsec (default radius_arcsec/2). Extra rows are random and
    far apart so they do not match. Returns (table_a, table_b, expected_pairs).
    """
    if separation_arcsec is None:
        separation_arcsec = radius_arcsec / 2.0
    rng = _rng(seed)
    expected: list[ExpectedPair] = []
    a_ra, a_dec, a_ids = [], [], []
    b_ra, b_dec, b_ids = [], [], []

    for i in range(n_pairs):
        ra = 360.0 * next(rng)
        dec_rad = math.asin(2.0 * next(rng) - 1.0)
        dec = math.degrees(dec_rad)
        a_ra.append(ra)
        a_dec.append(dec)
        a_ids.append(i + 1)
        ra_b, dec_b = _offset_for_separation_arcsec(ra, dec, separation_arcsec)
        b_ra.append(ra_b)
        b_dec.append(dec_b)
        b_ids.append(f"B{i + 1}")
        expected.append((i + 1, f"B{i + 1}", separation_arcsec))

    for i in range(n_a_extra):
        a_ra.append(360.0 * next(rng))
        a_dec.append(math.degrees(math.asin(2.0 * next(rng) - 1.0)))
        a_ids.append(n_pairs + i + 1)
    for i in range(n_b_extra):
        b_ra.append(360.0 * next(rng))
        b_dec.append(math.degrees(math.asin(2.0 * next(rng) - 1.0)))
        b_ids.append(f"B{n_pairs + i + 1}")

    table_a = pa.table(
        {"source_id": a_ids, "ra": a_ra, "dec": a_dec},
        schema=pa.schema(
            [
                ("source_id", pa.int64()),
                ("ra", pa.float64()),
                ("dec", pa.float64()),
            ]
        ),
    )
    table_b = pa.table(
        {"object_id": b_ids, "ra": b_ra, "dec": b_dec},
        schema=pa.schema(
            [
                ("object_id", pa.string()),
                ("ra", pa.float64()),
                ("dec", pa.float64()),
            ]
        ),
    )
    return table_a, table_b, expected


def make_catalogs_random(
    n_a: int,
    n_b: int,
    seed: int = 0,
) -> tuple[pa.Table, pa.Table]:
    """Two random catalogs (ra, dec uniformly on sphere). For use with reference cross-match."""
    rng = _rng(seed)
    a_ra = [360.0 * next(rng) for _ in range(n_a)]
    a_dec = [math.degrees(math.asin(2.0 * next(rng) - 1.0)) for _ in range(n_a)]
    b_ra = [360.0 * next(rng) for _ in range(n_b)]
    b_dec = [math.degrees(math.asin(2.0 * next(rng) - 1.0)) for _ in range(n_b)]
    table_a = pa.table(
        {"source_id": list(range(1, n_a + 1)), "ra": a_ra, "dec": a_dec},
        schema=pa.schema(
            [("source_id", pa.int64()), ("ra", pa.float64()), ("dec", pa.float64())]
        ),
    )
    table_b = pa.table(
        {"object_id": [f"B{i}" for i in range(1, n_b + 1)], "ra": b_ra, "dec": b_dec},
        schema=pa.schema(
            [("object_id", pa.string()), ("ra", pa.float64()), ("dec", pa.float64())]
        ),
    )
    return table_a, table_b


def make_catalogs_no_pairs(
    n_a: int = 20,
    n_b: int = 20,
    radius_arcsec: float = 2.0,
    seed: int = 123,
) -> tuple[pa.Table, pa.Table]:
    """Generate A and B with random positions; radius is small so no pairs match."""
    rng = _rng(seed)
    a_ra = [360.0 * next(rng) for _ in range(n_a)]
    a_dec = [math.degrees(math.asin(2.0 * next(rng) - 1.0)) for _ in range(n_a)]
    b_ra = [360.0 * next(rng) for _ in range(n_b)]
    b_dec = [math.degrees(math.asin(2.0 * next(rng) - 1.0)) for _ in range(n_b)]
    table_a = pa.table(
        {"source_id": list(range(1, n_a + 1)), "ra": a_ra, "dec": a_dec},
        schema=pa.schema(
            [("source_id", pa.int64()), ("ra", pa.float64()), ("dec", pa.float64())]
        ),
    )
    table_b = pa.table(
        {"object_id": [f"B{i}" for i in range(1, n_b + 1)], "ra": b_ra, "dec": b_dec},
        schema=pa.schema(
            [("object_id", pa.string()), ("ra", pa.float64()), ("dec", pa.float64())]
        ),
    )
    return table_a, table_b


def make_catalogs_pair_at_exact_radius(
    radius_arcsec: float,
    seed: int = 1,
) -> tuple[pa.Table, pa.Table, ExpectedPair]:
    """One pair with separation exactly radius_arcsec (boundary). Returns (table_a, table_b, (id_a, id_b, sep))."""
    rng = _rng(seed)
    ra = 360.0 * next(rng)
    dec = math.degrees(math.asin(2.0 * next(rng) - 1.0))
    ra_b, dec_b = _offset_for_separation_arcsec(ra, dec, radius_arcsec)
    table_a = pa.table(
        {"source_id": [1], "ra": [ra], "dec": [dec]},
        schema=pa.schema(
            [("source_id", pa.int64()), ("ra", pa.float64()), ("dec", pa.float64())]
        ),
    )
    table_b = pa.table(
        {"object_id": ["B1"], "ra": [ra_b], "dec": [dec_b]},
        schema=pa.schema(
            [("object_id", pa.string()), ("ra", pa.float64()), ("dec", pa.float64())]
        ),
    )
    actual_sep = _haversine_arcsec(ra, dec, ra_b, dec_b)
    return table_a, table_b, (1, "B1", actual_sep)


def make_catalogs_one_to_many(
    n_b_matches: int,
    radius_arcsec: float,
    separation_arcsec: float | None = None,
    seed: int = 10,
) -> tuple[pa.Table, pa.Table, list[ExpectedPair]]:
    """One source in A, n_b_matches sources in B all within radius. Expected n_b_matches matches."""
    if separation_arcsec is None:
        separation_arcsec = radius_arcsec / 2.0
    rng = _rng(seed)
    ra = 360.0 * next(rng)
    dec = math.degrees(math.asin(2.0 * next(rng) - 1.0))
    a_ra, a_dec = [ra], [dec]
    b_ra, b_dec, b_ids = [], [], []
    expected: list[ExpectedPair] = []
    for i in range(n_b_matches):
        ra_b, dec_b = _offset_for_separation_arcsec(
            ra, dec, separation_arcsec * (i + 1) / (n_b_matches + 1)
        )
        b_ra.append(ra_b)
        b_dec.append(dec_b)
        b_ids.append(f"B{i + 1}")
        sep = _haversine_arcsec(ra, dec, ra_b, dec_b)
        expected.append((1, f"B{i + 1}", sep))
    table_a = pa.table(
        {"source_id": [1], "ra": a_ra, "dec": a_dec},
        schema=pa.schema(
            [("source_id", pa.int64()), ("ra", pa.float64()), ("dec", pa.float64())]
        ),
    )
    table_b = pa.table(
        {"object_id": b_ids, "ra": b_ra, "dec": b_dec},
        schema=pa.schema(
            [("object_id", pa.string()), ("ra", pa.float64()), ("dec", pa.float64())]
        ),
    )
    return table_a, table_b, expected


def make_catalogs_many_to_one(
    n_a_matches: int,
    radius_arcsec: float,
    separation_arcsec: float | None = None,
    seed: int = 20,
) -> tuple[pa.Table, pa.Table, list[ExpectedPair]]:
    """n_a_matches sources in A, one in B; all A within radius of B. Expected n_a_matches matches."""
    if separation_arcsec is None:
        separation_arcsec = radius_arcsec / 2.0
    rng = _rng(seed)
    ra_b = 360.0 * next(rng)
    dec_b = math.degrees(math.asin(2.0 * next(rng) - 1.0))
    a_ra, a_dec, a_ids = [], [], []
    expected: list[ExpectedPair] = []
    for i in range(n_a_matches):
        ra_a, dec_a = _offset_for_separation_arcsec(
            ra_b, dec_b, separation_arcsec * (i + 1) / (n_a_matches + 1)
        )
        a_ra.append(ra_a)
        a_dec.append(dec_a)
        a_ids.append(i + 1)
        sep = _haversine_arcsec(ra_a, dec_a, ra_b, dec_b)
        expected.append((i + 1, "B1", sep))
    table_a = pa.table(
        {"source_id": a_ids, "ra": a_ra, "dec": a_dec},
        schema=pa.schema(
            [("source_id", pa.int64()), ("ra", pa.float64()), ("dec", pa.float64())]
        ),
    )
    table_b = pa.table(
        {"object_id": ["B1"], "ra": [ra_b], "dec": [dec_b]},
        schema=pa.schema(
            [("object_id", pa.string()), ("ra", pa.float64()), ("dec", pa.float64())]
        ),
    )
    return table_a, table_b, expected


def write_catalogs(
    table_a: pa.Table,
    table_b: pa.Table,
    dir_path: Path,
    *,
    name_a: str = "catalog_a.parquet",
    name_b: str = "catalog_b.parquet",
) -> tuple[Path, Path]:
    """Write tables to dir_path; return (path_a, path_b)."""
    dir_path.mkdir(parents=True, exist_ok=True)
    path_a = dir_path / name_a
    path_b = dir_path / name_b
    pq.write_table(table_a, path_a)
    pq.write_table(table_b, path_b)
    return path_a, path_b


def read_matches(path: Path) -> pa.Table:
    """Read output Parquet from cross_match."""
    return pq.read_table(path)


def match_table_id_b_col(t: pa.Table) -> str:
    """Return the id_b column name in a cross_match output table (id_b or object_id)."""
    return "id_b" if "id_b" in t.column_names else "object_id"


def match_set_from_table(
    t: pa.Table, id_a_col: str, id_b_col: str
) -> set[tuple[Any, Any]]:
    """Return set of (id_a, id_b) from match table for comparison."""
    col_a = t.column(id_a_col)
    col_b = t.column(id_b_col)
    return {(col_a[i].as_py(), col_b[i].as_py()) for i in range(t.num_rows)}


def reference_cross_match_brute_force(
    table_a: pa.Table,
    table_b: pa.Table,
    radius_arcsec: float,
    *,
    id_col_a: str = "source_id",
    id_col_b: str = "object_id",
    ra_col: str = "ra",
    dec_col: str = "dec",
) -> set[tuple[Any, Any]]:
    """
    Ground-truth cross-match: compare every row of A with every row of B,
    haversine distance, emit (id_a, id_b) if sep <= radius_arcsec.
    Same formula as production; no HEALPix, no partitioning. Use for comparison.
    """
    ra_a = table_a.column(ra_col)
    dec_a = table_a.column(dec_col)
    id_a = table_a.column(id_col_a)
    ra_b = table_b.column(ra_col)
    dec_b = table_b.column(dec_col)
    id_b_col = table_b.column(id_col_b)
    n_a, n_b = len(table_a), len(table_b)
    result: set[tuple[Any, Any]] = set()
    for i in range(n_a):
        ra_i = ra_a[i].as_py() if hasattr(ra_a[i], "as_py") else float(ra_a[i])
        dec_i = dec_a[i].as_py() if hasattr(dec_a[i], "as_py") else float(dec_a[i])
        id_i = id_a[i].as_py() if hasattr(id_a[i], "as_py") else id_a[i]
        for j in range(n_b):
            ra_j = ra_b[j].as_py() if hasattr(ra_b[j], "as_py") else float(ra_b[j])
            dec_j = dec_b[j].as_py() if hasattr(dec_b[j], "as_py") else float(dec_b[j])
            sep = _haversine_arcsec(ra_i, dec_i, ra_j, dec_j)
            if sep <= radius_arcsec:
                id_j = (
                    id_b_col[j].as_py()
                    if hasattr(id_b_col[j], "as_py")
                    else id_b_col[j]
                )
                result.add((id_i, id_j))
    return result
