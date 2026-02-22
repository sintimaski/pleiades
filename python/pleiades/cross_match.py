"""Out-of-core spatial cross-match: HEALPix index + chunked stream (cdshealpix)."""

from __future__ import annotations

import math
import os
import tempfile
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import astropy.units as u
import cdshealpix.nested as cds_nested
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.coordinates import Latitude, Longitude

from pleiades.models import CrossMatchResult
from pleiades.validation import (
    validate_catalog_schema,
    validate_cross_match_args,
    validate_prepartitioned_dir,
)

RA_COL = "ra"
DEC_COL = "dec"
RA_DEC_UNITS_DEFAULT = "deg"  # "deg" or "rad" for input catalog columns
# cdshealpix nested: nside = 2^depth, so depth 8 -> nside 256
DEPTH_DEFAULT = 8
# Benchmark-style defaults (1M rows: batch = rows/4, n_shards = 16)
BATCH_SIZE_A = 250_000
BATCH_SIZE_B = 250_000
# Block size for matrix path: block_n * block_m * 8 bytes (e.g. 4000*4000 = 128 MB)
MATRIX_BLOCK_SIZE = 4000
# Shards for partitioned B (fewer = less I/O overhead, benchmark uses 16)
B_PARTITION_SHARDS = 16


def _lonlat_deg_to_healpix(
    ra_deg: np.ndarray, dec_deg: np.ndarray, depth: int
) -> np.ndarray:
    """Convert ra/dec in degrees to nested HEALPix pixel IDs (cdshealpix expects units)."""
    lon = Longitude(ra_deg, unit=u.deg)
    lat = Latitude(dec_deg, unit=u.deg)
    return cds_nested.lonlat_to_healpix(lon, lat, depth)


def _haversine_matrix_arcsec(
    ra1_deg: np.ndarray,
    dec1_deg: np.ndarray,
    ra2_deg: np.ndarray,
    dec2_deg: np.ndarray,
) -> np.ndarray:
    """Full (n x m) angular distance matrix in arcsec. Vectorized; shape (len(ra1), len(ra2))."""
    ra1 = np.radians(ra1_deg)
    dec1 = np.radians(dec1_deg)
    ra2 = np.radians(ra2_deg)
    dec2 = np.radians(dec2_deg)
    # (n,) and (m,) -> (n, m) via broadcasting
    ddec = dec2 - dec1[:, np.newaxis]
    dra = ra2 - ra1[:, np.newaxis]
    x = (
        np.sin(ddec / 2) ** 2
        + np.cos(dec1[:, np.newaxis]) * np.cos(dec2) * np.sin(dra / 2) ** 2
    )
    dist_rad = 2 * np.arcsin(np.minimum(np.sqrt(x), 1.0))
    return np.degrees(dist_rad) * 3600.0


def _angular_distance_arcsec(
    ra1_deg: np.ndarray,
    dec1_deg: np.ndarray,
    ra2_deg: float,
    dec2_deg: float,
) -> np.ndarray:
    """Angular distance in arcsec from (ra1, dec1) to (ra2, dec2). Vectorized over 1."""
    ra1 = np.radians(ra1_deg)
    dec1 = np.radians(dec1_deg)
    ra2 = math.radians(ra2_deg)
    dec2 = math.radians(dec2_deg)
    ddec = dec2 - dec1
    dra = ra2 - ra1
    x = np.sin(ddec / 2) ** 2 + np.cos(dec1) * np.cos(dec2) * np.sin(dra / 2) ** 2
    dist_rad = 2 * np.arcsin(np.minimum(np.sqrt(x), 1.0))
    return np.degrees(dist_rad) * 3600.0


def _id_column(table: pa.Table, id_col: str | None) -> str:
    """Return first non-ra/dec column as ID if id_col not set."""
    if id_col is not None:
        return id_col
    for name in table.column_names:
        if name.lower() not in (RA_COL, DEC_COL):
            return name
    return "index"


def _cross_match_chunk(
    index: dict[int, list[tuple[Any, float, float]]],
    ra_b: float,
    dec_b: float,
    id_b: Any,
    radius_arcsec: float,
    depth: int,
) -> list[tuple[Any, Any, float]]:
    """Find all A rows within radius_arcsec of (ra_b, dec_b). Returns [(id_a, id_b, sep_arcsec), ...]."""
    center_pix = _lonlat_deg_to_healpix(
        np.array([ra_b], dtype=np.float64),
        np.array([dec_b], dtype=np.float64),
        depth,
    )[0]
    nb = cds_nested.neighbours(np.array([center_pix], dtype=np.uint64), depth)
    pixels = nb[0]
    pixels = pixels[pixels >= 0]
    matches: list[tuple[Any, Any, float]] = []
    for pix in pixels:
        for id_a, ra_a, dec_a in index.get(pix, []):
            sep = _angular_distance_arcsec(
                np.array([ra_a]), np.array([dec_a]), ra_b, dec_b
            )[0]
            if sep <= radius_arcsec:
                matches.append((id_a, id_b, float(sep)))
    return matches


def _infer_id_type(arr: pa.Array) -> pa.DataType:
    """Infer Arrow type from first non-null value or column type."""
    if arr.type == pa.int64() or arr.type == pa.int32():
        return pa.int64()
    if arr.type == pa.string():
        return pa.string()
    if arr.type == pa.float64() or arr.type == pa.float32():
        return pa.float64()
    return pa.string()


def _partition_b_to_shards(
    catalog_b: Path,
    shard_dir: Path,
    depth: int,
    n_shards: int,
    ra_col: str,
    dec_col: str,
    id_col_b: str | None,
    batch_size_b: int,
) -> tuple[str, pa.DataType]:
    """Stream B, compute pixel per row, write to shard Parquet files. Returns (id_b_name, type_b)."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(catalog_b)
    writers: list[pq.ParquetWriter | None] = [None] * n_shards
    id_b_name: str | None = None
    type_b: pa.DataType = pa.string()
    shard_schema: pa.Schema | None = None
    for batch in pf.iter_batches(batch_size=batch_size_b):
        table = pa.Table.from_batches([batch])
        if id_b_name is None:
            id_b_name = _id_column(table, id_col_b)
            type_b = _infer_id_type(table.column(id_b_name))
            shard_schema = pa.schema(
                [
                    ("pixel_id", pa.uint64()),
                    ("id_b", type_b),
                    ("ra", pa.float64()),
                    ("dec", pa.float64()),
                ]
            )
            for s in range(n_shards):
                writers[s] = pq.ParquetWriter(
                    shard_dir / f"shard_{s:04d}.parquet", shard_schema
                )
        ra = table.column(ra_col).to_numpy().astype(np.float64)
        dec = table.column(dec_col).to_numpy().astype(np.float64)
        pix = _lonlat_deg_to_healpix(ra, dec, depth)
        id_col = table.column(id_b_name)
        for i in range(len(table)):
            shard = int(pix[i]) % n_shards
            row = pa.table(
                {
                    "pixel_id": [pix[i]],
                    "id_b": [
                        id_col[i].as_py() if hasattr(id_col[i], "as_py") else id_col[i]
                    ],
                    "ra": [float(ra[i])],
                    "dec": [float(dec[i])],
                },
                schema=shard_schema,
            )
            w = writers[shard]
            assert w is not None
            w.write_table(row)
    for w in writers:
        if w is not None:
            w.close()
    if id_b_name is None:
        schema = pq.read_schema(catalog_b)
        for name in schema.names:
            if name.lower() not in (ra_col, dec_col):
                id_b_name = name
                type_b = schema.field(name).type
                break
        else:
            id_b_name = "object_id"
            type_b = pa.string()
    assert id_b_name is not None
    return id_b_name, type_b


def partition_catalog(
    catalog_path: str | Path,
    output_dir: str | Path,
    *,
    depth: int = DEPTH_DEFAULT,
    n_shards: int = B_PARTITION_SHARDS,
    ra_col: str = RA_COL,
    dec_col: str = DEC_COL,
    id_col: str | None = None,
    batch_size: int = BATCH_SIZE_B,
) -> None:
    """
    Partition a catalog by HEALPix pixel into shard Parquet files.

    Writes shard_0000.parquet, ... to output_dir with columns pixel_id, id_b, ra, dec
    (same layout as pre-partitioned B for cross_match). The catalog's ID column
    is written as id_b so the result can be used as catalog_b in cross_match.

    Raises:
        FileNotFoundError: If catalog_path does not exist.
        CatalogValidationError: If schema validation fails.
    """
    catalog_path = Path(catalog_path)
    output_dir = Path(output_dir)
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")
    validate_catalog_schema(
        catalog_path,
        ra_col=ra_col,
        dec_col=dec_col,
        id_col=id_col,
        must_have_id=True,
    )
    _partition_b_to_shards(
        catalog_path,
        output_dir,
        depth,
        n_shards,
        ra_col,
        dec_col,
        id_col,
        batch_size,
    )


def _pixels_in_chunk_with_neighbors(pix_a: np.ndarray, depth: int) -> set[int]:
    """Return set of pixel IDs in chunk plus all 8 neighbors (for halo)."""
    unique_pix = np.unique(pix_a).astype(np.uint64)
    if len(unique_pix) == 0:
        return set()
    nb = cds_nested.neighbours(unique_pix, depth)
    out = {int(p) for p in nb.ravel() if p >= 0}
    return out


def _load_b_subset_for_pixels(
    shard_dir: Path,
    pixels_wanted: set[int],
    n_shards: int,
    id_b_name: str,
    ra_col: str,
    dec_col: str,
) -> tuple[np.ndarray, np.ndarray, list[Any]]:
    """Load B rows whose pixel_id is in pixels_wanted; return (ra_b, dec_b, id_b_list)."""
    shards_to_read = {p % n_shards for p in pixels_wanted}
    ra_list: list[float] = []
    dec_list: list[float] = []
    id_list: list[Any] = []
    for s in shards_to_read:
        path = shard_dir / f"shard_{s:04d}.parquet"
        if not path.exists():
            continue
        t = pq.read_table(path)
        pix_col = t.column("pixel_id")
        for i in range(len(t)):
            if int(pix_col[i].as_py()) in pixels_wanted:
                ra_list.append(t.column("ra")[i].as_py())
                dec_list.append(t.column("dec")[i].as_py())
                id_list.append(t.column("id_b")[i].as_py())
    return np.array(ra_list), np.array(dec_list), id_list


def _cross_match_chunk_matrix(
    ra_a: np.ndarray,
    dec_a: np.ndarray,
    id_a_list: list[Any],
    ra_b: np.ndarray,
    dec_b: np.ndarray,
    id_b_list: list[Any],
    radius_arcsec: float,
    block_size: int,
) -> tuple[list[Any], list[Any], list[float]]:
    """Vectorized: compute (A x B) distance in blocks, return (id_a, id_b, sep) for sep <= radius."""
    n, m = len(ra_a), len(ra_b)
    buf_a: list[Any] = []
    buf_b: list[Any] = []
    buf_sep: list[float] = []
    for start_i in range(0, n, block_size):
        end_i = min(start_i + block_size, n)
        for start_j in range(0, m, block_size):
            end_j = min(start_j + block_size, m)
            dist = _haversine_matrix_arcsec(
                ra_a[start_i:end_i],
                dec_a[start_i:end_i],
                ra_b[start_j:end_j],
                dec_b[start_j:end_j],
            )
            ii, jj = np.where(radius_arcsec >= dist)
            for i, j in zip(ii, jj, strict=True):
                buf_a.append(id_a_list[start_i + i])
                buf_b.append(id_b_list[start_j + j])
                buf_sep.append(float(dist[i, j]))
    return buf_a, buf_b, buf_sep


def _apply_n_nearest(
    output_path: Path,
    id_a_name: str,
    id_b_name: str,
    n_nearest: int,
    schema: pa.Schema,
) -> None:
    """Keep only the n_nearest smallest-separation matches per id_a; overwrite output."""
    table = pq.read_table(output_path)
    if table.num_rows == 0:
        return
    id_a_col = table.column(id_a_name)
    id_b_col = table.column(id_b_name)
    sep_col = table.column("separation_arcsec")

    # Group by id_a, keep n_nearest smallest separation per group
    by_a: dict[Any, list[tuple[Any, Any, float]]] = {}
    for i in range(table.num_rows):
        a = id_a_col[i].as_py()
        b = id_b_col[i].as_py()
        sep = sep_col[i].as_py()
        by_a.setdefault(a, []).append((a, b, sep))

    new_a: list[Any] = []
    new_b: list[Any] = []
    new_sep: list[float] = []
    for _a, triples in by_a.items():
        triples.sort(key=lambda t: t[2])
        for t in triples[:n_nearest]:
            new_a.append(t[0])
            new_b.append(t[1])
            new_sep.append(t[2])

    out_table = pa.table(
        {id_a_name: new_a, id_b_name: new_b, "separation_arcsec": new_sep},
        schema=schema,
    )
    pq.write_table(out_table, output_path)


def cross_match(
    catalog_a: str | Path,
    catalog_b: str | Path,
    radius_arcsec: float,
    output_path: str | Path,
    *,
    id_col_a: str | None = None,
    id_col_b: str | None = None,
    ra_col: str = RA_COL,
    dec_col: str = DEC_COL,
    ra_dec_units: str = RA_DEC_UNITS_DEFAULT,
    depth: int = DEPTH_DEFAULT,
    batch_size_a: int = BATCH_SIZE_A,
    batch_size_b: int = BATCH_SIZE_B,
    use_matrix: bool = True,
    matrix_block_size: int = MATRIX_BLOCK_SIZE,
    partition_b: bool = True,
    n_shards: int = B_PARTITION_SHARDS,
    use_rust: bool = True,
    n_nearest: int | None = None,
    keep_b_in_memory: bool = False,
    progress_callback: Callable[[int, int | None, int, int], None | bool] | None = None,
    include_coords: bool = False,
) -> CrossMatchResult:
    """
    Cross-match two Parquet catalogs by angular distance (out-of-core).

    When partition_b=True (default), B is first written to shards by HEALPix pixel;
    each A chunk then loads only B rows in the same/neighbor pixels (avoids re-reading
    all of B per chunk). When use_matrix=True, distances are block matrix vectorized.
    Writes id_a, id_b, separation_arcsec.

    If catalog_b is a directory containing shard_0000.parquet, ... (same layout as
    from partitioning), B is treated as pre-partitioned and not re-partitioned.

    If n_nearest is set (e.g. 1 for best match only), only the n_nearest smallest-
    separation matches per id_a are kept in the output.

    Matching is done by the Rust engine (use_rust=True, default). Python is used
    for the API, validation, and helpers; the heavy work runs in the Rust extension.
    If the extension is not installed, install with ``pip install pleiades`` (wheels
    include it) or from source: ``uv run maturin develop``. Set use_rust=False to
    use the Python implementation (slow, for testing or environments without the wheel).

    If include_coords=True (Python path only, catalog_b must be a file), the
    output is augmented with ra_a, dec_a, ra_b, dec_b from the two catalogs.

    ra_dec_units: "deg" (default) or "rad" — units of ra/dec columns in the
    catalogs. Used when use_rust=True; Python fallback assumes degrees.

    batch_size_a, batch_size_b: rows per Parquet read chunk (default 250k each, benchmark-style).
    Smaller values use less RAM and are better on memory-constrained machines
    (e.g. laptops); larger values reduce I/O overhead.

    keep_b_in_memory: when True and catalog_b is a file, partition B into RAM
    instead of temp shard files (faster but uses more memory). Default False
    on purpose: the pipeline is out-of-core and should run on limited RAM.
    Set True only when B is small enough to fit comfortably in memory.

    progress_callback: if provided, called after each chunk with
    (chunk_ix, total_or_none, rows_a_read, matches_count). Return False to
    cancel (Rust engine stops at next chunk boundary; reduces Ctrl+C delay).

    Returns:
        CrossMatchResult with output_path, row counts, match count, and time.
    """
    catalog_a = Path(catalog_a)
    catalog_b = Path(catalog_b)
    output_path = Path(output_path)
    validate_cross_match_args(
        radius_arcsec, n_nearest=n_nearest, depth=depth, n_shards=n_shards
    )
    if not catalog_a.is_file():
        raise FileNotFoundError(f"Catalog A not found: {catalog_a}")

    validate_catalog_schema(
        catalog_a, ra_col=ra_col, dec_col=dec_col, id_col=id_col_a, must_have_id=True
    )

    b_is_prepartitioned = catalog_b.is_dir()
    if b_is_prepartitioned:
        n_shards, _ = validate_prepartitioned_dir(catalog_b)
        shard_path = catalog_b
    else:
        if not catalog_b.is_file():
            raise FileNotFoundError(f"Catalog B not found: {catalog_b}")
        validate_catalog_schema(
            catalog_b,
            ra_col=ra_col,
            dec_col=dec_col,
            id_col=id_col_b,
            must_have_id=True,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    if use_rust:
        try:
            import pleiades_core  # type: ignore[import-untyped]

            gpu_env = os.environ.get("PLEIADES_GPU")
            if (
                gpu_env == "wgpu"
                and not getattr(pleiades_core, "has_wgpu_feature", lambda: False)()
            ):
                import sys

                print(
                    "pleiades: PLEIADES_GPU=wgpu is set but the extension was not built "
                    "with GPU support. Rebuild with: uv run maturin develop --features wgpu",
                    file=sys.stderr,
                )

            try:
                result = pleiades_core.cross_match(
                    str(catalog_a),
                    str(catalog_b),
                    radius_arcsec,
                    str(output_path),
                    depth=depth,
                    batch_size_a=batch_size_a,
                    batch_size_b=batch_size_b,
                    n_shards=n_shards,
                    ra_col=ra_col,
                    dec_col=dec_col,
                    id_col_a=id_col_a,
                    id_col_b=id_col_b,
                    ra_dec_units=ra_dec_units,
                    n_nearest=n_nearest,
                    keep_b_in_memory=keep_b_in_memory,
                    progress_callback=progress_callback,
                )
            except TypeError as e:
                err_str = str(e)
                if "keep_b_in_memory" in err_str and keep_b_in_memory:
                    # Older extension doesn't support keep_b_in_memory; retry without it
                    import warnings

                    warnings.warn(
                        "Rust extension does not support keep_b_in_memory; rebuild with "
                        "uv run maturin develop to enable in-memory B (faster when B fits in RAM).",
                        UserWarning,
                        stacklevel=2,
                    )
                    result = pleiades_core.cross_match(
                        str(catalog_a),
                        str(catalog_b),
                        radius_arcsec,
                        str(output_path),
                        depth=depth,
                        batch_size_a=batch_size_a,
                        batch_size_b=batch_size_b,
                        n_shards=n_shards,
                        ra_col=ra_col,
                        dec_col=dec_col,
                        id_col_a=id_col_a,
                        id_col_b=id_col_b,
                        ra_dec_units=ra_dec_units,
                        n_nearest=n_nearest,
                        progress_callback=progress_callback,
                    )
                elif "progress_callback" in err_str or "n_nearest" in err_str:
                    result = pleiades_core.cross_match(
                        str(catalog_a),
                        str(catalog_b),
                        radius_arcsec,
                        str(output_path),
                        depth=depth,
                        batch_size_a=batch_size_a,
                        batch_size_b=batch_size_b,
                        n_shards=n_shards,
                        ra_col=ra_col,
                        dec_col=dec_col,
                        id_col_a=id_col_a,
                        id_col_b=id_col_b,
                        ra_dec_units=ra_dec_units,
                    )
                elif "ra_dec_units" not in err_str:
                    raise
                elif ra_dec_units.lower() != "deg":
                    raise ValueError(
                        "ra_dec_units='rad' requires rebuilding the Rust extension "
                        "(uv run maturin develop)"
                    ) from e
                else:
                    result = pleiades_core.cross_match(
                        str(catalog_a),
                        str(catalog_b),
                        radius_arcsec,
                        str(output_path),
                        depth=depth,
                        batch_size_a=batch_size_a,
                        batch_size_b=batch_size_b,
                        n_shards=n_shards,
                        ra_col=ra_col,
                        dec_col=dec_col,
                        id_col_a=id_col_a,
                        id_col_b=id_col_b,
                    )
            if result is not None and isinstance(result, dict):
                out_path = result["output_path"]
                if include_coords and catalog_b.is_file():
                    from pleiades.analysis import attach_match_coords

                    attach_match_coords(
                        out_path,
                        catalog_a,
                        catalog_b,
                        out_path,
                        id_col_a=id_col_a,
                        id_col_b=id_col_b,
                        ra_col=ra_col,
                        dec_col=dec_col,
                    )
                return CrossMatchResult(
                    output_path=out_path,
                    rows_a_read=int(result["rows_a_read"]),
                    rows_b_read=int(result["rows_b_read"]),
                    matches_count=int(result["matches_count"]),
                    chunks_processed=int(result["chunks_processed"]),
                    time_seconds=float(result["time_seconds"]),
                )
            # Older Rust extension returns None
            elapsed = time.perf_counter() - t0
            pf_out = pq.ParquetFile(output_path)
            n_match = pf_out.metadata.num_rows if pf_out.metadata else 0
            pf_a = pq.ParquetFile(catalog_a)
            rows_b = (
                sum(
                    pq.read_table(p).num_rows
                    for p in sorted(catalog_b.glob("shard_*.parquet"))
                )
                if catalog_b.is_dir()
                else (
                    pq.ParquetFile(catalog_b).metadata.num_rows
                    if pq.ParquetFile(catalog_b).metadata
                    else 0
                )
            )
            return CrossMatchResult(
                output_path=str(output_path),
                rows_a_read=pf_a.metadata.num_rows if pf_a.metadata else 0,
                rows_b_read=rows_b,
                matches_count=n_match,
                chunks_processed=0,
                time_seconds=elapsed,
            )
        except ImportError:
            if use_rust:
                raise ImportError(
                    "Pleiades requires the Rust engine for cross-match. "
                    "Install it with: pip install pleiades (wheels include it), "
                    "or from source: uv run maturin develop"
                ) from None
            # use_rust=False: fall back to Python implementation

    out_schema: pa.Schema | None = None
    writer: pq.ParquetWriter | None = None
    id_a_name: str | None = None
    id_b_name: str | None = None
    type_a: pa.DataType = pa.int64()
    type_b: pa.DataType = pa.string()
    rows_a_read = 0
    rows_b_read = 0
    matches_count = 0
    chunks_processed = 0

    if partition_b:
        if b_is_prepartitioned:
            shard_path = catalog_b
            id_b_name = "id_b"
            first_shard = next(shard_path.glob("shard_*.parquet"))
            t0_shard = pq.read_table(first_shard)
            type_b = _infer_id_type(t0_shard.column("id_b"))
            for p in sorted(shard_path.glob("shard_*.parquet")):
                rows_b_read += pq.read_table(p).num_rows
        else:
            with tempfile.TemporaryDirectory(prefix="pleiades_b_") as shard_dir:
                shard_path = Path(shard_dir)
                id_b_name, type_b = _partition_b_to_shards(
                    catalog_b,
                    shard_path,
                    depth,
                    n_shards,
                    ra_col,
                    dec_col,
                    id_col_b,
                    batch_size_b,
                )
                for s in range(n_shards):
                    p = shard_path / f"shard_{s:04d}.parquet"
                    if p.exists():
                        rows_b_read += pq.read_table(p).num_rows
                pf_a = pq.ParquetFile(catalog_a)
                for batch_a in pf_a.iter_batches(batch_size=batch_size_a):
                    table_a = pa.Table.from_batches([batch_a])
                    if id_a_name is None:
                        id_a_name = _id_column(table_a, id_col_a)
                        type_a = _infer_id_type(table_a.column(id_a_name))
                    rows_a_read += table_a.num_rows
                    chunks_processed += 1
                    ra_a = table_a.column(ra_col).to_numpy()
                    dec_a = table_a.column(dec_col).to_numpy()
                    id_a_arr = table_a.column(id_a_name)
                    pix_a = _lonlat_deg_to_healpix(ra_a, dec_a, depth)
                    pixels_wanted = _pixels_in_chunk_with_neighbors(pix_a, depth)
                    ra_b, dec_b, id_b_list = _load_b_subset_for_pixels(
                        shard_path, pixels_wanted, n_shards, id_b_name, ra_col, dec_col
                    )
                    if len(id_b_list) == 0:
                        if progress_callback is not None:
                            progress_callback(
                                chunks_processed, None, rows_a_read, matches_count
                            )
                        continue
                    id_a_list = [id_a_arr[i].as_py() for i in range(len(table_a))]
                    buf_a, buf_b, buf_sep = _cross_match_chunk_matrix(
                        ra_a,
                        dec_a,
                        id_a_list,
                        ra_b,
                        dec_b,
                        id_b_list,
                        radius_arcsec,
                        matrix_block_size,
                    )
                    matches_count += len(buf_a)
                    if progress_callback is not None:
                        progress_callback(
                            chunks_processed, None, rows_a_read, matches_count
                        )
                    if buf_a and id_a_name and id_b_name:
                        if out_schema is None:
                            out_schema = pa.schema(
                                [
                                    (id_a_name, type_a),
                                    (id_b_name, type_b),
                                    ("separation_arcsec", pa.float64()),
                                ]
                            )
                            writer = pq.ParquetWriter(output_path, out_schema)
                        tbl = pa.table(
                            {
                                id_a_name: buf_a,
                                id_b_name: buf_b,
                                "separation_arcsec": buf_sep,
                            },
                            schema=out_schema,
                        )
                        assert writer is not None
                        writer.write_table(tbl)

        if b_is_prepartitioned:
            pf_a = pq.ParquetFile(catalog_a)
            for batch_a in pf_a.iter_batches(batch_size=batch_size_a):
                table_a = pa.Table.from_batches([batch_a])
                if id_a_name is None:
                    id_a_name = _id_column(table_a, id_col_a)
                    type_a = _infer_id_type(table_a.column(id_a_name))
                rows_a_read += table_a.num_rows
                chunks_processed += 1
                ra_a = table_a.column(ra_col).to_numpy()
                dec_a = table_a.column(dec_col).to_numpy()
                id_a_arr = table_a.column(id_a_name)
                pix_a = _lonlat_deg_to_healpix(ra_a, dec_a, depth)
                pixels_wanted = _pixels_in_chunk_with_neighbors(pix_a, depth)
                ra_b, dec_b, id_b_list = _load_b_subset_for_pixels(
                    shard_path, pixels_wanted, n_shards, id_b_name, ra_col, dec_col
                )
                if len(id_b_list) == 0:
                    if progress_callback is not None:
                        progress_callback(
                            chunks_processed, None, rows_a_read, matches_count
                        )
                    continue
                id_a_list = [id_a_arr[i].as_py() for i in range(len(table_a))]
                buf_a, buf_b, buf_sep = _cross_match_chunk_matrix(
                    ra_a,
                    dec_a,
                    id_a_list,
                    ra_b,
                    dec_b,
                    id_b_list,
                    radius_arcsec,
                    matrix_block_size,
                )
                matches_count += len(buf_a)
                if progress_callback is not None:
                    progress_callback(
                        chunks_processed, None, rows_a_read, matches_count
                    )
                if buf_a and id_a_name and id_b_name:
                    if out_schema is None:
                        out_schema = pa.schema(
                            [
                                (id_a_name, type_a),
                                (id_b_name, type_b),
                                ("separation_arcsec", pa.float64()),
                            ]
                        )
                        writer = pq.ParquetWriter(output_path, out_schema)
                    tbl = pa.table(
                        {
                            id_a_name: buf_a,
                            id_b_name: buf_b,
                            "separation_arcsec": buf_sep,
                        },
                        schema=out_schema,
                    )
                    assert writer is not None
                    writer.write_table(tbl)
        else:
            # Recompute rows_b_read from catalog B (partitioning already done in temp dir)
            pf_b = pq.ParquetFile(catalog_b)
            rows_b_read = pf_b.metadata.num_rows if pf_b.metadata else 0
    else:
        pf_a = pq.ParquetFile(catalog_a)
        pf_b = pq.ParquetFile(catalog_b)
        for batch_a in pf_a.iter_batches(batch_size=batch_size_a):
            table_a = pa.Table.from_batches([batch_a])
            if id_a_name is None:
                id_a_name = _id_column(table_a, id_col_a)
                type_a = _infer_id_type(table_a.column(id_a_name))
            rows_a_read += table_a.num_rows
            chunks_processed += 1
            ra_a = table_a.column(ra_col).to_numpy()
            dec_a = table_a.column(dec_col).to_numpy()
            id_a_arr = table_a.column(id_a_name)
            index: dict[int, list[tuple[Any, float, float]]] = {}
            if not use_matrix:
                pix_a = _lonlat_deg_to_healpix(ra_a, dec_a, depth)
                for i in range(len(table_a)):
                    pix = int(pix_a[i])
                    ra = float(ra_a[i])
                    dec = float(dec_a[i])
                    id_val = (
                        id_a_arr[i].as_py()
                        if hasattr(id_a_arr[i], "as_py")
                        else id_a_arr[i]
                    )
                    index.setdefault(pix, []).append((id_val, ra, dec))

            for batch_b in pf_b.iter_batches(batch_size=batch_size_b):
                table_b = pa.Table.from_batches([batch_b])
                if id_b_name is None:
                    id_b_name = _id_column(table_b, id_col_b)
                    type_b = _infer_id_type(table_b.column(id_b_name))
                ra_b = table_b.column(ra_col).to_numpy()
                dec_b = table_b.column(dec_col).to_numpy()
                id_b_col = table_b.column(id_b_name)
                id_b_list = [id_b_col[j].as_py() for j in range(len(table_b))]
                if use_matrix:
                    id_a_list = [id_a_arr[i].as_py() for i in range(len(table_a))]
                    buf_a, buf_b, buf_sep = _cross_match_chunk_matrix(
                        ra_a,
                        dec_a,
                        id_a_list,
                        ra_b,
                        dec_b,
                        id_b_list,
                        radius_arcsec,
                        matrix_block_size,
                    )
                else:
                    buf_a, buf_b, buf_sep = [], [], []
                    for j in range(len(table_b)):
                        ra_b_j = float(ra_b[j])
                        dec_b_j = float(dec_b[j])
                        id_b_val = id_b_list[j]
                        for id_a, id_b, sep in _cross_match_chunk(
                            index, ra_b_j, dec_b_j, id_b_val, radius_arcsec, depth
                        ):
                            buf_a.append(id_a)
                            buf_b.append(id_b)
                            buf_sep.append(sep)
                matches_count += len(buf_a)
                if progress_callback is not None:
                    progress_callback(
                        chunks_processed, None, rows_a_read, matches_count
                    )
                if buf_a and id_a_name and id_b_name:
                    if out_schema is None:
                        out_schema = pa.schema(
                            [
                                (id_a_name, type_a),
                                (id_b_name, type_b),
                                ("separation_arcsec", pa.float64()),
                            ]
                        )
                        writer = pq.ParquetWriter(output_path, out_schema)
                    tbl = pa.table(
                        {
                            id_a_name: buf_a,
                            id_b_name: buf_b,
                            "separation_arcsec": buf_sep,
                        },
                        schema=out_schema,
                    )
                    assert writer is not None
                    writer.write_table(tbl)
        rows_b_read = pf_b.metadata.num_rows if pf_b.metadata else 0

    if writer is not None:
        writer.close()
    elif out_schema is None:
        out_schema = pa.schema(
            [
                ("source_id", pa.int64()),
                ("object_id", pa.string()),
                ("separation_arcsec", pa.float64()),
            ]
        )
        pq.write_table(
            pa.table(
                {"source_id": [], "object_id": [], "separation_arcsec": []},
                schema=out_schema,
            ),
            output_path,
        )

    if (
        n_nearest is not None
        and out_schema is not None
        and id_a_name is not None
        and id_b_name is not None
    ):
        _apply_n_nearest(output_path, id_a_name, id_b_name, n_nearest, out_schema)
        matches_count = pq.read_table(output_path).num_rows

    if include_coords and not b_is_prepartitioned:
        from pleiades.analysis import attach_match_coords

        attach_match_coords(
            output_path,
            catalog_a,
            catalog_b,
            output_path,
            id_col_a=id_col_a,
            id_col_b=id_col_b,
            ra_col=ra_col,
            dec_col=dec_col,
        )

    elapsed = time.perf_counter() - t0
    return CrossMatchResult(
        output_path=str(output_path),
        rows_a_read=rows_a_read,
        rows_b_read=rows_b_read,
        matches_count=matches_count,
        chunks_processed=chunks_processed,
        time_seconds=elapsed,
    )


def cross_match_iter(
    catalog_a: str | Path,
    catalog_b: str | Path,
    radius_arcsec: float,
    *,
    id_col_a: str | None = None,
    id_col_b: str | None = None,
    ra_col: str = RA_COL,
    dec_col: str = DEC_COL,
    depth: int = DEPTH_DEFAULT,
    batch_size_a: int = BATCH_SIZE_A,
    batch_size_b: int = BATCH_SIZE_B,
    matrix_block_size: int = MATRIX_BLOCK_SIZE,
    partition_b: bool = True,
    n_shards: int = B_PARTITION_SHARDS,
) -> Iterator[tuple[Any, Any, float]]:
    """
    Stream cross-match results as (id_a, id_b, separation_arcsec) tuples.

    Same semantics as cross_match() but yields match tuples instead of writing
    to a file. Use for pipelines that process matches in a streaming fashion.
    Only the partition_b path is supported (catalog_b may be a file or a
    pre-partitioned directory).
    """
    catalog_a = Path(catalog_a)
    catalog_b = Path(catalog_b)
    if not catalog_a.is_file():
        raise FileNotFoundError(f"Catalog A not found: {catalog_a}")
    validate_catalog_schema(
        catalog_a, ra_col=ra_col, dec_col=dec_col, id_col=id_col_a, must_have_id=True
    )
    b_is_prepartitioned = catalog_b.is_dir()
    if b_is_prepartitioned:
        n_shards, _ = validate_prepartitioned_dir(catalog_b)
        shard_path = catalog_b
        id_b_name = "id_b"
        first_shard = next(shard_path.glob("shard_*.parquet"))
        type_b = _infer_id_type(pq.read_table(first_shard).column("id_b"))
    else:
        if not catalog_b.is_file():
            raise FileNotFoundError(f"Catalog B not found: {catalog_b}")
        validate_catalog_schema(
            catalog_b,
            ra_col=ra_col,
            dec_col=dec_col,
            id_col=id_col_b,
            must_have_id=True,
        )
        with tempfile.TemporaryDirectory(prefix="pleiades_b_") as shard_dir:
            shard_path = Path(shard_dir)
            id_b_name, type_b = _partition_b_to_shards(
                catalog_b,
                shard_path,
                depth,
                n_shards,
                ra_col,
                dec_col,
                id_col_b,
                batch_size_b,
            )
            yield from _cross_match_iter_core(
                catalog_a,
                shard_path,
                ra_col,
                dec_col,
                id_col_a,
                id_b_name,
                depth,
                n_shards,
                batch_size_a,
                radius_arcsec,
                matrix_block_size,
            )
        return
    yield from _cross_match_iter_core(
        catalog_a,
        shard_path,
        ra_col,
        dec_col,
        id_col_a,
        id_b_name,
        depth,
        n_shards,
        batch_size_a,
        radius_arcsec,
        matrix_block_size,
    )


def _cross_match_iter_core(
    catalog_a: Path,
    shard_path: Path,
    ra_col: str,
    dec_col: str,
    id_col_a: str | None,
    id_b_name: str,
    depth: int,
    n_shards: int,
    batch_size_a: int,
    radius_arcsec: float,
    matrix_block_size: int,
) -> Iterator[tuple[Any, Any, float]]:
    pf_a = pq.ParquetFile(catalog_a)
    id_a_name: str | None = None
    for batch_a in pf_a.iter_batches(batch_size=batch_size_a):
        table_a = pa.Table.from_batches([batch_a])
        if id_a_name is None:
            id_a_name = _id_column(table_a, id_col_a)
        ra_a = table_a.column(ra_col).to_numpy()
        dec_a = table_a.column(dec_col).to_numpy()
        id_a_arr = table_a.column(id_a_name)
        pix_a = _lonlat_deg_to_healpix(ra_a, dec_a, depth)
        pixels_wanted = _pixels_in_chunk_with_neighbors(pix_a, depth)
        ra_b, dec_b, id_b_list = _load_b_subset_for_pixels(
            shard_path, pixels_wanted, n_shards, id_b_name, ra_col, dec_col
        )
        if len(id_b_list) == 0:
            continue
        id_a_list = [id_a_arr[i].as_py() for i in range(len(table_a))]
        buf_a, buf_b, buf_sep = _cross_match_chunk_matrix(
            ra_a,
            dec_a,
            id_a_list,
            ra_b,
            dec_b,
            id_b_list,
            radius_arcsec,
            matrix_block_size,
        )
        for i in range(len(buf_a)):
            yield (buf_a[i], buf_b[i], buf_sep[i])
