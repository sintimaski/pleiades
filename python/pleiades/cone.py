"""Cone search: find all catalog rows within radius_arcsec of (ra_deg, dec_deg)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pleiades.cross_match import (
    DEC_COL,
    RA_COL,
    RA_DEC_UNITS_DEFAULT,
    _angular_distance_arcsec,
)
from pleiades.validation import validate_catalog_schema

BATCH_SIZE = 100_000
RAD_TO_DEG = 180.0 / math.pi


def cone_search(
    catalog_path: str | Path,
    ra_deg: float,
    dec_deg: float,
    radius_arcsec: float,
    output_path: str | Path,
    *,
    ra_col: str = RA_COL,
    dec_col: str = DEC_COL,
    ra_dec_units: str = RA_DEC_UNITS_DEFAULT,
    id_col: str | None = None,
    batch_size: int = BATCH_SIZE,
) -> int:
    """
    Find all rows in the catalog within radius_arcsec of (ra_deg, dec_deg).

    Reads the catalog in chunks, computes angular distance to the center,
    and writes rows with separation_arcsec <= radius_arcsec to output_path.
    Output schema is the same as the catalog plus a "separation_arcsec" column.

    ra_deg, dec_deg are always in degrees. ra_dec_units ("deg" or "rad") specifies
    the units of the catalog's ra/dec columns; they are converted to degrees
    internally when needed.

    Returns:
        Number of rows written.
    """
    path = Path(catalog_path)
    path_out = Path(output_path)
    if not path.is_file():
        raise FileNotFoundError(f"Catalog not found: {path}")

    validate_catalog_schema(
        path, ra_col=ra_col, dec_col=dec_col, id_col=id_col, must_have_id=True
    )
    path_out.parent.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(path)
    writer: pq.ParquetWriter | None = None
    out_schema: pa.Schema | None = None
    total_written = 0

    to_deg = RAD_TO_DEG if ra_dec_units.lower() == "rad" else 1.0
    for batch in pf.iter_batches(batch_size=batch_size):
        table = pa.Table.from_batches([batch])
        ra = table.column(ra_col).to_numpy().astype(np.float64) * to_deg
        dec = table.column(dec_col).to_numpy().astype(np.float64) * to_deg
        sep = _angular_distance_arcsec(ra, dec, ra_deg, dec_deg)
        mask = sep <= radius_arcsec
        if not np.any(mask):
            continue
        subset = table.filter(pa.array(mask))
        sep_subset = sep[mask]
        new_columns = list(subset.columns)
        new_names = list(subset.column_names)
        new_columns.append(pa.array(sep_subset.tolist()))
        new_names.append("separation_arcsec")
        extended = pa.table(
            dict(zip(new_names, new_columns, strict=True)),
            schema=pa.schema(
                list(subset.schema) + [("separation_arcsec", pa.float64())]
            ),
        )
        if out_schema is None:
            out_schema = extended.schema
            writer = pq.ParquetWriter(path_out, out_schema)
        writer.write_table(extended)
        total_written += extended.num_rows

    if writer is not None:
        writer.close()
    else:
        schema_in = pq.read_schema(path)
        out_schema = pa.schema(list(schema_in) + [("separation_arcsec", pa.float64())])
        empty_cols = {
            n: pa.array([], type=schema_in.field(n).type) for n in schema_in.names
        }
        empty_cols["separation_arcsec"] = pa.array([], type=pa.float64())
        pq.write_table(pa.table(empty_cols, schema=out_schema), path_out)

    return total_written


def batch_cone_search(
    catalog_path: str | Path,
    queries: list[tuple[float, float, float]],
    output_path: str | Path,
    *,
    ra_col: str = RA_COL,
    dec_col: str = DEC_COL,
    ra_dec_units: str = RA_DEC_UNITS_DEFAULT,
    id_col: str | None = None,
    batch_size: int = BATCH_SIZE,
) -> int:
    """
    Find all catalog rows that fall within at least one of the given cones.

    Each query is (ra_deg, dec_deg, radius_arcsec). For each row written,
    the output includes the catalog columns plus separation_arcsec and
    query_index (0-based index of the first matching query).

    Returns:
        Number of rows written.
    """
    path = Path(catalog_path)
    path_out = Path(output_path)
    if not path.is_file():
        raise FileNotFoundError(f"Catalog not found: {path}")
    validate_catalog_schema(
        path, ra_col=ra_col, dec_col=dec_col, id_col=id_col, must_have_id=True
    )
    path_out.parent.mkdir(parents=True, exist_ok=True)
    to_deg = RAD_TO_DEG if ra_dec_units.lower() == "rad" else 1.0
    writer: pq.ParquetWriter | None = None
    out_schema: pa.Schema | None = None
    total_written = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
        table = pa.Table.from_batches([batch])
        ra = table.column(ra_col).to_numpy().astype(np.float64) * to_deg
        dec = table.column(dec_col).to_numpy().astype(np.float64) * to_deg
        best_sep = np.full(len(ra), np.inf)
        best_idx = np.full(len(ra), -1, dtype=np.int32)
        for qi, (qra, qdec, qrad) in enumerate(queries):
            sep = _angular_distance_arcsec(ra, dec, qra, qdec)
            within = sep <= qrad
            better = within & (sep < best_sep)
            best_sep = np.where(better, sep, best_sep)
            best_idx = np.where(better, qi, best_idx)
        mask = best_idx >= 0
        if not np.any(mask):
            continue
        subset = table.filter(pa.array(mask))
        sep_subset = best_sep[mask].tolist()
        idx_subset = best_idx[mask].tolist()
        new_columns = list(subset.columns)
        new_names = list(subset.column_names)
        new_columns.append(pa.array(sep_subset))
        new_names.append("separation_arcsec")
        new_columns.append(pa.array(idx_subset))
        new_names.append("query_index")
        extended = pa.table(
            dict(zip(new_names, new_columns, strict=True)),
            schema=pa.schema(
                list(subset.schema)
                + [
                    ("separation_arcsec", pa.float64()),
                    ("query_index", pa.int32()),
                ]
            ),
        )
        if out_schema is None:
            out_schema = extended.schema
            writer = pq.ParquetWriter(path_out, out_schema)
        writer.write_table(extended)
        total_written += extended.num_rows
    if writer is not None:
        writer.close()
    else:
        schema_in = pq.read_schema(path)
        out_schema = pa.schema(
            list(schema_in)
            + [("separation_arcsec", pa.float64()), ("query_index", pa.int32())]
        )
        empty_cols = {
            n: pa.array([], type=schema_in.field(n).type) for n in schema_in.names
        }
        empty_cols["separation_arcsec"] = pa.array([], type=pa.float64())
        empty_cols["query_index"] = pa.array([], type=pa.int32())
        pq.write_table(pa.table(empty_cols, schema=out_schema), path_out)
    return total_written
