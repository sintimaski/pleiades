"""POC: out-of-core Parquet processing (stream read → process → stream write).

Reads a Parquet file in batches without loading it fully into memory, adds a
spatial index column (simple grid cell id), and writes the result in chunks.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# Simple equatorial grid: cell index from (ra, dec) in degrees (vectorized).
GRID_RA_DEG = 1.0
GRID_DEC_DEG = 1.0


def add_pixel_column(batch: pa.RecordBatch, ra_name: str = "ra", dec_name: str = "dec") -> pa.Table:
    """Add a 'pixel_id' int64 column from ra/dec; vectorized for large chunks."""
    table = pa.Table.from_batches([batch])
    ra = table.column(ra_name)
    dec = table.column(dec_name)
    i_ra = pc.floor(pc.divide(ra, GRID_RA_DEG))
    i_ra = pc.if_else(pc.less(i_ra, 0), pc.add(i_ra, 360), i_ra)
    i_ra = pc.subtract(i_ra, pc.multiply(360, pc.floor(pc.divide(i_ra, 360))))
    i_ra = pc.cast(i_ra, pa.int64())
    i_dec = pc.floor(pc.divide(pc.add(dec, 90.0), GRID_DEC_DEG))
    i_dec = pc.if_else(pc.less(i_dec, 0), 0, i_dec)
    i_dec = pc.if_else(pc.greater(i_dec, 179), 179, i_dec)
    i_dec = pc.cast(i_dec, pa.int64())
    pixel = pc.add(pc.multiply(i_ra, 180), i_dec)
    return table.append_column("pixel_id", pixel)


def stream_process(
    input_path: Path,
    output_path: Path,
    batch_size: int = 50_000,
) -> tuple[int, int, float]:
    """
    Stream read → process (add pixel_id) → stream write.
    Returns (total_rows, num_batches, elapsed_seconds).
    """
    total_rows = 0
    num_batches = 0
    output_schema = None
    writer = None
    start = time.perf_counter()
    pf = pq.ParquetFile(input_path)
    for batch in pf.iter_batches(batch_size=batch_size):
        processed = add_pixel_column(batch)
        if output_schema is None:
            output_schema = processed.schema
            writer = pq.ParquetWriter(output_path, output_schema)
        writer.write_table(processed)
        total_rows += processed.num_rows
        num_batches += 1
        print(f"  chunk {num_batches}: {processed.num_rows} rows (total {total_rows})")
    if writer is not None:
        writer.close()
    elapsed = time.perf_counter() - start
    return total_rows, num_batches, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description="POC: stream process large Parquet (add pixel_id).")
    parser.add_argument("input", type=Path, nargs="?", help="Input Parquet path")
    parser.add_argument("-o", "--output", type=Path, help="Output Parquet path")
    parser.add_argument("-b", "--batch-size", type=int, default=100_000, help="Rows per chunk")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    input_path = args.input or data_dir / "catalog_large.parquet"
    output_path = args.output or data_dir / "catalog_large_indexed.parquet"
    if not input_path.is_file():
        print(f"Input not found: {input_path}", file=sys.stderr)
        print("Generate it first: uv run python scripts/generate_large_catalog.py [n_rows]", file=sys.stderr)
        return 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Batch size: {args.batch_size}")
    total, batches, elapsed = stream_process(input_path, output_path, batch_size=args.batch_size)
    print(f"Done: {total} rows, {batches} chunks, {elapsed:.2f}s ({total / elapsed:.0f} rows/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
