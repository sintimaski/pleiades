"""Generate a single large Parquet catalog for out-of-core POC.

Writes in row-groups so streaming readers see chunked layout.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# Real-world scale: Gaia has ~1.8e9; 10M is a realistic single-node test.
ROWS_DEFAULT = 10_000_000
ROWS_PER_ROW_GROUP = 100_000


def _simple_rng(seed: int):
    state = seed & 0xFFFFFFFF
    while True:
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        yield state / 0x7FFFFFFF


def generate_large_catalog(
    output_path: Path,
    n_rows: int = ROWS_DEFAULT,
    rows_per_row_group: int = ROWS_PER_ROW_GROUP,
    seed: int = 999,
) -> None:
    """Write a large Parquet file in row-groups (chunked on disk)."""
    schema = pa.schema([
        ("source_id", pa.int64()),
        ("ra", pa.float64()),
        ("dec", pa.float64()),
    ])
    rng = _simple_rng(seed)
    written = 0
    with pq.ParquetWriter(output_path, schema) as writer:
        while written < n_rows:
            chunk = min(rows_per_row_group, n_rows - written)
            ra = [360.0 * next(rng) for _ in range(chunk)]
            dec_rad = [math.asin(2.0 * next(rng) - 1.0) for _ in range(chunk)]
            dec = [math.degrees(d) for d in dec_rad]
            source_id = list(range(written + 1, written + chunk + 1))
            table = pa.table(
                {"source_id": source_id, "ra": ra, "dec": dec},
                schema=schema,
            )
            writer.write_table(table, row_group_size=chunk)
            written += chunk
    print(f"Wrote {output_path} ({written} rows, {written // rows_per_row_group} row-groups)")


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "catalog_large.parquet"
    n_rows = int(sys.argv[1]) if len(sys.argv) > 1 else ROWS_DEFAULT
    generate_large_catalog(output_path, n_rows=n_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
