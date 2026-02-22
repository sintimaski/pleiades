# AstroJoin

**High-performance out-of-core spatial cross-matcher for astronomical catalogs.**

Cross-match two Parquet catalogs by angular distance (e.g. Gaia × SDSS) using HEALPix indexing (cdshealpix) and chunked streaming. Python implementation is default; optional Rust engine for scale via `use_rust=True` (requires `maturin develop`).

## Installation

- **Requirements**: Python ≥3.10. For building the optional Rust extension: Rust toolchain.
- **From source**:
  ```bash
  uv sync
  uv run maturin develop   # optional: build Rust extension
  ```

## Usage (MVP)

```python
import astrojoin

astrojoin.cross_match(
    catalog_a="catalog_a.parquet",
    catalog_b="catalog_b.parquet",
    radius_arcsec=2.0,
    output_path="matches.parquet",
)
```

- Catalogs must have **ra** and **dec** columns (configurable via `ra_col` / `dec_col`). Coordinates can be in **degrees** (default) or **radians** — set `ra_dec_units="rad"` for LSST/Rubin and other radian-based catalogs. **Float32 or Float64** ra/dec are supported (Rust engine).
- ID columns are auto-detected (e.g. `source_id`, `object_id`) or set via `id_col_a=` / `id_col_b=`.
- Output: Parquet with **id_a**, **id_b**, **separation_arcsec**. `cross_match()` returns a **CrossMatchResult** (output_path, row counts, match count, time).

**Options**: `partition_b=True` (default); `ra_dec_units="deg"` | `"rad"`; `n_nearest=1` for best match only; `progress_callback=(chunk_ix, total, rows_a, matches)`; `catalog_b` may be a **directory** of pre-partitioned shards (`shard_0000.parquet`, ...). Rust: `use_rust=True`; full API parity (`depth`, `batch_size_*`, `ra_col`, `dec_col`, `id_col_*`, `ra_dec_units`).

### Analysis and helpers

- **summarize_matches**(`matches.parquet`) → **MatchSummary** (counts, separation min/max/mean/median).
- **merge_match_to_catalog**(matches_path, catalog_path, output_path, catalog_side=`"a"`) — attach match columns to a catalog.
- **filter_matches_by_radius**(matches_path, max_radius_arcsec, output_path) — subset by separation.
- **multi_radius_cross_match**(catalog_a, catalog_b, radii=[1, 2, 5], output_dir) — one run, one file per radius.

### Cone search and streaming

- **cone_search**(catalog_path, ra_deg, dec_deg, radius_arcsec, output_path, ra_dec_units=`"deg"`) — all rows within radius of a point (center in degrees; catalog ra/dec in deg or rad).
- **cross_match_iter**(catalog_a, catalog_b, radius_arcsec) — yields `(id_a, id_b, separation_arcsec)` tuples.

## Project layout

- `python/astrojoin/` – Python API, cross-match, analysis, cone search (HEALPix + stream I/O).
- `src/` – Rust engine (arrow, cdshealpix, HEALPix join); optional, built with `maturin develop`.
- `tests/` – Unit and integration tests; `tests/fixtures/` – small Parquet catalogs.

## Development

- **Lint/format**: `uv run ruff check . && uv run ruff format .`
- **Type check**: `uv run mypy python/`
- **Tests** (Python + Rust): `uv run python run_tests.py` — or Python only: `uv run pytest`
- **Pre-commit**: `uv run pre-commit install` then `pre-commit run --all-files`

## License

See repository license.
