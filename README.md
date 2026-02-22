# AstroJoin

**High-performance out-of-core spatial cross-matcher for astronomical catalogs.**

Cross-match two Parquet catalogs by angular distance (e.g. Gaia × SDSS) using HEALPix indexing and chunked streaming. **Python is the interface** (API, validation, CLI, analysis helpers); **the Rust engine does the matching** and is included in PyPI wheels. If the extension is not installed, you get a clear error; set `use_rust=False` to use the slow Python implementation (e.g. for testing).

## Installation

**Plug and play** (no Rust required):

```bash
pip install astrojoin
# or
uv add astrojoin
```

This installs the Python package and the Rust extension wheel for your platform (Python 3.10–3.12). You can use `use_rust=True` immediately. For preparing real catalogs (Gaia, LSST, SDSS, etc.), see [DATA_SOURCES.md](DATA_SOURCES.md).

**From source** (developers):

```bash
git clone <repo> && cd astro
uv sync
uv run maturin develop   # build Rust extension for current env
```

## Usage

### Cross-match (Python API)

```python
import astrojoin

result = astrojoin.cross_match(
    catalog_a="catalog_a.parquet",
    catalog_b="catalog_b.parquet",
    radius_arcsec=2.0,
    output_path="matches.parquet",
)
# result: CrossMatchResult(output_path, rows_a_read, rows_b_read, matches_count, time_seconds)
```

- Catalogs must have **ra** and **dec** columns (configurable via `ra_col` / `dec_col`). Coordinates can be in **degrees** (default) or **radians** — set `ra_dec_units="rad"` for LSST/Rubin. **Float32 or Float64** ra/dec supported (Rust engine).
- ID columns are auto-detected (e.g. `source_id`, `object_id`) or set via `id_col_a` / `id_col_b`.
- Output: Parquet with **id_a**, **id_b**, **separation_arcsec**. Optionally **ra_a**, **dec_a**, **ra_b**, **dec_b** with `include_coords=True` (Python path, catalog_b must be a file).

**Options**: `partition_b=True` (default); `ra_dec_units="deg"` | `"rad"`; `n_nearest=1` for best match only; `progress_callback=(chunk_ix, total, rows_a, matches)`; `catalog_b` may be a **directory** of pre-partitioned shards (`shard_0000.parquet`, ...). The Rust engine is used by default (`use_rust=True`); set `use_rust=False` for the Python implementation (slow).

### Command-line interface (CLI)

```bash
# Cross-match (Rust engine by default; use --no-rust for Python path)
astrojoin cross-match catalog_a.parquet catalog_b.parquet -r 2.0 -o matches.parquet

# Summarize matches
astrojoin summarize-matches matches.parquet

# Cone search
astrojoin cone-search catalog.parquet 180.0 0.0 -r 3600 -o cone.parquet

# Partition a catalog into HEALPix shards (for use as pre-partitioned B)
astrojoin partition-catalog catalog.parquet ./shards --depth 8 --n-shards 512
```

### Analysis and helpers

- **summarize_matches**(matches_path) → **MatchSummary** (counts, separation min/max/mean/median).
- **match_stats**(matches_path, separation_percentiles=[25, 50, 75]) → dict with stats and optional percentiles.
- **match_quality_summary**(matches_path, rows_a, rows_b) → fraction of A/B rows with at least one match.
- **merge_match_to_catalog**(matches_path, catalog_path, output_path, catalog_side=`"a"`) — attach match columns to a catalog.
- **filter_matches_by_radius**(matches_path, max_radius_arcsec, output_path) — subset by separation.
- **multi_radius_cross_match**(catalog_a, catalog_b, radii=[1, 2, 5], output_dir) — one run, one file per radius.
- **attach_match_coords**(matches_path, catalog_a, catalog_b, output_path) — add ra_a, dec_a, ra_b, dec_b to matches.

### Cone search and partitioning

- **cone_search**(catalog_path, ra_deg, dec_deg, radius_arcsec, output_path, ra_dec_units=`"deg"`) — all rows within radius of a point.
- **batch_cone_search**(catalog_path, queries, output_path) — queries = list of `(ra_deg, dec_deg, radius_arcsec)`; one pass, output includes `query_index` and `separation_arcsec`.
- **partition_catalog**(catalog_path, output_dir, depth=8, n_shards=512, ...) — write HEALPix shards (pixel_id, id_b, ra, dec) for use as pre-partitioned B.

### Streaming

- **cross_match_iter**(catalog_a, catalog_b, radius_arcsec) — yields `(id_a, id_b, separation_arcsec)` tuples.

### Progress (UX)

Use a progress callback for long runs. With **tqdm** (optional):

```python
from tqdm import tqdm
pbar = tqdm(desc="cross-match", unit=" rows")
def progress(chunk_ix, total, rows_a, matches):
    pbar.n = rows_a
    pbar.set_postfix(matches=matches)
    pbar.refresh()
astrojoin.cross_match(..., progress_callback=progress)
pbar.close()
```

## Benchmark

```bash
uv run python scripts/benchmark_cross_match.py --rows 100000
uv run python scripts/benchmark_cross_match.py --rows 50000 --rust   # compare Python vs Rust
```

For tuning (batch sizes, n_shards, depth) and notes on CPU SIMD vs GPU (CUDA, wgpu, OpenCL), see [PERFORMANCE.md](PERFORMANCE.md).

## Project layout

- `python/astrojoin/` – Python API, cross-match, analysis, cone search, CLI (HEALPix + stream I/O).
- `src/` – Rust engine (arrow, cdshealpix, HEALPix join, pre-partitioned B, rayon); optional, built with `maturin develop`.
- `tests/` – Unit and integration tests; `tests/fixtures/` – small Parquet catalogs.
- `scripts/` – `benchmark_cross_match.py`, `generate_large_catalog.py`, etc.

## Development

- **Rust extension**: Recommended: `uv run maturin develop` (uses the project’s Python 3.10–3.12). If you use `cargo build --features python` and your default Python is outside that range, PyO3 may error; either set `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` to build with the stable ABI, or point to a supported interpreter, e.g. `PYO3_PYTHON=python3.12 cargo build --features python`.
- **Lint/format**: `uv run ruff check . && uv run ruff format .`
- **Type check**: `uv run mypy python/`
- **Tests** (Python + Rust): `uv run python run_tests.py` — or Python only: `uv run pytest`. Add `--benchmark` to run a small cross-match benchmark after tests; use `--benchmark-only` to run only the benchmark (no tests); use `--rust` to include the Rust engine.
- **Coverage**: `uv run pytest tests/ --cov=python/astrojoin --cov-report=term-missing`
- **Pre-commit**: `uv run pre-commit install` then `pre-commit run --all-files`
- **Publishing (plug-and-play wheels)**: From the project root, `uv run maturin build --release` builds wheels for the current platform; upload to PyPI with `uv run maturin publish`. For many platforms (Linux/macOS/Windows × Python 3.10–3.12), use CI (e.g. the provided GitHub Actions workflow) to build and publish on tag push.

## License

MIT License. See [LICENSE](LICENSE).
