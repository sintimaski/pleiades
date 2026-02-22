# Pleiades

*A high-performance spatial cross-matcher for astronomical catalogs — match billions of rows without blowing up your RAM.*

---

## The problem

Astronomical surveys (Gaia, SDSS, LSST, and others) publish huge catalogs: hundreds of millions to billions of sources, each with sky coordinates. A common task is **cross-matching**: finding which sources from one catalog correspond to the same object in another, within a given angular distance. Doing this with naive nested loops or in-memory joins does not scale. You either run out of memory or wait forever.

## The solution

**Pleiades** cross-matches two Parquet catalogs by angular distance using **HEALPix indexing** and **chunked streaming**. It never loads full catalogs into memory: it processes them in batches, so you can match billion-row datasets on a normal machine. The heavy lifting runs in a **Rust engine** (included in the Python wheels); the **Python layer** gives you a simple API, validation, CLI, and analysis helpers. The Rust extension is required — install with `pip install pleiades` or build from source with `uv run maturin develop`.

---

## Installation

No Rust toolchain needed — install and go:

```bash
pip install pleiades
# or
uv add pleiades
```

You get the Python package and the prebuilt Rust extension for your platform (Python 3.10–3.12). For pointers on obtaining real catalogs (Gaia, LSST, SDSS, etc.), see [DATA_SOURCES.md](DATA_SOURCES.md).

**From source** (if you’re hacking on the code):

```bash
git clone <repo> && cd astro
uv sync
uv run maturin develop   # build the Rust extension for your env
```

---

## Usage

### Cross-match (Python API)

```python
import pleiades

result = pleiades.cross_match(
    catalog_a="catalog_a.parquet",
    catalog_b="catalog_b.parquet",
    radius_arcsec=2.0,
    output_path="matches.parquet",
)
# result: CrossMatchResult(output_path, rows_a_read, rows_b_read, matches_count, time_seconds)
```

- Catalogs need **ra** and **dec** columns (configurable via `ra_col` / `dec_col`). Coordinates can be in **degrees** (default) or **radians** — use `ra_dec_units="rad"` for LSST/Rubin. Float32 or Float64 are both supported.
- ID columns are auto-detected (e.g. `source_id`, `object_id`) or set with `id_col_a` / `id_col_b`.
- Output is Parquet with **id_a**, **id_b**, **separation_arcsec**. Optionally **ra_a**, **dec_a**, **ra_b**, **dec_b** via `include_coords=True` (Python path; catalog_b must be a file).

**Handy options**: `partition_b=True` (default); `batch_size_a` / `batch_size_b` (default 250k); `n_shards` (default 16); `ra_dec_units="deg"` | `"rad"`; `n_nearest=1` for best match only; `progress_callback=(chunk_ix, total, rows_a, matches)`; `catalog_b` can be a **directory** of pre-partitioned shards.

### Command-line interface

```bash
# Cross-match (defaults: batch 250k, n_shards 16)
pleiades cross-match catalog_a.parquet catalog_b.parquet -r 2.0 -o matches.parquet
pleiades cross-match a.parquet b.parquet -r 2 -o out.parquet --verbose --batch-size 500000

# Summarize matches
pleiades summarize-matches matches.parquet

# Cone search
pleiades cone-search catalog.parquet 180.0 0.0 -r 3600 -o cone.parquet

# Partition a catalog into HEALPix shards (for use as pre-partitioned B)
pleiades partition-catalog catalog.parquet ./shards --depth 8 --n-shards 16
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

### Progress (long runs)

Use a progress callback. With **tqdm** (optional):

```python
from tqdm import tqdm
pbar = tqdm(desc="cross-match", unit=" rows")
def progress(chunk_ix, total, rows_a, matches):
    pbar.n = rows_a
    pbar.set_postfix(matches=matches)
    pbar.refresh()
pleiades.cross_match(..., progress_callback=progress)
pbar.close()
```

---

## Benchmarking

Quick run:

```bash
uv run python scripts/benchmark_cross_match.py --rows 100000
uv run python scripts/benchmark_cross_match.py --rows 50000 --rust   # compare Python vs Rust
```

With pregenerated fixtures (e.g. 100k, 1M, 50M rows) and logging:

```bash
uv run python scripts/generate_benchmark_fixtures.py   # one-time or when missing
./scripts/run_benchmarks.sh                            # runs 1M-row benchmark, logs to logs/
./scripts/run_benchmarks.sh --rows 500000 --rust --verbose
```

For tuning (batch sizes, n_shards, depth) and notes on CPU SIMD vs GPU (CUDA, wgpu, OpenCL), see [PERFORMANCE.md](PERFORMANCE.md).

---

## Project layout

- **python/pleiades/** — Python API: cross-match, analysis, cone search, CLI (HEALPix + stream I/O).
- **python/pleiades_core/** — Python bindings for the Rust extension (built by maturin).
- **src/** — Rust engine (arrow, cdshealpix, HEALPix join, pre-partitioned B, rayon); build with `maturin develop`.
- **tests/** — Unit and integration tests; **tests/fixtures/** — small Parquet catalogs.
- **scripts/** — benchmark_cross_match.py, generate_benchmark_fixtures.py, run_benchmarks.sh, generate_large_catalog.py, etc.

---

## Development

- **Rust extension**: `uv run maturin develop` is recommended (uses the project’s Python 3.10–3.12). If you use `cargo build --features python` and your default Python is outside that range, PyO3 may error; set `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` or point to a supported interpreter, e.g. `PYO3_PYTHON=python3.12 cargo build --features python`.
- **Lint/format**: `uv run ruff check . && uv run ruff format .`
- **Type check**: `uv run mypy python/`
- **Tests** (Python + Rust): `uv run python run_tests.py` — or Python only: `uv run pytest`. Add `--benchmark` for a small cross-match benchmark after tests; `--benchmark-only` runs only the benchmark; `--rust` includes the Rust engine.
- **Coverage**: `uv run pytest tests/ --cov=python/pleiades --cov-report=term-missing`
- **Pre-commit**: `uv run pre-commit install` then `pre-commit run --all-files`
- **Publishing**: From the project root, `uv run maturin build --release` builds wheels; `uv run maturin publish` uploads to PyPI. For many platforms (Linux/macOS/Windows × Python 3.10–3.12), use CI (e.g. the provided GitHub Actions workflow) on tag push.

---

## License

MIT License. See [LICENSE](LICENSE).
