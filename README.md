# Pleiades

*Match two sky catalogs by position — without loading them into RAM.*

- **Cross-match:** find pairs (id_a, id_b) whose sky positions are within a given radius (arcsec). Same idea as a spatial join, but out-of-core.
- **Out-of-core:** HEALPix indexing + chunked streaming — never load full catalogs; process in batches so billion-row runs fit on one machine.
- **Stack:** Rust engine (speed, parallelism) + Python API (validation, CLI, analysis helpers).

**When to use:** Gaia vs SDSS, LSST vs external catalog, any two Parquet catalogs with **ra**, **dec**, and an ID column. For how it works under the hood, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Requirements

- **Input:** Parquet files with numeric **ra**, **dec** (degrees or radians) and at least one ID column (Int64, Int32, or string). Column names are configurable.
- **Runtime:** The Rust extension is required (included in wheels). No extension → clear `ImportError` with install instructions.

---

## Install

```bash
pip install pleiades
# or
uv add pleiades
```

- Wheels include the Rust extension (Python 3.10–3.12)
- From source: `uv sync` then `uv run maturin develop`
- **macOS:** for faster shard I/O, build with read-ahead enabled: `uv run maturin develop --features macos_readahead`
- Real catalogs: [DATA_SOURCES.md](DATA_SOURCES.md)

---

## Use

### Python: cross-match

```python
import pleiades

result = pleiades.cross_match(
    catalog_a="catalog_a.parquet",
    catalog_b="catalog_b.parquet",
    radius_arcsec=2.0,
    output_path="matches.parquet",
)
# result: output_path, rows_a_read, rows_b_read, matches_count, time_seconds
```

**Input**

- Parquet with **ra**, **dec**, and one ID column
- Column names: auto-detected (e.g. `source_id`, `object_id`) or set via `ra_col`, `dec_col`, `id_col_a`, `id_col_b`
- Units: degrees (default) or `ra_dec_units="rad"` for radians (e.g. LSST)

**Output**

- **File:** Parquet with `id_a`, `id_b`, `separation_arcsec` (column names follow input ID columns). Optionally `ra_a`, `dec_a`, `ra_b`, `dec_b` if `include_coords=True` (catalog_b must be a file).
- **Return value** (`CrossMatchResult`): `output_path`, `rows_a_read`, `rows_b_read`, `matches_count`, `chunks_processed`, `time_seconds`

**Options**

- `include_coords=True` — add ra/dec columns (catalog_b must be a file)
- `n_nearest=1` — keep only best match per id_a
- `progress_callback=(chunk_ix, total, rows_a, matches)` — return `False` to cancel
- `ra_dec_units="rad"` — for LSST/Rubin
- `keep_b_in_memory=True` — when B is small
- `catalog_b` — can be a **directory** of pre-partitioned shards (see below)
- `partition_b=True` (default) — partition B by HEALPix once, then each A chunk only reads B rows in same/neighbor pixels
- `batch_size_a`, `batch_size_b` (default 250k) — rows per chunk; smaller = less RAM, more I/O
- `n_shards` (default 16) — HEALPix shards for B; fewer = less I/O overhead
- `depth` (default 8) — HEALPix depth (nside = 2^depth)

**Pre-partitioned B (large catalog_b):** Partition once, reuse for many runs:

```python
# One-time: write B to shards
pleiades.partition_catalog("large_b.parquet", "./shards_b", depth=8, n_shards=16)

# Then use the directory as catalog_b (no re-partitioning)
result = pleiades.cross_match("a.parquet", "./shards_b", radius_arcsec=2.0, output_path="out.parquet")
```

### CLI

```bash
# Cross-match
pleiades cross-match catalog_a.parquet catalog_b.parquet -r 2.0 -o matches.parquet

# Summarize
pleiades summarize-matches matches.parquet

# Cone search
pleiades cone-search catalog.parquet 180 0 -r 3600 -o cone.parquet

# Partition into shards (for use as catalog_b)
pleiades partition-catalog catalog.parquet ./shards --depth 8 --n-shards 16
```

**cross-match flags**

- `--batch-size`, `--n-shards`, `--keep-b-in-memory`, `--verbose`
- `--n-nearest`, `--ra-col`, `--dec-col`, `--id-col-a`, `--id-col-b`

### Helpers (Python)

| Area        | Functions | Purpose |
|------------|-----------|--------|
| Analysis   | `summarize_matches`, `match_stats`, `match_quality_summary` | Counts, unique id_a/id_b, separation min/max/mean/median (and percentiles) |
| Merge/filter | `merge_match_to_catalog`, `filter_matches_by_radius`, `multi_radius_cross_match`, `attach_match_coords` | Join matches to a catalog; subset by radius; one run several radii; add ra/dec to match rows |
| Cone       | `cone_search`, `batch_cone_search` | Rows within radius of (ra, dec); multi-query in one pass |
| Streaming  | `cross_match_iter` | Yield `(id_a, id_b, separation_arcsec)` tuples (Python path) |
| Types      | `CrossMatchResult`, `MatchSummary`; `pleiades.__version__` | Result of `cross_match()`; result of `summarize_matches()`; package version string |

### Progress (e.g. tqdm)

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

### Typical workflow

1. **Cross-match** → `cross_match(catalog_a, catalog_b, radius_arcsec, output_path)`
2. **Inspect** → `summarize_matches(output_path)` or CLI `pleiades summarize-matches output_path`
3. **Attach catalog columns** (optional) → `merge_match_to_catalog(matches_path, catalog_a, enriched.parquet, catalog_side="a")` to join match rows back to catalog A (or B) for magnitudes, parallax, etc.

### Errors

- **FileNotFoundError** — catalog path missing or not a file (or not a directory when passing pre-partitioned B).
- **CatalogValidationError** — e.g. non-positive radius, invalid `n_nearest`/`depth`/`n_shards`.
- **ImportError** — Rust extension not installed; message includes `pip install pleiades` or `uv run maturin develop`.

---

## Benchmarks

```bash
uv run python scripts/benchmark_cross_match.py --rows 100000
uv run python scripts/generate_benchmark_fixtures.py   # big fixtures
./scripts/run_benchmarks.sh                            # 1M-row, logs to logs/
```

- **Tuning:** batch sizes, `n_shards`, depth (see script options and [ARCHITECTURE.md](ARCHITECTURE.md#bottlenecks-and-knobs))

---

## More docs

| Doc | Contents |
|-----|----------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Flow diagram, Rust vs Python roles, bottlenecks |
| [DATA_SOURCES.md](DATA_SOURCES.md) | Gaia, IRSA, SDSS, LSST — where to get Parquet catalogs and column names |

---

## Layout

| Path                 | Role |
|----------------------|------|
| `python/pleiades/`   | API, analysis, cone search, CLI |
| `python/pleiades_core/` | Rust bindings (maturin) |
| `src/`               | Rust engine |
| `tests/`             | Unit + integration |
| `scripts/`           | Benchmarks, fixtures, I/O profiling (`profile_io.sh`) |

### Dev

```bash
uv run maturin develop
uv run ruff check . && uv run ruff format .
uv run mypy python/
uv run pytest tests/
# or: uv run python run_tests.py --benchmark
```

- **I/O profiling (macOS):** `./scripts/profile_io.sh` runs the benchmark under `time -l`; use `--fs-usage` for file-level I/O (requires sudo).
- Publish: `maturin build --release`, `maturin publish`; use CI for multi-platform wheels

---

## License

MIT. See [LICENSE](LICENSE).
