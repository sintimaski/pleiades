# Pleiades

An out-of-core spatial cross-matcher for astronomical catalogs. Match two sky catalogs by position without loading them into RAM.

## Highlights

- **Cross-match** two Parquet catalogs by angular distance (arcsec). Find pairs (id_a, id_b) within a radius — a spatial join that streams.

- **Out-of-core** by design: HEALPix indexing and chunked streaming. Process billion-row catalogs on one machine; batch sizes control memory.

- **Rust engine** for the join (speed, parallelism); **Python API** for validation, CLI, and analysis helpers.

- **Pre-partition** the larger catalog once, reuse shards across many runs. Optional in-memory B when it fits.

- **Cone search** and **match summarization** built in. Works with Gaia, SDSS, LSST, or any Parquet with `ra`, `dec`, and an ID column.

See [ARCHITECTURE.md](ARCHITECTURE.md) for how it works and [DATA_SOURCES.md](DATA_SOURCES.md) for where to get catalogs.

## Installation

Install from PyPI (wheels include the Rust extension for Python 3.10–3.12):

```bash
pip install pleiades
```

Or with uv:

```bash
uv add pleiades
```

From source (requires Rust and maturin):

```bash
uv sync
uv run maturin develop
```

On macOS, enable read-ahead for faster shard I/O:

```bash
uv run maturin develop --features macos_readahead
```

## Quick start

Cross-match two catalogs from Python:

```python
import pleiades

result = pleiades.cross_match(
    catalog_a="catalog_a.parquet",
    catalog_b="catalog_b.parquet",
    radius_arcsec=2.0,
    output_path="matches.parquet",
)
# result.output_path, result.rows_a_read, result.matches_count, result.time_seconds
```

Or from the CLI:

```bash
$ pleiades cross-match catalog_a.parquet catalog_b.parquet -r 2.0 -o matches.parquet
$ pleiades summarize-matches matches.parquet
```

Input: Parquet files with numeric **ra**, **dec** (degrees or radians) and at least one ID column. Column names are configurable (`ra_col`, `dec_col`, `id_col_a`, `id_col_b`). Output: Parquet with `id_a`, `id_b`, `separation_arcsec`.

## Features

### Cross-match

The main entry point is `pleiades.cross_match()`. It streams catalog A in chunks, partitions or loads catalog B by HEALPix pixel, and writes matches to a Parquet file. Options include `n_nearest=1` (best match only), `progress_callback` for progress bars, `ra_dec_units="rad"` for LSST, and `keep_b_in_memory=True` when B is small.

```python
import pleiades

result = pleiades.cross_match(
    catalog_a="gaia_subsample.parquet",
    catalog_b="sdss.parquet",
    radius_arcsec=2.0,
    output_path="matches.parquet",
    n_nearest=1,
)
print(f"{result.matches_count} matches in {result.time_seconds:.1f}s")
```

See the docstring for `cross_match()` for all options. Batch sizes and `n_shards` tune memory and I/O; see [ARCHITECTURE.md](ARCHITECTURE.md#bottlenecks-and-knobs).

### Pre-partitioned catalog B

For large catalog B, partition once and reuse the shards so multiple runs skip re-partitioning:

```bash
$ pleiades partition-catalog large_b.parquet ./shards_b --depth 8 --n-shards 16
```

```python
# Use the directory as catalog_b
pleiades.cross_match("a.parquet", "./shards_b", radius_arcsec=2.0, output_path="out.parquet")
```

### CLI

```bash
$ pleiades cross-match <catalog_a> <catalog_b> -r <radius_arcsec> -o <output.parquet>
$ pleiades summarize-matches <matches.parquet>
$ pleiades cone-search <catalog.parquet> <ra> <dec> -r <radius_arcsec> -o <output.parquet>
$ pleiades partition-catalog <catalog.parquet> <output_dir> [--depth 8] [--n-shards 16]
```

Use `pleiades --help` and `pleiades cross-match --help` for flags (`--batch-size`, `--n-shards`, `--keep-b-in-memory`, `--verbose`, `--n-nearest`, `--ra-col`, `--dec-col`, etc.).

### Analysis and helpers

Summarize match output, merge matches back onto a catalog, or filter by radius:

```python
summary = pleiades.summarize_matches("matches.parquet")
# summary.num_matches, summary.separation_arcsec_min/max/mean

pleiades.merge_match_to_catalog("matches.parquet", "catalog_a.parquet", "enriched.parquet", catalog_side="a")
pleiades.filter_matches_by_radius("matches.parquet", max_radius_arcsec=1.0, output_path="tight.parquet")
```

Streaming: `cross_match_iter()` yields `(id_a, id_b, separation_arcsec)` tuples. Cone search: `cone_search()` and `batch_cone_search()` for rows within radius of a point (or multiple points).

## Requirements

- Parquet inputs with numeric **ra**, **dec** and at least one ID column (Int64, Int32, or string). Names configurable.
- The Rust extension is required; it is included in PyPI wheels. Without it you get a clear `ImportError` with install instructions.

## Benchmarks

```bash
$ uv run python scripts/benchmark_cross_match.py --rows 100000
$ ./scripts/run_benchmarks.sh   # 1M-row run, logs to logs/
```

Tuning: see script options and [ARCHITECTURE.md](ARCHITECTURE.md#bottlenecks-and-knobs).

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — flow, Rust vs Python roles, bottlenecks and knobs
- [DATA_SOURCES.md](DATA_SOURCES.md) — Gaia, IRSA, SDSS, LSST: where to get Parquet catalogs

Command-line help: `pleiades --help`, `pleiades cross-match --help`.

## Errors

- **FileNotFoundError** — catalog path missing or not a file (or not a directory for pre-partitioned B).
- **CatalogValidationError** — invalid args (e.g. non-positive radius, invalid depth/n_shards).
- **ImportError** — Rust extension not installed; message includes `pip install pleiades` or `uv run maturin develop`.

## Development

```bash
$ uv run maturin develop
$ uv run ruff check . && uv run ruff format .
$ uv run mypy python/
$ uv run pytest tests/
# or: uv run python run_tests.py --benchmark
```

I/O profiling on macOS: `./scripts/profile_io.sh` (optionally `--fs-usage` with sudo). Publish: use CI for multi-platform wheels; see `.github/workflows/wheels.yml`.

## License

MIT. See [LICENSE](LICENSE).
