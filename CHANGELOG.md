# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Rust engine required**: `cross_match()` always uses the Rust engine. The `use_rust` parameter and the pure-Python fallback have been removed. Install with `pip install pleiades` or build from source with `uv run maturin develop`. CLI flags `--rust` / `--no-rust` removed.
- **Documentation**: README benchmark section documents `generate_benchmark_fixtures.py` and `run_benchmarks.sh`; project layout updated (pleiades_core, scripts list).
- **Code quality**: Ruff and test cleanups (import order, unused imports, loop variable, assertion on specific exception). Mypy overrides for untyped third-party libs (pyarrow, astropy, cdshealpix); assertions for optional ParquetWriter and partition writers.
- **Rust engine**: Parquet 52 compatibility — use `File` directly for `ParquetRecordBatchReaderBuilder::try_new()` (parquet 52’s `ChunkReader` is implemented for `File`, not `BufReader`). Removed unused `BufReader` import and `PARQUET_READ_BUFFER_BYTES`.

## [0.1.0] - 2025-02-21

### Added

- **Rust engine**
  - Pre-partitioned catalog B: when `catalog_b` is a directory of HEALPix shards (`shard_*.parquet`), the engine loads only B rows for pixels needed per A chunk.
  - `n_nearest`: optional limit to the N smallest-separation matches per `id_a` (post-pass over match file).
  - Progress callback: optional Python callable invoked per A chunk with (chunk_ix, total_chunks, rows_in_chunk, matches_count).
  - Return value: engine returns match counts and timing; Python exposes this as `CrossMatchResult` (output_path, rows_a_read, rows_b_read, matches_count, chunks_processed, time_seconds).
  - Inner B-loop parallelized with Rayon.
- **Python API**
  - `partition_catalog(catalog_path, output_dir, ...)`: write a catalog to HEALPix shards for use as pre-partitioned B.
  - `match_stats(matches_path, ...)`: row counts and optional separation percentiles.
  - `match_quality_summary(matches_path, rows_a, rows_b)`: fraction of A/B matched and match counts.
  - `attach_match_coords(matches_path, catalog_a_path, catalog_b_path, output_path, ...)`: add ra_a, dec_a, ra_b, dec_b to a match file.
  - `batch_cone_search(catalog_path, queries, output_path)`: multi-query cone search with query_index and separation_arcsec.
  - `cross_match(..., include_coords=False)`: when True (and B is a file), append ra/dec columns to the match output.
- **CLI** (`pleiades`): subcommands `cross-match`, `summarize-matches`, `cone-search`, `partition-catalog`.
- **Validation**: `validate_cross_match_args(radius_arcsec, n_nearest, depth, n_shards)`; clearer column-not-found errors with suggested names.
- **Benchmark**: `scripts/benchmark_cross_match.py` for synthetic catalogs (Python and optional Rust).

### Changed

- Rust: progress callback and stats return value are supported when the extension is built.
- Pre-partitioned B can be used with `use_rust=True` (directory of shards as `catalog_b`).

### Fixed

- Rust: Parquet row count uses row-group sums (metadata `num_rows()` not available).
- Rust: progress callback GIL and type handling for PyO3.
- Validation: reject NaN and infinite `radius_arcsec`.

[Unreleased]: https://github.com/your-org/pleiades/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/your-org/pleiades/releases/tag/v0.1.0
