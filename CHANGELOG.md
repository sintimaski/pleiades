# Changelog

[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) · [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- **B prefetch overlap:** Send current and next chunk’s B load requests at chunk start; index runs in parallel with load B for the first chunk, join with load B for the next (two requests in flight).
- **Verbose timing:** With `PLEIADES_VERBOSE=1`, the engine logs per-chunk phases: `pixels+index` (time and pixel count), `load B`, `join`, `write`, and `chunk total`.
- **load_one_shard fast path:** Downcast columns once per batch, slice access for ra/dec, reserve capacity, Int64 id_b specialization; 128k-row batch size for shard Parquet reads.
- **macOS:** Optional feature `macos_readahead` (build with `--features macos_readahead`) enables kernel read-ahead on shard files via `fcntl(F_RDAHEAD)`.
- **I/O profiling:** `scripts/profile_io.sh` runs the benchmark under `time -l` and optionally `fs_usage` (macOS) for disk I/O and resource stats.

### Changed

- **Rust only:** `cross_match()` always uses the Rust engine
  - `use_rust` and Python fallback removed
  - No extension → clear error + install instructions
  - CLI `--rust` / `--no-rust` removed
- Docs and benchmarks updated (README, scripts)
- Parquet 52 compatibility in Rust; ruff/mypy and test cleanups
- Crate: `#![forbid(unsafe_code)]` relaxed only when `macos_readahead` is enabled (single fcntl FFI)
- **Perf — pixels and index:** Single pass `pixels_and_index()` builds both the HEALPix index and the pixel set (one hash per row). Chunk 1+ reuses the pixel set from the previous B-prefetch and only builds the index via `index_only()`. Pixel-set construction uses parallel `fold_with`/`reduce_with` for HashSet merge.
- **Perf — join:** Columnar B (`BColumns`: `id_b`, `ra_b`, `dec_b`); hot path touches only ra/dec. B grouped by HEALPix pixel so `pixels_to_look` is computed once per pixel. Haversine in batches of 8 then 4; cheap reject (radius_deg) before haversine in remainder path. n_nearest applied per chunk before write (`merge_to_n_nearest`); `apply_n_nearest` still merges across chunks.
- **Perf — partition B:** Batches processed in parallel (`partition_batch_to_row_results` via Rayon); merge and flush remain single-threaded.

---

## [0.1.0] - 2025-02-21

### Added

- **Rust engine**
  - HEALPix join, pre-partitioned B (directory of shards as `catalog_b`)
  - `n_nearest`, `progress_callback`, `keep_b_in_memory`
  - Rayon-parallel inner loop; returns match counts and timing
- **Python**
  - `cross_match`, `cross_match_iter`, `partition_catalog`
  - `cone_search`, `batch_cone_search`
  - Analysis: `summarize_matches`, `match_stats`, `match_quality_summary`, `merge_match_to_catalog`, `filter_matches_by_radius`, `multi_radius_cross_match`, `attach_match_coords`
  - CLI: `cross-match`, `summarize-matches`, `cone-search`, `partition-catalog`
  - Models: `CrossMatchResult`, `MatchSummary`; `CatalogValidationError`; `__version__`
- Validation for radius and columns; benchmark script and fixtures

### Fixed

- Rust: row counts from Parquet row groups; progress callback GIL/type handling
- Validation: reject NaN/infinite radius

---

[Unreleased]: https://github.com/your-org/pleiades/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/your-org/pleiades/releases/tag/v0.1.0
