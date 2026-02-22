# Changelog

[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) · [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Changed

- **Rust only:** `cross_match()` always uses the Rust engine
  - `use_rust` and Python fallback removed
  - No extension → clear error + install instructions
  - CLI `--rust` / `--no-rust` removed
- Docs and benchmarks updated (README, scripts)
- Parquet 52 compatibility in Rust; ruff/mypy and test cleanups

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
