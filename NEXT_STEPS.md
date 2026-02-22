# Project analysis and next steps

## Current state (summary)

- **Scope**: Out-of-core spatial cross-match for Parquet catalogs (HEALPix + chunked I/O). Python API with optional Rust engine; CLI; analysis helpers (match_stats, merge_match_to_catalog, cone search, partition_catalog).
- **Quality**: ~82% Python coverage, ≥80% required; validation (radius NaN/Inf, column hints); CHANGELOG and README updated; wheels workflow for PyPI.
- **CI**: `.github/workflows/wheels.yml` runs tests on Python 3.10–3.12, builds wheels and sdist, and publish job flattens artifacts to `dist/out` for PyPI.

---

## Next steps (prioritized)

### 1. Release and verification

| Step | Action |
|------|--------|
| **1.1** | **Test PyPI install**: After a release, run `pip install pleiades` in a clean env and `python -c "import pleiades; pleiades.cross_match(...)"` to confirm plug-and-play. |
| **1.2** | **Publish job**: If needed, verify artifact paths (wheels in `dist` after merge-multiple; sdist in `dist`; flatten copies `*.whl` and `*.tar.gz` to `dist/out`). |

### 2. Documentation

| Step | Action |
|------|--------|
| **2.1** | **Docstrings**: Ensure all public functions in `analysis.py`, `cone.py`, `cross_match.py`, `validation.py` have consistent docstrings (parameters, returns, raises). |
| **2.2** | **DATA_SOURCES**: README already links to DATA_SOURCES.md for preparing real catalogs. |

### 3. Testing

| Step | Action |
|------|--------|
| **3.1** | **CLI coverage**: Optionally add unit tests that import `pleiades.cli` and call `main()` with mocked `sys.argv` so the CLI is exercised for coverage. |
| **3.2** | **Python 3.13**: When PyO3 supports 3.13, add it to `requires-python` and to the CI matrix. |

### 4. Robustness and UX

| Step | Action |
|------|--------|
| **4.1** | **Optional tqdm helper**: Provide `pleiades.progress_tqdm()` that returns a `progress_callback` wrapping tqdm (if installed). Make tqdm an optional dependency. |
| **4.2** | **Rust edge cases**: Consider validating in Rust that ra/dec are finite (or document that non-finite coords are undefined). |

### 5. Roadmap (from ARCHITECTURE)

| Idea | Notes |
|------|--------|
| **Pre-partition A** | Use `partition_catalog` for A; extend the join to read only A shards for pixels needed per B (or per A chunk). Currently only B is pre-partitioned. |
| **Parallel A chunks** | Run multiple A chunks in parallel (multiprocessing or Rust rayon over chunks) to use more CPU. |
| **Optional pass-through columns** | Allow users to request extra columns from A/B in the match output without a separate `merge_match_to_catalog` step. |
| **Index build in Rust** | Replace the per-chunk Python "pixel → list of (id, ra, dec)" with a Rust/Arrow structure for faster index build. |
