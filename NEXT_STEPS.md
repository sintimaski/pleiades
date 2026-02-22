# Project analysis and next steps

## Current state (summary)

- **Scope**: Out-of-core spatial cross-match for Parquet catalogs (HEALPix + chunked I/O). Python API with optional Rust engine; CLI; analysis helpers (match_stats, merge_match_to_catalog, cone search, partition_catalog).
- **Quality**: ~82% Python coverage, ≥80% required; validation (radius NaN/Inf, column hints); CHANGELOG and README updated; wheels workflow for PyPI.
- **Gaps**: CI sdist/publish need verification; some doc and UX polish; ARCHITECTURE table still has old “No pre-partitioning” text; no automated test run in CI.

---

## Next steps (prioritized)

### 1. Release and CI (high)

| Step | Action |
|------|--------|
| **1.1** | **Wheels workflow**: Sdist job uses `uv run maturin` but CI does not install `uv`. Add a step to install uv (e.g. `curl -LsSf https://astral.sh/uv/install.sh \| sh` or `pip install uv`) or use `maturin build --release --sdist-only` after installing maturin. |
| **1.2** | **Sdist artifact path**: Maturin puts sdist in `target/wheels/`. Confirm the sdist job uploads the correct path (e.g. `target/wheels/*.tar.gz` or the single file produced there). |
| **1.3** | **Publish job**: First download uses `path: dist` and `pattern: wheels-*` with `merge-multiple: true`; second download uses `path: dist` for sdist. With merge, wheel files may be in `dist/`; sdist may create `dist/sdist/`. Add a “Flatten” step that collects all `dist/**/*.whl` and `dist/**/*.tar.gz` into one directory (e.g. `dist/out/`) and run `twine upload dist/out/*`. Fix the current “Flatten” step that references `artifacts` (should be `dist` or the actual download path). |
| **1.4** | **Test PyPI install**: Create a TestPyPI release (or use the real PyPI after 0.1.0) and run `pip install pleiades` in a clean env; then `python -c "import pleiades; pleiades.cross_match(...)"` to confirm plug-and-play. |
| **1.5** | **CI tests**: Add a job that runs on push/PR: checkout, install project (e.g. `uv sync` + `uv run maturin develop` or install from built wheel), run `uv run pytest tests/ -q` (and optionally `run_tests.py`) on Python 3.10, 3.11, 3.12. |

### 2. Documentation (medium)

| Step | Action |
|------|--------|
| **2.1** | **ARCHITECTURE.md**: Update the “No pre-partitioning” row in the bottlenecks table to state that B is addressed via `partition_catalog` + directory as `catalog_b`; A pre-partitioning / join-by-pixel is a possible future extension. (Previous replace failed due to quote character; fix manually or with a small script.) |
| **2.2** | **README**: Add a one-line pointer to `DATA_SOURCES.md` for “Preparing real catalogs (Gaia, LSST, etc.)”. |
| **2.3** | **Docstrings**: Ensure all public functions in `analysis.py`, `cone.py`, `cross_match.py`, `validation.py` have consistent docstrings (parameters, returns, raises). |

### 3. Testing (medium)

| Step | Action |
|------|--------|
| **3.1** | **CLI coverage**: `cli.py` is 0% in coverage because tests invoke the CLI via subprocess. Optionally add unit tests that import `pleiades.cli` and call `main()` with `sys.argv` mocked (or use a small in-process helper) so the CLI branch is exercised for coverage. |
| **3.2** | **Integration**: Add (or document) one integration test that uses real-sized fixtures (or skipped if no large fixtures) to stress the Rust path and pre-partitioned B. |
| **3.3** | **Python 3.13**: When PyO3 supports 3.13, add it to `requires-python` and to the CI/test matrix. |

### 4. Robustness and UX (lower)

| Step | Action |
|------|--------|
| **4.1** | **Optional tqdm helper**: Provide `pleiades.progress_tqdm()` that returns a `progress_callback` wrapping tqdm (if tqdm is installed), so users can do `cross_match(..., progress_callback=pleiades.progress_tqdm())` without writing the closure. Make tqdm an optional dependency. |
| **4.2** | **Rust edge cases**: Consider validating in Rust that ra/dec are finite (or document that non-finite coords are undefined). |

### 5. Roadmap (from ARCHITECTURE)

| Idea | Notes |
|------|--------|
| **Pre-partition A** | Use `partition_catalog` for A; extend the join to read only A shards for pixels needed per B (or per A chunk). Currently only B is pre-partitioned. |
| **Parallel A chunks** | Run multiple A chunks in parallel (multiprocessing or Rust rayon over chunks) to use more CPU. |
| **Optional pass-through columns** | Allow users to request extra columns from A/B in the match output (e.g. magnitude, parallax) without a separate `merge_match_to_catalog` step. |
| **Index build in Rust** | Replace the per-chunk Python “pixel → list of (id, ra, dec)” with a Rust/Arrow structure for faster index build. |

---

## Suggested order of work

1. **Immediate**: Fix wheels workflow (1.1–1.3) and add a CI job that runs tests (1.5).  
2. **Before first public release**: Do a TestPyPI or PyPI publish and verify install (1.4); update ARCHITECTURE “No pre-partitioning” row (2.1).  
3. **Soon after**: README link to DATA_SOURCES (2.2); optional tqdm helper (4.1).  
4. **Backlog**: CLI coverage (3.1), Python 3.13 (3.3), roadmap items (section 5).
