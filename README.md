# Pleiades

*Match two sky catalogs by position — without loading them into RAM.*

Big surveys (Gaia, SDSS, LSST) give you billions of rows. Cross-matching “which object in A is the same as in B?” with naive loops doesn’t scale. **Pleiades** does it out-of-core: HEALPix indexing + chunked streaming, with a **Rust** engine and a **Python** API. Install and run.

---

## Install

```bash
pip install pleiades
# or: uv add pleiades
```

Wheels include the Rust extension (Python 3.10–3.12). Catalogs: [DATA_SOURCES.md](DATA_SOURCES.md). From source: `uv sync` then `uv run maturin develop`.

---

## Use

**Python:**

```python
import pleiades

result = pleiades.cross_match(
    catalog_a="catalog_a.parquet",
    catalog_b="catalog_b.parquet",
    radius_arcsec=2.0,
    output_path="matches.parquet",
)
# result has output_path, rows_a_read, rows_b_read, matches_count, time_seconds
```

Catalogs need **ra**, **dec** and an ID column. Output: Parquet with `id_a`, `id_b`, `separation_arcsec`. Optional: `include_coords=True` (adds ra/dec columns), `n_nearest=1`, `progress_callback`, `ra_dec_units="rad"`, `keep_b_in_memory` (when B is small), and `catalog_b` can be a **directory** of pre-partitioned shards.

**CLI:**

```bash
pleiades cross-match catalog_a.parquet catalog_b.parquet -r 2.0 -o matches.parquet
pleiades summarize-matches matches.parquet
pleiades cone-search catalog.parquet 180 0 -r 3600 -o cone.parquet
pleiades partition-catalog catalog.parquet ./shards --depth 8 --n-shards 16
```

`cross-match` also accepts `--batch-size`, `--n-shards`, `--keep-b-in-memory`, `--verbose`, `--n-nearest`, and `--ra-col` / `--dec-col` / `--id-col-a` / `--id-col-b`.

**Helpers:** `summarize_matches`, `match_stats`, `match_quality_summary`, `merge_match_to_catalog`, `filter_matches_by_radius`, `multi_radius_cross_match`, `attach_match_coords`. Cone: `cone_search`, `batch_cone_search`. Streaming: `cross_match_iter`. Types: `CrossMatchResult`, `MatchSummary`; `pleiades.__version__` for the version string.

**Progress (e.g. tqdm):** pass `progress_callback=(chunk_ix, total, rows_a, matches)`; return `False` to cancel.

---

## Benchmarks

```bash
uv run python scripts/benchmark_cross_match.py --rows 100000
uv run python scripts/generate_benchmark_fixtures.py   # when you need big fixtures
./scripts/run_benchmarks.sh                            # 1M-row run, logs to logs/
```

Tuning: [PERFORMANCE.md](PERFORMANCE.md).

---

## Layout

- **python/pleiades/** — API, cross-match glue, analysis, cone search, CLI  
- **python/pleiades_core/** — Rust bindings (maturin)  
- **src/** — Rust engine  
- **tests/** — unit + integration; **scripts/** — benchmarks, fixtures

**Dev:** `uv run maturin develop` · `uv run ruff check . && ruff format .` · `uv run mypy python/` · `uv run pytest tests/` (or `uv run python run_tests.py`; add `--benchmark` to run a small cross-match after tests). Publish: `maturin build --release` and `maturin publish`; use CI for multi-platform wheels.

---

## License

MIT. See [LICENSE](LICENSE).
