# Production Readiness

This document summarizes production-oriented features and how to use them.

## Checklist

- **CI**: Lint (ruff + mypy) and tests run on every push/PR. Wheels are built for Linux, macOS, and Windows.
- **Logging**: Python uses a `pleiades` logger. Set `PLEIADES_VERBOSE=1` for INFO-level messages and Rust engine timing.
- **Configuration**: Env vars are centralized in `pleiades.config`; see below.
- **Path safety**: Set `PLEIADES_OUTPUT_BASE_DIR` to restrict where output files can be written.
- **Validation**: `CatalogValidationError` for invalid args; schema and path validation before I/O.
- **Type hints & lint**: mypy and ruff are run in CI; use `uv run python run_tests.py --lint` locally.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `PLEIADES_VERBOSE` | Set to `1` or `true` for INFO logging and Rust timing. |
| `PLEIADES_PROFILE` | Set to `1` for Rust engine profiling output. |
| `PLEIADES_JOIN_STRATEGY` | Rust join strategy (e.g. `matrix`, `nested`). |
| `PLEIADES_GPU` | Set to `wgpu` to use GPU join (if built with `wgpu` feature). |
| `PLEIADES_GPU_MIN_PAIRS` | Minimum pair count to enable GPU (Rust). |
| `PLEIADES_OUTPUT_BASE_DIR` | If set, all output paths must resolve under this directory. |

## Output path restriction

In production, restrict where cross-match and partition output can be written:

```bash
export PLEIADES_OUTPUT_BASE_DIR=/data/pleiades/output
```

Then `cross_match(..., output_path="matches.parquet")` will resolve `matches.parquet` relative to the current working directory, and validation will require the resolved path to be under `/data/pleiades/output`. Paths outside that directory raise `CatalogValidationError`.

## Logging

- Default: WARNING (only warnings and errors).
- With `PLEIADES_VERBOSE=1`: INFO (cross_match start/done, CLI commands).
- The Rust engine also prints per-chunk timing to stderr when `PLEIADES_VERBOSE=1` is set.

To attach your own handler:

```python
import logging
logging.getLogger("pleiades").setLevel(logging.DEBUG)
```

## Running lint and tests

```bash
# Lint only (ruff + mypy)
uv run python run_tests.py --lint

# Full test suite + optional benchmark
uv run python run_tests.py
uv run python run_tests.py --benchmark --benchmark-rows 2000
```

CI runs lint and tests on every push to `main` and on pull requests.

## Versioning and release

- Version is in `pyproject.toml`; single source of truth.
- Changelog: `CHANGELOG.md` (Keep a Changelog style).
- Release: tag a release in GitHub; CI builds wheels and sdist and publishes to PyPI when the release is published (requires `PYPI_API_TOKEN` in the `pypi` environment).

## Deployment notes

- Install from PyPI: `pip install pleiades` (wheels include the Rust extension for Python 3.10–3.12).
- From source: `uv sync` then `uv run maturin develop --release` (optionally `--features macos_readahead` on macOS).
- No Docker image is provided; use a Python 3.10+ image and install from PyPI or build from source.
- For large catalogs, ensure sufficient disk for temp shards and output; tune `batch_size_*` and `n_shards` for memory (see [ARCHITECTURE.md](ARCHITECTURE.md)).
