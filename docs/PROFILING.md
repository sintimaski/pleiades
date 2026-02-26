# Profiling Pleiades cross_match

How to profile and interpret performance for speedup.

---

## Quick: benchmark breakdown

Run with `--verbose` to see per-phase timing:

```bash
PLEIADES_VERBOSE=1 uv run python scripts/benchmark_cross_match.py \
    --catalog-a data/benchmark_fixtures/catalog_a_10000000.parquet \
    --catalog-b data/benchmark_fixtures/catalog_b_10000000.parquet
```

Typical breakdown (10M×10M, 4 chunks, 2.3k matches, ~11.5s):

| Phase | Time | % | Notes |
|-------|-----|---|-------|
| **Partition B** | ~1.1s | 10% | One-time; parallel row-group decode in use |
| **Pixels+index** | ~2.2s | 19% | Chunk 1 does full index; others index-only |
| **Load B** | ~1.2s | 10% | From shards; parallel decode per shard |
| **Join** | ~6.3s | **55%** | Dominates; CPU haversine loop |
| **Write** | ~0s | 0% | Matches stream to Parquet |

**Main target: join** — haversine distance loop over candidate pairs.

---

## Sub-phase timing (join breakdown)

Set `PLEIADES_PROFILE=1` to log join sub-phases:

```bash
PLEIADES_VERBOSE=1 PLEIADES_PROFILE=1 uv run python scripts/benchmark_cross_match.py \
    --catalog-a data/benchmark_fixtures/catalog_a_1000000.parquet \
    --catalog-b data/benchmark_fixtures/catalog_b_1000000.parquet --verbose
```

Output includes:
- `profile join_group_b: X.XXXs (N pixels)` — group B by pixel
- `profile join_haversine_loop: X.XXXs (per-pixel)` — candidate merge + haversine

---

## CPU sampling (macOS)

**Sample** — built-in, no install:

```bash
./scripts/run_profile.sh
```

Runs the benchmark and samples the process for 15s. Output: `logs/profile_<timestamp>.txt`. Open in a text editor and search for hot functions (e.g. `haversine`, `pleiades_core`).

---

## Flamegraph (macOS)

For a visual flamegraph:

1. Install inferno: `cargo install inferno`
2. Run with DTrace (requires sudo):

```bash
./scripts/run_profile.sh --dtrace
```

Output: `logs/flamegraph_<timestamp>.svg`. Open in a browser. Wider bars = more CPU time.

---

## Linux

On Linux, use `perf`:

```bash
# Record
perf record -g -- python -c "
import pleiades
pleiades.cross_match('catalog_a.parquet', 'catalog_b.parquet', 2.0, 'out.parquet')
"
# Flamegraph (install cargo-flamegraph, inferno)
perf script | inferno-collapse-perf | inferno-flamegraph > flamegraph.svg
```

---

## What to optimize

1. **Join (~55%)** — haversine loop. Already SIMD (AVX2) for 4/8-lane. Consider: `PLEIADES_JOIN_STRATEGY=adynamic` when |A| < |B| per pixel; GPU for very large pair counts.
2. **Pixels+index (~19%)** — HEALPix + HashMap build. First chunk is heavier (pixels_and_index).
3. **Load B (~10%)** — parallel Parquet decode already in use; mmap (`--features parquet_mmap`) may help on Linux.
4. **Partition B (~10%)** — parallel decode in use; parallel merge+write (see PERFORMANCE_ANALYSIS §4.3) is the next step.

---

## Tuning knobs

| Env / arg | Effect |
|-----------|--------|
| `RAYON_NUM_THREADS` | Parallelism (default: num CPUs) |
| `PLEIADES_JOIN_STRATEGY` | `bdynamic` (default), `adynamic`, `rtree` |
| `PLEIADES_GPU=0` | Force CPU (avoids GPU upload/sync overhead when pair count < 80M) |
| `PLEIADES_VERBOSE=1` | Timing logs |
| `PLEIADES_PROFILE=1` | Join sub-phase timing |
