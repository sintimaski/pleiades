# Pleiades: Performance Analysis & Speedup Options

This document summarizes the project flow, pain points, similar projects, and concrete ways to make cross-match **much faster** while preserving **single-machine** execution.

---

## 1. Project and flow (concise)

**What Pleiades does:** Out-of-core spatial cross-match of two Parquet catalogs (A, B) using HEALPix indexing and haversine distance. Output: `(id_a, id_b, separation_arcsec)`.

**High-level flow:**

```
catalog_a.parquet          catalog_b.parquet (or shard dir)
       |                              |
       v                              v
 Stream A in chunks            Partition B by HEALPix (or use existing shards)
       |                              |
       +------------------------------+
       v
 For each A chunk:
   · Pixels + index: one parallel pass → HEALPix index + pixel set (center + 8 neighbors)
   · Load B rows for those pixels only (from shards or in-memory)
   · CPU (or GPU) join: haversine → (id_a, id_b, sep)
   · Optional n_nearest; write matches to Parquet
       v
 matches.parquet
```

**Key implementation details:**

- **Rust engine** (`src/engine.rs`): streaming A, HEALPix index per chunk, columnar B (`BColumns`), Rayon over pixels for join, prefetch threads for A and (when shards) B.
- **B handling:** If B is a file: one partition pass to temp shards or in-memory; if B is a directory of `shard_*.parquet`, use it directly. Per chunk only B rows for needed pixels are loaded.
- **Join:** Group B by pixel; for each pixel, candidates from index (center + neighbors); haversine in batches of 8 → 4 → remainder with cheap reject; optional GPU when pair count ≥ 80M (default).

---

## 2. Pain points (bottlenecks)

### 2.1 Single-threaded Parquet decode (high impact)

- **Where:** Each Parquet file (catalog A, each B shard) is decoded by **one** reader in **one** thread. Row groups are read sequentially.
- **Effect:** Large A or large B shards become a decode bottleneck; extra CPU cores sit idle during that phase.
- **Already in place:** A prefetch thread (and B prefetch when using shards) overlaps I/O with compute, but **decode** of a given file is still single-threaded.

### 2.2 Partition-B merge and flush (medium impact)

- **Where:** `partition_b_to_temp`: after `batches.par_iter().map(partition_batch_to_row_results)`, results are merged into per-shard buffers and flushed **single-threaded** (loop over `batch_results`, drain buffers at `FLUSH_AT`, build RecordBatches, `writer.write()`).
- **Effect:** First run when B is a file: partition pass is long; merge/write doesn’t use multiple cores.

### 2.3 Channel backpressure

- **Where:** A channel capacity 1 or 2 (for A and B prefetch). Main thread can block the A reader if the join is slow.
- **Effect:** If join dominates, prefetch may stall; increasing capacity (e.g. 3–4) could improve overlap.

### 2.4 Haversine SIMD (addressed)

- **Implementation:** With the `simd` feature (default), `haversine_a_4_rad` and `haversine_a_8_rad` use AVX2 on x86_64 for vectorized loads, deg-to-rad conversion, dlat/dlon arithmetic, and mul_add. Sin/cos remain scalar (no native SIMD transcendentals). See `src/haversine_simd.rs`.
- **Effect:** Reduces CPU join time when GPU is not used.

### 2.5 Match output write (lower impact)

- **Where:** Main thread builds one RecordBatch per chunk and calls `ArrowWriter::write()`; single-threaded, synchronous.
- **Effect:** Usually smaller than decode/join; could be overlapped later (e.g. background writer thread) if profiling shows it matters.

### 2.6 GPU threshold and sync

- **Where:** GPU is used only when pair count ≥ `PLEIADES_GPU_MIN_PAIRS` (default 80M); otherwise CPU. Upload/sync overhead makes small pair counts faster on CPU.
- **Effect:** For “medium” pair counts, neither path may be optimal; tuning threshold or adding a hybrid policy could help.

---

## 3. Similar projects and how they address these problems

| Project / area        | Approach | Relevance to Pleiades |
|-----------------------|----------|------------------------|
| **HEALPix Alchemy**   | SQL + HEALPix indexing in a DB for all-sky geometry | Same idea: partition by HEALPix to reduce work; we already do this; DB adds distribution/query layer we don’t need for single-machine. |
| **smatch**             | HEALPix-based matching on the sphere | Same partitioning idea; we already use cdshealpix; could compare depth/neighbor strategies. |
| **healsparse**        | Sparse HEALPix maps, reduced memory | Confirms HEALPix + sparse representation is standard; we could consider sparse B representation per pixel if memory becomes an issue. |
| **HLC2**              | Heterogeneous (CPU+GPU) cross-match for large catalogs | Similar goal (fast cross-match); we already have optional wgpu; can borrow ideas for when to use GPU and how to chunk. |
| **Apache Arrow (Acero)** | Hash join with row table, 64-bit offsets; streaming limitations | Arrow’s hash join is in-memory; we do spatial join (haversine). Takeaway: parallel row-group decode and buffering (e.g. async reader) are the direction for “same file, more throughput.” |
| **parquet / parquet2 (Rust)** | Metadata first, then row groups; async reader with `next_row_group()` and concurrent decode | **Directly addresses pain #1:** parallel row-group decode (multiple row groups in parallel) instead of one sequential reader. |
| **SimSIMD / haversine SIMD** | SIMD kernels for haversine and other distances | **Addressed:** custom AVX2 in `haversine_simd.rs` for 4–8 haversine per call on CPU path. |

**Summary:** The highest-leverage improvements are: **(1) parallel Parquet row-group decode** for A and B (not yet done), and **(2) SIMD haversine** (implemented, default feature). Partition merge and channel sizing are secondary.

---

## 4. Recommendations (faster, still single-machine)

### 4.1 Parallel Parquet decode (implemented)

- **Status:** Implemented. Decodes multiple row groups of the same file in parallel via Rayon.
- **Mechanism:** `parquet_parallel::read_parquet_parallel()` splits row group indices across workers; each worker opens the file, uses `ParquetRecordBatchReaderBuilder::with_row_groups()` for its subset, decodes to RecordBatches, and returns. Falls back to sequential read when 0–1 row groups.
- **Integration:** Used in `partition_file_to_shard_dir` (catalog B partitioning) and `load_one_shard` (B loading from pre-partitioned shards).
- **Files:** `src/parquet_parallel.rs`; `set_readahead_for_parquet_input` moved to `parquet_mmap.rs`.

### 4.2 SIMD haversine on CPU path (implemented)

- **Status:** Implemented via `simd` feature (default). Uses `std::arch` AVX2 on x86_64 for 4×f64 haversine `a` term (cheap-reject path); 8-lane batches call 4-lane twice. Runtime feature detection falls back to scalar on CPUs without AVX2.
- **Files:** `src/haversine_simd.rs`; engine dispatches via `haversine_a_4_rad` / `haversine_a_8_rad`.

### 4.2b Parquet mmap (implemented, opt-in)

- **Status:** Implemented via `parquet_mmap` feature (opt-in). Uses memory-mapped I/O (`memmap2`) for Parquet reads. On macOS with multi-shard workloads, mmap can be slower than File+read (partition B and load B); on Linux/single-file sequential reads it may help. Build with `--features parquet_mmap` to enable.
- **Files:** `src/parquet_mmap.rs`; all Parquet read paths use `ParquetInput::open()`.

### 4.3 Partition B: parallel merge and write (implemented)

- **Status:** Implemented. After parallel decode and `partition_batch_to_row_results`, results are grouped by shard in parallel, then each shard's data is written in parallel via Rayon.

### 4.4 Prefetch and backpressure (implemented)

- **Status:** A channel capacity increased to 4 (was 2) when B prefetch used, 2 otherwise. B request/response channels increased to 4 (was 2).

### 4.5 Overlap match output write (lower impact)

- **Goal:** Don’t block the main loop on `writer.write(batch)`.
- **Options:**
  - Background writer thread: main thread sends match RecordBatches via a channel; writer thread does `ArrowWriter::write` and buffers; main thread only builds batches and enqueues. Requires careful close/flush on finish.
- **Single-machine:** Same process; better overlap of join and I/O.

### 4.6 Join strategy: A-driven and R-tree (experimental)

- **PLEIADES_JOIN_STRATEGY** env var selects the CPU join algorithm:
  - **bdynamic** (default): B-driven — for each B row, find A candidates. Uses dec pre-filter, block bbox, a≤a_max cheap reject.
  - **adynamic**: A-driven — for each A candidate, find B rows in dec window. Use when |A| < |B| in a pixel.
  - **rtree**: R-tree — build spatial index on A candidates (≥2000), range-query per B row. Requires `--features rtree`.
- Benchmark with your catalogs to see which strategy is faster.

### 4.7 GPU threshold and tuning (situational)

- **Goal:** Use GPU for more workloads where it wins, without hurting small runs.
- **Options:**
  - Lower `PLEIADES_GPU_MIN_PAIRS` (e.g. 20M–40M) and benchmark; or add a small benchmark step that picks CPU vs GPU by timing a sample.
  - Document `PLEIADES_GPU` and `PLEIADES_GPU_MIN_PAIRS` for users who want to force CPU or encourage GPU.

---

## 5. Benchmark profile (10M×10M, 4 chunks, ~11.5s)

From `run_benchmarks.sh` with `--verbose`:

| Phase | Time | % of total | Hot path |
|-------|------|------------|----------|
| **Join** | ~6.3s | **55%** | haversine loop, cheap reject, SIMD 4/8-lane |
| **Pixels+index** | ~2.2s | 19% | Chunk 1: full index; others: index lookup |
| **Load B** | ~1.2s | 10% | Parallel row-group decode per shard |
| **Partition B** | ~1.1s | 10% | Parallel decode + single-thread merge |
| **Write** | ~0s | 0% | Negligible |

**Profile with:** `./scripts/run_profile.sh` (sample) or `./scripts/run_profile.sh --profile` (sub-phase timing). See `docs/PROFILING.md`.

---

## 5.1 CPU profile analysis (sample, 10M×10M)

From `profile_20260226_130908.txt` (macOS sample, Python process):

**Sort by top of stack (application hotspots):**

| Symbol | Samples | % | Interpretation |
|--------|---------|---|-----------------|
| `Vec::from_iter` | 5730 | ~6% | Building vectors from parallel collect (pixels merge) |
| `HashMap::insert` | 2500 | ~2.5% | Index building (pixel→candidates) |
| `cross_match_impl` | 2477 | ~2.5% | Main engine loop |
| `Hasher::write` | 1426 | ~1.5% | Hashing for HashMap keys |
| `DashMap::_insert` | 1155 | ~1% | Concurrent index (chunk 1: pixels_and_index) |
| `extract_shard_batch_rows` | 867 | ~1% | Load B from shards, filter by pixel |
| `__sincos_stret` | 755 | ~1% | sin/cos in cdshealpix Layer::hash |
| `RawTable::reserve_rehash` | 738 | ~1% | HashMap rehashing |
| `quicksort` | 432 | ~0.5% | sort_unstable in join (candidates_by_dec) |

*(~66k samples in kernel wait states: `__psynch_cvwait`, `semaphore_wait` — Rayon workers; excluded from above.)*

**Improvement suggestions (highest impact first):**

1. **Pre-size index containers**  
   - In `merge_pixels_and_index`, `merge_index_only`, `group_b_by_pixel`: estimate pixel count (e.g. `ra_deg.len() / 100`) and call `HashMap::with_capacity` before the merge loop. Reduces `reserve_rehash` and allocation churn.
2. **DashMap → map+merge** — Implemented. Replaced `DashMap`/`DashSet` with parallel `map()` + single-threaded `merge_pixels_and_index` / `merge_index_only`. Eliminates lock contention; removed `dashmap` dependency.
3. **Batch HEALPix hash**
   - `cdshealpix::Layer::hash` + sin/cos dominate ~755 samples. cdshealpix 0.9 does not expose `hash_many` or bulk API; per-row `layer.hash(lon, lat)` remains.
4. **Pre-allocate Vecs in collect** — Implemented
   - `pixels_and_index`, `index_only`, `partition_batch_to_row_results`, partition in-memory, `id_a_flat`, and partition write use `collect_into_vec` with pre-allocated `Vec::with_capacity(n)`.
5. **Reserve in HashMap** — Implemented via `estimated_pixels()`; `HashMap`/`HashSet` use `with_capacity`.
6. **Try `PLEIADES_JOIN_STRATEGY=adynamic`**  
   - For chunks where |A candidates| < |B rows| per pixel, adynamic may do less work. Benchmark on your data.
7. **extract_shard_batch_rows** — Implemented
   - `pixels_wanted` now uses `FxHashSet<u64>` for faster lookups.

---

## 6. Priority order (single-machine, max speedup)

1. **Parallel Parquet decode** — done. §4.1.
2. **SIMD haversine** — done. AVX2 in `haversine_simd.rs` (default feature).
3. **Index pre-sizing and map+merge** (profile-driven) — Implemented: `HashMap::with_capacity` via `estimated_pixels()`; DashMap replaced with map+merge.
4. **Parallel partition-B merge + write** — faster first run when B is a file.
5. **Prefetch channel sizing** — quick change; validate with benchmarks.
6. **Background match writer** — if profiling shows write time is non-negligible.
7. **GPU threshold tuning** — per-environment.

---

## 7. Move more work to Rust

These Python-side operations are good candidates to move (or delegate) to Rust for speed.

| Feature | What Python does today | Impact | Approach |
|--------|-------------------------|--------|----------|
| **cross_match_iter** | Streams A in batches, partitions B (or uses shards), loads B subset per chunk, runs **NumPy haversine matrix** in blocks (`_cross_match_chunk_matrix`). **Does not call the Rust engine.** | **High** | Implement a Rust-backed streaming API: e.g. engine writes matches to a temp file or channel; Python yields from a reader. Or expose a Rust iterator/generator that yields `(id_a, id_b, sep)` batches. Then `cross_match_iter` becomes a thin wrapper. |
| **partition_catalog** | Streams B with PyArrow, computes HEALPix per row in Python/cdshealpix, writes one row at a time to per-shard Parquet writers. | **Medium** | Rust already has `partition_b_to_temp` (same logic). Expose it as a PyO3 function `partition_catalog(catalog_path, output_dir, ...)` and call it from Python `partition_catalog()`. Single read, parallel pixel assignment, batched writes. |
| **cone_search** | Reads catalog in batches, vectorized haversine to center, filter, write. | **Medium** | Add a Rust `cone_search_impl(path, ra_deg, dec_deg, radius_arcsec, output_path)` that streams, filters, and writes. Reduces Python/NumPy overhead and allows reuse of engine I/O and haversine. |
| **batch_cone_search** | Multiple cones; for each batch, distance to each query, keep row if within any. | **Medium** | Same as cone_search but with multiple centers; Rust can loop over queries or vectorize. |
| **attach_match_coords** | Reads matches + full catalogs A and B into memory, builds id→(ra,dec) dicts, joins, writes. | **Lower** (unless catalogs huge) | Rust could stream: read matches in chunks, lookup id_a/id_b from pre-indexed catalogs (or sorted merge), write extended rows. Helps when catalogs are large. |
| **summarize_matches** | Reads match file, computes count/min/max/mean of separation. | **Lower** | Single pass in Rust over Parquet match file; expose as PyO3. Nice-to-have. |

**Suggested order:** (1) **cross_match_iter** → Rust-backed stream (biggest win for iterator users). (2) **partition_catalog** → call existing Rust partition (easy, avoids duplicate logic). (3) **cone_search** / **batch_cone_search** if those paths are hot.

---

## 8. References

- **PROFILING.md** (this repo): how to profile and interpret results.
- **ARCHITECTURE.md** (this repo): flow, bottlenecks, knobs.
- **Arrow Rust async reader:** `parquet::arrow::async_reader`, `ParquetRecordBatchStream`, row-group concurrency.
- **parquet2:** Row-group–level read and parallel decode patterns.
- **SimSIMD:** https://github.com/ashvardanian/simsimd (SIMD haversine and distances).
- **HEALPix Alchemy, smatch, healsparse, HLC2:** Spatial indexing and cross-match in astronomy.
