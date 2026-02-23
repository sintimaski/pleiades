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

### 2.4 Haversine is scalar (medium impact on CPU path)

- **Where:** `haversine_rad`, `haversine_arcsec_8_rad`, `haversine_arcsec_4_rad` in `engine.rs` are hand-unrolled (8/4/1) but **not SIMD**.
- **Effect:** CPU join is compute-bound; SIMD (e.g. 4 or 8 distances per vector) could reduce time when GPU is not used or pair count &lt; threshold.

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
| **SimSIMD / haversine SIMD** | SIMD kernels for haversine and other distances | **Directly addresses pain #4:** use SIMD (e.g. simsimd or custom AVX2/NEON) for 4–8 haversine per call to speed up CPU path. |

**Summary:** The two highest-leverage, single-machine-friendly improvements are: **(1) parallel Parquet row-group decode** for A and B, and **(2) SIMD haversine** on the CPU path. Partition merge and channel sizing are secondary but straightforward.

---

## 4. Recommendations (faster, still single-machine)

### 4.1 Parallel Parquet decode (highest impact)

- **Goal:** Decode multiple row groups of the **same** file in parallel so multiple cores are used during read.
- **Options:**
  - **Arrow Rust:** Check if `parquet::arrow::async_reader` (e.g. `ParquetRecordBatchStream`, `next_row_group()`) or a row-group–level API is available in the version you use (arrow 52 / parquet 52). If so, drive multiple row groups in parallel (e.g. Rayon over row group indices) and feed batches into the existing channel.
  - **Alternative:** Use a reader that exposes row-group boundaries; spawn N workers that each decode a subset of row groups and send batches to the same channel. May require a different crate or a fork that exposes row-group–level read.
- **Single-machine:** Stays on one machine; no distributed I/O. Only CPU and possibly disk bandwidth are better utilized.

### 4.2 SIMD haversine on CPU path (high impact)

- **Goal:** Replace or augment the current scalar haversine (8/4/1 batches) with SIMD so 4–8 distances are computed per call.
- **Options:**
  - **simsimd** (or similar): Use a crate that provides SIMD haversine; integrate into `run_cpu_join` for the hot loops (batch-of-8 and batch-of-4, or a single larger batch).
  - **Custom:** Use `std::simd` (Rust nightly or stable when available) or architecture-specific intrinsics (AVX2/NEON) to compute 4×f64 or 8×f64 haversine; keep same API (inputs: slices of ra/dec; output: separations).
- **Single-machine:** Purely compute; no change to distribution model.

### 4.3 Partition B: parallel merge and write (medium impact)

- **Goal:** After parallel pixel assignment, avoid a single-threaded merge; write shards in parallel.
- **Options:**
  - Merge per-shard results in parallel (e.g. each shard’s Vec is filled by one worker, or merge by shard index in parallel), then **parallel flush**: one thread per shard (or a pool) so multiple `ArrowWriter::write()` and `close()` run concurrently. Requires one writer per shard (you already have that); only the merge logic and who calls `write` need to change.
- **Single-machine:** Better use of many cores during the first B partition pass.

### 4.4 Prefetch and backpressure (low–medium impact)

- **Goal:** Reduce chance that a slow join blocks the A (or B) prefetcher.
- **Options:**
  - Increase A channel capacity (e.g. 3–4 batches) and optionally B response capacity so the main thread can run join while more batches are read.
  - Optionally make batch size for A slightly larger so there are fewer handoffs (tune with benchmarks).

### 4.5 Overlap match output write (lower impact)

- **Goal:** Don’t block the main loop on `writer.write(batch)`.
- **Options:**
  - Background writer thread: main thread sends match RecordBatches via a channel; writer thread does `ArrowWriter::write` and buffers; main thread only builds batches and enqueues. Requires careful close/flush on finish.
- **Single-machine:** Same process; better overlap of join and I/O.

### 4.6 GPU threshold and tuning (situational)

- **Goal:** Use GPU for more workloads where it wins, without hurting small runs.
- **Options:**
  - Lower `PLEIADES_GPU_MIN_PAIRS` (e.g. 20M–40M) and benchmark; or add a small benchmark step that picks CPU vs GPU by timing a sample.
  - Document `PLEIADES_GPU` and `PLEIADES_GPU_MIN_PAIRS` for users who want to force CPU or encourage GPU.

---

## 5. Priority order (single-machine, max speedup)

1. **Parallel Parquet decode** (A and, where applicable, B shards) — removes the biggest single-threaded bottleneck.
2. **SIMD haversine** — large gain on CPU path with no architectural change.
3. **Parallel partition-B merge + write** — faster first run when B is a file.
4. **Prefetch channel sizing** — quick change; validate with benchmarks.
5. **Background match writer** — if profiling shows write time is non-negligible.
6. **GPU threshold tuning** — per-environment.

---

## 6. References

- **ARCHITECTURE.md** (this repo): flow, bottlenecks, knobs.
- **Arrow Rust async reader:** `parquet::arrow::async_reader`, `ParquetRecordBatchStream`, row-group concurrency.
- **parquet2:** Row-group–level read and parallel decode patterns.
- **SimSIMD:** https://github.com/ashvardanian/simsimd (SIMD haversine and distances).
- **HEALPix Alchemy, smatch, healsparse, HLC2:** Spatial indexing and cross-match in astronomy.
