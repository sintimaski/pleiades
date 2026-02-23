# How Pleiades works

High-level flow and where the work runs.

---

## Flow

```
catalog_a.parquet     catalog_b.parquet (or shard dir)
        |                        |
        v                        v
  Stream A in chunks        Partition B by HEALPix (or use existing shards)
        |                        |
        +------------------------+
        |
        v
  For each A chunk:
  · Pixels + index: one pass over (ra, dec) → HEALPix index (pixel → rows) and pixel set (centers + neighbors); chunk 1+ reuses pixel set from previous B-prefetch and only builds index
  · Load B rows for those pixels (from shards or memory)
  · For each B row in same/neighbor pixels: haversine → (id_a, id_b, sep)
  · Write matches to Parquet
        |
        v
  matches.parquet (id_a, id_b, separation_arcsec)
```

- **Rust** does the join
- **Python** does: CLI, validation, schema checks, helpers (summarize, cone, partition)
- B is partitioned once or read from a shard directory; each A chunk only touches B rows in same/neighbor pixels

---

## Main pieces

| Piece | Role |
|-------|------|
| **Rust engine** (`src/`) | Streams A, indexes by pixel, loads only relevant B from shards, haversine + rayon, writes matches. Supports `n_nearest`, `progress_callback`, `keep_b_in_memory`, pre-partitioned B (`shard_*.parquet` dir). |
| **Python API** (`python/pleiades/`) | `cross_match()` → Rust; `cross_match_iter()` → Python-only streaming. Analysis, cone search, `partition_catalog`, CLI. `include_coords` applied in Python after Rust. |
| **pleiades_core** | PyO3 bindings; built by maturin, shipped in wheels. |

- Cross-match is **Rust-only** (no Python fallback)
- Missing extension → clear error + install instructions

---

## Bottlenecks and knobs

| Knob | Effect |
|------|--------|
| **B reads** | One partition pass, or pass a shard directory to skip. Per A chunk: only B rows for needed pixels. |
| **Memory** | `batch_size_a`, `batch_size_b`, `n_shards`. Smaller = less RAM, more I/O. |
| **Throughput** | Rayon in Rust. Pre-partition B + pass directory; use `progress_callback` (e.g. tqdm). |

### Decode and copy (load B from shards)

When B is loaded from shards, cost is **Parquet decode** + **copy into Rust structs** + **memory bandwidth** (not disk if cached). Implemented:

- **Large batch size** (`SHARD_READ_BATCH_ROWS` 128k): fewer decode cycles and larger reads.
- **Fast path in `load_one_shard`**: Shard schema is fixed (UInt64 pixel_id, Float64 ra/dec from `partition_b_to_temp`). We downcast columns once per batch, use slice access for ra/dec (`ra_arr.values()[row]`), reserve capacity, and specialize Int64 `id_b` to avoid per-row `get_id_value` dispatch and extra branches.

**macOS:** Optional build feature `macos_readahead` enables `fcntl(F_RDAHEAD)` on opened shard files so the kernel does read-ahead for sequential Parquet reads. Build with `maturin develop --features macos_readahead` on macOS.

Columnar B is used in the join (`BColumns`: join hot path streams `ra_b`/`dec_b`; `id_b` fetched only when emitting a match). Possible later: predicate pushdown on shard Parquet by pixel_id if the format allows.

---

## Parallelism (threads and I/O)

**Already in place:**

- **A reads:** A dedicated thread prefetches catalog A (1–2 batches ahead) so Parquet I/O overlaps with join work. Main thread consumes batches; reader thread keeps filling the channel.
- **B reads (shards):** When `catalog_b` is a directory of shards, **multiple shard files are read in parallel** (Rayon `par_iter` over the shard indices needed for the chunk). So many threads read different `shard_*.parquet` files at once.
- **Join:** Columnar B (`BColumns`: `id_b`, `ra_b`, `dec_b`); the hot path touches only `ra_b`/`dec_b` and fetches `id_b` when emitting a match. B rows are grouped by HEALPix pixel so `pixels_to_look` (center + 8 neighbors) is computed once per pixel. Haversine runs in batches of 8 (then 4, then remainder); a cheap reject (|Δdec| / cos(dec) bound vs `radius_deg`) skips haversine for clearly outside pairs in the remainder. Rayon parallelizes over pixels.
- **B prefetch:** When using shards, a background thread loads B for the current and next chunk (two requests in flight). The main thread sends both at chunk start, builds the index (while current B loads), then runs the join (while next B loads). So index and load B overlap for the first chunk, and join and load-next-B overlap for every chunk.
- **Pixels + index (A chunk):** A single parallel pass builds both the HEALPix index (pixel → row indices) and the pixel set (centers + 8 neighbors) with one hash per row (`pixels_and_index`). For chunk 1+, the pixel set is reused from the previous chunk’s B-prefetch request and only the index is built (`index_only`). The “next” chunk’s pixel set is computed for B prefetch and stored for the next iteration. Pixel-set construction uses parallel `fold_with`/`reduce_with` for HashSet merge. With `PLEIADES_VERBOSE=1`, the engine logs `pixels+index` time and pixel count per chunk.
- **Partitioning B** (when `catalog_b` is a file): Batches are collected then processed in parallel (`partition_batch_to_row_results` per batch via Rayon); results are merged into shard buffers and flushed. Single-threaded read and write; pixel assignment is parallel.
- **n_nearest per chunk:** When `n_nearest` is set, each chunk keeps only the best n matches per `id_a` before writing (`merge_to_n_nearest`); `apply_n_nearest` still runs at the end to merge across chunks.

**Still single-threaded:**

- **Single-file Parquet decode:** Each Parquet file (A or one B shard) is decoded by one reader in one thread. The underlying crate reads row groups sequentially; parallel row-group decode would need API support.

**Could add later:**

- **Parallel row-group reads:** If the Parquet/Arrow API allows, decode different row groups of the same file in parallel (helps when one large file is the bottleneck).

So: **yes, multiple threads already read input** (A in a prefetch thread; B shards in parallel). Adding more threads to “read the same file” would only help if we split the file (e.g. by row groups) and decode in parallel, which the current reader API doesn’t expose; for **multiple files** (shards), we already do it.

---

## Further speedups (matching)

Implemented: columnar B (`BColumns`), reuse of `pixels_to_look` per pixel, cheap reject before haversine (remainder path), haversine batches of 8 (then 4), n_nearest per chunk before write, parallel partition of B (batches processed in parallel). See “Join” and “Partitioning B” above.

**Could add later:**

| Idea | Benefit | Notes |
|------|---------|--------|
| **GPU join by default when available** | Large speedup for huge pair counts | Today GPU is opt-in (`PLEIADES_GPU=wgpu`) and only used when pair count > threshold. Could default on when the wgpu feature is built and a GPU is available. |
