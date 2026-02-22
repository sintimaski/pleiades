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
  · Assign pixels (ra, dec → HEALPix)
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

Possible later improvements: columnar B representation (`ra_b: &[f64], dec_b: &[f64]`) so the join hot path streams ra/dec without touching ids; predicate pushdown on shard Parquet by pixel_id if the format allows.

---

## Parallelism (threads and I/O)

**Already in place:**

- **A reads:** A dedicated thread prefetches catalog A (1–2 batches ahead) so Parquet I/O overlaps with join work. Main thread consumes batches; reader thread keeps filling the channel.
- **B reads (shards):** When `catalog_b` is a directory of shards, **multiple shard files are read in parallel** (Rayon `par_iter` over the shard indices needed for the chunk). So many threads read different `shard_*.parquet` files at once.
- **Join:** The inner B-loop (haversine, candidate search) is parallelized with Rayon over B rows. Thread count follows `RAYON_NUM_THREADS` or `std::thread::available_parallelism()`.
- **B prefetch:** When using shards, a background thread loads B for the current and next chunk (two requests in flight). The main thread sends both at chunk start, builds the index (while current B loads), then runs the join (while next B loads). So index and load B overlap for the first chunk, and join and load-next-B overlap for every chunk.
- **Index build (A chunk):** Building the HEALPix index (pixel → row indices) and `id_a_flat` for each A chunk is done in parallel with Rayon (`par_iter` + `fold_with`/`reduce_with`), so this hot path is no longer sequential.

**Still single-threaded:**

- **Partitioning B** (when `catalog_b` is a file): one sequential read of B and sequential writes to shard files. Parallelizing would require splitting the B file by row ranges and merging shard writes (more complex).
- **Single-file Parquet decode:** Each Parquet file (A or one B shard) is decoded by one reader in one thread. The underlying crate reads row groups sequentially; parallel row-group decode would need API support.

**Could add later:**

- **Parallel partition of B:** e.g. stream B in chunks, assign chunks to a Rayon pool, each worker writes to its shard(s) with synchronized writers (or per-thread buffers then merge).
- **Parallel row-group reads:** If the Parquet/Arrow API allows, decode different row groups of the same file in parallel (helps when one large file is the bottleneck).

So: **yes, multiple threads already read input** (A in a prefetch thread; B shards in parallel). Adding more threads to “read the same file” would only help if we split the file (e.g. by row groups) and decode in parallel, which the current reader API doesn’t expose; for **multiple files** (shards), we already do it.
