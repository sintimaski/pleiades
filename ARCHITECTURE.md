# Pleiades MVP: flow, optimizations, bottlenecks, usability

## 1. Flow

```
catalog_a.parquet          catalog_b.parquet
        |                            |
        v                            |
  iter_batches(batch_size_a)         |
        |                            |
        v                            |
  For each chunk A:                  |
  - Compute HEALPix pixel (vectorized ang2pix)
  - Build index: pixel_id -> [(id_a, ra, dec), ...]
        |                            |
        +----------------------------+
        |                            |
        v                            v
  For each chunk B:            iter_batches(batch_size_b)
  - For each row B: (ra_b, dec_b, id_b)
    - Center pixel + 8 neighbors (get_all_neighbours)
    - Look up A candidates in index[pixels]
    - For each candidate: haversine distance
    - If sep <= radius_arcsec -> emit (id_a, id_b, sep)
        |
        v
  Buffer matches -> ParquetWriter.write_table (per B chunk)
        |
        v
  matches.parquet (id_a, id_b, separation_arcsec)
```

**Important**: We stream A in chunks and, **for each A chunk**, we **re-read the entire B** (full scan of B per A chunk). So total B reads = (number of A chunks) × (full B). Memory = O(batch_A) + O(batch_B) at any time; we never load full A or full B.

---

## 2. Optimizations

| What | Where | Effect |
|------|--------|--------|
| **HEALPix indexing** | Chunk A: group by pixel; B: lookup only center + 8 neighbors | Avoids O(N×M) brute force: we only compare B rows to A rows in the same or adjacent pixels. |
| **Chunked I/O** | `ParquetFile.iter_batches(batch_size)` for A and B | Keeps memory bounded; no full load of catalogs. |
| **Vectorized pixel assignment** | `hp.ang2pix(nside, ra_a, dec_a, lonlat=True)` on numpy arrays | One call per A chunk instead of per-row Python loop. |
| **Batch write** | Matches buffered per B chunk, then `writer.write_table(tbl)` | Fewer Parquet writes instead of one row at a time. |
| **Haversine** | Single scalar/array path, `np.minimum(sqrt(x), 1.0)` to avoid numerical issues | Correct and stable angular distance. |
| **A×B matrix (default)** | `use_matrix=True`: for each (chunk_A, chunk_B), compute distance in blocks of (block×block) with `_haversine_matrix_arcsec`, then `np.where(D <= radius)`. | Vectorized inner loop; much faster than per-row Python. Memory = block²×8 bytes (e.g. 4000² ≈ 128 MB). |

---

## 3. Persistent bottlenecks

| Bottleneck | Why | Possible direction |
|------------|-----|--------------------|
| **B read many times** | ~~For each A chunk we scan all of B.~~ | **Addressed**: `partition_b=True` (default) writes B to HEALPix shards once; each A chunk loads only B rows in the same/neighbor pixels from those shards. B is read once (partitioning) + per-chunk only relevant shards. |
| **Python inner loop** | Addressed when use_matrix=True (block matrix). | Addressed: block (A×B) matrix + np.where; partition path same. Remaining: Rust/rayon for more speed. |
| **Index build per A chunk** | We build `pixel -> list of (id, ra, dec)` in Python with many small lists. | Build index in Rust/Arrow (e.g. array of structs grouped by pixel) or use a faster in-memory structure. |
| **Single-threaded** | One process, one thread for the join. | Parallelize over A chunks (each chunk gets a B stream or B partitions) or over B chunks within an A chunk (rayon / multiprocessing). |
| **No pre-partitioning** | If A and B are not spatially ordered, we still do full scans. | **Addressed for B**: use **partition_catalog**(catalog_b, output_dir) to write shards once; pass the directory as catalog_b. For A, same helper can pre-partition; join-by-pixel is a possible future extension. |

**Matrix vs HEALPix path**: `use_matrix=True` (default) computes distances as (A_block × B_block) matrices in blocks (e.g. 4000×4000); fully vectorized, bounded memory. `use_matrix=False` builds a HEALPix index and only compares B rows to A candidates in the same/neighbor pixels (fewer distance ops, more Python overhead). The matrix path usually wins for typical chunk sizes.

---

## 4. Usability

| Aspect | Status | Notes |
|--------|--------|--------|
| **API** | Good | Single function `cross_match(catalog_a, catalog_b, radius_arcsec, output_path)`; optional `id_col_a`, `id_col_b`, `ra_col`, `dec_col`, `nside`, `batch_size_*`. |
| **Docs** | OK | README has usage; docstring describes behavior. ARCHITECTURE (this file) covers flow and limits. |
| **Input contract** | Clear | Parquet with `ra`, `dec` in degrees; ID columns inferred or set. No schema validation beyond “column exists”. |
| **Output** | Clear | Parquet with id_a, id_b, separation_arcsec. Empty file if no matches (schema still written). |
| **Errors** | Basic | FileNotFoundError if a catalog path is missing. No handling for wrong column names or types. |
| **Scale** | Limited | Suitable for “medium” catalogs (e.g. A and B in the millions). B is re-read per A chunk, so very large B becomes expensive. |
| **Correctness** | Good | HEALPix halo (center + 8 neighbors) avoids missing cross-boundary pairs. Fixture test: 10 planted pairs recovered with radius_arcsec=2.0. |
| **Dependencies** | Few | PyArrow, healpy, numpy (and pydantic). No Rust required for the Python MVP. |

**Summary**: Usable for production-like runs on modest catalog sizes and for validating the algorithm. For billion-row scale and better performance, the Rust engine (see below) addresses the bottlenecks above.

---

## 5. Rust engine (optional)

- **Location**: `src/engine.rs`, `src/lib.rs`; built with `maturin develop`.
- **Flow**: Stream A in batches → build HEALPix index (pixel → candidates). **If catalog_b is a file**: re-stream B per A chunk, each B row → center+8 neighbours via cdshealpix, lookup candidates, haversine filter (inner loop parallelized with rayon), write matches. **If catalog_b is a directory** (pre-partitioned B): list `shard_*.parquet`, for each A chunk compute pixels_wanted (chunk pixels + neighbours), load only B rows from shards for those pixels, then same haversine join. Supports `n_nearest` (post-pass filter) and **progress_callback** (invoked per A chunk).
- **Return value**: Rust engine returns **CrossMatchStats** (output_path, rows_a_read, rows_b_read, matches_count, chunks_processed, time_seconds), exposed to Python as a dict and converted to **CrossMatchResult**.
- **Use**: `cross_match(..., use_rust=True)` (falls back to Python if `pleiades_core` not installed). Pre-partitioned B and progress callback work with the Rust path.
- **Validation**: `test_rust_matches_equal_reference_brute_force` and `test_rust_with_prepartitioned_b_matches_reference` compare Rust output to reference; `test_rust_progress_callback_called` checks progress is invoked.
