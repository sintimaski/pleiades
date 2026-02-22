# How Pleiades works

High-level flow and where the heavy work runs.

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
  - Assign pixels (ra, dec → HEALPix)
  - For each B row in same/neighbor pixels: haversine, emit (id_a, id_b, sep)
  - Write matches to Parquet
        |
        v
  matches.parquet (id_a, id_b, separation_arcsec)
```

**Rust does the join.** Python does: CLI, validation, schema checks, and helpers (summarize, cone search, partition). B is either partitioned once into shards or read from a pre-partitioned directory; each A chunk only touches B rows in the same or neighboring HEALPix pixels, so we never load full catalogs.

## Main pieces

| Piece | Role |
|-------|------|
| **Rust engine** (`src/`, built with `maturin develop`) | Streams A, indexes by pixel, loads only relevant B from shards, haversine + rayon, writes matches. Supports `n_nearest`, `progress_callback`, `keep_b_in_memory`, pre-partitioned B (directory of `shard_*.parquet`). |
| **Python API** (`python/pleiades/`) | `cross_match()` calls the Rust extension; `cross_match_iter()` is a Python-only streaming path. Analysis, cone search, `partition_catalog`, CLI. Validates inputs; `include_coords` is applied in Python after the Rust run. |
| **pleiades_core** | PyO3 bindings to the Rust extension; built by maturin and shipped in wheels. |

Cross-match is **Rust-only**; there is no Python fallback. If the extension isn’t installed you get a clear error and install instructions.

## Bottlenecks and knobs

- **B reads**: One partition pass (or skip it by passing a shard directory). Per A chunk we only read B rows for the pixels we need.
- **Memory**: Governed by `batch_size_a`, `batch_size_b`, and `n_shards`. Smaller batches = less RAM, more I/O.
- **Throughput**: Rayon parallelizes the inner B-loop in Rust. For big runs, pre-partition B and pass the directory; use `progress_callback` (e.g. with tqdm) to watch progress.

Tuning and GPU notes: [PERFORMANCE.md](PERFORMANCE.md).
