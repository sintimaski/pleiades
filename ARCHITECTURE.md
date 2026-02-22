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

- Tuning and GPU: [PERFORMANCE.md](PERFORMANCE.md)
