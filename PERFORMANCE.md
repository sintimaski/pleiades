# Performance tuning and acceleration

## What the engine already does

- **Rust engine (default):** All I/O and join logic run in Rust (Parquet read/write, HEALPix indexing, haversine distance).
- **Parallelism:** Join runs over B rows with Rayon; B shards are loaded in parallel; A is read in a prefetch thread so I/O overlaps with compute.
- **Chunked streaming:** Catalog A is processed in batches; B is either partitioned once to temp shards or read from a pre-partitioned directory, so B is not re-read per A chunk.
- **Index by indices:** Match phase uses indices to avoid cloning ID values until building the output columns.
- **Batched haversine:** The join loop computes angular distances in batches of 4 (same B row, four A candidates) via `haversine_arcsec_4`, with scalar fallback for the remainder. The inner formula uses the standard haversine (sin²(Δφ/2) + cos φ₁ cos φ₂ sin²(Δλ/2); θ = 2 asin(√a)) with hoisted constants; the compiler can autovectorize the 4-lane loop.

## Tuning parameters

| Parameter        | Effect |
|-----------------|--------|
| `batch_size_a`   | Rows of A per chunk. Larger = fewer chunks, more memory per chunk, better CPU utilization. Try 200_000–500_000 for big runs. |
| `batch_size_b`   | Rows of B per read when partitioning. Affects partition phase only. |
| `n_shards`       | Number of HEALPix shards (default 512). More shards = smaller per-shard files and more targeted loading; fewer = less overhead, coarser granularity. |
| `depth`          | HEALPix depth (default 8). Higher = finer pixels, more neighbors to check, better selectivity on dense catalogs. |

**Verbose timing:** Set `ASTROJOIN_VERBOSE=1` (or any value) to log partition, per-chunk index/load/join/write, and total time. With verbose, the engine also logs detected parallelism and the Rayon thread pool size. If you see only one thread (e.g. in a container or restricted environment), set **`RAYON_NUM_THREADS`** to the number of threads to use (e.g. `export RAYON_NUM_THREADS=8`).

**Tune on your machine:** Run `uv run python scripts/tune_cross_match_params.py --rows 200000` to sweep batch sizes (and optionally `--sweep-shards`, `--sweep-memory`) and get a recommended config for your hardware.

**Pre-partitioned B:** If you run multiple cross-matches with the same B, partition B once (e.g. `astrojoin.partition_catalog(...)`) and pass the shard directory as `catalog_b`. The engine will skip partitioning and load only the shards needed per A chunk.

**keep_b_in_memory (default: False):** When `catalog_b` is a file, setting this to `True` partitions B into RAM instead of temp shard files, so there is no shard I/O after the initial B read. **This is not the default on purpose:** the project is *out-of-core* — it is designed to run on laptops and machines with limited RAM by streaming and using disk. Use `keep_b_in_memory=True` only when B is small enough to fit comfortably in memory (e.g. B is a few hundred MB and you have plenty of free RAM). On memory-constrained machines or for large B, leave it `False` so the engine stays within a bounded memory footprint.

**Cancellation (Ctrl+C):** The engine invokes the progress callback after each chunk. If the callback returns `False`, the run stops at the next chunk boundary and raises an error (so Ctrl+C can take effect within a chunk or two instead of after the whole run). The benchmark and tune scripts install a SIGINT handler and pass a progress callback that returns `False` when you press Ctrl+C.

## Low-level I/O ideas (cross-platform)

Concepts or tools that can reduce I/O time and work across OSes (Windows, macOS, Linux):

| Idea | What it does | How it helps |
|------|----------------|--------------|
| **Memory-mapped files (mmap)** | Map a file into the process address space; reads/writes become memory accesses, with the OS handling paging and read-ahead. | Fewer copies (no user-space buffer); OS can prefetch and evict pages. Good for large, sequential or random access to the same file. Rust: `memmap2` crate. Parquet/Arrow would need to read from a `&[u8]` or `Bytes`; some runtimes support that. |
| **Larger buffer sizes** | Use a bigger buffer for `BufReader`/`BufWriter` or for the Parquet reader’s internal reads. | Fewer syscalls; better throughput when I/O is sequential. Default is often 8 KiB; 256 KiB–1 MiB can help for large sequential reads. |
| **Fewer, larger reads** | Prefer reading whole row groups or big chunks instead of many small reads. | Same as above: fewer syscalls and better use of OS and device read-ahead. |
| **Sequential access hints** | Tell the OS that access will be sequential (e.g. `posix_fadvise(POSIX_FADV_SEQUENTIAL)` on Unix, or `MADV_SEQUENTIAL` for mmap). | OS can aggressize read-ahead and avoid treating the file as randomly accessed. Less universal on Windows but the pattern (sequential reads) still helps. |
| **Overlapped I/O / async** | Do I/O in the background while the main thread computes (you already prefetch A). | Can extend to overlapping “load next B shards” with “join current chunk”. Cross-platform with threads or async runtimes (e.g. tokio). |
| **Uncompressed or fast compression for temp data** | Write temp shards with no compression or Snappy. | Less CPU and sometimes faster end-to-end when the bottleneck is write or re-read of shards. |
| **Direct I/O (optional)** | Bypass OS page cache (e.g. `O_DIRECT` on Linux, `FILE_FLAG_NO_BUFFERING` on Windows). | Can help when you manage your own cache or when the working set is huge and you don’t want to pollute cache; use only if you measure. Not always faster. |
| **Place temp dir on fast storage** | Put `TMPDIR` (or the temp shard directory) on SSD/NVMe or a RAM disk. | Reduces latency and increase throughput for partition writes and shard reads. |

**Practical order to try:** (1) Larger batch sizes and buffer sizes (stay out-of-core); (2) temp dir on fast disk (SSD); (3) if still I/O-bound and B is small enough to fit in RAM, consider `keep_b_in_memory=True`; (4) mmap or overlapped B loading; (5) sequential access hints and compression choices for temp shards.

## Further CPU improvements

- **SIMD (easy, cross-platform):** The inner hot path is haversine distance per (A, B) pair. Vectorizing this is the most impactful “acceleration” that works on all platforms:
  - **Option A:** Use a crate that provides SIMD haversine (e.g. [simsimd](https://docs.rs/simsimd) if/when haversine is added, or [HaversineSimSIMD](https://github.com/ashvardanian/HaversineSimSIMD)).
  - **Option B:** Use Rust’s `std::simd` (nightly) or a portable SIMD crate to compute 2–4 distances per loop iteration.
- **Larger default batch sizes:** If your machine has enough RAM, increase `batch_size_a` (and optionally `batch_size_b`) so the join does more work per chunk and amortizes overhead.

## GPU acceleration

There is **no single “easy and cross-platform”** GPU path that works everywhere with minimal code.

| Approach | Pros | Cons |
|----------|------|------|
| **CUDA** | Mature, great tooling, used in RAPIDS/cuSpatial. | **NVIDIA only** (no macOS Metal, no AMD/Intel GPUs). Requires CUDA toolkit and Rust bindings (e.g. `cudarc`, custom kernels, or calling cuDF/cuSpatial from Python and not using this Rust engine for the hot path). |
| **wgpu** | **Cross-platform** (Vulkan, Metal, D3D12, WebGPU). Rust-native, compute shaders supported. | Non-trivial: you write WGSL compute shaders, manage buffers and dispatch. Good for “many pairs” distance kernels; integration with existing Arrow/Parquet pipeline is more work. |
| **OpenCL** | Runs on NVIDIA, AMD, Intel, some Macs. | API is verbose; support and drivers vary; Rust crates (e.g. `ocl`) require setup. Not “easy.” |
| **OneAPI / SYCL** | Cross-vendor. | Complex build and runtime; less common in Rust. |

**Practical recommendation:**

1. **First:** Optimize on CPU (tuning above + SIMD haversine if you can add it). That gets you a single code path that runs well on any machine.
2. **If you need GPU:**  
   - **NVIDIA-only is OK:** Add an optional CUDA path (e.g. feature-gated crate, or a separate Python path using CuPy/RAPIDS for the distance kernel) and keep the Rust engine for I/O and HEALPix.  
   - **Must support Mac/AMD/Intel:** Consider wgpu compute: implement a small compute shader that, given buffers of (ra, dec) for A and B candidates, writes distances; call it from Rust and merge results. This is more engineering but stays cross-platform.

### wgpu backend (optional)

An optional **wgpu** backend is available for the haversine distance kernel. It uses a WGSL compute shader to compute angular separation (arcsec) for the candidate (A, B) pairs produced by the HEALPix index; the rest of the pipeline (Parquet I/O, HEALPix indexing, match filtering, output) is unchanged.

**How to enable**

1. **Build the extension with the `wgpu` feature** (required; the default build does not include it). You need a Vulkan, Metal, or D3D12 backend on your machine.
   - **Development / editable install** (e.g. when using `uv run python` from the repo):
     ```bash
     uv run maturin develop --features wgpu
     ```
   - Wheel: `maturin build --features wgpu`
   - Rust-only: `cargo build --features wgpu`
2. At runtime, set the environment variable to use the GPU for the join phase:
   ```bash
   export ASTROJOIN_GPU=wgpu
   uv run python scripts/benchmark_cross_match.py --rust --verbose ...
   ```

If you set `ASTROJOIN_GPU=wgpu` but the extension was built *without* the wgpu feature, the Python layer prints a one-line warning to stderr and the join runs on CPU. With verbose logging, the engine reports `join (GPU)` when the GPU was used and `join (CPU)` when the CPU path was used.

If `ASTROJOIN_GPU=wgpu` is set and the extension has the feature but no GPU adapter is available, the engine falls back to the CPU join automatically.

**When the GPU is used:** Because of buffer size limits, the GPU path processes pairs in chunks (each with upload, compute, readback, and sync). For small or medium pair counts this overhead can make the GPU **slower** than the CPU (e.g. 15+ s vs 1–2 s per chunk). So by default the engine uses the GPU only when the number of candidate pairs in a chunk is **≥ 80M** (`ASTROJOIN_GPU_MIN_PAIRS`, default 80_000_000). Below that, it uses the CPU join even when `ASTROJOIN_GPU=wgpu` is set. To force GPU for all chunk sizes (for benchmarking), set `ASTROJOIN_GPU_MIN_PAIRS=0`.
