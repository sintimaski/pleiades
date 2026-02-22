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

**Verbose timing:** Set `PLEIADES_VERBOSE=1` (or any value) to log partition, per-chunk index/load/join/write, and total time. With verbose, the engine also logs detected parallelism and the Rayon thread pool size. If you see only one thread (e.g. in a container or restricted environment), set **`RAYON_NUM_THREADS`** to the number of threads to use (e.g. `export RAYON_NUM_THREADS=8`).

**Tune on your machine:** Run `uv run python scripts/tune_cross_match_params.py --rows 200000` to sweep batch sizes (and optionally `--sweep-shards`, `--sweep-memory`) and get a recommended config for your hardware.

**Pre-partitioned B:** If you run multiple cross-matches with the same B, partition B once (e.g. `pleiades.partition_catalog(...)`) and pass the shard directory as `catalog_b`. The engine will skip partitioning and load only the shards needed per A chunk.

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

## I/O as bottleneck (e.g. “load B” dominates)

When verbose timing shows **partition B** and **load B** taking most of the time (e.g. partition ~33s, load B ~8s per chunk so ~80s over 10 chunks), I/O is the bottleneck. The join and index phases are already in Rust and fast.

### Do you need an “assembly driver” to read files and push memory to Rust?

**No.** The engine already does all I/O in Rust; there is no Python read path that copies data into Rust. A separate C/assembly “driver” that reads files and hands memory to Rust would not remove a boundary—Rust is already that boundary. The gains come from *how* Rust does I/O (buffer sizes, mmap, overlapped reads, or format choice), not from moving I/O into another layer.

If you wanted a minimal “driver” in the sense of *fastest possible read path*, that would be: inside Rust, use memory-mapped files or a custom binary shard format and parse directly into the structures the join needs, avoiding Parquet decode per shard. That’s still Rust; no assembly required.

### Suggested solutions (in rough order: quick wins → larger changes)

| Solution | Effort | Impact | Notes |
|----------|--------|--------|--------|
| **1. `keep_b_in_memory=True`** | None (flag) | **Removes “load B” entirely** after partition | Use when B fits in RAM (e.g. ~10M rows × ~40 bytes ≈ hundreds of MB). Partition once into RAM; no shard I/O per chunk. |
| **2. Larger read buffers** | Small (code) | 10–30% faster shard reads in many cases | Wrap `File` in `BufReader::with_capacity(1 << 20, file)` (1 MiB) for every Parquet open (partition read, each shard in `load_b_from_shards`, A prefetch). Fewer syscalls, better throughput. |
| **3. Temp dir on fast storage** | None (env) | 2× or more if currently on HDD | Set `TMPDIR` to an SSD or RAM disk so partition write and shard reads are fast. |
| **4. Prefetch B for next chunk** | Medium (code) | Overlap “load B” with “join (CPU)” | Like A prefetch: in a background thread, load the *next* chunk’s B shards while the main thread joins the *current* chunk. Requires predicting next chunk’s pixels or loading a superset. |
| **5. Memory-mapped shard reads** | Medium (code + dep) | Can reduce copy and syscall overhead | Add `memmap2`; open each shard with `unsafe { Mmap::map(&file)? }` and build a Parquet reader from `Bytes::from(mmap.as_ref())` if the Parquet API accepts it. Or use mmap only for a custom binary shard format (see below). |
| **6. Custom binary shard format** | Larger (code) | Highest throughput for B load | Instead of Parquet shards, write a simple binary layout (e.g. fixed record: `pixel_id: u64`, `id_b: i64`, `ra: f64`, `dec: f64`) and memory-map or stream with large buffers. Parse with `std::io::Read` or pointer casts (safe if layout is fixed). No Parquet decode; “load B” becomes mostly I/O bound. |
| **7. Fewer, larger shards** | Tuning | Fewer file opens; more sequential I/O | Try e.g. `n_shards=128` instead of 512. Fewer shards = fewer small files and often better sequential read behavior; selectivity per chunk may drop slightly. |
| **8. Sequential access hints** | Medium (code, platform) | Better OS read-ahead | On Unix, after opening a file for sequential read, call `posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL)` (via `libc` or `nix`). Helps the kernel prefetch. |

**What the engine does for I/O:** Temp shard files (when not using `keep_b_in_memory`) are written with **no compression** and **256 KiB write buffers** to speed up partition and shard reads. Partition uses a **131k-row flush** so fewer Parquet `write()` calls and RecordBatch builds. **B prefetch** is on when using shard files (not in-memory B): the engine keeps two A batches ahead and loads B for the *next* chunk in a background thread while joining the *current* chunk, so "load B" and "join (CPU)" overlap. With `PLEIADES_VERBOSE=1` you’ll see "B prefetch: overlapping load B with join" when it’s active. For runs where B fits in RAM, use **`--keep-b-in-memory`** (or `keep_b_in_memory=True` in the API): B is partitioned once into RAM and no shard I/O happens per chunk. Also set **`TMPDIR`** to an SSD so partition writes and shard reads are fast.

### Can I/O be async?

**Yes.** The Parquet crate (52.x) has an **async API** when the `async` feature is enabled: `ParquetRecordBatchStreamBuilder`, `ParquetRecordBatchStream`, and async file readers (e.g. `tokio::fs::File`). You can move the engine’s I/O to async and run it on a Tokio runtime.

**What it would take:**

| Piece | Change |
|--------|--------|
| **Cargo** | Add `parquet = { ..., features = ["arrow", "snap", "async"] }`, add `tokio` with `rt-multi-thread` and `fs`. |
| **Engine** | Refactor to `async fn`: open files with `tokio::fs::File::open(...).await`, use `ParquetRecordBatchStreamBuilder::new(file).await` and streams instead of `ParquetRecordBatchReaderBuilder` + sync reader. Run CPU-heavy work (partition hash loop, join) in `tokio::task::spawn_blocking` so the async runtime isn’t blocked. |
| **Python boundary** | Keep the public API sync: from the PyO3 entry point run `tokio::runtime::Runtime::new()?.block_on(cross_match_impl_async(...))` (or use `Handle::current().block_on` if you already have a runtime). Python still sees a blocking call; inside Rust, I/O is async. |

**When it helps:**

- **Today:** The engine already overlaps I/O with compute using **threads** (A prefetch, B prefetch). So you already get “async-like” overlap without an async runtime.
- **With async:** You can have many I/O operations in flight (e.g. start several shard reads, await them, then run the join) without one OS thread per operation. That can reduce context switching and scale better if you add more concurrent reads (e.g. more overlapping shard loads). It also fits if you later integrate with other async code (e.g. object storage, async Python).

**Practical takeaway:** Async I/O is possible and the crates support it, but it’s a non-trivial refactor. Try the existing thread-based prefetch and `keep_b_in_memory` first; consider an async path if you need many concurrent I/O ops or an async-native design.

### Async migration: analysis and recommendation

**Current design (no async):**

| Component | Concurrency | Role |
|-----------|-------------|------|
| **A reads** | 1 dedicated thread | Prefetches next batch(es) of catalog A; main thread never blocks on A I/O. |
| **B shard reads** | **Rayon** (N threads) | `load_b_from_shards` uses `par_iter` over shard indices; many shards are read in parallel. |
| **B prefetch** | 1 dedicated thread | Loads B for the *next* chunk while the main thread joins the *current* chunk. |
| **Main thread** | 1 | Index A, receive prefetched B (or sync load first chunk), Rayon-based join, write. |

So I/O is already overlapped with compute (A and B prefetch), and B load is already **parallel across shards** (not one-thread-per-chunk). The only single-threaded I/O is the sequential partition read/write and the A stream.

**What async would change:**

- Replace dedicated threads with async tasks and a Tokio runtime.
- Use `ParquetRecordBatchStreamBuilder` + `tokio::fs::File` for A and for each shard (or keep sync in `spawn_blocking`).
- Run CPU-heavy work (partition hash loop, join) in `spawn_blocking` so the runtime isn’t starved.
- At the PyO3 boundary: `block_on(cross_match_impl_async(...))` so Python still sees a blocking call.

**Why async is not needed today:**

1. **Overlap already achieved:** Threads already overlap “load B” with “join” and A read with the rest of the pipeline. Benchmarks show chunk time ≈ max(load B, join) + index, not load B + join.
2. **B is already parallel:** Shard reads run in parallel via Rayon. Async would not increase B concurrency; it would only change how those reads are scheduled (cooperatively vs. OS threads).
3. **Single sync entry point:** The engine is called only from Python via PyO3, in a blocking way. There is no async caller or async ecosystem to plug into.
4. **Cost is high:** Full engine refactor to `async fn`, `spawn_blocking` for all CPU work, and a runtime at the boundary. Ongoing maintenance of two mental models (sync vs async) if only part of the code is migrated.
5. **Benefit is low:** Gains appear when you have many small, independent I/O operations and want to multiplex them on few threads. Here, B load is already N-way parallel (Rayon), and A is a single stream; the bottleneck is usually disk throughput or CPU (join), not lack of concurrency.

**Recommendation:** **Do not migrate to async** unless you have a concrete need, for example:

- Reading catalogs from **object storage** (S3, GCS, Azure) via async object-store APIs.
- Calling the engine from **async Rust** or **async Python** without blocking a thread.
- A future design with **many more concurrent I/O streams** (e.g. dozens of independent partition reads) where you want to cap OS thread count.

Until then, keep the current thread- and Rayon-based design and tune with the existing knobs (e.g. `keep_b_in_memory`, `TMPDIR`, batch sizes, B prefetch).

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

An optional **wgpu** backend is available for the join phase. It does the following on GPU:

- **Haversine + radius filter + compact output:** One compute shader computes angular separation (arcsec) for each candidate (A, B) pair, filters by radius on the GPU, and writes only matches `(a_ix, b_ix, sep)` via an atomic counter. So the CPU reads back only the match list (not all pair distances), which reduces transfer and avoids a separate CPU filter step.
- **Candidate pairs** are still built on the CPU from the HEALPix index (same as before). Moving HEALPix hashing and pair construction to the GPU would require porting the nested scheme to WGSL and a GPU-friendly index layout; it is left as future work.

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
   export PLEIADES_GPU=wgpu
   uv run python scripts/benchmark_cross_match.py --rust --verbose ...
   ```

If you set `PLEIADES_GPU=wgpu` but the extension was built *without* the wgpu feature, the Python layer prints a one-line warning to stderr and the join runs on CPU. With verbose logging, the engine reports `join (GPU)` when the GPU was used and `join (CPU)` when the CPU path was used.

If `PLEIADES_GPU=wgpu` is set and the extension has the feature but no GPU adapter is available, the engine falls back to the CPU join automatically.

**When the GPU is used:** Because of buffer size limits, the GPU path processes pairs in chunks (each with upload, compute, readback, and sync). For small or medium pair counts this overhead can make the GPU **slower** than the CPU (e.g. 15+ s vs 1–2 s per chunk). So by default the engine uses the GPU only when the number of candidate pairs in a chunk is **≥ 80M** (`PLEIADES_GPU_MIN_PAIRS`, default 80_000_000). Below that, it uses the CPU join even when `PLEIADES_GPU=wgpu` is set. To force GPU for all chunk sizes (for benchmarking), set `PLEIADES_GPU_MIN_PAIRS=0`.
