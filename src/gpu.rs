//! Optional wgpu-based GPU haversine kernel for cross-match distance computation.
//! Enabled with the `wgpu` feature and activated by setting PLEIADES_GPU=wgpu.
//!
//! Computes angular separation in arcsec for N (ra_a, dec_a, ra_b, dec_b) pairs in a single
//! compute dispatch. Used when the engine has collected candidate pairs from the HEALPix index.

use std::sync::OnceLock;

use pollster::block_on;
use wgpu::util::DeviceExt;

const WGSL_SOURCE: &str = include_str!("gpu.wgsl");
const WGSL_COMPACT_SOURCE: &str = include_str!("gpu_compact.wgsl");
const WORKGROUP_SIZE: u32 = 256;
/// Many adapters limit buffer binding size to 128 MiB.
const MAX_BUFFER_BYTES: u64 = 128 * 1024 * 1024;
/// Coords = 4 f32 per pair = 16 bytes (legacy path).
const MAX_PAIRS_PER_CHUNK: usize = (MAX_BUFFER_BYTES / 16) as usize;
/// Compact path: 24 bytes per pair (4 f32 + 2 u32).
const MAX_PAIRS_PER_CHUNK_COMPACT: usize = (MAX_BUFFER_BYTES / 24) as usize;

static GPU_CONTEXT: OnceLock<Option<GpuContext>> = OnceLock::new();

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PairRaw {
    ra_a: f32,
    dec_a: f32,
    ra_b: f32,
    dec_b: f32,
    a_ix: u32,
    b_ix: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MatchRaw {
    a_ix: u32,
    b_ix: u32,
    sep: f32,
}

struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline_compact: wgpu::ComputePipeline,
    bind_group_layout_compact: wgpu::BindGroupLayout,
}

fn init_gpu() -> Option<GpuContext> {
    block_on(async {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await?;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("pleiades-gpu"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .ok()?;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("haversine.wgsl"),
            source: wgpu::ShaderSource::Wgsl(WGSL_SOURCE.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("haversine-bind-group-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("haversine-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("haversine-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        // Compact pipeline: haversine + radius filter + atomic append of matches.
        let shader_compact = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("haversine-compact.wgsl"),
            source: wgpu::ShaderSource::Wgsl(WGSL_COMPACT_SOURCE.into()),
        });
        let bind_group_layout_compact = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("haversine-compact-bind-group-layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: std::num::NonZeroU64::new(4),
                        },
                        count: None,
                    },
                ],
            },
        );
        let pipeline_layout_compact = device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("haversine-compact-pipeline-layout"),
                bind_group_layouts: &[&bind_group_layout_compact],
                push_constant_ranges: &[],
            },
        );
        let pipeline_compact = device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("haversine-compact-pipeline"),
                layout: Some(&pipeline_layout_compact),
                module: &shader_compact,
                entry_point: Some("main"),
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
        );

        Some(GpuContext {
            device,
            queue,
            pipeline,
            bind_group_layout,
            pipeline_compact,
            bind_group_layout_compact,
        })
    })
}

/// Returns true if the wgpu backend is available and initialized.
pub fn gpu_available() -> bool {
    GPU_CONTEXT
        .get_or_init(init_gpu)
        .is_some()
}

/// Run the haversine compute pipeline for one chunk of pairs. Caller ensures chunk_len <= MAX_PAIRS_PER_CHUNK.
fn haversine_chunk(
    ctx: &GpuContext,
    ra_a: &[f32],
    dec_a: &[f32],
    ra_b: &[f32],
    dec_b: &[f32],
    chunk_len: usize,
) -> Result<Vec<f32>, String> {
    if chunk_len == 0 {
        return Ok(Vec::new());
    }
    let byte_len = chunk_len * std::mem::size_of::<f32>();
    let mut coords: Vec<f32> = Vec::with_capacity(chunk_len * 4);
    for i in 0..chunk_len {
        coords.push(ra_a[i]);
        coords.push(dec_a[i]);
        coords.push(ra_b[i]);
        coords.push(dec_b[i]);
    }
    let coords_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("coords"),
        contents: bytemuck::cast_slice(&coords),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out"),
        size: byte_len as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("haversine-bind-group"),
        layout: &ctx.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: coords_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("haversine-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("haversine-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (chunk_len as u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: byte_len as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, byte_len as u64);
    ctx.queue.submit(std::iter::once(encoder.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    ctx.device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|_| "map_async callback failed".to_string())?
        .map_err(|e| format!("buffer map failed: {}", e))?;
    let mapped = slice.get_mapped_range();
    let out_slice: &[f32] = bytemuck::cast_slice(&mapped);
    let result = out_slice.to_vec();
    drop(mapped);
    staging.unmap();
    Ok(result)
}

/// One chunk for compact path: compute haversine, filter by radius, write only matches via atomic.
fn haversine_chunk_compact(
    ctx: &GpuContext,
    pairs_raw: &[PairRaw],
    radius_arcsec: f32,
    chunk_len: usize,
) -> Result<Vec<(u32, u32, f32)>, String> {
    if chunk_len == 0 {
        return Ok(Vec::new());
    }
    let pairs_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pairs-compact"),
        contents: bytemuck::cast_slice(pairs_raw),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let params = [radius_arcsec];
    let params_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::cast_slice(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let count_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("count"),
        contents: bytemuck::cast_slice(&[0u32]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let matches_size = (chunk_len * std::mem::size_of::<MatchRaw>()) as u64;
    let matches_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("matches"),
        size: matches_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("haversine-compact-bind-group"),
        layout: &ctx.bind_group_layout_compact,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: pairs_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: count_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: matches_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("haversine-compact-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("haversine-compact-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.pipeline_compact);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (chunk_len as u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    let staging_count = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging-count"),
        size: 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&count_buf, 0, &staging_count, 0, 4);
    ctx.queue.submit(std::iter::once(encoder.finish()));
    let slice_count = staging_count.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice_count.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    ctx.device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|_| "map_async count failed".to_string())?
        .map_err(|e| format!("buffer map failed: {}", e))?;
    let mapped_count = slice_count.get_mapped_range();
    let count_val = *bytemuck::from_bytes::<u32>(mapped_count.as_ref());
    drop(mapped_count);
    staging_count.unmap();

    if count_val == 0 {
        return Ok(Vec::new());
    }
    let count_val = count_val as usize;
    let matches_read_size = count_val * std::mem::size_of::<MatchRaw>();
    let staging_matches = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging-matches"),
        size: matches_read_size as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder2 = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("copy-matches"),
        });
    encoder2.copy_buffer_to_buffer(
        &matches_buf,
        0,
        &staging_matches,
        0,
        matches_read_size as u64,
    );
    ctx.queue.submit(std::iter::once(encoder2.finish()));
    let slice_m = staging_matches.slice(..);
    let (tx2, rx2) = std::sync::mpsc::channel();
    slice_m.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx2.send(r);
    });
    ctx.device.poll(wgpu::Maintain::Wait);
    rx2.recv()
        .map_err(|_| "map_async matches failed".to_string())?
        .map_err(|e| format!("buffer map failed: {}", e))?;
    let mapped_m = slice_m.get_mapped_range();
    let matches_slice: &[MatchRaw] = bytemuck::cast_slice(&mapped_m);
    let result: Vec<(u32, u32, f32)> = matches_slice[..count_val]
        .iter()
        .map(|m| (m.a_ix, m.b_ix, m.sep))
        .collect();
    drop(mapped_m);
    staging_matches.unmap();
    Ok(result)
}

/// Compute haversine, filter by radius on GPU, return only (a_ix, b_ix, sep) for matches.
/// Reduces readback vs haversine_pairs_gpu. a_ix and b_ix must fit in u32.
pub fn haversine_pairs_gpu_compact(
    ra_a: &[f32],
    dec_a: &[f32],
    ra_b: &[f32],
    dec_b: &[f32],
    a_ix: &[u32],
    b_ix: &[u32],
    radius_arcsec: f32,
) -> Result<Vec<(u32, u32, f32)>, String> {
    let n = ra_a.len();
    if dec_a.len() != n
        || ra_b.len() != n
        || dec_b.len() != n
        || a_ix.len() != n
        || b_ix.len() != n
    {
        return Err("haversine_pairs_gpu_compact: slice length mismatch".to_string());
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    let ctx = GPU_CONTEXT.get_or_init(init_gpu).as_ref().ok_or_else(|| {
        "wgpu not available (no adapter or device)".to_string()
    })?;
    let mut result = Vec::new();
    let mut start = 0;
    while start < n {
        let chunk_len = (n - start).min(MAX_PAIRS_PER_CHUNK_COMPACT);
        let mut pairs_raw: Vec<PairRaw> = Vec::with_capacity(chunk_len);
        for i in 0..chunk_len {
            let j = start + i;
            pairs_raw.push(PairRaw {
                ra_a: ra_a[j],
                dec_a: dec_a[j],
                ra_b: ra_b[j],
                dec_b: dec_b[j],
                a_ix: a_ix[j],
                b_ix: b_ix[j],
            });
        }
        let chunk = haversine_chunk_compact(ctx, &pairs_raw, radius_arcsec, chunk_len)?;
        result.extend(chunk);
        start += chunk_len;
    }
    Ok(result)
}

/// Compute haversine angular separation in arcsec for each (ra_a, dec_a, ra_b, dec_b) pair.
/// Inputs are in degrees (f32). Returns distances in arcsec (f32).
/// All four slices must have the same length N. Processes in chunks to stay under adapter buffer limits (e.g. 256 MiB).
pub fn haversine_pairs_gpu(
    ra_a: &[f32],
    dec_a: &[f32],
    ra_b: &[f32],
    dec_b: &[f32],
) -> Result<Vec<f32>, String> {
    let n = ra_a.len();
    if dec_a.len() != n || ra_b.len() != n || dec_b.len() != n {
        return Err("haversine_pairs_gpu: slice length mismatch".to_string());
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    let ctx = GPU_CONTEXT.get_or_init(init_gpu).as_ref().ok_or_else(|| {
        "wgpu not available (no adapter or device)".to_string()
    })?;
    let mut result = Vec::with_capacity(n);
    let mut start = 0;
    while start < n {
        let chunk_len = (n - start).min(MAX_PAIRS_PER_CHUNK);
        let chunk = haversine_chunk(
            ctx,
            &ra_a[start..][..chunk_len],
            &dec_a[start..][..chunk_len],
            &ra_b[start..][..chunk_len],
            &dec_b[start..][..chunk_len],
            chunk_len,
        )?;
        result.extend(chunk);
        start += chunk_len;
    }
    Ok(result)
}
