//! Optional wgpu-based GPU haversine kernel for cross-match distance computation.
//! Enabled with the `wgpu` feature and activated by setting ASTROJOIN_GPU=wgpu.
//!
//! Computes angular separation in arcsec for N (ra_a, dec_a, ra_b, dec_b) pairs in a single
//! compute dispatch. Used when the engine has collected candidate pairs from the HEALPix index.

use std::sync::OnceLock;

use pollster::block_on;
use wgpu::util::DeviceExt;

const WGSL_SOURCE: &str = include_str!("gpu.wgsl");
const WORKGROUP_SIZE: u32 = 256;
/// Many adapters limit buffer binding size to 128 MiB. Coords = 4 f32 per pair = 16 bytes per pair.
const MAX_BUFFER_BYTES: u64 = 128 * 1024 * 1024;
const MAX_PAIRS_PER_CHUNK: usize = (MAX_BUFFER_BYTES / 16) as usize;

static GPU_CONTEXT: OnceLock<Option<GpuContext>> = OnceLock::new();

struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
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
                    label: Some("astrojoin-gpu"),
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
        Some(GpuContext {
            device,
            queue,
            pipeline,
            bind_group_layout,
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
