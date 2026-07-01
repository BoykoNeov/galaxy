//! FEASIBILITY SPIKE (throwaway) — proves headless wgpu works on this box before
//! M3's render stage is built around it. NOT test-first code; it is orientation.
//!
//! What it exercises (the crux of the whole renderer):
//!   1. request a headless adapter (no window / no surface),
//!   2. request a device with FLOAT32_BLENDABLE (additive blend into 32F needs it),
//!   3. additive-blend two overlapping Gaussian splats into an Rgba32Float target,
//!   4. copy the texture to a buffer, map it, read pixels back.
//!
//! Success criteria printed at the end:
//!   - center pixel accumulates BOTH splats and exceeds 1.0 (32F headroom, no clamp),
//!   - a far-corner pixel stays ~0 (splats are local).
//!
//! Run: `cargo run -p galaxy-render --bin spike`

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

const DIM: u32 = 256;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

/// One Gaussian splat: clip-space center + emissive color. Additively blended.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Splat {
    center: [f32; 2],
    color: [f32; 3],
    _pad: f32,
}

const SHADER: &str = r#"
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) local: vec2<f32>,
    @location(1) color: vec3<f32>,
};

// Unit quad (two triangles) in local [-1,1] space; scaled to a splat half-size.
const HALF: f32 = 0.25;

@vertex
fn vs(@location(0) corner: vec2<f32>,
      @location(1) center: vec2<f32>,
      @location(2) color: vec3<f32>) -> VsOut {
    var out: VsOut;
    out.pos = vec4<f32>(center + corner * HALF, 0.0, 1.0);
    out.local = corner;
    out.color = color;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    // Gaussian falloff from the quad center; brightness 4.0 so two overlapping
    // splats push the shared center well past 1.0 (the 32F-headroom probe).
    let g = exp(-6.0 * dot(in.local, in.local)) * 4.0;
    return vec4<f32>(in.color * g, g);
}
"#;

fn main() {
    pollster::block_on(run());
}

async fn run() {
    let instance = wgpu::Instance::default();

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None, // headless: no surface
            force_fallback_adapter: false,
        })
        .await
        .expect("SPIKE FAIL: no wgpu adapter available (headless)");

    let info = adapter.get_info();
    println!(
        "adapter: {} ({:?}, backend {:?})",
        info.name, info.device_type, info.backend
    );

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("spike-device"),
            required_features: wgpu::Features::FLOAT32_BLENDABLE,
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            trace: wgpu::Trace::Off,
        })
        .await
        .expect("SPIKE FAIL: could not get a device with FLOAT32_BLENDABLE");

    // Offscreen HDR accumulation target.
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("hdr-accum"),
        size: wgpu::Extent3d {
            width: DIM,
            height: DIM,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());

    // Geometry: unit quad corners (two triangles), instanced by splats.
    let quad: [[f32; 2]; 6] = [
        [-1.0, -1.0],
        [1.0, -1.0],
        [1.0, 1.0],
        [-1.0, -1.0],
        [1.0, 1.0],
        [-1.0, 1.0],
    ];
    let quad_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("quad"),
        contents: bytemuck::cast_slice(&quad),
        usage: wgpu::BufferUsages::VERTEX,
    });

    // Two splats overlapping near the center so their center pixel accumulates both.
    let splats = [
        Splat {
            center: [-0.05, 0.0],
            color: [1.0, 0.4, 0.2],
            _pad: 0.0,
        },
        Splat {
            center: [0.05, 0.0],
            color: [0.2, 0.5, 1.0],
            _pad: 0.0,
        },
    ];
    let splat_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("splats"),
        contents: bytemuck::cast_slice(&splats),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("splat-shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("splat-pipeline"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[
                wgpu::VertexBufferLayout {
                    array_stride: 8,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2],
                },
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Splat>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![1 => Float32x2, 2 => Float32x3],
                },
            ],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: FORMAT,
                // Additive blend: dst = src*1 + dst*1 (order-independent accumulation).
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    // Readback buffer: DIM rows * DIM * 16 bytes/pixel. bytes_per_row = 4096 (256-aligned).
    let bytes_per_row = DIM * 16;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (bytes_per_row * DIM) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("splat-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_vertex_buffer(0, quad_buf.slice(..));
        pass.set_vertex_buffer(1, splat_buf.slice(..));
        pass.draw(0..6, 0..splats.len() as u32);
    }
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(DIM),
            },
        },
        wgpu::Extent3d {
            width: DIM,
            height: DIM,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([enc.finish()]);

    // Map and read back.
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map failed"));
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll failed");
    let data = slice.get_mapped_range();
    let pixels: &[f32] = bytemuck::cast_slice(&data);

    let px = |x: u32, y: u32| -> [f32; 4] {
        let i = ((y * DIM + x) * 4) as usize;
        [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
    };

    let center = px(DIM / 2, DIM / 2);
    let corner = px(2, 2);
    println!("center pixel RGBA = {center:?}");
    println!("corner pixel RGBA = {corner:?}");

    let headroom_ok = center.iter().take(3).any(|&c| c > 1.0);
    let local_ok = corner.iter().take(3).all(|&c| c < 0.01);
    println!("32F headroom (center > 1.0): {headroom_ok}");
    println!("splats local (corner ~0):   {local_ok}");
    if headroom_ok && local_ok {
        println!("SPIKE PASS: headless wgpu additive 32F render + readback works.");
    } else {
        println!("SPIKE INCONCLUSIVE: check values above.");
    }
}
