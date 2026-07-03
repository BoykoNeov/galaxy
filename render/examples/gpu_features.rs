//! Diagnostic: print the headless adapter and the texture-sampling features the
//! volumetric path cares about (M7e risk note: FLOAT32_FILTERABLE is distinct
//! from FLOAT32_BLENDABLE and not universal). Run:
//! `cargo run -p galaxy-render --example gpu_features`

fn main() {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("no wgpu adapter");

    let info = adapter.get_info();
    println!("adapter: {} ({:?}, {:?})", info.name, info.backend, info.device_type);
    let f = adapter.features();
    for feat in [
        wgpu::Features::FLOAT32_BLENDABLE,
        wgpu::Features::FLOAT32_FILTERABLE,
    ] {
        println!("{feat:?}: {}", f.contains(feat));
    }
    let l = adapter.limits();
    println!(
        "max_texture_dimension_3d: {}, max_storage_buffers_per_shader_stage: {}",
        l.max_texture_dimension_3d, l.max_storage_buffers_per_shader_stage
    );
}
