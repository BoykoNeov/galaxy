//! [`GpuSorter`]: the GPU Morton sort — stage 2 of the GPU-resident LBVH build (DESIGN
//! M4d), the load-bearing risk of the whole port.
//!
//! Given the per-particle 30-bit Morton `codes` (the output of [`crate::GpuMortonBuilder`]
//! or the CPU [`galaxy_solvers::reference_morton`]) it produces, on the GPU (wgpu compute),
//! the permutation `order` that sorts the bodies by `(code, original index)` — the exact
//! ordering [`galaxy_solvers::reference_sort`] defines and the Karras tree-build stage
//! consumes.
//!
//! ## Pure integer — determinism is nearly free, correctness is the risk
//! Unlike the f32 Morton/direct-sum kernels, this stage touches **no floats**: `u32` codes
//! in, a `u32` permutation out. So its result is not merely deterministic on a given device
//! but **bit-for-bit equal to the f64 CPU reference** (the sort of a code array is the same
//! whether the arithmetic around it is f32 or f64). The real hazard is therefore not
//! nondeterminism but **scatter/scan correctness**.
//!
//! ## Single-invocation stable counting sort (correctness made unarguable)
//! The kernel is an LSD radix sort: `NUM_PASSES` passes of an [`RADIX_BITS`]-bit digit, one
//! dispatch per pass, host-side ping-pong between two `(key, payload)` buffer pairs. Each
//! pass runs in a **single invocation** (`@workgroup_size(1)`): it builds a 256-bucket
//! histogram, exclusive-scans it to bucket bases, then scatters every element in ascending
//! source order, incrementing a per-bucket cursor. That serial scatter is **stable by
//! construction** — equal digits keep their input order — so with the payload initialized to
//! `0..n` (index order) the ties break by ascending original index, exactly reproducing
//! `reference_sort`. There are no atomics and no cross-invocation ordering, so the result is
//! trivially deterministic; the point of the single invocation is unarguable *correctness*,
//! not determinism.
//!
//! This is a **reference-grade** sort, not the scale sort: one thread doing all the work is
//! O(passes·N) serial. The named performance refinement (the deferred scale build, alongside
//! keeping state GPU-resident) is a parallel stable scatter — per-tile local ranks plus a
//! scanned global offset, the standard multi-workgroup radix — which reintroduces the scatter
//! ordering this landing deliberately avoids. Land the simple correct thing; name the fast
//! one (same pattern as the deferred 63-bit two-word sort).

use bytemuck::{Pod, Zeroable};

use crate::GpuError;

/// Bits consumed per radix pass. 8 → a 256-bucket histogram (a `vec<u32,256>` workgroup
/// array, 1 KiB, well under the shared-memory floor).
const RADIX_BITS: u32 = 8;
/// Passes to cover a `u32` key. 4 × 8 = 32 bits ≥ the 30-bit Morton range (the top passes
/// are all-zero digits for in-range codes — correct, just cheap).
const NUM_PASSES: u32 = 4;

/// One radix pass as a single-invocation stable counting sort. `src_*` → `dst_*`, digit =
/// `(key >> params.shift) & 0xff`. Histogram (order-independent), exclusive scan, then a
/// stable serial scatter in ascending source index.
const SHADER: &str = r#"
struct Params { n: u32, shift: u32, pad0: u32, pad1: u32 };

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       src_keys: array<u32>;
@group(0) @binding(2) var<storage, read>       src_idx:  array<u32>;
@group(0) @binding(3) var<storage, read_write> dst_keys: array<u32>;
@group(0) @binding(4) var<storage, read_write> dst_idx:  array<u32>;

var<workgroup> hist: array<u32, 256>;
var<workgroup> offs: array<u32, 256>;

@compute @workgroup_size(1)
fn radix_pass() {
    let n = params.n;
    let sh = params.shift;

    // histogram of the current digit (order-independent count)
    for (var b = 0u; b < 256u; b = b + 1u) { hist[b] = 0u; }
    for (var i = 0u; i < n; i = i + 1u) {
        let d = (src_keys[i] >> sh) & 0xffu;
        hist[d] = hist[d] + 1u;
    }
    // exclusive scan → per-bucket base offset (fixed order)
    var running = 0u;
    for (var b = 0u; b < 256u; b = b + 1u) {
        offs[b] = running;
        running = running + hist[b];
    }
    // stable scatter in ascending source index (equal digits keep input order)
    for (var i = 0u; i < n; i = i + 1u) {
        let k = src_keys[i];
        let d = (k >> sh) & 0xffu;
        let p = offs[d];
        offs[d] = p + 1u;
        dst_keys[p] = k;
        dst_idx[p]  = src_idx[i];
    }
}
"#;

/// Result of the GPU Morton sort. `order` is the permutation of `0..n` sorting by
/// `(code, index)`; `sorted_codes[k] == input_codes[order[k]]` is the (non-decreasing)
/// key array, returned for free from the final key buffer so callers can check sortedness
/// without re-gathering.
pub struct GpuSort {
    /// The permutation of `0..n` that sorts bodies by `(code, original index)`.
    pub order: Vec<u32>,
    /// The codes in sorted order (`sorted_codes[k] == codes[order[k]]`), non-decreasing.
    pub sorted_codes: Vec<u32>,
}

/// Uniform block mirroring the WGSL `Params` (16-byte aligned): body count + current
/// radix shift (`0`, `8`, `16`, `24`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    n: u32,
    shift: u32,
    _pad: [u32; 2],
}

/// GPU Morton radix sort. Holds a reusable wgpu compute context built once and driven per
/// [`sort`](Self::sort) call; storage buffers grow lazily with N. Same bring-up idiom as
/// [`crate::GpuMortonBuilder`].
pub struct GpuSorter {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    // One params buffer per pass (shift is a per-pass constant; `n` is rewritten each sort).
    params_bufs: [wgpu::Buffer; NUM_PASSES as usize],
    // Ping-pong storage grown lazily to the largest N seen; A holds the final result (even
    // pass count). The bind group per pass is rebuilt with these.
    key_a: Option<wgpu::Buffer>,
    key_b: Option<wgpu::Buffer>,
    idx_a: Option<wgpu::Buffer>,
    idx_b: Option<wgpu::Buffer>,
    keys_readback: Option<wgpu::Buffer>,
    idx_readback: Option<wgpu::Buffer>,
    // bind_groups[p] drives pass p: even passes A→B, odd passes B→A.
    bind_groups: Option<[wgpu::BindGroup; NUM_PASSES as usize]>,
    capacity: usize,
}

impl GpuSorter {
    /// Bring up a headless wgpu compute device and build the radix-sort pipeline. Returns a
    /// typed [`GpuError`] (never panics) when no adapter is available.
    pub fn new() -> Result<Self, GpuError> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None, // headless
                force_fallback_adapter: false,
            })
            .await
            .map_err(|_| GpuError::NoAdapter)?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("galaxy-gpu-sort-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| GpuError::Device(e.to_string()))?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu-sort-shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu-sort-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage(1, true),  // src_keys (read)
                storage(2, true),  // src_idx  (read)
                storage(3, false), // dst_keys (read-write)
                storage(4, false), // dst_idx  (read-write)
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gpu-sort-pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gpu-sort-radix"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("radix_pass"),
            compilation_options: Default::default(),
            cache: None,
        });

        // One params buffer per pass, holding this pass's constant shift.
        let params_bufs = std::array::from_fn(|_| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu-sort-params"),
                size: std::mem::size_of::<Params>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        Ok(GpuSorter {
            device,
            queue,
            pipeline,
            bgl,
            params_bufs,
            key_a: None,
            key_b: None,
            idx_a: None,
            idx_b: None,
            keys_readback: None,
            idx_readback: None,
            bind_groups: None,
            capacity: 0,
        })
    }

    /// Ensure the ping-pong storage / readback buffers hold at least `n` bodies, rebuilding
    /// the per-pass bind groups when they are (re)allocated. Only grows. Caller guarantees
    /// `n > 0`.
    fn ensure_capacity(&mut self, n: usize) {
        if n <= self.capacity && self.bind_groups.is_some() {
            return;
        }
        let bytes = (n * std::mem::size_of::<u32>()) as u64;
        let storage_buf = |label: &str| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let key_a = storage_buf("gpu-sort-key-a");
        let key_b = storage_buf("gpu-sort-key-b");
        let idx_a = storage_buf("gpu-sort-idx-a");
        let idx_b = storage_buf("gpu-sort-idx-b");
        let readback = |label: &str| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        };
        let keys_readback = readback("gpu-sort-keys-readback");
        let idx_readback = readback("gpu-sort-idx-readback");

        // Pass p reads src, writes dst: even A→B, odd B→A. Result lands in A after an even
        // number of passes.
        let make_bg = |p: usize,
                       src_k: &wgpu::Buffer,
                       src_i: &wgpu::Buffer,
                       dst_k: &wgpu::Buffer,
                       dst_i: &wgpu::Buffer| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gpu-sort-bind-group"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.params_bufs[p].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: src_k.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: src_i.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: dst_k.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: dst_i.as_entire_binding(),
                    },
                ],
            })
        };
        let bind_groups: [wgpu::BindGroup; NUM_PASSES as usize] = std::array::from_fn(|p| {
            if p % 2 == 0 {
                make_bg(p, &key_a, &idx_a, &key_b, &idx_b)
            } else {
                make_bg(p, &key_b, &idx_b, &key_a, &idx_a)
            }
        });

        self.key_a = Some(key_a);
        self.key_b = Some(key_b);
        self.idx_a = Some(idx_a);
        self.idx_b = Some(idx_b);
        self.keys_readback = Some(keys_readback);
        self.idx_readback = Some(idx_readback);
        self.bind_groups = Some(bind_groups);
        self.capacity = n;
    }

    /// Sort the bodies by `(code, original index)` on the GPU, returning the permutation
    /// `order` and the sorted key array. `codes` may be empty (yields empty output).
    pub fn sort(&mut self, codes: &[u32]) -> GpuSort {
        let n = codes.len();
        if n == 0 {
            return GpuSort {
                order: Vec::new(),
                sorted_codes: Vec::new(),
            };
        }
        self.ensure_capacity(n);

        // Seed buffer A: keys = codes, payload = index order 0..n.
        let idx0: Vec<u32> = (0..n as u32).collect();
        let key_a = self.key_a.as_ref().expect("buffers ensured above");
        let idx_a = self.idx_a.as_ref().expect("buffers ensured above");
        self.queue
            .write_buffer(key_a, 0, bytemuck::cast_slice(codes));
        self.queue
            .write_buffer(idx_a, 0, bytemuck::cast_slice(&idx0));

        // Per-pass params (n + this pass's shift).
        for p in 0..NUM_PASSES as usize {
            let params = Params {
                n: n as u32,
                shift: p as u32 * RADIX_BITS,
                _pad: [0; 2],
            };
            self.queue
                .write_buffer(&self.params_bufs[p], 0, bytemuck::bytes_of(&params));
        }

        let bind_groups = self
            .bind_groups
            .as_ref()
            .expect("bind groups ensured above");
        let bytes = std::mem::size_of_val(codes) as u64;

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for bg in bind_groups.iter() {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu-sort-radix-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(1, 1, 1); // single-invocation counting sort
        }
        // Even NUM_PASSES ⇒ result is back in buffer A.
        let keys_rb = self.keys_readback.as_ref().expect("buffers ensured above");
        let idx_rb = self.idx_readback.as_ref().expect("buffers ensured above");
        enc.copy_buffer_to_buffer(key_a, 0, keys_rb, 0, bytes);
        enc.copy_buffer_to_buffer(idx_a, 0, idx_rb, 0, bytes);
        self.queue.submit([enc.finish()]);

        // Map both readbacks, wait once, read. A map failure here is a genuine GPU loss
        // (new() already validated the device), so — like the CPU solvers' asserts on
        // infallible paths — it panics rather than silently corrupting state.
        let keys_slice = keys_rb.slice(..bytes);
        let idx_slice = idx_rb.slice(..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        for slice in [&keys_slice, &idx_slice] {
            let tx = tx.clone();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
        }
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("gpu poll failed");
        for _ in 0..2 {
            rx.recv()
                .expect("map channel closed")
                .expect("gpu buffer map failed");
        }

        let keys_data = keys_slice.get_mapped_range();
        let sorted_codes: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&keys_data)[..n].to_vec();
        let idx_data = idx_slice.get_mapped_range();
        let order: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&idx_data)[..n].to_vec();

        drop(keys_data);
        drop(idx_data);
        keys_rb.unmap();
        idx_rb.unmap();

        GpuSort {
            order,
            sorted_codes,
        }
    }
}
