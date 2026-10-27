use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::{mpsc, Mutex};

use crate::backend::dtype::DataKind;
use crate::backend::shader::{BindingRole, Shader, ShaderBinary, Signature, Workgroups};
use crate::backend::{Access, Buffer, DeviceInfo, Dispatch, Node, Storage, Topology, COPY_ALIGN};

// Every opened `WgpuBackend` gets its own value; its buffers and nodes carry
// it, so another device's are rejected.
static NEXT_OWNER: AtomicU64 = AtomicU64::new(0);

// wgpu buffers have no address; each buffer takes its own range of these
// instead, so no two buffers share one.
static NEXT_ADDRESS: AtomicU64 = AtomicU64::new(0);

// Every buffer is at least this big: a binding can't be 0 bytes.
const MIN_BYTES: u64 = COPY_ALIGN;

// Vulkan, Metal and DX12; `WGPU_BACKEND` can narrow it. GL's compute path
// doesn't work.
fn backends() -> wgpu::Backends {
    wgpu::Backends::PRIMARY.with_env()
}

pub struct WgpuBuffer {
    buf: wgpu::Buffer,
    bytes: u64,
    address: u64,
    owner: u64,
}

impl Buffer for WgpuBuffer {
    fn size_bytes(&self) -> u64 {
        self.bytes
    }

    fn address(&self) -> u64 {
        self.address
    }
}

pub struct WgpuNode {
    signature: &'static Signature,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    meta: wgpu::Buffer,
    workgroups: Workgroups,
    owner: u64,
}

impl Node for WgpuNode {
    // WebGPU's default limit, which every adapter allows.
    const MAX_WORKGROUPS_PER_DIM: u32 = 65535;

    fn signature(&self) -> &'static Signature {
        self.signature
    }

    fn workgroups(&self) -> Workgroups {
        self.workgroups
    }
}

#[derive(Clone, Debug)]
pub struct WgpuInfo {
    adapter: wgpu::Adapter,
    index: usize,
    name: String,
    backend: wgpu::Backend,
}

impl DeviceInfo for WgpuInfo {
    fn label(&self) -> String {
        format!("{} (wgpu:{}, {:?})", self.name, self.index, self.backend)
    }

    fn total_memory_bytes(&self) -> u64 {
        // TODO: wgpu doesn't tell a device's memory; SysTopology needs it.
        0
    }

    // Buffers are plain bytes; which kinds a shader reads is its own matter.
    // A shader that uses WGSL's `f16` fails to build where the adapter lacks
    // `SHADER_F16`.
    fn supports_kind(&self, _kind: DataKind) -> bool {
        true
    }
}

// Commands recorded but not submitted yet.
#[derive(Default)]
struct Pending {
    encoder: Option<wgpu::CommandEncoder>,
    commands: u32,
}

pub struct WgpuBackend {
    // Submit after this many queued runs, copies and clears; 0 submits only
    // at `synchronize`, downloads, uploads and `update_meta`. 1 when opened.
    // TODO: one compute pass with several dispatches dies with `Parent
    // device is lost` on some runs; above 1 is for finding out why.
    pub submit_every: AtomicU8,
    info: WgpuInfo,
    owner: u64,
    device: wgpu::Device,
    queue: wgpu::Queue,
    limits: wgpu::Limits,
    pending: Mutex<Pending>,
}

impl WgpuBackend {
    fn check_owner(&self, owner: u64, what: &str) -> Result<(), String> {
        if owner == self.owner {
            Ok(())
        } else {
            Err(format!("wgpu: {what} belongs to another device"))
        }
    }

    // Records one command, then submits if `submit_every` is reached.
    fn record(&self, command: impl FnOnce(&mut wgpu::CommandEncoder)) {
        let mut pending = self.pending.lock().unwrap();
        let encoder = pending
            .encoder
            .get_or_insert_with(|| self.device.create_command_encoder(&Default::default()));
        command(encoder);
        pending.commands += 1;
        let every = self.submit_every.load(Ordering::Relaxed);
        if every != 0 && pending.commands >= u32::from(every) {
            self.submit(&mut pending);
        }
    }

    // Callers hold the lock across the submit and what must follow it, so
    // no other thread's command lands in between.
    fn submit(&self, pending: &mut Pending) {
        if let Some(encoder) = pending.encoder.take() {
            self.queue.submit(Some(encoder.finish()));
            self.poll(wgpu::PollType::Poll);
        }
        pending.commands = 0;
    }

    // For the contract's methods that can't fail: a poll error means the
    // device is gone.
    fn poll(&self, ty: wgpu::PollType) {
        if let Err(e) = self.device.poll(ty) {
            panic!("wgpu: poll: {e}");
        }
    }
}

impl Topology for WgpuBackend {
    type Info = WgpuInfo;

    const NAME: &'static str = "wgpu";

    fn choosable_devices() -> Vec<WgpuInfo> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = backends();
        let instance = wgpu::Instance::new(desc);
        let adapters = pollster::block_on(instance.enumerate_adapters(backends()));
        adapters
            .into_iter()
            .enumerate()
            .map(|(index, adapter)| {
                let info = adapter.get_info();
                WgpuInfo {
                    adapter,
                    index,
                    name: info.name,
                    backend: info.backend,
                }
            })
            .collect()
    }

    fn attach(info: WgpuInfo) -> Result<Self, String> {
        let limits = info.adapter.limits();
        let (device, queue) =
            pollster::block_on(info.adapter.request_device(&wgpu::DeviceDescriptor {
                required_features: info.adapter.features() & wgpu::Features::SHADER_F16,
                required_limits: limits.clone(),
                ..Default::default()
            }))
            .map_err(|e| format!("wgpu: open the device: {e}"))?;
        Ok(Self {
            submit_every: AtomicU8::new(1),
            info,
            owner: NEXT_OWNER.fetch_add(1, Ordering::Relaxed),
            device,
            queue,
            limits,
            pending: Mutex::new(Pending::default()),
        })
    }

    fn info(&self) -> &WgpuInfo {
        &self.info
    }
}

impl Storage for WgpuBackend {
    type Buffer = WgpuBuffer;

    fn alloc_raw(&self, bytes: u64) -> Result<WgpuBuffer, String> {
        let bytes = bytes.max(MIN_BYTES).next_multiple_of(COPY_ALIGN);
        if bytes > self.limits.max_buffer_size {
            return Err(format!(
                "wgpu: {bytes} bytes asked, over this device's buffer limit of {}",
                self.limits.max_buffer_size
            ));
        }
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if let Some(e) = pollster::block_on(scope.pop()) {
            return Err(format!("wgpu: allocate: {e}"));
        }
        Ok(WgpuBuffer {
            buf,
            bytes,
            address: NEXT_ADDRESS.fetch_add(bytes, Ordering::Relaxed),
            owner: self.owner,
        })
    }

    // `write_buffer` lands before every command of the next submit, so the
    // commands recorded before it are submitted first.
    unsafe fn upload_raw(&self, buf: &mut WgpuBuffer, offset: u64, data: &[u8]) {
        let mut pending = self.pending.lock().unwrap();
        self.submit(&mut pending);
        self.queue.write_buffer(&buf.buf, offset, data);
    }

    unsafe fn download_raw(&self, buf: &WgpuBuffer, offset: u64, bytes: u64) -> Vec<u8> {
        if bytes == 0 {
            self.synchronize();
            return Vec::new();
        }
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        {
            let mut pending = self.pending.lock().unwrap();
            let encoder = pending
                .encoder
                .get_or_insert_with(|| self.device.create_command_encoder(&Default::default()));
            encoder.copy_buffer_to_buffer(&buf.buf, offset, &staging, 0, bytes);
            self.submit(&mut pending);
        }
        let slice = staging.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |mapped| {
            let _ = tx.send(mapped);
        });
        self.poll(wgpu::PollType::wait_indefinitely());
        match rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => panic!("wgpu: download: {e}"),
            Err(_) => panic!("wgpu: download: the buffer was never mapped"),
        }
        let out = slice
            .get_mapped_range()
            .unwrap_or_else(|e| panic!("wgpu: download: {e}"))
            .to_vec();
        staging.unmap();
        out
    }

    unsafe fn copy_raw(
        &self,
        src: &WgpuBuffer,
        src_offset: u64,
        dst: &mut WgpuBuffer,
        dst_offset: u64,
        bytes: u64,
    ) -> Result<(), String> {
        self.check_owner(src.owner, "copy_raw: the source")?;
        self.check_owner(dst.owner, "copy_raw: the destination")?;
        if bytes > 0 {
            self.record(|encoder| {
                encoder.copy_buffer_to_buffer(&src.buf, src_offset, &dst.buf, dst_offset, bytes);
            });
        }
        Ok(())
    }

    unsafe fn clear_raw(&self, buf: &mut WgpuBuffer) -> Result<(), String> {
        self.check_owner(buf.owner, "clear_raw: the buffer")?;
        self.record(|encoder| encoder.clear_buffer(&buf.buf, 0, None));
        Ok(())
    }
}

impl Dispatch for WgpuBackend {
    type Node = WgpuNode;

    fn accepts(binary: &ShaderBinary) -> bool {
        matches!(
            binary,
            ShaderBinary::Wgsl(_) | ShaderBinary::Glsl(_) | ShaderBinary::Spirv(_)
        )
    }

    fn build_node(
        &self,
        shader: &Shader,
        meta: &[u32],
        workgroups: Workgroups,
    ) -> Result<WgpuNode, String> {
        let signature = shader.signature();
        let name = signature.name;
        let source = match shader.binary() {
            ShaderBinary::Wgsl(wgsl) => wgpu::ShaderSource::Wgsl(wgsl.0.clone()),
            ShaderBinary::Glsl(glsl) => wgpu::ShaderSource::Glsl {
                shader: glsl.0.clone(),
                stage: wgpu::naga::ShaderStage::Compute,
                defines: &[],
            },
            ShaderBinary::Spirv(spirv) => wgpu::ShaderSource::SpirV(spirv.0.clone()),
            _ => {
                return Err(format!(
                    "wgpu: shader `{name}`: neither WGSL, GLSL nor SPIR-V"
                ))
            }
        };

        // Slot 0 is the meta; the layout's roles give the rest, so a shader
        // whose bindings disagree with its signature fails here.
        let entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let entries: Vec<_> =
            std::iter::once(entry(0, true))
                .chain(
                    signature.layout.iter().zip(1..).map(|(role, binding)| {
                        entry(binding, matches!(role, BindingRole::Input(_)))
                    }),
                )
                .collect();

        // Validation errors are the shader's or the signature's; internal
        // ones are the driver failing to compile it.
        let internal = self.device.push_error_scope(wgpu::ErrorFilter::Internal);
        let validation = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source,
            });
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(name),
                entries: &entries,
            });
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(name),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("entry"),
                compilation_options: Default::default(),
                cache: None,
            });
        for scope in [validation, internal] {
            if let Some(e) = pollster::block_on(scope.pop()) {
                return Err(format!("wgpu: shader `{name}`: {e}"));
            }
        }

        // At least one word: every shader binds a meta. A new buffer is
        // zeroed, and nothing ran on it, so the write needs no ordering.
        let meta_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(name),
            size: (meta.len() as u64 * 4).max(MIN_BYTES),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !meta.is_empty() {
            self.queue
                .write_buffer(&meta_buf, 0, bytemuck::cast_slice(meta));
        }
        Ok(WgpuNode {
            signature,
            pipeline,
            layout,
            meta: meta_buf,
            workgroups,
            owner: self.owner,
        })
    }

    // Like `upload_raw`: runs recorded before it are submitted first, so
    // they read the old words.
    fn update_meta(&self, node: &mut WgpuNode, meta: &[u32]) {
        let mut pending = self.pending.lock().unwrap();
        self.submit(&mut pending);
        self.queue
            .write_buffer(&node.meta, 0, bytemuck::cast_slice(meta));
    }

    unsafe fn execute_raw(
        &self,
        node: &WgpuNode,
        bindings: &mut [Access<'_, WgpuBuffer>],
    ) -> Result<(), String> {
        let name = node.signature.name;
        self.check_owner(node.owner, &format!("shader `{name}`: the node"))?;
        let limit = self.limits.max_storage_buffer_binding_size;
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: node.meta.as_entire_binding(),
        }];
        for (binding, slot) in bindings.iter().zip(1..) {
            let buf = match binding {
                Access::Read(buf) => &**buf,
                Access::Write(buf) => &**buf,
            };
            self.check_owner(buf.owner, &format!("shader `{name}`: a bound buffer"))?;
            if buf.bytes > limit {
                return Err(format!(
                    "wgpu: shader `{name}`: slot {slot} is {} bytes, over this device's binding limit of {limit}",
                    buf.bytes
                ));
            }
            entries.push(wgpu::BindGroupEntry {
                binding: slot,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buf.buf,
                    offset: 0,
                    size: NonZeroU64::new(buf.bytes),
                }),
            });
        }
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(name),
            layout: &node.layout,
            entries: &entries,
        });
        let wg = node.workgroups;
        // One compute pass per dispatch.
        self.record(|encoder| {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&node.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(wg.x, wg.y, wg.z);
        });
        Ok(())
    }

    fn synchronize(&self) {
        {
            let mut pending = self.pending.lock().unwrap();
            self.submit(&mut pending);
        }
        self.poll(wgpu::PollType::wait_indefinitely());
    }
}

#[cfg(test)]
#[path = "../tests/wgpu.rs"]
mod tests;
