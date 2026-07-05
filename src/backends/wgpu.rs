use crate::backend::{Backend, Binding, TensorMode};
use crate::pool::BufferPool;
use crate::shader::Shader;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use wgpu::util::DeviceExt;

pub type WgpuBuffer = Arc<wgpu::Buffer>;

#[derive(Clone)]
pub struct WgpuNode {
    pipeline: Arc<wgpu::ComputePipeline>,
    bind_group: Arc<wgpu::BindGroup>,
    #[allow(dead_code)]
    buffers: Vec<WgpuBuffer>,
    workgroups: [u32; 3],
}

// ========================
//       WgpuBackend
// ========================

pub struct WgpuBackend {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline_cache: Mutex<HashMap<usize, (Arc<wgpu::BindGroupLayout>, Arc<wgpu::ComputePipeline>)>>,
    submit_count: AtomicU64,
    pool: BufferPool<WgpuBuffer>,
}

impl WgpuBackend {
    pub async fn new() -> Self {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("[wgpu] no adapter");
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_limits: adapter.limits(),
                    ..Default::default()
                },
                None,
            )
            .await
            .expect("[wgpu] device creation failed");
        Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            pipeline_cache: Mutex::new(HashMap::new()),
            submit_count: AtomicU64::new(0),
            pool: BufferPool::new(),
        }
    }
}

impl Backend for WgpuBackend {
    type Buffer = WgpuBuffer;
    type Node = WgpuNode;

    fn name(&self) -> &'static str {
        "wgpu"
    }

    fn alloc(&self, size_bytes: u64) -> WgpuBuffer {
        if let Some(buf) = self.pool.take(size_bytes) {
            return buf;
        }
        Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }))
    }

    fn alloc_from_cpu<T: bytemuck::Pod>(&self, data: &[T]) -> WgpuBuffer {
        let bytes = bytemuck::cast_slice(data);
        if let Some(buf) = self.pool.take(bytes.len() as u64) {
            self.queue.write_buffer(&buf, 0, bytes);
            return buf;
        }
        Arc::new(
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytes,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                }),
        )
    }

    fn copy_from_cpu<T: bytemuck::Pod>(&self, buf: &WgpuBuffer, data: &[T]) {
        self.queue.write_buffer(buf, 0, bytemuck::cast_slice(data));
    }

    fn copy_to_cpu<T: bytemuck::Pod + Default + Clone>(&self, buf: &WgpuBuffer) -> Vec<T> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: buf.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, buf.size());
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        self.device.poll(wgpu::Maintain::Wait);
        pollster::block_on(async { rx.receive().await.unwrap().unwrap() });

        let mapped = slice.get_mapped_range();
        let result = bytemuck::cast_slice::<_, T>(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        result
    }

    fn free_buffer(&self, buf: WgpuBuffer) {
        buf.destroy();
    }

    fn recycle(&self, size_bytes: u64, buf: WgpuBuffer) {
        self.pool.recycle(size_bytes, buf);
    }

    fn is_sole_owner(buf: &WgpuBuffer) -> bool {
        Arc::strong_count(buf) == 1
    }

    fn build_node(
        &self,
        shader: &'static Shader,
        bindings: &[Binding<WgpuBuffer>],
        workgroups: [u32; 3],
    ) -> WgpuNode {
        let src = shader
            .wgpu
            .unwrap_or_else(|| panic!("[wgpu] shader `{}` has no wgpu impl", shader.name));
        let layout = shader.layout;
        let key = shader as *const Shader as usize;

        let (bgl, pipeline) = {
            let mut cache = self.pipeline_cache.lock().unwrap();

            if let Some((l, p)) = cache.get(&key) {
                (l.clone(), p.clone())
            } else {
                let module = self
                    .device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some(shader.name),
                        source: wgpu::ShaderSource::Wgsl(src.into()),
                    });

                let layout_entries: Vec<_> = layout
                    .iter()
                    .enumerate()
                    .map(|(i, mode)| wgpu::BindGroupLayoutEntry {
                        binding: i as u32,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage {
                                read_only: matches!(mode, TensorMode::Input | TensorMode::Meta),
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    })
                    .collect();

                let bgl = Arc::new(self.device.create_bind_group_layout(
                    &wgpu::BindGroupLayoutDescriptor {
                        label: None,
                        entries: &layout_entries,
                    },
                ));

                let pl = self
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[&bgl],
                        push_constant_ranges: &[],
                    });

                let pipeline = Arc::new(self.device.create_compute_pipeline(
                    &wgpu::ComputePipelineDescriptor {
                        label: Some(shader.name),
                        layout: Some(&pl),
                        module: &module,
                        entry_point: "main",
                    },
                ));

                cache.insert(key, (bgl.clone(), pipeline.clone()));
                (bgl, pipeline)
            }
        };

        let bind_entries: Vec<_> = bindings
            .iter()
            .map(|b| wgpu::BindGroupEntry {
                binding: b.slot,
                resource: b.buffer.as_entire_binding(),
            })
            .collect();

        let bind_group = Arc::new(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &bind_entries,
        }));

        WgpuNode {
            pipeline,
            bind_group,
            buffers: bindings.iter().map(|b| b.buffer.clone()).collect(),
            workgroups,
        }
    }

    fn execute(&self, nodes: &[WgpuNode]) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            for node in nodes {
                cpass.set_pipeline(&node.pipeline);
                cpass.set_bind_group(0, &node.bind_group, &[]);
                cpass.dispatch_workgroups(
                    node.workgroups[0],
                    node.workgroups[1],
                    node.workgroups[2],
                );
            }
        }
        self.queue.submit(Some(encoder.finish()));

        const POLL_INTERVAL: u64 = 2;
        let n = self.submit_count.fetch_add(1, Ordering::Relaxed) + 1;
        if n % POLL_INTERVAL == 0 {
            self.device.poll(wgpu::Maintain::Wait);
        } else {
            self.device.poll(wgpu::Maintain::Poll);
        }
    }

    fn synchronize(&self) {
        self.device.poll(wgpu::Maintain::Wait);
    }
}
