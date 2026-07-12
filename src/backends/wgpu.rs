use crate::backend::{Backend, Binding, Dtype, TensorMode};
use crate::pool::{size_class, BufferPool};
use crate::shader::Shader;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

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
    in_flight: Mutex<VecDeque<wgpu::SubmissionIndex>>,
    pool: BufferPool<WgpuBuffer>,
}

/// Only block once this many submissions are unfinished
/// then wait for the oldest
/// keeping the queue fed instead of stalling on every other submit.
const MAX_IN_FLIGHT: usize = 16;

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
            in_flight: Mutex::new(VecDeque::new()),
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

    // Buffers are created at their == power-of-two == size class so slightly
    // different lengths share pool buckets. Tensor tracks the logical size;
    // to_cpu truncates back down to it.
    fn alloc(&self, size_bytes: u64) -> WgpuBuffer {
        let class = size_class(size_bytes);
        if let Some(buf) = self.pool.take(class) {
            return buf;
        }
        Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: class,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }))
    }

    fn alloc_from_cpu<T: bytemuck::Pod>(&self, data: &[T]) -> WgpuBuffer {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        let buf = self.alloc(bytes.len() as u64);
        self.queue.write_buffer(&buf, 0, bytes);
        buf
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

    // TODO: f16 and other types supportations
    fn alloc_dtype(&self, elem_count: usize, dtype: Dtype) -> WgpuBuffer {
        assert_eq!(
            dtype,
            Dtype::F32,
            "[wgpu] backend only supports F32 today, got {dtype:?}"
        );
        self.alloc((elem_count * dtype.elem_size()) as u64)
    }

    fn upload_as(&self, buf: &WgpuBuffer, data: &[f32], dtype: Dtype) {
        assert_eq!(
            dtype,
            Dtype::F32,
            "[wgpu] backend only supports F32 today, got {dtype:?}"
        );
        self.copy_from_cpu(buf, data);
    }

    fn download_as(&self, buf: &WgpuBuffer, dtype: Dtype) -> Vec<f32> {
        assert_eq!(
            dtype,
            Dtype::F32,
            "[wgpu] backend only supports F32 today, got {dtype:?}"
        );
        self.copy_to_cpu(buf)
    }

    fn free_buffer(&self, buf: WgpuBuffer) {
        buf.destroy();
    }

    fn recycle(&self, size_bytes: u64, buf: WgpuBuffer) {
        if let Some(evicted) = self.pool.recycle(size_class(size_bytes), buf) {
            evicted.destroy();
        }
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
        let idx = self.queue.submit(Some(encoder.finish()));

        let oldest = {
            let mut q = self.in_flight.lock().unwrap();
            q.push_back(idx);
            if q.len() > MAX_IN_FLIGHT {
                q.pop_front()
            } else {
                None
            }
        };
        match oldest {
            Some(i) => {
                self.device.poll(wgpu::Maintain::WaitForSubmissionIndex(i));
            }
            None => {
                self.device.poll(wgpu::Maintain::Poll);
            }
        }
    }

    fn synchronize(&self) {
        self.device.poll(wgpu::Maintain::Wait);
        self.in_flight.lock().unwrap().clear();
    }
}
