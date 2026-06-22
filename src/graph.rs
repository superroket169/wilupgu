use crate::context::WgpuContext;
use crate::tensor::Tensor;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TensorMode {
    Input,  // (Read-Only)
    Output, // (Read-Write)
    Meta,   // (Read-Only)
}

pub struct ShaderDef {
    pub name: String,
    pub source: String,
    pub expected_layout: Vec<TensorMode>,
}

impl ShaderDef {
    pub fn new(name: &str, source: &str, expected_layout: Vec<TensorMode>) -> Self {
        Self {
            name: name.to_string(),
            source: source.to_string(),
            expected_layout,
        }
    }
}

pub struct TensorBind<'a> {
    pub binding: u32,
    pub tensor: &'a Tensor,
    pub mode: TensorMode,
}

pub struct ComputeNode {
    pub name: String,
    pub pipeline: Arc<wgpu::ComputePipeline>,
    pub bind_group: Arc<wgpu::BindGroup>,
    pub workgroups: [u32; 3],
}

pub struct ComputeGraph {
    ctx: Arc<WgpuContext>,
    nodes: Vec<ComputeNode>,
}

impl ComputeGraph {
    pub fn new(ctx: Arc<WgpuContext>) -> Self {
        Self {
            ctx,
            nodes: Vec::new(),
        }
    }

    pub fn add_node(&mut self, shader: &ShaderDef, bindings: &[TensorBind], workgroups: [u32; 3]) {
        // ---------------- VALIDATION ------------------
        if bindings.len() != shader.expected_layout.len() {
            panic!(
                "[ERROR] Layout mismatch error: This shader [{}]: Wants {} tensor, but theres {} tensor",
                shader.name, shader.expected_layout.len(), bindings.len()
            );
        }

        for bind in bindings {
            let expected_mode = shader.expected_layout[bind.binding as usize];
            if bind.mode != expected_mode {
                panic!(
                    "[ERROR] Layout mismatch error: This shader [{}] wants this binding {} to this {:?} , but theres {:?}",
                    shader.name, bind.binding, expected_mode, bind.mode
                );
            }
        }
        // ---------------------------------------------------

        let shader_module = self
            .ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("{}_Shader", shader.name)),
                source: wgpu::ShaderSource::Wgsl(shader.source.as_str().into()),
            });

        let mut layout_entries = Vec::new();
        let mut bind_entries = Vec::new();

        for bind in bindings {
            let read_only = match bind.mode {
                TensorMode::Input | TensorMode::Meta => true,
                TensorMode::Output => false,
            };

            layout_entries.push(wgpu::BindGroupLayoutEntry {
                binding: bind.binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if read_only {
                        wgpu::BufferBindingType::Storage { read_only: true }
                    } else {
                        wgpu::BufferBindingType::Storage { read_only: false }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });

            bind_entries.push(wgpu::BindGroupEntry {
                binding: bind.binding,
                resource: bind.tensor.buffer.as_entire_binding(),
            });
        }

        let bind_group_layout =
            self.ctx
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{}_Layout", shader.name)),
                    entries: &layout_entries,
                });

        let bind_group = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("{}_BindGroup", shader.name)),
                layout: &bind_group_layout,
                entries: &bind_entries,
            });

        let pipeline_layout =
            self.ctx
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(&format!("{}_PipelineLayout", shader.name)),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = self
            .ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("{}_Pipeline", shader.name)),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: "main",
            });

        self.nodes.push(ComputeNode {
            name: shader.name.clone(),
            pipeline: Arc::new(pipeline),
            bind_group: Arc::new(bind_group),
            workgroups,
        });
    }

    pub fn execute(&self) {
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Wilupgu_Execute_Encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Wilupgu_Compute_Pass"),
                timestamp_writes: None,
            });

            for node in &self.nodes {
                cpass.set_pipeline(&node.pipeline);
                cpass.set_bind_group(0, &node.bind_group, &[]);
                cpass.dispatch_workgroups(
                    node.workgroups[0],
                    node.workgroups[1],
                    node.workgroups[2],
                );
            }
        }

        self.ctx.queue.submit(Some(encoder.finish()));
    }
}
