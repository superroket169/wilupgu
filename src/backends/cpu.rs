use std::sync::atomic::{AtomicU64, Ordering};

use crate::backend::dtype::DataKind;
use crate::backend::shader::{NativeBinding, NativeCode, Shader, ShaderFormat, Workgroups};
use crate::backend::{Access, Buffer, DeviceInfo, Dispatch, Node, Storage, Topology};

// Every opened `CpuBackend` gets its own value; its buffers carry it, so a
// buffer from another device is rejected in `execute_raw`.
static NEXT_OWNER: AtomicU64 = AtomicU64::new(0);

/// A [`CpuBackend`] buffer: host memory, kept as `u32` words so the bytes
/// the native shaders see are 4-aligned.
pub struct CpuBuffer {
    words: Vec<u32>,
    owner: u64,
}

impl CpuBuffer {
    fn bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.words)
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        bytemuck::cast_slice_mut(&mut self.words)
    }
}

impl Buffer for CpuBuffer {
    fn size_bytes(&self) -> u64 {
        self.words.len() as u64 * 4
    }

    fn address(&self) -> u64 {
        self.words.as_ptr() as u64
    }
}

/// A [`CpuBackend`] node: the shader's native code and its meta.
pub struct CpuNode {
    shader: &'static Shader,
    code: NativeCode,
    meta: Vec<u32>,
    workgroups: Workgroups,
}

impl Node for CpuNode {
    fn shader(&self) -> &'static Shader {
        self.shader
    }

    fn workgroups(&self) -> Workgroups {
        self.workgroups
    }
}

/// The one device [`CpuBackend`] offers: the host.
#[derive(Clone, Debug)]
pub struct CpuInfo;

impl DeviceInfo for CpuInfo {
    fn label(&self) -> String {
        "cpu".to_string()
    }

    fn total_memory_bytes(&self) -> u64 {
        // TODO: the host's real memory size, for SysTopology.
        u64::MAX
    }

    fn supports_kind(&self, _kind: DataKind) -> bool {
        true
    }
}

/// Runs [`NativeCode`] on the host, single-threaded. `execute_raw` finishes
/// the work before it returns, so `synchronize` has nothing to wait for.
pub struct CpuBackend {
    info: CpuInfo,
    owner: u64,
}

impl Topology for CpuBackend {
    type Info = CpuInfo;

    const NAME: &'static str = "cpu";

    fn choosable_devices() -> Vec<CpuInfo> {
        vec![CpuInfo]
    }

    fn attach(info: CpuInfo) -> Result<Self, String> {
        Ok(Self {
            info,
            owner: NEXT_OWNER.fetch_add(1, Ordering::Relaxed),
        })
    }

    fn info(&self) -> &CpuInfo {
        &self.info
    }
}

impl Storage for CpuBackend {
    type Buffer = CpuBuffer;

    fn alloc_raw(&self, bytes: u64) -> Result<CpuBuffer, String> {
        Ok(CpuBuffer {
            words: vec![0; bytes.div_ceil(4) as usize],
            owner: self.owner,
        })
    }

    unsafe fn upload_raw(&self, buf: &mut CpuBuffer, data: &[u8]) {
        buf.bytes_mut()[..data.len()].copy_from_slice(data);
    }

    unsafe fn download_raw(&self, buf: &CpuBuffer, bytes: u64) -> Vec<u8> {
        buf.bytes()[..bytes as usize].to_vec()
    }
}

// Work runs right away inside `execute_raw`, so `synchronize` has nothing to wait for.
impl Dispatch for CpuBackend {
    type Node = CpuNode;

    const FORMAT: ShaderFormat = ShaderFormat::Native;

    fn build_node(&self, shader: &'static Shader, meta: &[u32], workgroups: Workgroups) -> CpuNode {
        let code = shader
            .shader_code
            .native
            .unwrap_or_else(|| panic!("shader `{}` has no native code", shader.name));
        CpuNode {
            shader,
            code,
            meta: meta.to_vec(),
            workgroups,
        }
    }

    fn update_meta(&self, node: &mut CpuNode, meta: &[u32]) {
        node.meta.copy_from_slice(meta);
    }

    unsafe fn execute_raw(
        &self,
        node: &CpuNode,
        bindings: &mut [Access<'_, CpuBuffer>],
    ) -> Result<(), String> {
        let foreign = bindings.iter().any(|a| match a {
            Access::Read(b) => b.owner != self.owner,
            Access::Write(b) => b.owner != self.owner,
        });
        if foreign {
            return Err(format!(
                "shader `{}`: a bound buffer belongs to another device",
                node.shader.name
            ));
        }
        let mut native: Vec<NativeBinding> = bindings
            .iter_mut()
            .map(|a| match a {
                Access::Read(b) => NativeBinding::Read(b.bytes()),
                Access::Write(b) => NativeBinding::Write(b.bytes_mut()),
            })
            .collect();
        node.code.run(&node.meta, &mut native);
        Ok(())
    }

    fn synchronize(&self) {}
}
