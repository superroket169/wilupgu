use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use crate::backend::dtype::DataKind;
use crate::backend::shader::{NativeBinding, NativeCode, Shader, ShaderFormat, Workgroups};
use crate::backend::{Access, Buffer, Capturable, DeviceInfo, Dispatch, Node, Storage, Topology};

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
    /// `Some` between `begin_capture` and `end_capture`.
    recording: Mutex<Option<Vec<CpuCall>>>,
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
            recording: Mutex::new(None),
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
        if let Some(calls) = self.recording.lock().unwrap().as_mut() {
            calls.push(CpuCall {
                node,
                slots: bindings
                    .iter_mut()
                    .map(|a| match a {
                        Access::Read(b) => {
                            let bytes = b.bytes();
                            CpuSlot::Read(bytes.as_ptr(), bytes.len())
                        }
                        Access::Write(b) => {
                            let bytes = b.bytes_mut();
                            CpuSlot::Write(bytes.as_mut_ptr(), bytes.len())
                        }
                    })
                    .collect(),
            });
            return Ok(());
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

/// One recorded `execute_raw`. The node by address, so a replay sees its
/// current meta; each slot by its bytes' address, which is what
/// `Buffer::address` promises stays put.
struct CpuCall {
    node: *const CpuNode,
    slots: Vec<CpuSlot>,
}

enum CpuSlot {
    Read(*const u8, usize),
    Write(*mut u8, usize),
}

pub struct CpuCapture {
    calls: Vec<CpuCall>,
    owner: u64,
}

/// SAFETY: the pointers are only followed in `replay`, whose caller promises
/// that the nodes and buffers live and are left alone until it returns.
unsafe impl Send for CpuCall {}
unsafe impl Sync for CpuCall {}

impl Capturable for CpuBackend {
    type Capture = CpuCapture;

    fn begin_capture(&self) {
        let mut recording = self.recording.lock().unwrap();
        assert!(
            recording.is_none(),
            "begin_capture: this device is already recording"
        );
        *recording = Some(Vec::new());
    }

    fn end_capture(&self) -> Result<CpuCapture, String> {
        let calls = self
            .recording
            .lock()
            .unwrap()
            .take()
            .ok_or("end_capture: this device isn't recording")?;
        Ok(CpuCapture {
            calls,
            owner: self.owner,
        })
    }

    unsafe fn replay(&self, capture: &CpuCapture) -> Result<(), String> {
        if capture.owner != self.owner {
            return Err("replay: the capture belongs to another device".to_string());
        }
        for call in &capture.calls {
            // SAFETY: the caller keeps every recorded node and buffer alive and
            // in place, and the recorded calls followed `execute_raw`'s rules,
            // so no written slot aliases another.
            let node = unsafe { &*call.node };
            let mut native: Vec<NativeBinding> = call
                .slots
                .iter()
                .map(|slot| match *slot {
                    CpuSlot::Read(ptr, len) => {
                        NativeBinding::Read(unsafe { std::slice::from_raw_parts(ptr, len) })
                    }
                    CpuSlot::Write(ptr, len) => {
                        NativeBinding::Write(unsafe { std::slice::from_raw_parts_mut(ptr, len) })
                    }
                })
                .collect();
            node.code.run(&node.meta, &mut native);
        }
        Ok(())
    }

    fn release_capture(&self, capture: CpuCapture) {
        drop(capture);
    }
}
