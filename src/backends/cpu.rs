use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::backend::dtype::DataKind;
use crate::backend::shader::{Native, NativeBinding, Shader, ShaderBinary, Signature, Workgroups};
use crate::backend::{
    Access, Area, Areable, Buffer, Capturable, Carved, DeviceInfo, Dispatch, Node, Storage,
    Topology,
};

// Every opened `CpuBackend` gets its own value; its buffers carry it, so a
// buffer from another device is rejected in `execute_raw`.
static NEXT_OWNER: AtomicU64 = AtomicU64::new(0);

/// A [`CpuBackend`] buffer: host memory, kept as `u32` words so the bytes
/// the native shaders see are 4-aligned.
pub struct CpuBuffer {
    mem: CpuMem,
    owner: u64,
}

enum CpuMem {
    Owned(Vec<u32>),
    // A part of a `CpuArea`'s words. Only reachable through the `Carved`
    // around it, which borrows the area, so the words outlive it.
    Part { ptr: *mut u32, words: usize },
}

// SAFETY: an owned buffer is a plain `Vec`. A part is a range of its area that
// no other buffer covers, read through `&self` and written through `&mut self`
// only, like the `Vec`.
unsafe impl Send for CpuBuffer {}
unsafe impl Sync for CpuBuffer {}

impl CpuBuffer {
    fn bytes(&self) -> &[u8] {
        match &self.mem {
            CpuMem::Owned(words) => bytemuck::cast_slice(words),
            // SAFETY: see `CpuMem::Part`.
            CpuMem::Part { ptr, words } => unsafe {
                std::slice::from_raw_parts(ptr.cast::<u8>(), words * 4)
            },
        }
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        match &mut self.mem {
            CpuMem::Owned(words) => bytemuck::cast_slice_mut(words),
            // SAFETY: see `CpuMem::Part`.
            CpuMem::Part { ptr, words } => unsafe {
                std::slice::from_raw_parts_mut(ptr.cast::<u8>(), *words * 4)
            },
        }
    }
}

impl Buffer for CpuBuffer {
    fn size_bytes(&self) -> u64 {
        self.bytes().len() as u64
    }

    fn address(&self) -> u64 {
        self.bytes().as_ptr() as u64
    }
}

/// A [`CpuBackend`] node: the shader's native code and its meta.
pub struct CpuNode {
    signature: &'static Signature,
    code: Native,
    meta: Vec<u32>,
    workgroups: Workgroups,
}

impl Node for CpuNode {
    fn signature(&self) -> &'static Signature {
        self.signature
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

/// Runs [`Native`] code on the host, single-threaded. `execute_raw` finishes
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
            mem: CpuMem::Owned(vec![0; bytes.div_ceil(4) as usize]),
            owner: self.owner,
        })
    }

    unsafe fn upload_raw(&self, buf: &mut CpuBuffer, offset: u64, data: &[u8]) {
        let start = offset as usize;
        buf.bytes_mut()[start..start + data.len()].copy_from_slice(data);
    }

    unsafe fn download_raw(&self, buf: &CpuBuffer, offset: u64, bytes: u64) -> Vec<u8> {
        let start = offset as usize;
        buf.bytes()[start..start + bytes as usize].to_vec()
    }

    // Runs right away, like `execute_raw`.
    unsafe fn copy_raw(
        &self,
        src: &CpuBuffer,
        src_offset: u64,
        dst: &mut CpuBuffer,
        dst_offset: u64,
        bytes: u64,
    ) -> Result<(), String> {
        if src.owner != self.owner || dst.owner != self.owner {
            return Err("copy_raw: a buffer belongs to another device".to_string());
        }
        let (from, to, len) = (src_offset as usize, dst_offset as usize, bytes as usize);
        dst.bytes_mut()[to..to + len].copy_from_slice(&src.bytes()[from..from + len]);
        Ok(())
    }

    unsafe fn clear_raw(&self, buf: &mut CpuBuffer) -> Result<(), String> {
        if buf.owner != self.owner {
            return Err("clear_raw: the buffer belongs to another device".to_string());
        }
        buf.bytes_mut().fill(0);
        Ok(())
    }
}

// Work runs right away inside `execute_raw`, so `synchronize` has nothing to wait for.
impl Dispatch for CpuBackend {
    type Node = CpuNode;

    fn accepts(binary: &ShaderBinary) -> bool {
        matches!(binary, ShaderBinary::Native(_))
    }

    fn build_node(
        &self,
        shader: &Shader,
        meta: &[u32],
        workgroups: Workgroups,
    ) -> Result<CpuNode, String> {
        let ShaderBinary::Native(code) = *shader.binary() else {
            return Err(format!(
                "shader `{}`: not native code",
                shader.signature().name
            ));
        };
        Ok(CpuNode {
            signature: shader.signature(),
            code,
            meta: meta.to_vec(),
            workgroups,
        })
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
                node.signature.name
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
        (node.code.0)(&node.meta, &mut native);
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

/// A [`CpuBackend`] recording: the recorded calls, by address.
pub struct CpuCapture {
    calls: Vec<CpuCall>,
    owner: u64,
}

// SAFETY: the pointers are only followed in `replay`, whose caller promises
// that the nodes and buffers live and are left alone until it returns.
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
            (node.code.0)(&node.meta, &mut native);
        }
        Ok(())
    }

    fn release_capture(&self, capture: CpuCapture) {
        drop(capture);
    }
}

/// A [`CpuBackend`] area: one block of host words, cut out front to back.
pub struct CpuArea {
    /// `UnsafeCell` because carved buffers write into it through `&CpuArea`.
    words: Box<[UnsafeCell<u32>]>,
    /// Words already cut out; parts are handed out front to back.
    next: AtomicUsize,
    owner: u64,
}

// SAFETY: the area itself never reads or writes its words; each carved buffer
// gets a range no other one covers (`next` only grows, atomically).
unsafe impl Sync for CpuArea {}

impl Area for CpuArea {
    fn size_bytes(&self) -> u64 {
        self.words.len() as u64 * 4
    }

    fn remaining_bytes(&self) -> u64 {
        (self.words.len() - self.next.load(Ordering::Relaxed)) as u64 * 4
    }
}

impl Areable for CpuBackend {
    type Area = CpuArea;

    fn reserve(&self, total_bytes: u64) -> CpuArea {
        CpuArea {
            words: (0..total_bytes.div_ceil(4))
                .map(|_| UnsafeCell::new(0))
                .collect(),
            next: AtomicUsize::new(0),
            owner: self.owner,
        }
    }

    fn release(&self, area: CpuArea) {
        drop(area);
    }

    fn carve<'a>(&self, area: &'a CpuArea, bytes: u64) -> Result<Carved<'a, CpuBuffer>, String> {
        if area.owner != self.owner {
            return Err("carve: the area belongs to another device".to_string());
        }
        let words = bytes.div_ceil(4) as usize;
        let total = area.words.len();
        let start = area
            .next
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                (words <= total - next).then_some(next + words)
            })
            .map_err(|_| {
                format!(
                    "carve: {bytes} bytes asked, the area has {} left",
                    area.remaining_bytes()
                )
            })?;
        // `UnsafeCell<u32>` has the layout of `u32`.
        let base = UnsafeCell::raw_get(area.words.as_ptr());
        Ok(Carved::new(CpuBuffer {
            // SAFETY: `start + words <= total`, so the part is inside the area.
            mem: CpuMem::Part {
                ptr: unsafe { base.add(start) },
                words,
            },
            owner: self.owner,
        }))
    }
}
