use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use cudarc::driver::result::{self as cu, DriverError};
use cudarc::driver::sys::{self, CUgraphInstantiate_flags, CUstreamCaptureMode};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaGraph, CudaSlice, CudaStream, DevicePtr, LaunchConfig,
    PushKernelArg,
};

use crate::backend::dtype::DataKind;
use crate::backend::shader::{Shader, ShaderBinary, Signature, Workgroups};
use crate::backend::{
    Access, Area, Areable, Buffer, Capturable, Carved, DeviceInfo, Dispatch, Node, Storage,
    Topology, COPY_ALIGN,
};

// Every opened `CudaBackend` gets its own value; its buffers, nodes, areas
// and captures carry it, so another device's are rejected.
static NEXT_OWNER: AtomicU64 = AtomicU64::new(0);

// Every buffer is at least this big: `cuMemAlloc` refuses 0 bytes, and so
// no two buffers share an address.
const MIN_BYTES: u64 = COPY_ALIGN;

// Where buffers cut out of an area start, like `cuMemAlloc`'s own alignment.
const CARVE_ALIGN: u64 = 256;

// Runs record without timing or memory nodes; the flag only matters for
// graphs that allocate.
const INSTANTIATE: CUgraphInstantiate_flags =
    CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;

// Uploads and downloads stay allowed on the main stream while a capture runs.
const CAPTURE_MODE: CUstreamCaptureMode = CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED;

fn driver_err(what: &str, e: DriverError) -> String {
    format!("cuda: {what}: {e}")
}

// For the contract's methods that can't fail: a driver error there means
// the device is gone.
fn must<T>(result: Result<T, DriverError>, what: &str) -> T {
    result.unwrap_or_else(|e| panic!("{}", driver_err(what, e)))
}

/// A [`CudaBackend`] buffer: device memory, by its address.
pub struct CudaBuffer {
    ptr: sys::CUdeviceptr,
    bytes: u64,
    owner: u64,
    // `Some` for an allocated buffer, which frees itself on the main stream,
    // after the work queued before; `None` for a part of an area.
    _mem: Option<CudaSlice<u8>>,
}

impl Buffer for CudaBuffer {
    fn size_bytes(&self) -> u64 {
        self.bytes
    }

    fn address(&self) -> u64 {
        self.ptr
    }
}

/// A [`CudaBackend`] node: the loaded `entry` function and its meta buffer.
pub struct CudaNode {
    signature: &'static Signature,
    function: CudaFunction,
    workgroups: Workgroups,
    // At least one word: every kernel takes a `shader_meta` pointer.
    _meta: CudaSlice<u32>,
    meta_ptr: sys::CUdeviceptr,
    owner: u64,
}

impl Node for CudaNode {
    fn signature(&self) -> &'static Signature {
        self.signature
    }

    fn workgroups(&self) -> Workgroups {
        self.workgroups
    }

    // CUDA's grid limits differ by dimension.
    fn validate_workgroups(wg: Workgroups) -> Result<(), String> {
        if wg.x <= i32::MAX as u32 && wg.y <= 65535 && wg.z <= 65535 {
            Ok(())
        } else {
            Err(format!(
                "workgroup count {wg:?} exceeds CUDA's grid limit of (2^31 - 1, 65535, 65535)"
            ))
        }
    }
}

/// One CUDA device, as found by [`CudaBackend::choosable_devices`].
#[derive(Clone, Debug)]
pub struct CudaInfo {
    ordinal: usize,
    name: String,
    total_bytes: u64,
}

impl DeviceInfo for CudaInfo {
    fn label(&self) -> String {
        format!("{} (cuda:{})", self.name, self.ordinal)
    }

    fn total_memory_bytes(&self) -> u64 {
        self.total_bytes
    }

    // Buffers are plain bytes; which kinds a kernel reads is its own matter.
    fn supports_kind(&self, _kind: DataKind) -> bool {
        true
    }
}

/// An NVIDIA GPU, through the CUDA driver. Runs [`Ptx`](crate::backend::shader::Ptx)
/// and [`Cubin`](crate::backend::shader::Cubin) binaries; every upload,
/// download, copy, clear and run goes to one stream, in order.
///
/// The CUDA library is loaded when it is first needed; without it there are
/// no devices. The CUDA version its bindings follow is picked like for any
/// cudarc user: `CUDARC_CUDA_VERSION=13030`, or a `cuda-13030` feature on
/// cudarc in the user's own `Cargo.toml`.
pub struct CudaBackend {
    info: CudaInfo,
    owner: u64,
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    // Runs are recorded here between `begin_capture` and `end_capture`, so
    // uploads and copies in between still run on `stream`.
    capture_stream: Arc<CudaStream>,
    capturing: AtomicBool,
    // Some drivers and GPUs can't capture; found once, in `attach`.
    capture_works: bool,
}

impl CudaBackend {
    // The main stream, with the context bound to the calling thread, for the
    // raw driver calls.
    fn raw_stream(&self) -> sys::CUstream {
        must(self.ctx.bind_to_thread(), "bind the context");
        self.stream.cu_stream()
    }

    fn check_owner(&self, owner: u64, what: &str) -> Result<(), String> {
        if owner == self.owner {
            Ok(())
        } else {
            Err(format!("cuda: {what} belongs to another device"))
        }
    }
}

// Records a memset on `capture` and replays it on `stream`.
fn capture_works(stream: &Arc<CudaStream>, capture: &Arc<CudaStream>) -> bool {
    // SAFETY: the bytes are only ever set, never read.
    let Ok(buf) = (unsafe { stream.alloc::<u8>(4) }) else {
        return false;
    };
    let ptr = buf.device_ptr(stream).0;
    let tried = || -> Result<bool, DriverError> {
        stream.synchronize()?;
        capture.begin_capture(CAPTURE_MODE)?;
        // SAFETY: `buf` lives until the end of this function, after the sync.
        let set = unsafe { cu::memset_d8_async(ptr, 0, 4, capture.cu_stream()) };
        let graph = capture.end_capture(INSTANTIATE)?;
        set?;
        let Some(graph) = graph else {
            return Ok(false);
        };
        // SAFETY: `graph` and `buf` live until `stream` is synchronized.
        unsafe { cu::graph::launch(graph.cu_graph_exec(), stream.cu_stream()) }?;
        stream.synchronize()?;
        Ok(true)
    };
    matches!(tried(), Ok(true))
}

impl Topology for CudaBackend {
    type Info = CudaInfo;

    const NAME: &'static str = "cuda";

    fn choosable_devices() -> Vec<CudaInfo> {
        // SAFETY: it only tries to open the library.
        if !unsafe { sys::is_culib_present() } {
            return Vec::new();
        }
        let Ok(count) = CudaContext::device_count() else {
            return Vec::new();
        };
        (0..count)
            .filter_map(|ordinal| {
                let device = cu::device::get(ordinal).ok()?;
                Some(CudaInfo {
                    ordinal: ordinal as usize,
                    name: cu::device::get_name(device).ok()?,
                    // SAFETY: `device` came from `device::get` above.
                    total_bytes: unsafe { cu::device::total_mem(device) }.ok()? as u64,
                })
            })
            .collect()
    }

    fn attach(info: CudaInfo) -> Result<Self, String> {
        let ctx = CudaContext::new(info.ordinal).map_err(|e| driver_err("open the device", e))?;
        // SAFETY: every use of a buffer goes to `stream`, so nothing needs
        // cudarc's events between streams; this comes before any allocation.
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx
            .new_stream()
            .map_err(|e| driver_err("make a stream", e))?;
        let capture_stream = ctx
            .new_stream()
            .map_err(|e| driver_err("make a stream", e))?;
        let capture_works = capture_works(&stream, &capture_stream);
        Ok(Self {
            info,
            owner: NEXT_OWNER.fetch_add(1, Ordering::Relaxed),
            ctx,
            stream,
            capture_stream,
            capturing: AtomicBool::new(false),
            capture_works,
        })
    }

    fn info(&self) -> &CudaInfo {
        &self.info
    }
}

impl Storage for CudaBackend {
    type Buffer = CudaBuffer;

    fn alloc_raw(&self, bytes: u64) -> Result<CudaBuffer, String> {
        let bytes = bytes.max(MIN_BYTES);
        // SAFETY: the contract leaves a new buffer's contents unspecified.
        let mem = unsafe { self.stream.alloc::<u8>(bytes as usize) }
            .map_err(|e| driver_err("allocate", e))?;
        let ptr = mem.device_ptr(&self.stream).0;
        Ok(CudaBuffer {
            ptr,
            bytes,
            owner: self.owner,
            _mem: Some(mem),
        })
    }

    // `data` is pageable, so the driver copies it aside before returning.
    unsafe fn upload_raw(&self, buf: &mut CudaBuffer, offset: u64, data: &[u8]) {
        let stream = self.raw_stream();
        // SAFETY: the caller keeps the range inside `buf`.
        must(
            unsafe { cu::memcpy_htod_async(buf.ptr + offset, data, stream) },
            "upload",
        );
    }

    unsafe fn download_raw(&self, buf: &CudaBuffer, offset: u64, bytes: u64) -> Vec<u8> {
        let stream = self.raw_stream();
        let mut out = vec![0u8; bytes as usize];
        // SAFETY: the caller keeps the range inside `buf`; `out` lives until
        // the sync below.
        must(
            unsafe { cu::memcpy_dtoh_async(&mut out, buf.ptr + offset, stream) },
            "download",
        );
        must(self.stream.synchronize(), "download");
        out
    }

    unsafe fn copy_raw(
        &self,
        src: &CudaBuffer,
        src_offset: u64,
        dst: &mut CudaBuffer,
        dst_offset: u64,
        bytes: u64,
    ) -> Result<(), String> {
        self.check_owner(src.owner, "copy_raw: the source")?;
        self.check_owner(dst.owner, "copy_raw: the destination")?;
        let stream = self.raw_stream();
        // SAFETY: the caller keeps both ranges inside their buffers, and both
        // alive until `synchronize`.
        unsafe {
            cu::memcpy_dtod_async(
                dst.ptr + dst_offset,
                src.ptr + src_offset,
                bytes as usize,
                stream,
            )
        }
        .map_err(|e| driver_err("copy", e))
    }

    unsafe fn clear_raw(&self, buf: &mut CudaBuffer) -> Result<(), String> {
        self.check_owner(buf.owner, "clear_raw: the buffer")?;
        let stream = self.raw_stream();
        // SAFETY: the caller keeps `buf` alive until `synchronize`.
        unsafe { cu::memset_d8_async(buf.ptr, 0, buf.bytes as usize, stream) }
            .map_err(|e| driver_err("clear", e))
    }
}

impl Dispatch for CudaBackend {
    type Node = CudaNode;

    fn accepts(binary: &ShaderBinary) -> bool {
        matches!(binary, ShaderBinary::Ptx(_) | ShaderBinary::Cubin(_))
    }

    fn build_node(
        &self,
        shader: &Shader,
        meta: &[u32],
        workgroups: Workgroups,
    ) -> Result<CudaNode, String> {
        let name = shader.signature().name;
        let image = match shader.binary() {
            ShaderBinary::Ptx(ptx) if ptx.0.contains('\0') => {
                return Err(format!("cuda: shader `{name}`: the PTX holds a NUL"));
            }
            ShaderBinary::Ptx(ptx) => cudarc::nvrtc::Ptx::from_src(ptx.0.to_string()),
            ShaderBinary::Cubin(cubin) => cudarc::nvrtc::Ptx::from_binary(cubin.0.to_vec()),
            _ => return Err(format!("cuda: shader `{name}`: neither PTX nor cubin")),
        };
        let module = self
            .ctx
            .load_module(image)
            .map_err(|e| driver_err(&format!("shader `{name}`: load"), e))?;
        let function = module
            .load_function("entry")
            .map_err(|e| driver_err(&format!("shader `{name}`: find `entry`"), e))?;

        // SAFETY: every word the kernel reads is written below; with no meta,
        // the kernel reads none.
        let meta_buf = unsafe { self.stream.alloc::<u32>(meta.len().max(1)) }
            .map_err(|e| driver_err("allocate a meta buffer", e))?;
        let meta_ptr = meta_buf.device_ptr(&self.stream).0;
        if !meta.is_empty() {
            let stream = self.raw_stream();
            // SAFETY: the buffer holds `meta.len()` words.
            unsafe { cu::memcpy_htod_async(meta_ptr, meta, stream) }
                .map_err(|e| driver_err("write the meta", e))?;
        }
        Ok(CudaNode {
            signature: shader.signature(),
            function,
            workgroups,
            _meta: meta_buf,
            meta_ptr,
            owner: self.owner,
        })
    }

    // On the main stream: runs queued before it read the old words, runs and
    // replays queued after it the new ones.
    fn update_meta(&self, node: &mut CudaNode, meta: &[u32]) {
        let stream = self.raw_stream();
        // SAFETY: the meta buffer holds the shader's meta words.
        must(
            unsafe { cu::memcpy_htod_async(node.meta_ptr, meta, stream) },
            "write the meta",
        );
    }

    unsafe fn execute_raw(
        &self,
        node: &CudaNode,
        bindings: &mut [Access<'_, CudaBuffer>],
    ) -> Result<(), String> {
        let name = node.signature.name;
        self.check_owner(node.owner, &format!("shader `{name}`: the node"))?;
        let mut ptrs = vec![node.meta_ptr];
        for binding in bindings.iter() {
            let buf = match binding {
                Access::Read(buf) => &**buf,
                Access::Write(buf) => &**buf,
            };
            self.check_owner(buf.owner, &format!("shader `{name}`: a bound buffer"))?;
            ptrs.push(buf.ptr);
        }
        let capturing = self.capturing.load(Ordering::Relaxed);
        if capturing && !self.capture_works {
            // Recorded into nothing; `end_capture` fails.
            return Ok(());
        }
        let stream = if capturing {
            &self.capture_stream
        } else {
            &self.stream
        };
        let [bx, by, bz] = node.signature.workgroup_size;
        let wg = node.workgroups;
        let config = LaunchConfig {
            grid_dim: (wg.x, wg.y, wg.z),
            block_dim: (bx, by, bz),
            shared_mem_bytes: 0,
        };
        let mut launch = stream.launch_builder(&node.function);
        for ptr in &ptrs {
            launch.arg(ptr);
        }
        // SAFETY: the kernel takes the meta pointer, then one pointer per
        // slot (docs/SHADERS.md); the caller keeps the buffers alive, of the
        // slots' kinds and at least `min_lens` long, until `synchronize`.
        unsafe { launch.launch(config) }
            .map(drop)
            .map_err(|e| driver_err(&format!("shader `{name}`: launch"), e))
    }

    fn synchronize(&self) {
        must(self.stream.synchronize(), "synchronize");
    }
}

/// A [`CudaBackend`] recording: one CUDA graph of the recorded runs, or
/// none when nothing ran.
pub struct CudaCapture {
    // A graph must not be launched from two threads at once.
    graph: Mutex<Option<CudaGraph>>,
    owner: u64,
}

// SAFETY: the graph is only reached under the mutex, one thread at a time,
// which is all CUDA asks of graph objects.
unsafe impl Send for CudaCapture {}
unsafe impl Sync for CudaCapture {}

impl Capturable for CudaBackend {
    type Capture = CudaCapture;

    fn begin_capture(&self) {
        assert!(
            !self.capturing.swap(true, Ordering::Relaxed),
            "begin_capture: this device is already recording"
        );
        if self.capture_works {
            must(
                self.capture_stream.begin_capture(CAPTURE_MODE),
                "begin a capture",
            );
        }
    }

    fn end_capture(&self) -> Result<CudaCapture, String> {
        if !self.capturing.swap(false, Ordering::Relaxed) {
            return Err("end_capture: this device isn't recording".to_string());
        }
        if !self.capture_works {
            return Err(format!(
                "end_capture: `{}` or its driver can't capture",
                self.info.label()
            ));
        }
        let graph = self
            .capture_stream
            .end_capture(INSTANTIATE)
            .map_err(|e| driver_err("end the capture", e))?;
        Ok(CudaCapture {
            graph: Mutex::new(graph),
            owner: self.owner,
        })
    }

    // On the main stream, so it is ordered with everything else.
    unsafe fn replay(&self, capture: &CudaCapture) -> Result<(), String> {
        self.check_owner(capture.owner, "replay: the capture")?;
        let graph = capture.graph.lock().unwrap();
        let Some(graph) = graph.as_ref() else {
            return Ok(());
        };
        let stream = self.raw_stream();
        // SAFETY: the caller keeps every recorded node and buffer alive and in
        // place until `synchronize`.
        unsafe { cu::graph::launch(graph.cu_graph_exec(), stream) }
            .map_err(|e| driver_err("replay", e))
    }

    fn release_capture(&self, capture: CudaCapture) {
        drop(capture);
    }
}

/// A [`CudaBackend`] area: one allocation, cut out front to back.
pub struct CudaArea {
    _mem: CudaSlice<u8>,
    base: sys::CUdeviceptr,
    bytes: u64,
    // Bytes already cut out; always a multiple of `CARVE_ALIGN`.
    next: AtomicU64,
    owner: u64,
}

impl Area for CudaArea {
    fn size_bytes(&self) -> u64 {
        self.bytes
    }

    fn remaining_bytes(&self) -> u64 {
        self.bytes - self.next.load(Ordering::Relaxed)
    }
}

impl Areable for CudaBackend {
    type Area = CudaArea;

    fn reserve(&self, total_bytes: u64) -> CudaArea {
        let bytes = total_bytes.max(MIN_BYTES);
        // SAFETY: carved buffers' contents are unspecified, like allocated ones.
        let mem = must(
            unsafe { self.stream.alloc::<u8>(bytes as usize) },
            "reserve an area",
        );
        let base = mem.device_ptr(&self.stream).0;
        CudaArea {
            base,
            _mem: mem,
            bytes,
            next: AtomicU64::new(0),
            owner: self.owner,
        }
    }

    fn release(&self, area: CudaArea) {
        drop(area);
    }

    fn carve<'a>(&self, area: &'a CudaArea, bytes: u64) -> Result<Carved<'a, CudaBuffer>, String> {
        self.check_owner(area.owner, "carve: the area")?;
        let bytes = bytes.max(MIN_BYTES);
        let taken = bytes.next_multiple_of(CARVE_ALIGN);
        let start = area
            .next
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                (taken <= area.bytes - next).then_some(next + taken)
            })
            .map_err(|_| {
                format!(
                    "carve: {bytes} bytes asked, the area has {} left",
                    area.remaining_bytes()
                )
            })?;
        Ok(Carved::new(CudaBuffer {
            ptr: area.base + start,
            bytes,
            owner: self.owner,
            _mem: None,
        }))
    }
}

#[cfg(all(test, feature = "compile-nvrtc"))]
#[path = "../tests/cuda.rs"]
mod tests;
