use crate::backend::{Backend, Binding, Dtype, TensorMode};
use crate::builtin::cuda_kernels as k;
use crate::pool::BufferPool;
use crate::shader::{CudaShape, MetaField, Shader};
use cudarc::cublas::sys::{cublasMath_t, cublasOperation_t, cublasSetMathMode, cublasStatus_t};
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::result::DriverError;
use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
use cudarc::driver::{
    CudaContext as CuDevice, CudaFunction, CudaGraph, CudaSlice, CudaStream, LaunchConfig,
    PushKernelArg,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub enum CudaBuffer {
    F32(Arc<Mutex<CudaSlice<f32>>>),
    F16(Arc<Mutex<CudaSlice<half::f16>>>),
    Bf16(Arc<Mutex<CudaSlice<half::bf16>>>),
}

impl CudaBuffer {
    pub fn dtype(&self) -> Dtype {
        match self {
            CudaBuffer::F32(_) => Dtype::F32,
            CudaBuffer::F16(_) => Dtype::F16,
            CudaBuffer::Bf16(_) => Dtype::Bf16,
        }
    }

    pub fn as_f32(&self) -> &Arc<Mutex<CudaSlice<f32>>> {
        match self {
            CudaBuffer::F32(s) => s,
            other => panic!("[cuda] expected F32 buffer, got {:?}", other.dtype()),
        }
    }

    pub fn as_f16(&self) -> &Arc<Mutex<CudaSlice<half::f16>>> {
        match self {
            CudaBuffer::F16(s) => s,
            other => panic!("[cuda] expected F16 buffer, got {:?}", other.dtype()),
        }
    }

    pub fn as_bf16(&self) -> &Arc<Mutex<CudaSlice<half::bf16>>> {
        match self {
            CudaBuffer::Bf16(s) => s,
            other => panic!("[cuda] expected Bf16 buffer, got {:?}", other.dtype()),
        }
    }
}

#[derive(Clone)]
pub struct CudaBinding {
    pub slot: u32,
    pub slice: CudaBuffer,
    pub mode: TensorMode,
    pub cached_meta: Option<Vec<u8>>,
}

#[derive(Clone)]
pub struct CudaNode {
    shader: &'static Shader,
    bindings: Vec<CudaBinding>,
    workgroups: [u32; 3],
}

// ========================
//       CudaBackend
// ========================

struct GraphCell {
    graph: CudaGraph,
    owner: std::thread::ThreadId,
}
unsafe impl Send for GraphCell {}
unsafe impl Sync for GraphCell {}

enum GraphState {
    Warmed,
    Captured(GraphCell),
}

pub struct CudaBackend {
    device: Arc<CuDevice>,
    pub stream: Arc<CudaStream>,
    blas: CudaBlas,
    kernel_cache: Mutex<HashMap<usize, CudaFunction>>,
    pool: BufferPool<CudaBuffer, (u64, Dtype)>,
    graph_cache: Mutex<HashMap<usize, GraphState>>,
    capturing: std::sync::atomic::AtomicBool,
}

impl CudaBackend {
    pub fn new(ordinal: usize) -> Result<Self, DriverError> {
        let device = CuDevice::new(ordinal)?;
        let stream = device.new_stream()?;

        let blas = CudaBlas::new(stream.clone()).map_err(|e| {
            eprintln!("[cuda] cuBLAS init failed: {e:?}");
            DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN)
        })?;

        unsafe {
            let status =
                cublasSetMathMode(*blas.handle(), cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH);
            if status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                eprintln!(
                    "[cuda] cublasSetMathMode(TF32) failed: {status:?} -- matmuls stay full FP32"
                );
            }
        }

        Ok(Self {
            device,
            stream,
            blas,
            kernel_cache: Mutex::new(HashMap::new()),
            pool: BufferPool::new(),
            graph_cache: Mutex::new(HashMap::new()),
            capturing: std::sync::atomic::AtomicBool::new(false),
        })
    }

    pub(crate) fn compile(&self, key: usize, src: &str, func: &str) -> CudaFunction {
        {
            let cache = self.kernel_cache.lock().unwrap();
            if let Some(f) = cache.get(&key) {
                return f.clone();
            }
        }
        let ptx = cudarc::nvrtc::compile_ptx(src)
            .unwrap_or_else(|e| panic!("[cuda] NVRTC failed for '{func}': {e:?}"));
        let module = self
            .device
            .load_module(ptx)
            .unwrap_or_else(|e| panic!("[cuda] load PTX for '{func}': {e:?}"));
        let func = module
            .load_function(func)
            .unwrap_or_else(|e| panic!("[cuda] fn '{func}' not found: {e:?}"));

        self.kernel_cache.lock().unwrap().insert(key, func.clone());

        func
    }
}

// ========================================
//            Dispatch helpers
// ========================================

use super::cuda_launch_macros::{define_launch, launch, read_meta};

#[allow(dead_code)]
fn lock_f32(bindings: &[CudaBinding], slot: u32) -> std::sync::MutexGuard<'_, CudaSlice<f32>> {
    find(bindings, slot).slice.as_f32().lock().unwrap()
}

#[allow(dead_code)]
fn lock_f16(
    bindings: &[CudaBinding],
    slot: u32,
) -> std::sync::MutexGuard<'_, CudaSlice<half::f16>> {
    find(bindings, slot).slice.as_f16().lock().unwrap()
}

fn find(bindings: &[CudaBinding], slot: u32) -> &CudaBinding {
    bindings
        .iter()
        .find(|b| b.slot == slot)
        .expect("missing binding slot")
}

fn meta_bytes(b: &CudaBinding) -> Vec<u8> {
    b.cached_meta.clone().expect("meta_bytes: no cached data")
}

fn meta_u32(b: &CudaBinding) -> Vec<u32> {
    bytemuck::cast_slice::<u8, u32>(&meta_bytes(b)).to_vec()
}

fn cfg_1d(n: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n + 255) / 256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

impl CudaBackend {
    fn gemm_dispatch<T>(
        &self,
        bg: &CudaSlice<T>,
        ag: &CudaSlice<T>,
        cg: &mut CudaSlice<T>,
        transpose_b: bool,
        alpha: T,
        beta: T,
        m: u32,
        n: u32,
        ki: u32,
    ) where
        CudaBlas: Gemm<T>,
    {
        let (op_b, ldb) = if transpose_b {
            (cublasOperation_t::CUBLAS_OP_T, ki)
        } else {
            (cublasOperation_t::CUBLAS_OP_N, n)
        };
        let cfg = GemmConfig {
            transa: op_b,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: ki as i32,
            alpha,
            lda: ldb as i32,
            ldb: ki as i32,
            beta,
            ldc: n as i32,
        };
        unsafe { self.blas.gemm(cfg, bg, ag, cg).expect("[cuda] gemm failed") }
    }

    fn gemm_meta_u32(&self, b: &CudaBinding) -> Vec<u32> {
        if self.capturing.load(std::sync::atomic::Ordering::Relaxed) {
            return meta_u32(b);
        }
        let g = b.slice.as_f32().lock().unwrap();
        let f32s = self
            .stream
            .clone_dtoh(&*g)
            .expect("[cuda] live meta dtoh failed");
        bytemuck::cast_slice::<f32, u32>(&f32s).to_vec()
    }

    fn gemm_matmul(&self, bindings: &[CudaBinding], transpose_b: bool, beta: f32) {
        let a = find(bindings, 0);
        let b = find(bindings, 1);
        let c = find(bindings, 2);
        let dims = self.gemm_meta_u32(find(bindings, 3));
        let (m, n, ki) = (dims[0], dims[1], dims[2]);

        match a.slice.dtype() {
            Dtype::F32 => {
                let ag = a.slice.as_f32().lock().unwrap();
                let bg = b.slice.as_f32().lock().unwrap();
                let mut cg = c.slice.as_f32().lock().unwrap();
                self.gemm_dispatch(&*bg, &*ag, &mut *cg, transpose_b, 1.0f32, beta, m, n, ki);
            }
            Dtype::F16 => {
                let ag = a.slice.as_f16().lock().unwrap();
                let bg = b.slice.as_f16().lock().unwrap();
                let mut cg = c.slice.as_f16().lock().unwrap();
                self.gemm_dispatch(
                    &*bg,
                    &*ag,
                    &mut *cg,
                    transpose_b,
                    half::f16::from_f32(1.0),
                    half::f16::from_f32(beta),
                    m,
                    n,
                    ki,
                );
            }
            Dtype::Bf16 => {
                let ag = a.slice.as_bf16().lock().unwrap();
                let bg = b.slice.as_bf16().lock().unwrap();
                let mut cg = c.slice.as_bf16().lock().unwrap();
                self.gemm_dispatch(
                    &*bg,
                    &*ag,
                    &mut *cg,
                    transpose_b,
                    half::bf16::from_f32(1.0),
                    half::bf16::from_f32(beta),
                    m,
                    n,
                    ki,
                );
            }
        }
    }

    fn gemm_weight_bwd(&self, bindings: &[CudaBinding]) {
        let a = find(bindings, 0);
        let dc = find(bindings, 1);
        let db = find(bindings, 2);
        let dims = self.gemm_meta_u32(find(bindings, 3));
        let (m, n, ki) = (dims[0], dims[1], dims[2]);

        let ag = a.slice.as_f32().lock().unwrap();
        let dcg = dc.slice.as_f32().lock().unwrap();
        let mut dbg = db.slice.as_f32().lock().unwrap();

        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_T,
            m: n as i32,
            n: ki as i32,
            k: m as i32,
            alpha: 1.0,
            lda: n as i32,
            ldb: ki as i32,
            beta: 1.0,
            ldc: n as i32,
        };
        unsafe {
            self.blas
                .gemm(cfg, &*dcg, &*ag, &mut *dbg)
                .expect("[cuda] sgemm weight_bwd failed")
        }
    }

    // -------- Structured ----------

    fn launch_adamw(&self, bindings: &[CudaBinding], key: usize) {
        let size = meta_u32(find(bindings, 4))[0];

        let const_bytes = meta_bytes(find(bindings, 6));
        let beta1 = f32::from_ne_bytes(const_bytes[0..4].try_into().unwrap());
        let beta2 = f32::from_ne_bytes(const_bytes[4..8].try_into().unwrap());
        let eps = f32::from_ne_bytes(const_bytes[8..12].try_into().unwrap());
        let wd = f32::from_ne_bytes(const_bytes[12..16].try_into().unwrap());

        let f = self.compile(key, k::ADAMW, "adamw_kernel");
        let mut wg = find(bindings, 0).slice.as_f32().lock().unwrap();
        let gg = find(bindings, 1).slice.as_f32().lock().unwrap();
        let mut mg = find(bindings, 2).slice.as_f32().lock().unwrap();
        let mut vg = find(bindings, 3).slice.as_f32().lock().unwrap();
        let schedule_g = find(bindings, 5).slice.as_f32().lock().unwrap();
        launch!(
            self,
            f,
            cfg_1d(size),
            &mut *wg,
            &*gg,
            &mut *mg,
            &mut *vg,
            &size,
            &*schedule_g,
            &beta1,
            &beta2,
            &eps,
            &wd
        );
    }

    define_launch!(
        launch_adamw_schedule,
        meta_slot: 1, meta: [lr_max: f32, lr_min: f32, warmup_steps: u32, max_steps: u32],
        buffers: [mut state_g: 0],
        grid: LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 },
        launch: [&mut *state_g, &lr_max, &lr_min, &warmup_steps, &max_steps]
    );

    fn dispatch_node(&self, node: &CudaNode) {
        let b = &node.bindings;
        let wg = node.workgroups;

        let spec = node
            .shader
            .cuda
            .as_ref()
            .unwrap_or_else(|| panic!("[cuda] shader `{}` has no cuda impl", node.shader.name));

        match &spec.shape {
            CudaShape::Generic {
                meta_fields,
                block_dim,
                append_len,
            } => self.dispatch_generic(
                node.shader,
                spec.src,
                spec.entry,
                *meta_fields,
                *block_dim,
                *append_len,
                b,
                wg,
            ),
            CudaShape::Custom(f) => f(node.shader, self, b, wg),
        }
    }

    /// Data-driven dispatch for CudaShape::Generic.
    ///
    /// Meta is passed to the kernel as a device pointer
    /// and a captured CUDA graph stays valid because only the pointer is baked into it, not the values.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_generic(
        &self,
        shader: &'static Shader,
        src: &str,
        entry: &str,
        meta_fields: &[MetaField],
        block_dim: (u32, u32, u32),
        append_len: bool,
        bindings: &[CudaBinding],
        workgroups: [u32; 3],
    ) {
        let meta_slots: Vec<u32> = shader
            .layout
            .iter()
            .enumerate()
            .filter(|(_, m)| **m == TensorMode::Meta)
            .map(|(i, _)| i as u32)
            .collect();
        assert!(
            meta_slots.len() <= 1,
            "[cuda] '{}': dispatch_generic supports at most one Meta slot ({} declared in layout) \
             — use CudaShape::Custom for kernels with more",
            shader.name,
            meta_slots.len()
        );

        if let Some(&slot) = meta_slots.first() {
            assert_eq!(
                slot as usize,
                shader.layout.len() - 1,
                "[cuda] '{}': the Meta slot must be the last binding so kernel arg order matches the layout",
                shader.name
            );
        } else {
            assert!(
                meta_fields.is_empty(),
                "[cuda] '{}': meta_fields declares {} field(s) but layout has no Meta slot",
                shader.name,
                meta_fields.len()
            );
        }

        let mut buf_guards: Vec<(TensorMode, std::sync::MutexGuard<'_, CudaSlice<f32>>)> =
            Vec::new();

        for (slot, mode) in shader.layout.iter().enumerate() {
            let binding = find(bindings, slot as u32);
            buf_guards.push((*mode, binding.slice.as_f32().lock().unwrap()));
        }

        if let Some(&slot) = meta_slots.first() {
            let len_bytes = buf_guards[slot as usize].1.len() * 4;
            assert!(
                len_bytes >= meta_fields.len() * 4,
                "[cuda] '{}': meta buffer is only {} bytes but meta_fields declares {} field(s) ({} bytes needed) \
                 — layout/meta_fields are out of sync",
                shader.name,
                len_bytes,
                meta_fields.len(),
                meta_fields.len() * 4
            );
        }

        let len_arg: u32 = if append_len {
            buf_guards
                .first()
                .unwrap_or_else(|| {
                    panic!(
                        "[cuda] '{}': append_len needs at least one buffer",
                        shader.name
                    )
                })
                .1
                .len() as u32
        } else {
            0
        };

        let key = shader_key(shader);
        let f = self.compile(key, src, entry);
        let cfg = LaunchConfig {
            grid_dim: (
                workgroups[0].max(1),
                workgroups[1].max(1),
                workgroups[2].max(1),
            ),
            block_dim,
            shared_mem_bytes: 0,
        };

        let mut b = self.stream.launch_builder(&f);
        for (mode, guard) in buf_guards.iter_mut() {
            match mode {
                TensorMode::Input | TensorMode::Meta => {
                    b.arg(&**guard);
                }
                TensorMode::Output | TensorMode::InOut => {
                    b.arg(&mut **guard);
                }
            }
        }
        if append_len {
            b.arg(&len_arg);
        }
        unsafe { b.launch(cfg) }.expect("[cuda] kernel launch failed");
    }
}

fn shader_key(shader: &'static Shader) -> usize {
    shader as *const Shader as usize
}

// ==========================================================================
//                          Custom-shape dispatches
// ==========================================================================

pub(crate) fn custom_matmul(
    _s: &'static Shader,
    b: &CudaBackend,
    bindings: &[CudaBinding],
    _wg: [u32; 3],
) {
    b.gemm_matmul(bindings, false, 0.0)
}

pub(crate) fn custom_matmul_trp(
    _s: &'static Shader,
    b: &CudaBackend,
    bindings: &[CudaBinding],
    _wg: [u32; 3],
) {
    b.gemm_matmul(bindings, true, 0.0)
}

pub(crate) fn custom_matmul_add(
    _s: &'static Shader,
    b: &CudaBackend,
    bindings: &[CudaBinding],
    _wg: [u32; 3],
) {
    b.gemm_matmul(bindings, false, 1.0)
}

pub(crate) fn custom_matmul_weight_bwd(
    _s: &'static Shader,
    b: &CudaBackend,
    bindings: &[CudaBinding],
    _wg: [u32; 3],
) {
    b.gemm_weight_bwd(bindings)
}

pub(crate) fn custom_adamw(
    s: &'static Shader,
    b: &CudaBackend,
    bindings: &[CudaBinding],
    _wg: [u32; 3],
) {
    b.launch_adamw(bindings, shader_key(s))
}

pub(crate) fn custom_adamw_schedule(
    s: &'static Shader,
    b: &CudaBackend,
    bindings: &[CudaBinding],
    _wg: [u32; 3],
) {
    b.launch_adamw_schedule(
        bindings,
        shader_key(s),
        k::ADAMW_SCHEDULE,
        "adamw_schedule_kernel",
    )
}

pub(crate) mod dispatch {
    pub(crate) use super::{
        custom_adamw, custom_adamw_schedule, custom_matmul, custom_matmul_add, custom_matmul_trp,
        custom_matmul_weight_bwd,
    };
}

impl Backend for CudaBackend {
    type Buffer = CudaBuffer;
    type Node = CudaNode;

    fn name(&self) -> &'static str {
        "cuda"
    }

    fn alloc(&self, size_bytes: u64) -> CudaBuffer {
        if let Some(buf) = self.pool.take((size_bytes, Dtype::F32)) {
            return buf;
        }
        let n = (size_bytes as usize) / std::mem::size_of::<f32>();
        let slice = self
            .stream
            .alloc_zeros::<f32>(n)
            .expect("[cuda] alloc failed");
        CudaBuffer::F32(Arc::new(Mutex::new(slice)))
    }

    fn alloc_from_cpu<T: bytemuck::Pod>(&self, data: &[T]) -> CudaBuffer {
        let f32s: &[f32] = bytemuck::cast_slice(data);
        let size_bytes = (f32s.len() * std::mem::size_of::<f32>()) as u64;
        if let Some(buf) = self.pool.take((size_bytes, Dtype::F32)) {
            let mut g = buf.as_f32().lock().unwrap();
            self.stream
                .memcpy_htod(f32s, &mut *g)
                .expect("[cuda] htod copy (recycled) failed");
            drop(g);
            return buf;
        }
        let slice = self.stream.clone_htod(f32s).expect("[cuda] htod failed");
        CudaBuffer::F32(Arc::new(Mutex::new(slice)))
    }

    fn copy_from_cpu<T: bytemuck::Pod>(&self, buf: &CudaBuffer, data: &[T]) {
        let f32s: &[f32] = bytemuck::cast_slice(data);
        let mut g = buf.as_f32().lock().unwrap();
        self.stream
            .memcpy_htod(f32s, &mut *g)
            .expect("[cuda] htod copy failed");
    }

    fn copy_to_cpu<T: bytemuck::Pod + Default + Clone>(&self, buf: &CudaBuffer) -> Vec<T> {
        let g = buf.as_f32().lock().unwrap();
        let f32s = self.stream.clone_dtoh(&*g).expect("[cuda] dtoh failed");
        bytemuck::cast_slice::<f32, T>(&f32s).to_vec()
    }

    fn alloc_dtype(&self, elem_count: usize, dtype: Dtype) -> CudaBuffer {
        let size_bytes = (elem_count * dtype.elem_size()) as u64;
        if let Some(buf) = self.pool.take((size_bytes, dtype)) {
            return buf;
        }
        match dtype {
            Dtype::F32 => {
                let slice = self
                    .stream
                    .alloc_zeros::<f32>(elem_count)
                    .expect("[cuda] alloc (f32) failed");
                CudaBuffer::F32(Arc::new(Mutex::new(slice)))
            }
            Dtype::F16 => {
                let slice = self
                    .stream
                    .alloc_zeros::<half::f16>(elem_count)
                    .expect("[cuda] alloc (f16) failed");
                CudaBuffer::F16(Arc::new(Mutex::new(slice)))
            }
            Dtype::Bf16 => {
                let slice = self
                    .stream
                    .alloc_zeros::<half::bf16>(elem_count)
                    .expect("[cuda] alloc (bf16) failed");
                CudaBuffer::Bf16(Arc::new(Mutex::new(slice)))
            }
        }
    }

    fn upload_as(&self, buf: &CudaBuffer, data: &[f32], dtype: Dtype) {
        match dtype {
            Dtype::F32 => {
                let mut g = buf.as_f32().lock().unwrap();
                self.stream
                    .memcpy_htod(data, &mut *g)
                    .expect("[cuda] htod (f32) failed");
            }
            Dtype::F16 => {
                let converted: Vec<half::f16> =
                    data.iter().map(|&x| half::f16::from_f32(x)).collect();
                let mut g = buf.as_f16().lock().unwrap();
                self.stream
                    .memcpy_htod(&converted, &mut *g)
                    .expect("[cuda] htod (f16) failed");
            }
            Dtype::Bf16 => {
                let converted: Vec<half::bf16> =
                    data.iter().map(|&x| half::bf16::from_f32(x)).collect();
                let mut g = buf.as_bf16().lock().unwrap();
                self.stream
                    .memcpy_htod(&converted, &mut *g)
                    .expect("[cuda] htod (bf16) failed");
            }
        }
    }

    fn download_as(&self, buf: &CudaBuffer, dtype: Dtype) -> Vec<f32> {
        match dtype {
            Dtype::F32 => {
                let g = buf.as_f32().lock().unwrap();
                self.stream
                    .clone_dtoh(&*g)
                    .expect("[cuda] dtoh (f32) failed")
            }
            Dtype::F16 => {
                let g = buf.as_f16().lock().unwrap();
                let raw = self
                    .stream
                    .clone_dtoh(&*g)
                    .expect("[cuda] dtoh (f16) failed");
                raw.iter().map(|v| v.to_f32()).collect()
            }
            Dtype::Bf16 => {
                let g = buf.as_bf16().lock().unwrap();
                let raw = self
                    .stream
                    .clone_dtoh(&*g)
                    .expect("[cuda] dtoh (bf16) failed");
                raw.iter().map(|v| v.to_f32()).collect()
            }
        }
    }

    fn free_buffer(&self, _buf: CudaBuffer) {
        // CudaSlice frees device memory on Drop — nothing to do explicitly
    }

    fn recycle(&self, size_bytes: u64, buf: CudaBuffer) {
        let dtype = buf.dtype();
        self.pool.recycle((size_bytes, dtype), buf);
    }

    fn is_sole_owner(buf: &CudaBuffer) -> bool {
        match buf {
            CudaBuffer::F32(a) => Arc::strong_count(a) == 1,
            CudaBuffer::F16(a) => Arc::strong_count(a) == 1,
            CudaBuffer::Bf16(a) => Arc::strong_count(a) == 1,
        }
    }

    fn build_node(
        &self,
        shader: &'static Shader,
        bindings: &[Binding<CudaBuffer>],
        workgroups: [u32; 3],
    ) -> CudaNode {
        let cuda_bindings = bindings
            .iter()
            .map(|b| {
                let cached_meta = if b.mode == TensorMode::Meta {
                    let g = b.buffer.as_f32().lock().unwrap();
                    let data = self
                        .stream
                        .clone_dtoh(&*g)
                        .expect("[cuda] meta dtoh at build time failed");
                    Some(bytemuck::cast_slice::<f32, u8>(&data).to_vec())
                } else {
                    None
                };
                CudaBinding {
                    slot: b.slot,
                    slice: b.buffer.clone(),
                    mode: b.mode,
                    cached_meta,
                }
            })
            .collect();

        CudaNode {
            shader,
            bindings: cuda_bindings,
            workgroups,
        }
    }

    fn execute(&self, nodes: &[CudaNode]) {
        for node in nodes {
            self.dispatch_node(node);
        }
    }

    fn execute_captured(&self, key: usize, nodes: &[CudaNode]) {
        let warmed = {
            let cache = self.graph_cache.lock().unwrap();
            match cache.get(&key) {
                Some(GraphState::Captured(cell)) => {
                    debug_assert_eq!(
                        cell.owner,
                        std::thread::current().id(),
                        "[cuda] captured graph launched from a different thread than it was captured on (CudaGraph is not thread safe)"
                    );
                    cell.graph.launch().expect("[cuda] graph launch failed");
                    return;
                }
                Some(GraphState::Warmed) => true,
                None => false,
            }
        };

        if !warmed {
            self.execute(nodes);
            self.graph_cache
                .lock()
                .unwrap()
                .insert(key, GraphState::Warmed);
            return;
        }

        self.capturing
            .store(true, std::sync::atomic::Ordering::Relaxed);
        self.stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .expect("[cuda] begin_capture failed");
        for node in nodes {
            self.dispatch_node(node);
        }

        let graph = self
            .stream
            .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
            .expect("[cuda] end_capture failed")
            .expect("[cuda] end_capture recorded no graph (nothing was dispatched?)");
        self.capturing
            .store(false, std::sync::atomic::Ordering::Relaxed);

        graph.launch().expect("[cuda] first graph launch failed");

        self.graph_cache.lock().unwrap().insert(
            key,
            GraphState::Captured(GraphCell {
                graph,
                owner: std::thread::current().id(),
            }),
        );
    }

    fn release_captured(&self, key: usize) {
        self.graph_cache.lock().unwrap().remove(&key);
    }

    fn synchronize(&self) {
        self.stream
            .synchronize()
            .expect("[cuda] stream sync failed");
    }
}

/// cargo test --features cuda -- --test-threads=1
#[cfg(test)]
mod gemm_dtype_validation {
    use super::CudaBackend;
    use crate::{builtin, Backend, Binding, ComputeGraph, Dtype, Tensor, TensorMode};
    use std::sync::Arc;

    fn cuda() -> Arc<CudaBackend> {
        Arc::new(CudaBackend::new(0).expect("[cuda] no CUDA device on this machine"))
    }

    fn run_matmul<B: Backend>(ctx: Arc<B>, dtype: Dtype) -> Vec<f32> {
        let a_data = [1.0f32, 2.0, 3.0, 4.0];
        let b_data = [5.0f32, 6.0, 7.0, 8.0];
        let meta_data: [u32; 3] = [2, 2, 2]; // m, n, ki

        let meta = Tensor::init_from_cpu(ctx.clone(), &meta_data);
        let a = Tensor::init_from_cpu_dtype(ctx.clone(), &a_data, dtype);
        let b = Tensor::init_from_cpu_dtype(ctx.clone(), &b_data, dtype);
        let c = Tensor::new_dtype(ctx.clone(), 4, dtype);

        let mut graph = ComputeGraph::new(ctx.clone());
        graph.add_node(
            &builtin::MATMUL,
            &[
                Binding::new(0, &a.buffer, TensorMode::Input),
                Binding::new(1, &b.buffer, TensorMode::Input),
                Binding::new(2, &c.buffer, TensorMode::Output),
                Binding::new(3, &meta.buffer, TensorMode::Meta),
            ],
            [1, 1, 1],
        );
        graph.execute();
        ctx.synchronize();

        c.to_cpu_as(dtype)
    }

    #[test]
    fn f16_matmul_matches_f32_matmul() {
        let ctx = cuda();
        let expected = [19.0f32, 22.0, 43.0, 50.0];

        let f32_result = run_matmul(ctx.clone(), Dtype::F32);
        for (got, want) in f32_result.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 1e-4,
                "f32 path: got {got}, want {want}"
            );
        }

        let f16_result = run_matmul(ctx, Dtype::F16);
        for (got, want) in f16_result.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 1e-2,
                "f16 path diverged from expected: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn bf16_matmul_matches_f32_matmul() {
        let ctx = cuda();
        let expected = [19.0f32, 22.0, 43.0, 50.0];

        let bf16_result = run_matmul(ctx, Dtype::Bf16);
        for (got, want) in bf16_result.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 3e-1,
                "bf16 path diverged from expected: got {got}, want {want}"
            );
        }
    }
}
