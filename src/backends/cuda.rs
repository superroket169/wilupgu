use crate::backend::{Backend, Binding, TensorMode};
use crate::builtin::cuda_kernels as k;
use crate::pool::BufferPool;
use crate::shader::{CudaShape, Shader};
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::result::DriverError;
use cudarc::driver::{
    CudaContext as CuDevice, CudaFunction, CudaStream, LaunchConfig, PushKernelArg,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub type CudaBuffer = Arc<Mutex<cudarc::driver::CudaSlice<f32>>>;

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

pub struct CudaBackend {
    device: Arc<CuDevice>,
    pub stream: Arc<CudaStream>,
    blas: CudaBlas,
    kernel_cache: Mutex<HashMap<usize, CudaFunction>>,
    pool: BufferPool<CudaBuffer>,
}

impl CudaBackend {
    pub fn new(ordinal: usize) -> Result<Self, DriverError> {
        let device = CuDevice::new(ordinal)?;
        let stream = device.default_stream();
        let blas = CudaBlas::new(stream.clone()).map_err(|e| {
            eprintln!("[cuda] cuBLAS init failed: {e:?}");
            DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN)
        })?;
        Ok(Self {
            device,
            stream,
            blas,
            kernel_cache: Mutex::new(HashMap::new()),
            pool: BufferPool::new(),
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

macro_rules! launch {
    ($self:expr, $f:expr, $cfg:expr, $($arg:expr),+ $(,)?) => {{
        let mut b = $self.stream.launch_builder(&$f);
        $(b.arg($arg);)+
        unsafe { b.launch($cfg) }.expect("[cuda] kernel launch failed")
    }};
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

fn live_meta_bytes(b: &CudaBinding, backend: &CudaBackend) -> Vec<u8> {
    let g = b.slice.lock().unwrap();
    let data = backend
        .stream
        .clone_dtoh(&*g)
        .expect("live meta dtoh failed");
    bytemuck::cast_slice::<f32, u8>(&data).to_vec()
}

fn n_elems(b: &CudaBinding) -> u32 {
    b.slice.lock().unwrap().len() as u32
}

fn cfg_1d(n: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n + 255) / 256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

impl CudaBackend {
    fn gemm_matmul(&self, bindings: &[CudaBinding], transpose_b: bool, beta: f32) {
        let a = find(bindings, 0);
        let b = find(bindings, 1);
        let c = find(bindings, 2);
        let dims = meta_u32(find(bindings, 3));
        let (m, n, ki) = (dims[0], dims[1], dims[2]);

        let ag = a.slice.lock().unwrap();
        let bg = b.slice.lock().unwrap();
        let mut cg = c.slice.lock().unwrap();

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
            alpha: 1.0,
            lda: ldb as i32,
            ldb: ki as i32,
            beta,
            ldc: n as i32,
        };

        unsafe {
            self.blas
                .gemm(cfg, &*bg, &*ag, &mut *cg)
                .expect("[cuda] sgemm failed")
        }
    }

    fn gemm_weight_bwd(&self, bindings: &[CudaBinding]) {
        let a = find(bindings, 0);
        let dc = find(bindings, 1);
        let db = find(bindings, 2);
        let dims = meta_u32(find(bindings, 3));
        let (m, n, ki) = (dims[0], dims[1], dims[2]);

        let ag = a.slice.lock().unwrap();
        let dcg = dc.slice.lock().unwrap();
        let mut dbg = db.slice.lock().unwrap();

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

    // -------- Elementwise --------

    pub fn launch_inout_1(&self, bindings: &[CudaBinding], key: usize, src: &str, func: &str) {
        let x = find(bindings, 0);
        let n = n_elems(x);
        let f = self.compile(key, src, func);
        let mut g = x.slice.lock().unwrap();
        launch!(self, f, cfg_1d(n), &mut *g, &n);
    }

    pub fn launch_in2_out1(&self, bindings: &[CudaBinding], key: usize, src: &str, func: &str) {
        let x = find(bindings, 0);
        let dy = find(bindings, 1);
        let dx = find(bindings, 2);
        let n = n_elems(x);
        let f = self.compile(key, src, func);
        let xg = x.slice.lock().unwrap();
        let dyg = dy.slice.lock().unwrap();
        let mut dxg = dx.slice.lock().unwrap();
        launch!(self, f, cfg_1d(n), &*xg, &*dyg, &mut *dxg, &n);
    }

    pub fn launch_add(&self, bindings: &[CudaBinding], key: usize, src: &str, func: &str) {
        let x = find(bindings, 0);
        let r = find(bindings, 1);
        let n = n_elems(x);
        let f = self.compile(key, src, func);
        let mut xg = x.slice.lock().unwrap();
        let rg = r.slice.lock().unwrap();
        launch!(self, f, cfg_1d(n), &mut *xg, &*rg, &n);
    }

    // -------- Structured ----------

    pub fn launch_embedding(
        &self,
        bindings: &[CudaBinding],
        key: usize,
        src: &str,
        func: &str,
        wg: [u32; 3],
    ) {
        let dims = meta_u32(find(bindings, 3));
        let (vocab, embed, seq) = (dims[0], dims[1], dims[2]);
        let f = self.compile(key, src, func);
        let g0 = find(bindings, 0).slice.lock().unwrap();
        let g1 = find(bindings, 1).slice.lock().unwrap();
        let mut g2 = find(bindings, 2).slice.lock().unwrap();
        let cfg = LaunchConfig {
            grid_dim: (wg[0].max(1), seq.max(1), 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch!(self, f, cfg, &*g0, &*g1, &mut *g2, &vocab, &embed, &seq);
    }

    fn launch_causal_mask(&self, bindings: &[CudaBinding], key: usize) {
        let bytes = meta_bytes(find(bindings, 1));
        let seq_len = u32::from_ne_bytes(bytes[0..4].try_into().unwrap());
        let scale = f32::from_ne_bytes(bytes[4..8].try_into().unwrap());
        let f = self.compile(key, k::CAUSAL_MASK, "causal_mask_kernel");
        let mut g = find(bindings, 0).slice.lock().unwrap();
        let grid = (seq_len + 15) / 16;
        let cfg = LaunchConfig {
            grid_dim: (grid, grid, 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };
        launch!(self, f, cfg, &mut *g, &seq_len, &scale);
    }

    pub fn launch_causal_softmax(
        &self,
        bindings: &[CudaBinding],
        key: usize,
        src: &str,
        func: &str,
    ) {
        let bytes = meta_bytes(find(bindings, 1));
        let seq_len = u32::from_ne_bytes(bytes[0..4].try_into().unwrap());
        let scale = f32::from_ne_bytes(bytes[4..8].try_into().unwrap());
        let f = self.compile(key, src, func);
        let mut g = find(bindings, 0).slice.lock().unwrap();
        launch!(self, f, cfg_1d(seq_len), &mut *g, &seq_len, &scale);
    }

    pub fn launch_head_move(&self, bindings: &[CudaBinding], key: usize, src: &str, func: &str) {
        let dims = meta_u32(find(bindings, 2));
        let (seq, full_dim, head_dim, offset) = (dims[0], dims[1], dims[2], dims[3]);
        let f = self.compile(key, src, func);
        let fg = find(bindings, 0).slice.lock().unwrap();
        let mut tg = find(bindings, 1).slice.lock().unwrap();
        let grid = ((head_dim + 15) / 16).max(1);
        let grid_y = ((seq + 15) / 16).max(1);
        let cfg = LaunchConfig {
            grid_dim: (grid, grid_y, 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };
        launch!(self, f, cfg, &*fg, &mut *tg, &seq, &full_dim, &head_dim, &offset);
    }

    pub fn launch_rope(&self, bindings: &[CudaBinding], key: usize, src: &str, func: &str) {
        let dims = meta_u32(find(bindings, 1));
        let (seq, dim, head_dim) = (dims[0], dims[1], dims[2]);
        let f = self.compile(key, src, func);
        let mut g = find(bindings, 0).slice.lock().unwrap();
        let gx = ((head_dim / 2 + 15) / 16).max(1);
        let gy = ((seq + 15) / 16).max(1);
        let cfg = LaunchConfig {
            grid_dim: (gx, gy, 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };
        launch!(self, f, cfg, &mut *g, &seq, &dim, &head_dim);
    }

    pub fn launch_softmax(&self, bindings: &[CudaBinding], key: usize, src: &str, func: &str) {
        let seq = meta_u32(find(bindings, 1))[0];
        let f = self.compile(key, src, func);
        let mut g = find(bindings, 0).slice.lock().unwrap();
        launch!(self, f, cfg_1d(seq), &mut *g, &seq);
    }

    pub fn launch_softmax_bwd(&self, bindings: &[CudaBinding], key: usize, src: &str, func: &str) {
        let bytes = meta_bytes(find(bindings, 3));
        let seq_len = u32::from_ne_bytes(bytes[0..4].try_into().unwrap());
        let scale = f32::from_ne_bytes(bytes[4..8].try_into().unwrap());
        let f = self.compile(key, src, func);
        let yg = find(bindings, 0).slice.lock().unwrap();
        let dyg = find(bindings, 1).slice.lock().unwrap();
        let mut dxg = find(bindings, 2).slice.lock().unwrap();
        launch!(
            self,
            f,
            cfg_1d(seq_len),
            &*yg,
            &*dyg,
            &mut *dxg,
            &seq_len,
            &scale
        );
    }

    pub fn launch_rmsnorm(&self, bindings: &[CudaBinding], key: usize, src: &str, func: &str) {
        let bytes = meta_bytes(find(bindings, 3));
        let seq_len = u32::from_ne_bytes(bytes[0..4].try_into().unwrap());
        let size = u32::from_ne_bytes(bytes[4..8].try_into().unwrap());
        let eps = f32::from_ne_bytes(bytes[8..12].try_into().unwrap());
        let f = self.compile(key, src, func);
        let xg = find(bindings, 0).slice.lock().unwrap();
        let wg = find(bindings, 1).slice.lock().unwrap();
        let mut og = find(bindings, 2).slice.lock().unwrap();
        let cfg = LaunchConfig {
            grid_dim: (seq_len.max(1), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch!(self, f, cfg, &*xg, &*wg, &mut *og, &seq_len, &size, &eps);
    }

    pub fn launch_rmsnorm_bwd(&self, bindings: &[CudaBinding], key: usize, src: &str, func: &str) {
        let bytes = meta_bytes(find(bindings, 5));
        let seq_len = u32::from_ne_bytes(bytes[0..4].try_into().unwrap());
        let size = u32::from_ne_bytes(bytes[4..8].try_into().unwrap());
        let eps = f32::from_ne_bytes(bytes[8..12].try_into().unwrap());
        let f = self.compile(key, src, func);
        let dyg = find(bindings, 0).slice.lock().unwrap();
        let xg = find(bindings, 1).slice.lock().unwrap();
        let wg = find(bindings, 2).slice.lock().unwrap();
        let mut dxg = find(bindings, 3).slice.lock().unwrap();
        let mut rsg = find(bindings, 4).slice.lock().unwrap();
        let cfg = LaunchConfig {
            grid_dim: (seq_len.max(1), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch!(self, f, cfg, &*dyg, &*xg, &*wg, &mut *dxg, &mut *rsg, &seq_len, &size, &eps);
    }

    pub fn launch_rmsnorm_weight_bwd(
        &self,
        bindings: &[CudaBinding],
        key: usize,
        src: &str,
        func: &str,
    ) {
        let dims = meta_u32(find(bindings, 4));
        let (seq, size) = (dims[0], dims[1]);
        let f = self.compile(key, src, func);
        let dyg = find(bindings, 0).slice.lock().unwrap();
        let xg = find(bindings, 1).slice.lock().unwrap();
        let rsg = find(bindings, 2).slice.lock().unwrap();
        let mut dwg = find(bindings, 3).slice.lock().unwrap();
        launch!(
            self,
            f,
            cfg_1d(size),
            &*dyg,
            &*xg,
            &*rsg,
            &mut *dwg,
            &seq,
            &size
        );
    }

    pub fn launch_cross_entropy(
        &self,
        bindings: &[CudaBinding],
        key: usize,
        src: &str,
        func: &str,
    ) {
        let dims = meta_u32(find(bindings, 4));
        let (vocab, rows) = (dims[0], dims[1]);
        let f = self.compile(key, src, func);
        let lg = find(bindings, 0).slice.lock().unwrap();
        let tg = find(bindings, 1).slice.lock().unwrap();
        let mut pg = find(bindings, 2).slice.lock().unwrap();
        let mut losg = find(bindings, 3).slice.lock().unwrap();
        launch!(
            self,
            f,
            cfg_1d(rows),
            &*lg,
            &*tg,
            &mut *pg,
            &mut *losg,
            &vocab,
            &rows
        );
    }

    pub fn launch_cross_entropy_bwd(
        &self,
        bindings: &[CudaBinding],
        key: usize,
        src: &str,
        func: &str,
    ) {
        let dims = meta_u32(find(bindings, 4));
        let (vocab, rows) = (dims[0], dims[1]);
        let f = self.compile(key, src, func);
        let pg = find(bindings, 0).slice.lock().unwrap();
        let tg = find(bindings, 1).slice.lock().unwrap();
        let dlg = find(bindings, 2).slice.lock().unwrap();
        let mut dlogg = find(bindings, 3).slice.lock().unwrap();
        launch!(
            self,
            f,
            cfg_1d(rows),
            &*pg,
            &*tg,
            &*dlg,
            &mut *dlogg,
            &vocab,
            &rows
        );
    }

    pub fn launch_softmax_rect(&self, bindings: &[CudaBinding], key: usize, src: &str, func: &str) {
        let bytes = meta_bytes(find(bindings, 1));
        let num_rows = u32::from_ne_bytes(bytes[0..4].try_into().unwrap());
        let width = u32::from_ne_bytes(bytes[4..8].try_into().unwrap());
        let scale = f32::from_ne_bytes(bytes[8..12].try_into().unwrap());
        let f = self.compile(key, src, func);
        let mut g = find(bindings, 0).slice.lock().unwrap();
        launch!(
            self,
            f,
            cfg_1d(num_rows),
            &mut *g,
            &num_rows,
            &width,
            &scale
        );
    }

    pub fn launch_cache_write(&self, bindings: &[CudaBinding], key: usize, src: &str, func: &str) {
        let dims = meta_u32(find(bindings, 2));
        let (row_count, width, dst_row_offset) = (dims[0], dims[1], dims[2]);
        let f = self.compile(key, src, func);
        let sg = find(bindings, 0).slice.lock().unwrap();
        let mut dg = find(bindings, 1).slice.lock().unwrap();
        let grid_x = ((width + 15) / 16).max(1);
        let grid_y = ((row_count + 15) / 16).max(1);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };
        launch!(
            self,
            f,
            cfg,
            &*sg,
            &mut *dg,
            &row_count,
            &width,
            &dst_row_offset
        );
    }

    pub fn launch_rope_offset(&self, bindings: &[CudaBinding], key: usize, src: &str, func: &str) {
        let dims = meta_u32(find(bindings, 1));
        let (seq, dim, head_dim, pos_offset) = (dims[0], dims[1], dims[2], dims[3]);
        let f = self.compile(key, src, func);
        let mut g = find(bindings, 0).slice.lock().unwrap();
        let gx = ((head_dim / 2 + 15) / 16).max(1);
        let gy = ((seq + 15) / 16).max(1);
        let cfg = LaunchConfig {
            grid_dim: (gx, gy, 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };
        launch!(self, f, cfg, &mut *g, &seq, &dim, &head_dim, &pos_offset);
    }

    fn launch_adamw(&self, bindings: &[CudaBinding], key: usize) {
        let size = meta_u32(find(bindings, 4))[0];
        // AdamW cfg changes every step — must read live, not from cached_meta
        let bytes = live_meta_bytes(find(bindings, 5), self);
        let step = u32::from_ne_bytes(bytes[0..4].try_into().unwrap());
        let lr = f32::from_ne_bytes(bytes[4..8].try_into().unwrap());
        let beta1 = f32::from_ne_bytes(bytes[8..12].try_into().unwrap());
        let beta2 = f32::from_ne_bytes(bytes[12..16].try_into().unwrap());
        let eps = f32::from_ne_bytes(bytes[16..20].try_into().unwrap());
        let wd = f32::from_ne_bytes(bytes[20..24].try_into().unwrap());

        let f = self.compile(key, k::ADAMW, "adamw_kernel");
        let mut wg = find(bindings, 0).slice.lock().unwrap();
        let gg = find(bindings, 1).slice.lock().unwrap();
        let mut mg = find(bindings, 2).slice.lock().unwrap();
        let mut vg = find(bindings, 3).slice.lock().unwrap();
        launch!(
            self,
            f,
            cfg_1d(size),
            &mut *wg,
            &*gg,
            &mut *mg,
            &mut *vg,
            &size,
            &step,
            &lr,
            &beta1,
            &beta2,
            &eps,
            &wd
        );
    }
}

fn shader_key(shader: &'static Shader) -> usize {
    shader as *const Shader as usize
}

// ==========================================================================
//                          Custom-shape dispatches
// ==========================================================================

fn custom_matmul(_s: &'static Shader, b: &CudaBackend, bindings: &[CudaBinding], _wg: [u32; 3]) {
    b.gemm_matmul(bindings, false, 0.0)
}

fn custom_matmul_trp(
    _s: &'static Shader,
    b: &CudaBackend,
    bindings: &[CudaBinding],
    _wg: [u32; 3],
) {
    b.gemm_matmul(bindings, true, 0.0)
}

fn custom_matmul_add(
    _s: &'static Shader,
    b: &CudaBackend,
    bindings: &[CudaBinding],
    _wg: [u32; 3],
) {
    b.gemm_matmul(bindings, false, 1.0)
}

fn custom_matmul_weight_bwd(
    _s: &'static Shader,
    b: &CudaBackend,
    bindings: &[CudaBinding],
    _wg: [u32; 3],
) {
    b.gemm_weight_bwd(bindings)
}

fn custom_causal_mask(
    s: &'static Shader,
    b: &CudaBackend,
    bindings: &[CudaBinding],
    _wg: [u32; 3],
) {
    b.launch_causal_mask(bindings, shader_key(s))
}

fn custom_adamw(s: &'static Shader, b: &CudaBackend, bindings: &[CudaBinding], _wg: [u32; 3]) {
    b.launch_adamw(bindings, shader_key(s))
}

pub(crate) mod dispatch {
    pub(crate) use super::{
        custom_adamw, custom_causal_mask, custom_matmul, custom_matmul_add, custom_matmul_trp,
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
        if let Some(buf) = self.pool.take(size_bytes) {
            return buf;
        }
        let n = (size_bytes as usize) / std::mem::size_of::<f32>();
        let slice = self
            .stream
            .alloc_zeros::<f32>(n)
            .expect("[cuda] alloc failed");
        Arc::new(Mutex::new(slice))
    }

    fn alloc_from_cpu<T: bytemuck::Pod>(&self, data: &[T]) -> CudaBuffer {
        let f32s: &[f32] = bytemuck::cast_slice(data);
        let size_bytes = (f32s.len() * std::mem::size_of::<f32>()) as u64;
        if let Some(buf) = self.pool.take(size_bytes) {
            let mut g = buf.lock().unwrap();
            self.stream
                .memcpy_htod(f32s, &mut *g)
                .expect("[cuda] htod copy (recycled) failed");
            drop(g);
            return buf;
        }
        let slice = self.stream.clone_htod(f32s).expect("[cuda] htod failed");
        Arc::new(Mutex::new(slice))
    }

    fn copy_from_cpu<T: bytemuck::Pod>(&self, buf: &CudaBuffer, data: &[T]) {
        let f32s: &[f32] = bytemuck::cast_slice(data);
        let mut g = buf.lock().unwrap();
        self.stream
            .memcpy_htod(f32s, &mut *g)
            .expect("[cuda] htod copy failed");
    }

    fn copy_to_cpu<T: bytemuck::Pod + Default + Clone>(&self, buf: &CudaBuffer) -> Vec<T> {
        let g = buf.lock().unwrap();
        let f32s = self.stream.clone_dtoh(&*g).expect("[cuda] dtoh failed");
        bytemuck::cast_slice::<f32, T>(&f32s).to_vec()
    }

    fn free_buffer(&self, _buf: CudaBuffer) {
        // CudaSlice frees device memory on Drop — nothing to do explicitly
    }

    fn recycle(&self, size_bytes: u64, buf: CudaBuffer) {
        self.pool.recycle(size_bytes, buf);
    }

    fn is_sole_owner(buf: &CudaBuffer) -> bool {
        Arc::strong_count(buf) == 1
    }

    fn build_node(
        &self,
        shader: &'static Shader,
        bindings: &[Binding<CudaBuffer>],
        workgroups: [u32; 3],
    ) -> CudaNode {
        let is_adamw = std::ptr::eq(shader, &crate::builtin::ADAMW);
        let cuda_bindings = bindings
            .iter()
            .map(|b| {
                let is_live = is_adamw && b.slot == 5;
                let cached_meta = if b.mode == TensorMode::Meta && !is_live {
                    let g = b.buffer.lock().unwrap();
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
            let b = &node.bindings;
            let wg = node.workgroups;
            let spec =
                node.shader.cuda.as_ref().unwrap_or_else(|| {
                    panic!("[cuda] shader `{}` has no cuda impl", node.shader.name)
                });
            let key = shader_key(node.shader);
            match &spec.shape {
                CudaShape::InOut1 => self.launch_inout_1(b, key, spec.src, spec.entry),
                CudaShape::In2Out1 => self.launch_in2_out1(b, key, spec.src, spec.entry),
                CudaShape::Add => self.launch_add(b, key, spec.src, spec.entry),
                CudaShape::Custom(f) => f(node.shader, self, b, wg),
            }
        }
    }

    fn synchronize(&self) {
        self.stream
            .synchronize()
            .expect("[cuda] stream sync failed");
    }
}
