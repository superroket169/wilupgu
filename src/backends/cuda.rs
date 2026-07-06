use crate::backend::{Backend, Binding, Dtype, TensorMode};
use crate::builtin::cuda_kernels as k;
use crate::pool::BufferPool;
use crate::shader::{CudaShape, Shader};
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

pub struct CudaBackend {
    device: Arc<CuDevice>,
    pub stream: Arc<CudaStream>,
    blas: CudaBlas,
    kernel_cache: Mutex<HashMap<usize, CudaFunction>>,
    pool: BufferPool<CudaBuffer, (u64, Dtype)>,
    graph_cache: Mutex<HashMap<usize, CudaGraph>>,
}

impl CudaBackend {
    pub fn new(ordinal: usize) -> Result<Self, DriverError> {
        let device = CuDevice::new(ordinal)?;
        let stream = device.default_stream();
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
//                  MACROS
// ========================================

macro_rules! launch {
    ($self:expr, $f:expr, $cfg:expr, $($arg:expr),+ $(,)?) => {{
        let mut b = $self.stream.launch_builder(&$f);
        $(b.arg($arg);)+
        unsafe { b.launch($cfg) }.expect("[cuda] kernel launch failed")
    }};
}

macro_rules! read_meta {
    ($bytes:expr, $($field:ident : $ty:ty),+ $(,)?) => {
        let __bytes = $bytes;
        let mut __off = 0usize;
        $(
            let $field: $ty = <$ty>::from_ne_bytes(
                __bytes[__off..__off + std::mem::size_of::<$ty>()].try_into().unwrap()
            );
            #[allow(unused_assignments)]
            { __off += std::mem::size_of::<$ty>(); }
        )+
    };
}

macro_rules! define_launch {
    (
        $name:ident,
        $(meta_slot: $meta_slot:expr, meta: [$($mf:ident : $mty:ty),* $(,)?],)?
        buffers: [$($bkind:ident $bname:ident : $bslot:expr),* $(,)?],
        $(let: [$($lname:ident = $lexpr:expr),* $(,)?],)?
        grid: $grid:expr,
        launch: [$($largs:expr),* $(,)?]
    ) => {
        pub fn $name(&self, bindings: &[CudaBinding], key: usize, src: &str, func: &str) {
            $( read_meta!(meta_bytes(find(bindings, $meta_slot)), $($mf : $mty),*); )?
            $(
                define_launch!(@lock $bkind $bname, bindings, $bslot);
            )*
            $( $( let $lname = $lexpr; )* )?
            let f = self.compile(key, src, func);
            let cfg = $grid;
            launch!(self, f, cfg, $($largs),*);
        }
    };
    (@lock mut $name:ident, $bindings:ident, $slot:expr) => {
        let mut $name = find($bindings, $slot).slice.as_f32().lock().unwrap();
    };
    (@lock ro $name:ident, $bindings:ident, $slot:expr) => {
        let $name = find($bindings, $slot).slice.as_f32().lock().unwrap();
    };
}

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

    fn gemm_matmul(&self, bindings: &[CudaBinding], transpose_b: bool, beta: f32) {
        let a = find(bindings, 0);
        let b = find(bindings, 1);
        let c = find(bindings, 2);
        let dims = meta_u32(find(bindings, 3));
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
        let dims = meta_u32(find(bindings, 3));
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

    // -------- Elementwise --------

    define_launch!(
        launch_inout_1,
        buffers: [mut g: 0],
        let: [n = g.len() as u32],
        grid: cfg_1d(n),
        launch: [&mut *g, &n]
    );

    define_launch!(
        launch_in2_out1,
        buffers: [ro xg: 0, ro dyg: 1, mut dxg: 2],
        let: [n = xg.len() as u32],
        grid: cfg_1d(n),
        launch: [&*xg, &*dyg, &mut *dxg, &n]
    );

    define_launch!(
        launch_add,
        buffers: [mut xg: 0, ro rg: 1],
        let: [n = xg.len() as u32],
        grid: cfg_1d(n),
        launch: [&mut *xg, &*rg, &n]
    );

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

        let g0 = find(bindings, 0).slice.as_f32().lock().unwrap();
        let g1 = find(bindings, 1).slice.as_f32().lock().unwrap();
        let mut g2 = find(bindings, 2).slice.as_f32().lock().unwrap();

        let cfg = LaunchConfig {
            grid_dim: (wg[0].max(1), seq.max(1), 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch!(self, f, cfg, &*g0, &*g1, &mut *g2, &vocab, &embed, &seq);
    }

    define_launch!(
        launch_causal_mask,
        meta_slot: 1, meta: [seq_len: u32, scale: f32],
        buffers: [mut g: 0],
        let: [grid = (seq_len + 15) / 16],
        grid: LaunchConfig { grid_dim: (grid, grid, 1), block_dim: (16, 16, 1), shared_mem_bytes: 0 },
        launch: [&mut *g, &seq_len, &scale]
    );

    define_launch!(
        launch_causal_softmax,
        meta_slot: 1, meta: [seq_len: u32, scale: f32],
        buffers: [mut g: 0],
        grid: cfg_1d(seq_len),
        launch: [&mut *g, &seq_len, &scale]
    );

    define_launch!(
        launch_head_move,
        meta_slot: 2, meta: [seq: u32, full_dim: u32, head_dim: u32, offset: u32],
        buffers: [ro fg: 0, mut tg: 1],
        grid: LaunchConfig {
            grid_dim: (((head_dim + 15) / 16).max(1), ((seq + 15) / 16).max(1), 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        },
        launch: [&*fg, &mut *tg, &seq, &full_dim, &head_dim, &offset]
    );

    define_launch!(
        launch_rope,
        meta_slot: 1, meta: [seq: u32, dim: u32, head_dim: u32],
        buffers: [mut g: 0],
        grid: LaunchConfig {
            grid_dim: (((head_dim / 2 + 15) / 16).max(1), ((seq + 15) / 16).max(1), 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        },
        launch: [&mut *g, &seq, &dim, &head_dim]
    );

    define_launch!(
        launch_softmax,
        meta_slot: 1, meta: [seq: u32],
        buffers: [mut g: 0],
        grid: cfg_1d(seq),
        launch: [&mut *g, &seq]
    );

    define_launch!(
        launch_softmax_bwd,
        meta_slot: 3, meta: [seq_len: u32, scale: f32],
        buffers: [ro yg: 0, ro dyg: 1, mut dxg: 2],
        grid: cfg_1d(seq_len),
        launch: [&*yg, &*dyg, &mut *dxg, &seq_len, &scale]
    );

    define_launch!(
        launch_rmsnorm,
        meta_slot: 3, meta: [seq_len: u32, size: u32, eps: f32],
        buffers: [ro xg: 0, ro wg: 1, mut og: 2],
        grid: LaunchConfig { grid_dim: (seq_len.max(1), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
        launch: [&*xg, &*wg, &mut *og, &seq_len, &size, &eps]
    );

    define_launch!(
        launch_rmsnorm_bwd,
        meta_slot: 5, meta: [seq_len: u32, size: u32, eps: f32],
        buffers: [ro dyg: 0, ro xg: 1, ro wg: 2, mut dxg: 3, mut rsg: 4],
        grid: LaunchConfig { grid_dim: (seq_len.max(1), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
        launch: [&*dyg, &*xg, &*wg, &mut *dxg, &mut *rsg, &seq_len, &size, &eps]
    );

    define_launch!(
        launch_rmsnorm_weight_bwd,
        meta_slot: 4, meta: [seq: u32, size: u32],
        buffers: [ro dyg: 0, ro xg: 1, ro rsg: 2, mut dwg: 3],
        grid: cfg_1d(size),
        launch: [&*dyg, &*xg, &*rsg, &mut *dwg, &seq, &size]
    );

    define_launch!(
        launch_cross_entropy,
        meta_slot: 4, meta: [vocab: u32, rows: u32],
        buffers: [ro lg: 0, ro tg: 1, mut pg: 2, mut losg: 3],
        grid: cfg_1d(rows),
        launch: [&*lg, &*tg, &mut *pg, &mut *losg, &vocab, &rows]
    );

    define_launch!(
        launch_cross_entropy_bwd,
        meta_slot: 4, meta: [vocab: u32, rows: u32],
        buffers: [ro pg: 0, ro tg: 1, ro dlg: 2, mut dlogg: 3],
        grid: cfg_1d(rows),
        launch: [&*pg, &*tg, &*dlg, &mut *dlogg, &vocab, &rows]
    );

    define_launch!(
        launch_softmax_rect,
        meta_slot: 1, meta: [num_rows: u32, width: u32, scale: f32],
        buffers: [mut g: 0],
        grid: cfg_1d(num_rows),
        launch: [&mut *g, &num_rows, &width, &scale]
    );

    define_launch!(
        launch_cache_write,
        meta_slot: 2, meta: [row_count: u32, width: u32, dst_row_offset: u32],
        buffers: [ro sg: 0, mut dg: 1],
        grid: LaunchConfig {
            grid_dim: (((width + 15) / 16).max(1), ((row_count + 15) / 16).max(1), 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        },
        launch: [&*sg, &mut *dg, &row_count, &width, &dst_row_offset]
    );

    define_launch!(
        launch_rope_offset,
        meta_slot: 1, meta: [seq: u32, dim: u32, head_dim: u32, pos_offset: u32],
        buffers: [mut g: 0],
        grid: LaunchConfig {
            grid_dim: (((head_dim / 2 + 15) / 16).max(1), ((seq + 15) / 16).max(1), 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        },
        launch: [&mut *g, &seq, &dim, &head_dim, &pos_offset]
    );

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

    define_launch!(
        launch_rope_qk,
        meta_slot: 2, meta: [seq: u32, dim: u32, head_dim: u32],
        buffers: [mut qg: 0, mut kg: 1],
        grid: LaunchConfig {
            grid_dim: (((head_dim / 2 + 15) / 16).max(1), ((seq + 15) / 16).max(1), 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        },
        launch: [&mut *qg, &mut *kg, &seq, &dim, &head_dim]
    );

    define_launch!(
        launch_qkv_split,
        meta_slot: 4, meta: [seq: u32, full_dim: u32, head_dim: u32, head_offset: u32],
        buffers: [ro sg: 0, mut qg: 1, mut kg: 2, mut vg: 3],
        grid: LaunchConfig {
            grid_dim: (((head_dim + 15) / 16).max(1), ((seq + 15) / 16).max(1), 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        },
        launch: [&*sg, &mut *qg, &mut *kg, &mut *vg, &seq, &full_dim, &head_dim, &head_offset]
    );

    define_launch!(
        launch_qkv_scatter,
        meta_slot: 4, meta: [seq: u32, full_dim: u32, head_dim: u32, head_offset: u32],
        buffers: [ro qg: 0, ro kg: 1, ro vg: 2, mut dg: 3],
        grid: LaunchConfig {
            grid_dim: (((head_dim + 15) / 16).max(1), ((seq + 15) / 16).max(1), 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        },
        launch: [&*qg, &*kg, &*vg, &mut *dg, &seq, &full_dim, &head_dim, &head_offset]
    );

    define_launch!(
        launch_flash_attention,
        meta_slot: 5, meta: [seq_len: u32, dim: u32, head_dim: u32, scale: f32],
        buffers: [ro qg: 0, ro kg: 1, ro vg: 2, mut og: 3, mut lg: 4],
        grid: LaunchConfig {
            grid_dim: (((seq_len + 63) / 64).max(1), (dim / head_dim).max(1), 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        },
        launch: [&*qg, &*kg, &*vg, &mut *og, &mut *lg, &seq_len, &dim, &head_dim, &scale]
    );

    define_launch!(
        launch_flash_attention_bwd_dq,
        meta_slot: 7, meta: [seq_len: u32, dim: u32, head_dim: u32, scale: f32],
        buffers: [ro qg: 0, ro kg: 1, ro vg: 2, ro og: 3, ro dog: 4, ro lg: 5, mut dqg: 6],
        grid: LaunchConfig {
            grid_dim: (((seq_len + 63) / 64).max(1), (dim / head_dim).max(1), 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        },
        launch: [&*qg, &*kg, &*vg, &*og, &*dog, &*lg, &mut *dqg, &seq_len, &dim, &head_dim, &scale]
    );

    define_launch!(
        launch_flash_attention_bwd_dkdv,
        meta_slot: 8, meta: [seq_len: u32, dim: u32, head_dim: u32, scale: f32],
        buffers: [ro qg: 0, ro kg: 1, ro vg: 2, ro og: 3, ro dog: 4, ro lg: 5, mut dkg: 6, mut dvg: 7],
        grid: LaunchConfig {
            grid_dim: (((seq_len + 63) / 64).max(1), (dim / head_dim).max(1), 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        },
        launch: [&*qg, &*kg, &*vg, &*og, &*dog, &*lg, &mut *dkg, &mut *dvg, &seq_len, &dim, &head_dim, &scale]
    );

    fn dispatch_node(&self, node: &CudaNode) {
        let b = &node.bindings;
        let wg = node.workgroups;

        let spec = node
            .shader
            .cuda
            .as_ref()
            .unwrap_or_else(|| panic!("[cuda] shader `{}` has no cuda impl", node.shader.name));
        let key = shader_key(node.shader);

        match &spec.shape {
            CudaShape::InOut1 => self.launch_inout_1(b, key, spec.src, spec.entry),
            CudaShape::In2Out1 => self.launch_in2_out1(b, key, spec.src, spec.entry),
            CudaShape::Add => self.launch_add(b, key, spec.src, spec.entry),
            CudaShape::Custom(f) => f(node.shader, self, b, wg),
        }
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
    b.launch_causal_mask(
        bindings,
        shader_key(s),
        k::CAUSAL_MASK,
        "causal_mask_kernel",
    )
}

fn custom_adamw(s: &'static Shader, b: &CudaBackend, bindings: &[CudaBinding], _wg: [u32; 3]) {
    b.launch_adamw(bindings, shader_key(s))
}

fn custom_adamw_schedule(
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
        custom_adamw, custom_adamw_schedule, custom_causal_mask, custom_matmul, custom_matmul_add,
        custom_matmul_trp, custom_matmul_weight_bwd,
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
        {
            let cache = self.graph_cache.lock().unwrap();
            if let Some(graph) = cache.get(&key) {
                graph.launch().expect("[cuda] graph launch failed");
                return;
            }
        }

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
        graph.launch().expect("[cuda] first graph launch failed");
        self.graph_cache.lock().unwrap().insert(key, graph);
    }

    fn synchronize(&self) {
        self.stream
            .synchronize()
            .expect("[cuda] stream sync failed");
    }
}

/// cargo test --features cuda -- --test-threads=1
#[cfg(test)]
mod f16_gemm_validation {
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
}
