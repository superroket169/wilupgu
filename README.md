# wilupgu

A small GPU compute-graph and tensor library for Rust, built to back a
from-scratch transformer training pipeline ([akasha-core](https://github.com/2j87/akasha-core)).
It exposes a single tensor/kernel/graph abstraction over **two interchangeable
GPU backends**:

- **Vulkan** (via [`wgpu`](https://github.com/gfx-rs/wgpu)) — the default, cross-platform backend. Kernels are written in WGSL.
- **CUDA** (via [`cudarc`](https://github.com/coreylowman/cudarc), optional, behind the `cuda` feature) — NVIDIA-only, using cuBLAS for matrix multiplies and NVRTC-compiled CUDA C for everything else.

Both backends implement the *same* named operations, so model code written
against `wilupgu`'s `Tensor`/`ComputeGraph` API runs unmodified on either
backend — switching is a Cargo feature flag, not a code change.

## Why two backends

Vulkan/wgpu is the safe default: it works on any GPU (NVIDIA, AMD, integrated)
without extra system dependencies. CUDA is opt-in for NVIDIA hardware where
cuBLAS's hand-tuned GEMM kernels give a meaningful speedup over generic
compute-shader matrix multiplication — in practice, on an RTX 4050 training a
162M-parameter transformer, CUDA ran at **~80 steps/min vs Vulkan's ~31
steps/min** for the same model and workload (roughly 2.5x faster).

## Core abstractions

- **`Backend`** (`src/backend.rs`) — a trait implemented once by
  `WgpuBackend` (`src/backends/wgpu.rs`) and once by `CudaBackend`
  (`src/backends/cuda.rs`). Everything else in the crate (and in model code
  built on top of it) is generic over `B: Backend`, so it compiles against
  either implementation unchanged.
- **`Tensor<B>`** — a GPU buffer tagged with its backend (`B::Buffer`
  underneath). `Tensor::new` allocates zeroed, `Tensor::init_from_cpu`
  uploads initial data; `to_cpu()`/`copy_from_cpu()` round-trip through host
  memory.
- **Kernels** are referenced purely by name (a plain `&str`, e.g. `"MatMul"`,
  `"RMSNormBwd"`, `"AdamW"`) — see `BuiltInShader` in `src/nn/shaders.rs` for
  the full catalog. Each backend maps that name to its own implementation
  (a WGSL module for Vulkan, CUDA C or a cuBLAS call for CUDA); callers never
  touch shader source directly.
- **`Binding`** — pairs a slot index, a `Tensor`'s buffer, and a
  `TensorMode` (`Input`/`Output`/`InOut`/`Meta`) for one dispatch argument.
  `kernel_layout()` (`src/backend.rs`) is the single source of truth for
  which modes a kernel expects at which slot; `ComputeGraph::add_node`
  validates every binding against it and panics with `"Tensor Mode
  Mismatch"` if a caller gets the mode wrong for a slot — this is checked
  once, generically, for both backends, rather than duplicated per backend.
- **`ComputeGraph<B>`** — an ordered list of dispatches sharing tensor
  bindings. Built once per layer (`add_node` per op), then re-executed every
  training step via `execute()`.
- **`fuse_compute_graphs`** — concatenates several layers' graphs (e.g. every
  transformer block's forward pass, end to end) into one graph, so a full
  model forward or backward pass is a single `execute()` call instead of
  dozens of small ones.

## Building

```bash
# Vulkan only (default, no extra system deps beyond a Vulkan driver)
cargo build --release

# With CUDA support (requires the NVIDIA CUDA Toolkit installed; this crate
# was developed/tested against CUDA 13.3)
cargo build --release --features cuda
```

`CudaBackend::new(0)` returns a `Result`, so a `cuda`-featured binary can
still check for a compatible NVIDIA GPU/driver at runtime and fall back to
`WgpuBackend` if none is found, rather than hard-failing.

## Architecture decisions worth knowing

- **Pipeline/kernel caching.** `WgpuBackend` compiles a shader module + bind
  group layout + pipeline once per unique kernel name and caches it; `CudaBackend`
  NVRTC-compiles each named kernel once and caches the resulting
  `CudaFunction`. A fused model graph reuses the same kernel (e.g.
  `HeadGather`) across every attention head in every layer — thousands of
  nodes — so without this, model construction would create thousands of
  redundant pipeline/compile objects.
- **Vulkan dispatch dimension limit.** Vulkan caps each compute dispatch
  dimension at 65535 workgroups. A 1D dispatch over a large tensor (e.g. a
  50257x768 embedding/lm_head table) exceeds that — a real driver doesn't
  reliably reject this cleanly, it can surface as device loss instead of a
  clean validation error. AdamW's shader (the one place that dispatches over
  a *whole* parameter tensor flattened to 1D) spreads across a 2D grid
  (`groups_x`/`groups_y`) instead; CUDA doesn't need this and ignores the
  extra meta field.
- **Meta-tensor caching (CUDA).** Meta tensors (`TensorMode::Meta`, e.g.
  shape configs) are read once at graph-construction time and cached as raw
  bytes, since they're normally write-once. The exception is AdamW's
  `StepConfig` (carries the live `step`/`lr`, mutated every training step),
  which is always re-read live instead.
- **No atomics/CAS in WGSL, ever.** Backward scatter-adds (e.g.
  `embedding_bwd.wgsl`) use plain non-atomic `+=`, deliberately accepting the
  resulting intra-kernel collision risk on colliding indices in exchange for
  simple, fully readable, zero-overhead WGSL. This is a standing policy, not
  an oversight — don't reintroduce atomics/CAS loops to fix a similar race.

## A real bug worth knowing about (now fixed)

`Linear`'s backward `grad_input` computation (in `akasha-core`, but it
exercises wilupgu's `MatMulTrp` kernel) used to pass the *forward* pass's
`[M, N, K]` meta tensor into the backward `grad_input` dispatch. `MatMulTrp`
computes `C[M,N] = A[M,K] @ B^T` where `B` must be stored `[N,K]` — but
`grad_input`'s actual contraction needs `N`/`K` swapped relative to the
forward meta. Both backends "faithfully" executed the mislabeled dispatch,
but via different mechanics (cuBLAS's column-major reinterpretation vs WGSL's
direct strided indexing), so they silently produced *different* wrong
results rather than consistently-wrong-but-matching ones. This corrupted the
gradient flowing back from the output head into the rest of the network on
*every* training step, on both backends, and was the actual root cause of a
training run that never converged properly. Confirmed fixed via a tiny
single-layer memorization test (loss → 0.0000 on both backends from
identical seeded initial weights) — see `akasha-core/src/bin/diagnose.rs`
CHECK 8.

## License

No license file is currently present in this repository.
