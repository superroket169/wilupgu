# wilupgu

Wilupgu is a backend-independent tensor/dispatch library for Rust, aiming to
run GPU math on low-end hardware (e.g. iGPUs) without heavy dependencies. It
has no autograd and no shape system — it focuses purely on GPU compute and on
building compute graphs that run independently of the CPU. A kernel is a
`Shader` static carrying its per-backend sources; you decide which backends
each kernel implements, and adding a whole new backend only requires
implementing the `Backend` trait. Buffers are recycled through an automatic
pool. On CUDA, f32-storage matmuls can run their compute in bf16 tensor cores
(`set_bf16_matmul`); full quantized storage (f16/int8/int4) is future work.

Currently used by **sequexa-core** (sequential model engine).

## Backends

| | |
|---|---|
| **wgpu** (default) | WGSL kernels; runs on any GPU with a Vulkan/Metal/DX12 driver, including integrated GPUs |
| **cuda** (feature) | cuBLAS for matmuls + NVRTC-compiled CUDA C for everything else; CUDA graph capture for hot loops. Developed against CUDA 13.3 |
| **cpu** (feature) | plain single-threaded Rust reference implementations |

`CudaBackend::new(0)` returns a `Result`, so a `cuda`-featured binary can probe
for an NVIDIA GPU at runtime and fall back to `WgpuBackend` instead of
hard-failing.

Measured on an RTX 4050 training a 162M-parameter transformer: CUDA ~80
steps/min vs Vulkan ~31 steps/min (~2.5x). *Stale measurement — taken before
buffer pooling, flash attention and the fused-CE rework; due for a re-run.*

## Built-in shaders

Twelve kernels ship ready to use on all three backends: `matmul`,
`matmul_trp`, `matmul_add`, `matmul_weight_bwd`, `gemv`, `gemv_add`,
`residual_add`, `bwd_add_inplace`, `zero_tensor`, `causal_mask`, `adamw`,
`adamw_schedule`. Downstream projects add their own kernels the same way — a
`Shader` static with per-backend sources; sequexa-core's `shaders/` directory
is the living example.

## Building & testing

```bash
cargo build --release                  # wgpu only
cargo build --release --features cuda  # + CUDA (needs the CUDA Toolkit)

cargo test -- --test-threads=1         # ALWAYS single-threaded: parallel tests
                                       # create concurrent GPU devices and segfault
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the layer map, core contracts,
buffer lifecycle, execute paths, backend differences, and the checklist for
adding a new kernel. See [SHADERS.md](SHADERS.md) for the per-shader
wgsl/cuda/cpu/meta/emitter catalog, covering both wilupgu's builtins and
sequexa-core's own shaders.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in this work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
