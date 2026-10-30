# wilupgu

Borrow-checked GPU dispatch for Rust, with optional tools on top.

- **The borrow checker guards device memory.** GPU work runs in a scope,
  like `std::thread::scope`. A buffer bound to it stays borrowed until the
  device is done with it, so the host can't read, write or free it early.
- **Every run is checked before it reaches the device:** slot count,
  read/write roles, data kinds, aliasing, and each buffer's length against
  what the shader declares it touches (`min_lens`).
- **One API on every backend:** a CPU reference backend, wgpu (Vulkan,
  Metal, DX12) and CUDA; so far tested on the CPU, on wgpu with Vulkan and
  on CUDA 13.0.
  Bring your own backend by implementing three traits.
- **Optional tools on the safe layer:** describe work once and run it many
  times (plan), check a shader on two backends from the same seeded data
  (parity), compile CUDA C++ at runtime (NVRTC).

A `Tensor` here is a flat, typed device buffer; there is no shape system.

The checks stop at the shader's code. Building a node from a shader is
safe, so wilupgu trusts the code to stay within the `min_lens` the shader
declares; code that goes past it is a bug in the shader, and on CUDA it is
undefined behavior. The builtins follow the standards in
[SHADERS.md](https://github.com/superroket169/wilupgu/blob/v0.9.2/docs/SHADERS.md).

**Status: early.** Tested on CPU, an Intel iGPU (wgpu, Vulkan) and an
NVIDIA GPU (CUDA 13.0). Aimed at low-end
hardware, iGPUs included.
[sequexa-core](https://github.com/superroket169/sequexa-core), a sequential
model engine, is built on wilupgu.

## Example

```rust
use wilupgu::backends::CpuBackend;
use wilupgu::prelude::*;
use wilupgu::tools::builtins::ADD;

fn main() -> Result<(), Error> {
    let dev = ComputeDevice::<CpuBackend>::first()?;
    let mut x = dev.tensor_from(&F32::wrap(vec![1.0, 2.0, 3.0]))?;
    let mut y = dev.tensor_from(&F32::wrap(vec![10.0, 20.0, 30.0]))?;

    // x[i] += y[i] for i < n; the meta is n.
    let n = 3;
    let shader = ADD.shader_for::<CpuBackend>()?;
    let wg = Workgroups::cover(n, &shader);
    let mut add = dev.node(&shader, &[MetaValue::Uint(n)], wg)?;

    // While bound, the host can't touch x or y; the scope waits for the
    // device before it returns.
    dev.scope(|s| {
        let (x, y, add) = (s.bind(&mut x)?, s.bind(&mut y)?, s.bind_node(&mut add)?);
        s.execute(add, &[Arg::Write(x), Arg::Read(y)])
    })?;
    assert_eq!(x.download_vec::<F32>()?, vec![11.0, 22.0, 33.0]);
    Ok(())
}
```

Swap `CpuBackend` for `WgpuBackend` and nothing else changes:
`shader_for` picks the code the backend runs. On `CudaBackend` it compiles
the builtin's CUDA C++ at runtime, so the `compile-nvrtc` feature is needed
too.

Reading `x` while the scope still holds it doesn't compile (a
`compile_fail` doctest in the crate docs checks it):

```rust,ignore
dev.scope(|s| {
    let (xs, ys, add) = (s.bind(&mut x)?, s.bind(&mut y)?, s.bind_node(&mut add)?);
    s.execute(add, &[Arg::Write(xs), Arg::Read(ys)])?;
    x.download_vec::<F32>()?; // error[E0502]: cannot borrow `x` as immutable
                              // because it is also borrowed as mutable
    Ok(())
})?;
```

## Compared to

- **wgpu**: the layer below wilupgu's wgpu backend. With wgpu you build
  pipelines and bind groups yourself; wilupgu adds buffer ownership, the
  per-run checks above, and the same API on CUDA.
- **CubeCL**: kernels written once in Rust and compiled for each backend.
  wilupgu doesn't generate kernels: a shader brings code per backend (WGSL,
  CUDA, native Rust; every builtin ships all three). In return, existing
  WGSL and CUDA kernels run as they are, and the safety lives in the host
  API, not in a kernel language.
- **krnl**: Rust kernels compiled to SPIR-V, run on Vulkan or the host.
  wilupgu runs CUDA directly as well, and takes existing kernel code.

```toml
[dependencies]
wilupgu = { version = "0.9", features = ["cpu"] }
```

More in [`examples/`](https://github.com/superroket169/wilupgu/tree/v0.9.2/examples):

- `add.rs`: the code above, generic over the backend; it also runs on wgpu
  when that feature is on. `cargo run --example add --features cpu`
- `plan.rs`: the same work described up front with the plan tool:
  `x *= factor`, then `y += x`, run twice with a factor set from the host
  each time. `cargo run --example plan --features cpu,tool-plan`
- `hollow.rs`: two runs of one plan sharing a tensor: `x *= 2` runs in one,
  then `x` moves to the other, not copied, and runs there.
  `cargo run --example hollow --features cpu,tool-plan`
- `capture.rs`: a run recorded once and replayed, `x *= 2` three times,
  then again on new contents. `cargo run --example capture --features cpu,tool-plan`
- `parity.rs`: every builtin on every pair of backends that are on and
  have a device here, one line per case.
  `cargo run --example parity --features cpu,wgpu,tool-parity`
- `custom_shader.rs`: a shader of your own, `y = a * x + y`: its
  `Signature` (meta, slots, `min_lens`), WGSL code for wgpu, native code for
  the CPU backend, a run on tensors too short for its meta, which fails
  before it reaches the device, and a parity check of it against the CPU.
  `cargo run --example custom_shader --features cpu,tool-parity`

## Status

In it:

- **The safe layer** (`wilupgu::safe`), the API users call. It is generic
  over the backend and holds every check that can be written once for all
  of them:
  - `ComputeDevice`: an opened device, with a buffer pool (`DeviceConfig`).
  - `Tensor`: a flat buffer (no shape) with its kind and length, from the
    pool or cut out of a `DeviceArea`. Helpers: `ComputeDevice::tensor_from`,
    typed `upload_slice`/`download_vec`, ranges (`upload_at`,
    `download_range`), `copy_into` and `zero` on the device, `copy_to`
    another device. `Scope::copy` copies between bound tensors.
  - `Node`: one dispatch of a shader, its meta given as typed `MetaValue`s.
  - `Scope`: GPU work runs in a closure, like `std::thread::scope`, and the
    scope waits for the device before it returns. `execute` checks slot count,
    roles, kinds, aliasing and buffer lengths before anything reaches the
    device.
  - `Recording`: `Scope::capture` records runs once, `Scope::replay` runs
    them again in a later scope.
- **The raw backend contract** (`wilupgu::backend`). A buffer has one owner;
  reading takes `&`, writing `&mut`. The few methods that can cause undefined
  behavior are `unsafe fn`, with their rules in `# Safety`.
- **The plan tool** (`tool-plan` feature, `wilupgu::tools::plan`): tensors
  and calls described before a device exists, with sizes and meta that can
  be known later (`Resolvable`, `Relation`) or change between runs
  (`Dynamic`). A `RunSpec` builds into a `Run`, bound into the caller's
  scope once and run as often as needed; on a backend with capture it can
  become a `CapturedRun`, which replays one recording.
- **The parity tool** (`tool-parity` feature, `wilupgu::tools::parity`):
  `compare` runs one `Case` of a shader on two devices, of the same backend
  or not, from the same seeded data, and checks each slot by its `Check`.
  The `Report` also flags `Output` slots left unwritten and writes past
  `min_lens`. `compare_builtins` does it for every builtin, with edge cases
  for each.
- **Backends**, each behind its feature:
  - `cpu`: single-threaded, for tests and reference results, with capture
    and areas;
  - `wgpu`: Vulkan, Metal and DX12 through wgpu, tested on an Intel iGPU
    (Vulkan); no capture, no areas;
  - `cuda`: PTX and cubin through the CUDA driver, with capture (CUDA
    graphs) and areas, tested on CUDA 13.0.
- **17 builtin shaders** (`tool-builtins` feature, on by default), each with
  WGSL, native and CUDA code, and the `min_lens` every shader declares. The
  catalog is
  [SHADERS.md](https://github.com/superroket169/wilupgu/blob/v0.9.2/docs/SHADERS.md).

Not in it yet:

- a Vulkan backend (ash), with WGSL compiled to SPIR-V by naga;
- a parallel CPU backend (rayon);
- spreading work over devices;
- quantization (int8/int4);
- autograd.

Open work is in
[TODO.md](https://github.com/superroket169/wilupgu/blob/main/docs/TODO.md).

## Philosophy

- **`safe` is the minimum safe API.** It is the only caller of the raw
  `unsafe fn`s, and it has to stay sound with no tool on top.
- **`backend` is the contract.** Backends implement it; its unsafe methods
  say exactly what the caller must hold, so `safe` can hold it once for
  every backend.
- **Tools are optional and make life easier**, the way `Arc` and `Mutex` do
  for plain values, and exactly as optional as those are. A user can pick a
  backend and use no tool at all.
- **Backend authors are users too.** Next to the contract, `backend` holds a
  few backend tools (`pool`, `io_log`) for them. User tools are built on top
  of those, so they stay minimal.
- **Ownership over bookkeeping.** A buffer has one owner and the device
  borrows what it reads and writes. A check that can be written once for
  every backend lives above the backends, not in each of them.
- **Kinds are runtime values.** A buffer's kind is a `DataKind`, not a type
  parameter: shaders declare their slots' kinds at runtime, and some kinds
  are only known at runtime, like a model loaded from a file.

## Features

| Feature | Default | Adds |
|---|---|---|
| `tool-builtins` | yes | `wilupgu::tools::builtins`, the builtin shader table |
| `tool-plan` | no | `wilupgu::tools::plan`, the plan tool |
| `tool-parity` | no | `wilupgu::tools::parity`, the parity tool |
| `tool-compile` | no | `wilupgu::tools::compile`, the `ShaderCompiler` trait |
| `compile-nvrtc` | no | `Nvrtc`: CUDA C++ into PTX or cubin; the CUDA version is picked with `CUDARC_CUDA_VERSION` or a cudarc `cuda-XXXXX` feature |
| `cpu` | no | `wilupgu::backends::CpuBackend` |
| `cuda` | no | `wilupgu::backends::CudaBackend`, runs PTX and cubin; the CUDA version is picked like for `compile-nvrtc` |
| `wgpu` | no | `wilupgu::backends::WgpuBackend` (Vulkan, Metal, DX12), runs WGSL, GLSL and SPIR-V |

## The plan tool

In the plan tool, between runs, `Run::edit` lends every tensor and node
as `&mut`; inside a scope, `execute_call` runs one call and `tensor` gives a
tensor's slot for the scope's own work.

`RunSpec::build` takes the spec, so a spec makes one live run.
`RunSpec::build_hollow` makes as many runs as needed whose tensors are
hollow: they live in another run. `Run::fill_from` moves a live tensor into
a hollow place, with no copy. A run with a hollow tensor doesn't bind.

## Capture and areas

On a backend that has them (CPU and CUDA; not wgpu):

- `Scope::capture` records the runs its closure makes instead of running
  them. `Scope::replay` runs them again in a later scope, once every
  recorded buffer and node is bound there and every run's checks pass
  again. A replay uses each node's current meta, so `Node::update_meta`
  between replays reaches the next one. Uploads and downloads inside a
  capture aren't recorded; they run right away.
- `ComputeDevice::reserve` gives a `DeviceArea`, one block reserved up
  front; `DeviceArea::alloc` cuts buffers out of it. They can't outlive the
  area, and their memory goes back with it, not to the pool.

## Writing a shader

A shader is a `Signature`, how it is called (its meta fields, tensor
layout, `min_lens` and workgroup size), and its code. A node is built from a
`Shader`: a signature and one `ShaderBinary`, code in a form the backend
takes (native code for the CPU backend). `min_lens` tells, from the meta and
the workgroup count, how many elements each slot needs; the safe layer
rejects a shorter buffer, so the code must never go past it. The
builtins follow the standards in
[SHADERS.md](https://github.com/superroket169/wilupgu/blob/v0.9.2/docs/SHADERS.md);
a shader of your own should too.

## Writing a backend

In your own crate, on your own type, implement:

- `Topology`: find devices and open one (`choosable_devices`, `attach`,
  `info`), with a `DeviceInfo` for each device;
- `Storage`: buffers (`alloc_raw`, `upload_raw`, `download_raw`, `copy_raw`,
  `clear_raw`); offsets and lengths are bytes, multiples of `COPY_ALIGN` (4);
- `Dispatch`: the binaries it takes (`accepts`), and nodes and runs
  (`build_node`, `update_meta`, `execute_raw`, `synchronize`).

Optional: `Areable` (buffers cut out of one reserved area) and `Capturable`
(record calls once, replay them, like CUDA graphs). A replay runs each node
with its current meta.

`execute_raw` must reject a buffer from another device. The CPU backend
(`src/backends/cpu.rs`) is a small, complete example.

## License

Licensed under either of [Apache License, Version 2.0](https://github.com/superroket169/wilupgu/blob/v0.9.2/licenses/LICENSE-APACHE) or
[MIT license](https://github.com/superroket169/wilupgu/blob/v0.9.2/licenses/LICENSE-MIT) at your option.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in this work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
