# wilupgu

Backend-independent GPU compute for Rust, aimed at low-end hardware (iGPUs
included). No autograd, no shape system: GPU memory, shaders and dispatches,
on a backend contract built on Rust ownership.

[sequexa-core](https://github.com/superroket169/sequexa-core), a sequential
model engine, is built on wilupgu.

## Status (0.6.0)

In this release:

- **The safe layer** (`wilupgu::safe`), the API users call. It is generic
  over the backend and holds every check that can be written once for all
  of them:
  - `ComputeDevice`: an opened device, with a buffer pool (`DeviceConfig`).
  - `Tensor`: a flat buffer (no shape) with its kind and length, from the
    pool or cut out of a `DeviceArea`.
  - `Node`: one dispatch of a shader, its meta given as typed `MetaValue`s.
  - `Scope`: GPU work runs in a closure, like `std::thread::scope`, and the
    scope waits for the device before it returns. `run` checks slot count,
    roles, kinds, aliasing and buffer lengths before anything reaches the
    device.
  - `Recording`: `Scope::capture` records runs once, `Scope::replay` runs
    them again in a later scope.
- **The raw backend contract** (`wilupgu::backend`). A buffer has one owner;
  reading takes `&`, writing `&mut`. The few methods that can cause undefined
  behavior are `unsafe fn`, with their rules in `# Safety`.
- **A CPU backend** (`cpu` feature): single-threaded, for tests and reference
  results, with capture and areas.
- **17 builtin shaders** (`tool-builtins` feature, on by default), each with
  WGSL, native and CUDA code, and the `min_lens` every shader declares. The
  catalog is
  [SHADERS.md](https://github.com/superroket169/wilupgu/blob/main/docs/SHADERS.md).

Not in it yet:

- the tools built on the safe layer: tensors, graphs, spreading work over
  devices;
- GPU backends (wgpu, Vulkan, CUDA).

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
| `cpu` | no | `wilupgu::backends::CpuBackend` |
| `wgpu` | no | only the WGSL standards test (parses WGSL with naga); no backend yet |

## Example

The builtin `add` (`x[i] += y[i]`) on the CPU backend.

```toml
[dependencies]
wilupgu = { version = "0.6", features = ["cpu"] }
```

```rust
use wilupgu::backend::dtype::{DataKind, DataType, F32};
use wilupgu::backend::shader::Workgroups;
use wilupgu::backend::Topology;
use wilupgu::backends::CpuBackend;
use wilupgu::safe::{Arg, ComputeDevice, DeviceConfig, MetaValue};
use wilupgu::tools::builtins::ADD;

fn main() -> Result<(), String> {
    let cpu = CpuBackend::attach(CpuBackend::choosable_devices().remove(0))?;
    let dev = ComputeDevice::new(cpu, DeviceConfig::default());

    let mut x = dev.alloc(DataKind::F32, 3)?;
    let mut y = dev.alloc(DataKind::F32, 3)?;
    x.upload(&F32::wrap(vec![1.0, 2.0, 3.0]))?;
    y.upload(&F32::wrap(vec![10.0, 20.0, 30.0]))?;

    // ADD is `x[i] += y[i]` for i < n; its meta is n.
    let mut add = dev.node(&ADD, &[MetaValue::Uint(3)], Workgroups::linear(1))?;

    // Buffers and nodes are bound for the scope; it waits for the device
    // before it returns.
    dev.scope(|s| {
        let (xs, ys) = (s.bind(&mut x)?, s.bind(&mut y)?);
        let add = s.bind_node(&mut add)?;
        s.run(add, &[Arg::Write(xs), Arg::Read(ys)])
    })?;

    assert_eq!(F32::unwrap(x.download()), Some(vec![11.0, 22.0, 33.0]));
    Ok(())
}
```

## Capture and areas

On a backend that has them (the CPU backend does):

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

A shader is a `Shader` value: its meta fields, tensor layout, `min_lens`,
workgroup size and code in each format it has. `min_lens` tells, from the
meta and the workgroup count, how many elements each slot needs; the safe
layer rejects a shorter buffer, so the code must never go past it. The
builtins follow the standards in
[SHADERS.md](https://github.com/superroket169/wilupgu/blob/main/docs/SHADERS.md);
a shader of your own should too.

## Writing a backend

In your own crate, on your own type, implement:

- `Topology`: find devices and open one (`choosable_devices`, `attach`,
  `info`), with a `DeviceInfo` for each device;
- `Storage`: buffers (`alloc_raw`, `upload_raw`, `download_raw`);
- `Dispatch`: the shader code format it runs (`FORMAT`), and nodes and runs
  (`build_node`, `update_meta`, `execute_raw`, `synchronize`).

Optional: `Areable` (buffers cut out of one reserved area) and `Capturable`
(record calls once, replay them, like CUDA graphs). A replay runs each node
with its current meta.

`execute_raw` must reject a buffer from another device. The CPU backend
(`src/backends/cpu.rs`) is a small, complete example.

## License

Licensed under either of [Apache License, Version 2.0](../licenses/LICENSE-APACHE) or
[MIT license](../licenses/LICENSE-MIT) at your option.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in this work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
