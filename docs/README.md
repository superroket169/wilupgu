# wilupgu

Backend-independent GPU compute for Rust, aimed at low-end hardware (iGPUs
included). No autograd, no shape system: GPU memory, shaders and dispatches,
on a backend contract built on Rust ownership.

[sequexa-core](https://github.com/superroket169/sequexa-core), a sequential
model engine, is built on wilupgu.

## Status (0.5.0)

In this release:

- **The raw backend contract** (`wilupgu::backend`). A buffer has one owner;
  reading takes `&`, writing `&mut`. The few methods that can cause undefined
  behavior are `unsafe fn`, with their rules in `# Safety`.
- **A CPU backend** (`cpu` feature): single-threaded, for tests and reference
  results.
- **17 builtin shaders** (`tool-builtins` feature, on by default), each with
  WGSL, native and CUDA code. The catalog is
  [SHADERS.md](https://github.com/superroket169/wilupgu/blob/main/docs/SHADERS.md).

Not in it yet:

- the safe user layer over the raw contract: scopes, typed buffers, and the
  checks written once for every backend;
- the tools built on it: tensors, graphs, spreading work over devices;
- GPU backends (wgpu, Vulkan, CUDA).

Open work is in
[TODO.md](https://github.com/superroket169/wilupgu/blob/main/docs/TODO.md).

## Philosophy

- **`backend` is the minimum safe API.** Backends implement it and users
  end up calling it. It has to stay sound with no tool on top.
- **Tools are optional and make life easier**, the way `Arc` and `Mutex` do
  for plain values, and exactly as optional as those are. A user can pick a
  backend and use no tool at all.
- **Backend authors are users too.** Next to the contract, `backend` holds a
  few backend tools (`pool`, `io_log`) for them. User tools are built on top
  of those, so they stay minimal.
- **Ownership over bookkeeping.** A buffer has one owner and the device
  borrows what it reads and writes. A check that can be written once for
  every backend lives above the backends, not in each of them.

## Features

| Feature | Default | Adds |
|---|---|---|
| `tool-builtins` | yes | `wilupgu::tools::builtins`, the builtin shader table |
| `cpu` | no | `wilupgu::backends::CpuBackend` |
| `wgpu` | no | only the WGSL standards test (parses WGSL with naga); no backend yet |

## Example

The builtin `add` (`x[i] += y[i]`) on the CPU backend, through the raw
contract. The `unsafe` blocks are what the safe layer will take over.

```toml
[dependencies]
wilupgu = { version = "0.5", features = ["cpu"] }
bytemuck = "1"
```

```rust
use wilupgu::backend::shader::Workgroups;
use wilupgu::backend::{Access, Dispatch, Storage, Topology};
use wilupgu::backends::CpuBackend;
use wilupgu::tools::builtins::ADD;

fn main() {
    let cpu = CpuBackend::attach(CpuBackend::choosable_devices().remove(0)).unwrap();

    let mut x = cpu.alloc_raw(12).unwrap();
    let mut y = cpu.alloc_raw(12).unwrap();
    // SAFETY: both buffers hold 12 bytes.
    unsafe {
        cpu.upload_raw(&mut x, bytemuck::cast_slice(&[1.0f32, 2.0, 3.0]));
        cpu.upload_raw(&mut y, bytemuck::cast_slice(&[10.0f32, 20.0, 30.0]));
    }

    // ADD is `x[i] += y[i]` for i < n; its meta is n.
    let node = cpu.build_node(&ADD, &[3], Workgroups::linear(1));
    // SAFETY: ADD's layout is (x: accumulate, y: input), both hold 3 F32s,
    // and nothing touches them before `synchronize`.
    unsafe {
        cpu.execute_raw(&node, &mut [Access::Write(&mut x), Access::Read(&y)])
            .unwrap();
    }
    cpu.synchronize();

    // SAFETY: 12 bytes fit in x.
    let out: Vec<f32> = bytemuck::pod_collect_to_vec(&unsafe { cpu.download_raw(&x, 12) });
    assert_eq!(out, [11.0, 22.0, 33.0]);
}
```

## Writing a backend

In your own crate, on your own type, implement:

- `Topology`: find devices and open one (`choosable_devices`, `attach`,
  `info`), with a `DeviceInfo` for each device;
- `Storage`: buffers (`alloc_raw`, `upload_raw`, `download_raw`);
- `Dispatch`: the shader code format it runs (`FORMAT`), and nodes and runs
  (`build_node`, `update_meta`, `execute_raw`, `synchronize`).

Optional: `Areable` (buffers cut out of one reserved area) and `Capturable`
(record calls once, replay them, like CUDA graphs).

`execute_raw` must reject a buffer from another device. The CPU backend
(`src/backends/cpu.rs`) is a small, complete example.

## License

Licensed under either of [Apache License, Version 2.0](../licenses/LICENSE-APACHE) or
[MIT license](../licenses/LICENSE-MIT) at your option.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in this work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
