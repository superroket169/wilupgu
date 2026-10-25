# TODO

Open wilupgu work only. Finished items are deleted; history is in git log.
Model-specific work (attention, rope, optimizers) belongs to sequexa-core's list.

## Next

1. **Plan tool**: `Runnable` and `CapturedRunnable` are in; their design is
   still being settled. Then release 0.7.0.
2. **Spread tool**, reworked on plan, back in the build.
   - Placement and `SpreadSpec` name devices with an id that no longer
     exists; how a plan names a device (an index into a device list, the
     `Device<X>` enum below, ...) is open.
   - Two unordered writes to one buffer are no problem on one in-order queue;
     across devices or queues they may need a check again.
3. **Autograd tool** (`tools/autograd`); parts may come from sequexa-core.
4. **Backend parity tool**: every shader on every backend against the CPU
   reference (strategy note 2), as a tool.

Each tool gets its own feature once it builds (`tool-plan`,
`tool-spread = ["tool-plan"]`). Tools import each other as a DAG; a check
for that is still to write.

- Pool size: how `DeviceConfig::pool_bytes` should be picked or adjusted
  over a run (rolling) is still open.

## Design

- **Owner-value backend tool** for raw backends (ash): the value a buffer
  carries so `execute_raw` can reject another device's buffer. `CpuBackend`
  has its own copy of it.
- **`Device<X>`** (idea). The tools' `Device` enum keeps one variant per
  in-tree backend, each behind its feature, plus `Extra(X)` for a user's
  own backend (`X: Backend`, default an empty enum). A user with several
  backends of their own writes their own enum, implements `Backend` on it
  and passes it as `X`. No `dyn`, no fork of wilupgu.
- **SysTopology** (no draft yet). Built when wilupgu starts up: discovers
  the machine's devices, enables the backends that can run on them,
  computes device capacities and exposes the result as data. That data is
  then used to pick sizes (batch size and the like) and to distribute a
  graph across devices (suggestion functions were planned for this). A
  `Resolvable` handed to SysTopology is resolved to the largest value these
  devices can hold. Which backends are usable at runtime is decided here,
  not by `cfg`s on shader code.
- **CUDA backend** (cudarc). Use a fixed CUDA build-system version feature
  (like `cuda-13030`) that matches the driver; `cuda-version-from-build-system`
  can't find new CUDA versions.
- **cuda-blas backend**, separate from the plain CUDA backend: cuBLAS for the
  matmul family, for training-sized GEMMs.
- **wgpu backend**, rewritten on the new contract, with f16 and other dtypes.
  Known wgpu bug: in one compute pass with several nodes, barriers between
  dispatches are missing or late and the run dies with `Parent device is
  lost`. The only working workaround is a submit+wait per node. Upgrading
  wgpu didn't fix it; the native Vulkan backend below is the real fix.
- **Native Vulkan backend** (vulkano). WGSL → `naga` → SPIR-V →
  `vulkano::ShaderModule`, with barriers derived from the nodes'
  `BindingRole`s (real RAW/WAW hazards) and recorded by hand. First step:
  prove "WGSL → naga → SPIR-V → vulkano, one dispatch" in a scratch crate.
- **Rayon backend**: real parallel native shader code for many-core servers.
  `CpuBackend` stays single-threaded for deterministic tests.
- **SpreadTensor review.** Known inconsistency: `CombineOp` returns a `Vec`
  (all-reduce copies) but `SpreadTensor::combine` keeps only the last one.
  `combine::sum` / `sum_to_all` are missing.
- **ComputeMesh**: still a skeleton (`todo!()` bodies, no constructor);
  `Placement` is unused outside tests. Several design parts don't exist yet.
- **Export the `builtins!` macro** so sequexa-core defines its shaders the
  same way (`include_str!` resolves relative to the calling file, so this works).
- **Pipeline-overridable constants** (WGSL `override`): the `Shader` /
  pipeline API has no such concept.
- **Scope's name.** `device.scope(|s| ..)` follows `std::thread::scope`;
  `device.submit(|s| ..)` says what it does (hands work to the device).
- **Error enum** in place of `String` errors, for the safe layer and the
  tools.

## Checks

- Run Miri over the CPU backend and the safe layer (their `unsafe` code:
  capture, areas, scope pointers). Nightly is installed; the `miri`
  component isn't.
- Keep the scope's compile-time guarantees as `compile_fail` doctests: a
  buffer allocated inside the closure can't be bound, a bound buffer can't
  be touched by the host, a slot can't leave its scope or enter another,
  a `Recording` can't outlive its device. They live in doc comments.
- `min_lens` is tested for native code only; WGSL and CUDA need a GPU backend
  for the same exact-size vs padded test (SHADERS.md standard 10).
- Prove the GPU really runs independently of the CPU: no hidden host sync or
  copy in a steady-state loop beyond deliberate traffic.

## Parked

- **ROCm/HIP backend** (rocBLAS-based). Low priority: wgpu/Vulkan already
  covers AMD through RADV. Candidate crates: `cubecl-hip-sys`, `rocm-rs`.
- **WASM/WebGPU browser demo**, a natural by-product of the wgpu backend.
- **Quantization (int8/int4 inference)**: decode is bandwidth-bound, so
  this is a general iGPU lever.

## Strategy notes

1. **Tie contracts to tests, not comments.** A label is only a claim. One
   "canary" test that fills every Output buffer with NaN/garbage before
   dispatch and checks the result catches mislabeled Output/Accumulate
   bindings for every shader, for free.
2. **Shrink the copy surface.** Many bugs are twin implementations drifting
   apart (a barrier in one format, missing in another). With CPU code for
   every shader, a "shader × backend × CPU reference" parity matrix keeps
   the twins from drifting silently.
3. **Test edges when writing a formula.** For every parameter, test t=0,
   t=limit and t=limit+1.
4. **Make periodic review a ritual.** An independent read-through every N
   commits or after each big refactor.
5. **Verify the panic line before theorizing.** Read which call a
   `file:line` actually belongs to before guessing a cause.
