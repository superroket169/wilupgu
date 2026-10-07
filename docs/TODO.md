# TODO

Open wilupgu work only. Finished items are deleted; history is in git log.
Model-specific work (attention, rope, optimizers) belongs to sequexa-core's list.

- Graph going to be deleted, will replaced by run.rs parts (RunPart)
- docs.rs creation
- pool size rolling discuison
- nn tools : tools/nn : maybe could come from sequexa-core
- tool import check, tool importations should be dag
- Device::contains and io_log is not using
- Toy backend will be deleted, after cpu backend implemented
- there is writing exploids at graph::run and device::upload_kind

## Design

- **SysTopology** (no draft yet). Built when wilupgu starts up: discovers
  the machine's devices, enables the backends that can run on them,
  computes device capacities and exposes the result as data. That data is
  then used to pick sizes (batch size and the like) and to distribute a
  graph across devices (suggestion functions were planned for this). A
  `Resolvable` handed to SysTopology is resolved to the largest value these
  devices can hold. Which backends are usable at runtime is decided here,
  not by `cfg`s on shader code.
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
- **Contract gaps in `rules.rs`:**
  - a binding's `DataKind` is never checked against the tensor's real kind;
  - the same slot can be bound twice.
- **A `TensorSpec` size is `Resolvable<u32>`, a live `Tensor`'s length is
  `usize`.**
- **Export the `builtins!` macro** so sequexa-core defines its shaders the
  same way (`include_str!` resolves relative to the calling file, so this works).
- **Pipeline-overridable constants** (WGSL `override`): the `Shader` /
  pipeline API has no such concept.

## Checks

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
