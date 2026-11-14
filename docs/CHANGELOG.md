# Changelog

What each release changed. Versions before 0.4.3 were git tags only and
were never published to crates.io.

## (unreleased)

### Added
- CI runs the CPU tests under Miri, with every freed address reused when
  it can be.
- Regression tests for three soundness holes in capture and replay; they
  fail until the holes are closed.
- `rust-version = "1.95"` in `Cargo.toml`, checked in CI with every
  feature on.
- `WgpuBackend::device` and `WgpuBackend::queue`, and
  `WgpuBackend::from_device`: wilupgu on a device a wgpu app already has.
- `shader_table!`, exported: the macro the builtins are defined with, so
  another crate defines its own shaders the same way.
- `scripts/setup.sh`: on Ubuntu or Debian (Colab included), installs Rust,
  and nvcc and NVRTC where an NVIDIA driver has no toolkit, then runs
  `scripts/tester_script.sh`.

### Fixed
- CUDA builtins compute global indices and index products in 64 bits; in
  32 bits they wrapped past 2^32 elements or threads and read or wrote the
  wrong element. SHADERS.md standard 11.
- The builtins' `min_lens` saturate instead of wrapping, so a length too
  big for `usize` (three workgroup counts, or any product on a 32-bit
  target) fails the check instead of passing a short buffer.

### Changed
- README: a warning on the capture/replay soundness holes, a "Known
  limits" section, and candle, burn and cudarc in "Compared to".

### Changed (breaking)
- `Builtin` is `PortableShader`: other crates' shaders are made with it too.

## 0.10.0 (2026-11-12)

### Added
- Mesh, in the plan tool: tensors, and routines of calls on them, each run
  by its key. `MeshSpec` (with `MeshDesc` and `Routine::key()`) is checked
  like a `RunSpec`; `MeshSpec::build` builds every tensor once, live, and
  every routine's calls. `Mesh::bind` binds them all into the caller's
  scope once; `MeshBound::execute(key)` queues a routine's calls, and
  `apply_dynamics` sets the `Dynamic` meta of every routine.
- `Error::DuplicateId`: two routines given one key.

### Changed
- README: the mesh, in the status and the plan tool section.

## 0.9.2 (2026-10-30)

### Added
- `examples/hollow.rs`: two runs of one plan sharing a tensor, moved with
  `Run::fill_from`.
- `examples/capture.rs`: a run recorded once and replayed.

### Changed
- `examples/custom_shader.rs` checks its shader with the parity tool
  against the CPU, so it needs `tool-parity`.
- `scripts/tester_script`: stops without cargo; without nvcc, skips the
  CUDA tests and runs parity without CUDA.

## 0.9.1 (2026-10-29)

### Added
- `examples/custom_shader.rs`: a shader of your own, with its `Signature`,
  WGSL and native code.

### Changed
- README: the CUDA backend is tested, on CUDA 13.0.

## 0.9.0 (2026-10-29)

### Added
- Hollow tensors: a tensor that lives in another run. `HollowTensor` (its
  kind and length) and `AnyTensor` (live or hollow).
- `TensorSpec::build_hollow` and `RunSpec::build_hollow`: a run whose
  tensors are all hollow, made as many times as needed. A run with a
  hollow tensor doesn't bind.
- `Run::fill_from`: moves a live tensor of one run into the hollow place of
  the same tensor in another, with no copy.
- `Run::any_tensor`, `Run::contains_hollow`; `Error::Hollow`,
  `Error::NotHollow`.

### Changed
- README: which backends are tested, where the checks stop (the shader's
  code), that CUDA builtins need `compile-nvrtc`, and the planned work.
- `cuda` replaces `tensor` in the crates.io keywords.

### Changed (breaking)
- Renamed: `RunPlan` to `RunSpec`, `PlanDesc` to `RunDesc`, `Runnable` to
  `Run`, `CapturedRunnable` to `CapturedRun`.
- Renamed: `Scope::run` to `Scope::execute`; on a bound run, `run` to
  `execute` and `run_call` to `execute_call`.
- `RunSpec::build` takes the spec, so it makes one live run; on a fail it
  gives the spec back with the error.
- `Run::tensors` and `CapturedRun::tensors` give `AnyTensor`s.
- `Run` and `CapturedRun` are in `tools::plan::run`; the `runnable` module
  is gone.

### Removed (breaking)
- The `Spec` trait. `new`, `id`, `ready` and `build` are methods of each
  spec (`TensorSpec`, `NodeSpec`, `CallSpec`, `RunSpec`) on its own.

## 0.8.1 (2026-10-28)

### Added
- `PlanBuilder` and `TensorBuilder`: a `RunPlan` made a step at a time;
  ids come back to name tensors in calls.
- `Workgroups::cover`: enough workgroups for a length, from the shader's
  workgroup width.
- `From<T> for Resolvable<T>`: a fixed value is given as `n.into()`.
- Crate docs with a runnable example and a `compile_fail` doctest for the
  scope's borrow.
- Feature labels on docs.rs.

### Changed
- README: what is checked, a compile-fail example, a comparison with wgpu,
  CubeCL and krnl.
- Examples use the builder and `Workgroups::cover`.
- README links point to the release's tag; TODO.md's stays on `main`.

### Fixed
- `examples/parity.rs` builds with every backend feature set.

## 0.8.0 (2026-10-27)

### Added
- wgpu backend (`wgpu`): WGSL, GLSL and SPIR-V.
- CUDA backend (`cuda`): PTX and cubin; not yet run on an NVIDIA GPU.
- `tool-compile` with the `ShaderCompiler` trait; `compile-nvrtc` compiles
  CUDA C++ into PTX or cubin.
- `tool-parity`: one shader case on two devices from the same seeded data,
  compared slot by slot; edge cases for every builtin.
- `Builtin::shader_for`, `ComputeDevice::first`, `wilupgu::prelude`.

### Changed (breaking)
- A shader is a `Signature` and a `ShaderBinary`; backends take binaries,
  and `build_node` can fail.

## 0.7.0 (2026-10-26)

### Added
- The plan tool (`tool-plan`): the `Spec` trait, `TensorSpec`, `NodeSpec`,
  `CallSpec`, `RunPlan`, `Runnable` and `CapturedRunnable`.
- Tensor ranges, copies and zero; `Scope::copy`; `Tensor::replace`,
  `Node::replace`, `Node::meta`.
- Backend contract: offsets on upload and download, `copy_raw`,
  `clear_raw`, `COPY_ALIGN`.
- The `add` and `plan` examples.

### Changed (breaking)
- `Buf` is `Tensor`, `BufSlot` is `TensorSlot`.
- One `Error` enum for the safe layer and the tools.

### Removed
- Tools whose work the safe layer now does.

## 0.6.0 (2026-10-19)

### Added
- The safe layer: `ComputeDevice`, buffers, `Node`, and `Scope` with
  checked runs.
- `min_lens` on every shader; a run checks buffer lengths against it.
- CPU backend: capture and replay, areas; `DeviceArea`.

## 0.5.0 (2026-10-08)

### Added
- `Resolvable`, `GlobalId`, placement and a first `SpreadTensor`.
- 17 builtins with WGSL, native and CUDA code: add, mul, scale, max, min,
  clamp, sum, dot, transpose, matmul, gemv and more. Some are general
  shaders from before 0.4.3, brought back.

## 0.4.3 (2026-09-30), first on crates.io

### Changed (breaking)
- The backend contract as traits (`Storage`, `Topology`, `Areable`, ...).

### Added
- `TensorSpec`; CUDA graph capture; wgpu GL fallback.

### Removed
- The graph and the tensor structures of 0.4.x. They are unrelated to
  today's `Tensor`, which came in 0.7.0 as the new name of `Buf`.

## Before 0.4.3 (git tags v0.4.0 to v0.4.2)

- wgpu and CUDA backends with model shaders (matmul, softmax, RMSNorm,
  attention, AdamW). Most of them moved to sequexa-core.
