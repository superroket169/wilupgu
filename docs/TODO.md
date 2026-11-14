# TODO

Open wilupgu work only. Finished items are deleted; history is in git log.
Model-specific work (attention, rope, optimizers) belongs to sequexa-core's list.

## Before announcing

The first public post (Reddit, HN) waits for these, whichever release they
land in. New tools are not on the list.

1. wgpu runs on at least one discrete GPU, and `examples/parity.rs`
   passes there.
2. README's top bullets match what has been tested by then.
3. The post names only what the release has: a naga compiler if it has
   landed, autograd and quantization as planned.
4. The holes under Soundness are closed, their regression tests pass, and
   the Miri job passes in CI.
5. Shader code only its author can vouch for (CUDA, PTX, Cubin) is built
   through an `unsafe` constructor.
6. README's soundness warning is gone, and its "Known limits" section
   matches the code.
7. A soundness review is asked on users.rust-lang.org first, and the post
   invites people to break the safe API and open issues.

## Soundness

- **CPU replay follows invalidated pointers.** `CpuCapture` keeps a
  `*const CpuNode` and slot pointers made from references at capture time.
  A `&mut` to the node or a buffer between capture and replay
  (`Node::update_meta`, an upload) invalidates them under Stacked Borrows,
  and `replay` reads through them. Miri, one test at a time, reports UB
  in six tests, all at `cpu.rs:319` (the node) or `cpu.rs:325` (a slot);
  the other 172 pass. Proposed:
  `Capturable::replay` gets the current node and buffers from the safe
  layer instead of following recorded pointers; it goes with the id fix
  below. CUDA keeps device pointers, which Rust's aliasing rules don't
  cover.
- **Identity by address.** `replay` finds recorded buffers and nodes by
  address.
  - A buffer freed after the recording and reallocated at the same address
    matches; `check` sees the new length, but the CPU replay slices the
    recorded one, so a shorter buffer is read and written past its end. The
    comment in `Scope::replay` that two addresses meet only for empty
    buffers is wrong.
  - A node replaced in place (`Node::replace`, or assignment) matches,
    while a CUDA graph keeps the old kernel, grid and freed meta buffer.
  - Proposed: each `Tensor` and `Node` gets a unique `u64` id from a device
    counter, and the recording holds ids. `replace` gives a new id;
    `update_meta` keeps it. Empty CPU buffers share an address (an empty
    `Vec` allocates nothing), which ids make moot too. Tests:
    `replay_rejects_another_tensor_at_a_recorded_address`,
    `replay_rejects_a_node_replaced_in_place`.
- **Shader trust.** `Shader::new` is safe, but CUDA code that goes past
  `min_lens` is UB, so safe code reaches UB. Proposed: safe for WGSL and
  Native, `unsafe` for CUDA, PTX and Cubin; builtins take that trust on
  themselves. Open: one `unsafe` constructor for all, or a variant for the
  trusted formats.
- **Break-it tests.** Every `// SAFETY:` note rests on assumptions; each
  gets a test that tries to break it. Patterns to try:
  - identity by address: free and reallocate, `replace`, assignment;
  - device-wide state reaching other scopes or threads;
  - re-entrance: the same API called from inside its own closure;
  - `mem::swap`, `mem::replace`, `mem::forget`;
  - drop order on panic.

  A test that has to hit a race isn't run thousands of times hoping for
  it: the state is set up so it happens every time (the pool for address
  reuse), `loom` runs a test under every interleaving of its threads (the
  capture lock in `ComputeDevice::capturing` first), and
  `SEEDS=N scripts/miri.sh <test>` tries N schedules under Miri.

## Design

Each tool gets its own feature once it builds (`tool-plan`,
`tool-spread = ["tool-plan"]`). Tools import each other as a DAG; a check
for that is still to write.

- Pool size: how `DeviceConfig::pool_bytes` should be picked or adjusted
  over a run (rolling) is still open.
- **`HollowSpec`**: a tensor's place in a `RunDesc` that another spec owns
  (id, kind, len), built as a hollow tensor. `TensorSpec::build_hollow`
  can't stand in: it needs the len known and carries no id. Wanted once a
  mesh routine is captured on its own (a captured run owns its tensors),
  or by spread.

- **Autograd** (`tools/passes/autograd.rs`, `tool-passes`), aimed at
  0.11.0. A struct holding a table from shader signature to derivative
  rule, with a method like `apply(spec, forward_key)` that gives back a
  `MeshSpec` with a backward routine and its key.
  - A rule belongs to a shader, not a call; builtins can ship theirs. A
    rule gets the forward call and the gradient ids of what it writes, and
    gives backward calls. `None` means stop-gradient.
  - Which slots are read or written comes from the signature's roles.
  - Gradients accumulate (`+=`); the backward routine starts by zeroing
    them.
  - It returns each tensor's gradient id (`x -> dx`); the user writes `dy`
    into the seed's tensor. `Dynamic` meta can't: it holds scalars.
  - In-place overwrites of tensors backward needs are checked at build.
  - The optimizer is not part of it; it is a routine of its own.
  - Passes that change a call (quantize) need its `CallSpec` id kept.

- **Owner-value backend tool** for raw backends (Vulkan through ash): the
  value a buffer carries so `execute_raw` can reject another device's
  buffer. `CpuBackend` has its own copy of it.
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
- **CUDA backend**: tested on CUDA 13.0 only, on two NVIDIA GPUs, one a
  Tesla T4 on Colab (`scripts/setup.sh`). Its tests
  (`CUDARC_CUDA_VERSION=13000 cargo test --features cpu,cuda,compile-nvrtc cuda`)
  skip without a device and NVRTC.
- **wgpu backend**: tested on an Intel iGPU (Mesa, Vulkan) only.
  - `Parent device is lost`: many nodes in one compute pass and one submit
    died with it in training runs, not in inference. A missing barrier, or
    a submit long enough for the driver's GPU timeout? `submit_every` (1
    by default; one compute pass per dispatch) is the knob to find out: a
    test with N independent dispatches in one submit tells the two apart.
    More likely it is how long one submit runs on the GPU (Windows TDR is
    about 2 s; i915/xe have their own hang check), not the dispatch count,
    so a fixed `submit_every` may not be the right knob: split submits by
    measured time (submit to done, or timestamp queries), and split a
    single dispatch that is too long by itself. Check how cubecl's wgpu
    runtime caps the tasks per submit. Lavapipe in CI doesn't show TDR: a
    rented cloud GPU, or volunteers running `scripts/tester_script`.
  - `submit_every` is a `pub` field (`AtomicU8`); it wants a setter or a
    `DeviceConfig` entry once the knob above is settled (#12).
  - A bind group is made per dispatch; cache it by node and buffers.
  - Fixed binding model: every slot is a whole storage buffer at offset 0;
    no uniforms, textures, push constants or indirect dispatch, and the API
    blocks through `pollster`. Fine for compute kernels, listed for later.
  - No `Areable`: WebGPU forbids one buffer bound both read-only and
    writable in one dispatch, for the whole buffer, so tensors cut out of
    one buffer can't be an `Input` and an `Output` of the same run. No
    `Capturable`: wgpu can't record compute work for replay.
  - `total_memory_bytes` is 0: wgpu doesn't tell it.
- **Builtins are F32 only**, though `DataKind` has F16, BF16, Int8, Int4.
- **CPU builtins** copy the whole slot into a `Vec` and back on every call
  (`codes/native/common.rs`), single-threaded. Fine for a reference
  backend; slow.
- **Pipeline-overridable constants** (WGSL `override`): the `Shader` /
  pipeline API has no such concept.
- **Backend errors.** The contract's methods return `String`, which the
  error enum carries as one variant. Turn it into an enum once more than
  one backend exists. If backends' errors turn out very different, each
  backend could name its own error type in the contract instead; that is
  not expected to be needed. `upload_raw`, `download_raw`, `update_meta`,
  `synchronize` and `reserve` return nothing: a driver error there means the
  device is gone, and the backend panics with the driver's message.
- **`Areable::reserve`** returns no `Result`, so a backend out of memory
  can only panic.

## Checks

- Run the CUDA tests under `compute-sanitizer` (it comes with the toolkit,
  so Colab has it): it reports every out-of-bounds read and write, where
  the exact-size vs padded `min_lens` test sees a read only when it changes
  a result.
- Prove the GPU really runs independently of the CPU: no hidden host sync or
  copy in a steady-state loop beyond deliberate traffic.

## Parked

- **ROCm/HIP backend** (rocBLAS-based). Low priority: wgpu/Vulkan already
  covers AMD through RADV. Candidate crates: `cubecl-hip-sys`, `rocm-rs`.
- **`tool-blas`** (`tools/blas`), after the ROCm backend: the matmul family
  as shaders whose code calls the vendor's BLAS (cuBLAS on CUDA, rocBLAS or
  hipBLAS on ROCm, maybe MPS on Metal), so the same shaders run on each of
  those backends. It runs on most backends, not all; elsewhere a node fails
  with `UnsupportedBinary`. Needs a `ShaderBinary` form that is a library
  call, not kernel code.
- **WASM/WebGPU browser demo**, a natural by-product of the wgpu backend.

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
