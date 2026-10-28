# wilupgu

General-purpose GPU compute: a backend contract, a safe layer over it, and
optional tools on top. No shape system and no NN-training-specific ops:
optimizers, schedules and model ops belong to sequexa-core, which uses
wilupgu. Autograd comes only as an optional tool (`tools/autograd`, planned).

## Layout

- `src/backend/` is the contract: everything a backend implements or uses.
  `mod.rs` holds the traits; next to it: dtype, shader, and the backend
  tools pool and io_log.
- `src/safe/` is the safe layer, the API users call and the only caller of
  the contract's `unsafe fn`s. Every check that can be written once for all
  backends lives here.
  - `mod.rs`: `ComputeDevice`, `DeviceConfig`, `Tensor`, `DeviceArea`, `Node`,
    `MetaValue`.
  - `scope.rs`: `Scope`, slots, `Arg`, `run`, and capture/replay
    (`Recording`).
  - `rules.rs`: every check made in more than one place, one function each,
    returning the `Error` it finds; the tools use them too.
- `src/error.rs`: `Error`, the one error enum of the safe layer and the
  tools. Backend errors stay `String`, carried by `Error::Backend`.
- `src/backends/` holds backend implementations, one file each. `cpu.rs` is
  behind the `cpu` feature and always built for tests; it implements
  `Capturable` and `Areable`. `cuda.rs` is behind `cuda`; its tests need
  `compile-nvrtc` and skip without a device. `wgpu.rs` is behind `wgpu`;
  it implements neither `Capturable` nor `Areable`, and its tests skip
  without an adapter.
- `src/tools/` holds optional packages, each behind its own feature. A user
  can pick a backend and use none of them.
  - `builtins/` (`tool-builtins`, default): the builtin shader table;
    `codes/{wgsl,native,cuda}/` has one file per shader per format.
    Directory names mirror `Builtin` fields.
  - `plan/` (`tool-plan`, always built for tests): `deferred` (`Resolvable`,
    `Relation`, `Dynamic`), `id`, `spec` (the `Spec` trait), `tensor`
    (`TensorSpec`), `node` (`NodeSpec`), `call` (`CallSpec`, `Call`), `run`
    (`RunPlan`), `builder` (`PlanBuilder`, `TensorBuilder`) and `runnable`
    (`Runnable`, `CapturedRunnable`).
  - `compile/` (`tool-compile`, always built for tests): the
    `ShaderCompiler` trait; each compiler goes in `compile/compilers/`, one
    file each, behind its own feature.
  - `parity/` (`tool-parity`, always built for tests): runs one `Case` of
    a shader on two devices from the same seeded data and compares each
    slot by its `Check`; the `Report` also flags `Output` slots left NaN
    and writes past `min_lens`. `builtins` holds the edge cases of every
    builtin and `compare_builtins`.
  - `spread/`: out of the build until it is rewritten on the safe layer.
  One file = one subject.
- `src/tests/`: one file per topic, included with `#[path]` from the module
  it tests. There is only one tests directory.
- `examples/`: runnable examples. The README shows `add.rs` (with a
  `main` on the CPU backend) and points to the rest; keep the two the same.

## Builtin shaders

A builtin must be general-purpose GPU math. Optimizers, schedules and
attention-specific ops are not builtins; they go to sequexa-core.
Every shader declares `min_lens`, and its code never goes past it; the
standards are in `docs/SHADERS.md`. Update the catalog there when a shader
is added or removed.

## Tools

- Everything a tool holds can be read back: a `&` accessor for each part
  (tensors, nodes, workgroups, meta, ...). Changes go through narrower
  methods that can't move what the tool owns.

## Checks

Before reporting a change as done:

- `cargo test --features cpu`
- `cargo clippy --features cpu --all-targets` (no warnings)
- `cargo fmt --check`
- `cargo build --no-default-features`

Compile-time guarantees (what the borrow checker must reject) are tried in
a scratch crate outside the repo, depending on wilupgu by path.

## Comments

- `// SAFETY:` notes are plain comments, not doc comments.

## Docs

- `docs/README.md`: crate overview, status, example.
- `docs/SHADERS.md`: shader standards and catalog.
- `docs/TODO.md`: open work. Planned work lives there, not in code comments
  (except `TODO:` lines).
- `docs/CHANGELOG.md`: what each release changed. It is the only file that
  refers to the past; everything else describes the code as it is. Work
  goes under the top `(unreleased)` heading; a release's last commit puts
  the date there, bumps the version in `Cargo.toml` and points README's
  links (all but TODO.md's) to the new tag.
