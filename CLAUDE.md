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
- `src/backends/` holds backend implementations, one file each. `cpu.rs` is
  behind the `cpu` feature and always built for tests; it implements
  `Capturable` and `Areable`.
- `src/tools/` holds optional packages, each behind its own feature. A user
  can pick a backend and use none of them.
  - `builtins/` (`tool-builtins`, default): the builtin shader table;
    `codes/{wgsl,native,cuda}/` has one file per shader per format.
    Directory names mirror `ShaderCode` fields.
  - `plan/` (`tool-plan`, always built for tests): `deferred` (`Resolvable`,
    `Relation`, `Dynamic`), `id`, `spec` (the `Spec` trait), `tensor`
    (`TensorSpec`) and `node` (`NodeSpec`); `run` is out of the build
    (`#[cfg(any())]`) until it is rewritten on the specs.
  - `spread/`: out of the build until it is rewritten on the safe layer.
  One file = one subject.
- `src/tests/`: one file per topic, included with `#[path]` from the module
  it tests. There is only one tests directory.

## Builtin shaders

A builtin must be general-purpose GPU math. Optimizers, schedules and
attention-specific ops are not builtins; they go to sequexa-core.
Every shader declares `min_lens`, and its code never goes past it; the
standards are in `docs/SHADERS.md`. Update the catalog there when a shader
is added or removed.

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
