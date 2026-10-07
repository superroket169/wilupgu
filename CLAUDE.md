# wilupgu

General-purpose GPU compute: graph + placement + backends.
No autograd, no shape system, no NN-training-specific ops — those belong to
sequexa-core, which uses wilupgu.

## Layout

- `src/backend/` is the contract: everything a backend implements or uses.
  `mod.rs` holds the traits; next to it: dtype, id, shader, table, pool, io_log.
- `src/backends/` holds backend implementations (cuda, cpu, ...).
  `toy.rs` is a test-only backend (`cfg(test)`).
- `src/tools/` holds optional packages built on the contract. A user can pick
  a backend and use none of them.
  - `core/`: device, tensor, node, graph, rules, deferred, run.
  - `spread/`: spread, placement, mesh. Uses `core`.
  - `builtins/`: the builtin shader table; `codes/{wgsl,native,cuda}/` has
    one file per shader per format. Directory names mirror `ShaderCode` variants.
  One file = one subject.
- `src/tests/`: one file per topic. There is only one tests directory.

## Builtin shaders

A builtin must be general-purpose GPU math. Optimizers, schedules and
attention-specific ops are not builtins; they go to sequexa-core.
The catalog is `docs/SHADERS.md`; update it when a shader is added or removed.

## Docs

- `docs/README.md`: crate overview.
- `docs/SHADERS.md`: shader catalog and per-format status.
- `docs/TODO.md`: open work. Planned work lives there
