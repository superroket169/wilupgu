# wilupgu

General-purpose GPU compute: graph + placement + backends.
No autograd, no shape system, no NN-training-specific ops — those belong to
sequexa-core, which uses wilupgu.

## Layout

- `src/backend.rs` is the contract: only what `src/backends/*` implements.
- `src/backends/` holds backend implementations (cuda, cpu, ...).
- `src/core/` is the crate's own domain: device, dtype, graph, node, tensor,
  spread, placement, resolver, rules, shader, mesh, id, io_log.
  One file = one subject.
- `src/shader-codes/{wgsl,native,cuda}/`: one file per shader per format.
  Directory names mirror `ShaderCode` variants.
- `src/tests/`: one file per topic. There is only one tests directory.

## Builtin shaders

A builtin must be general-purpose GPU math. Optimizers, schedules and
attention-specific ops are not builtins; they go to sequexa-core.
The catalog is `docs/SHADERS.md`; update it when a shader is added or removed.

## Docs

- `docs/README.md`: crate overview.
- `docs/SHADERS.md`: shader catalog and per-format status.
- `docs/TODO.md`: open work. Planned work lives there
