# Shaders

Builtin shaders are general-purpose GPU math. Each one has a source file per
format under `src/tools/builtins/codes/{wgsl,native,cuda}/`, named after the shader.

The table in `src/tools/builtins/mod.rs` is the single source for every builtin:
its name, meta fields, tensor layout, workgroup size and which formats it has.
File paths, file names and native function names are all derived from the
shader's name, so a missing or misnamed file is a compile error.

## Standards

Every builtin follows these. The tests in `src/tests/builtins.rs` check them
over `builtins::ALL`; the WGSL checks parse the source with `naga` and run
with `--features wgpu`.

| # | Standard | Checked by |
|---|---|---|
| 1 | The entry point is `entry` in every format. | native: the compiler (the table calls `native::<name>::entry`); WGSL: naga; CUDA: signature test |
| 2 | Slot 0 is the meta, tensors are slots `1..=layout.len()`, every shader has exactly one meta. | meta test; naga (binding slots) |
| 3 | The meta is `struct ShaderMeta`, bound as `shader_meta`, `var<storage, read>`, with scalar `u32`/`f32` fields only, named and ordered as in the table. | naga |
| 4 | Bounds come from a meta field (`n`, `M`/`N`/`K`, `rows`/`cols`), never from `arrayLength`: a buffer may be larger than the tensor it holds. | text test |
| 5 | `workgroup_size` lives in the table. WGSL's `@workgroup_size` must match it; CUDA launches with it as `blockDim`. | naga |
| 6 | CUDA signature: `extern "C" __global__ void entry(const unsigned int* shader_meta, ...)`, then one pointer per tensor slot in slot order, `const float*` for inputs, `float*` otherwise. No scalar arguments; floats in the meta are read with `__uint_as_float`. | signature test |
| 7 | Native code takes lengths from the meta, writes only the elements it owns, and uses `native/common.rs`. | parity tests (not yet written) |
| 8 | Comments are in English. | ASCII test |
| 9 | Every file starts with the same one-line formula comment in every format. | formula test |

## Catalog

Every builtin has all three formats: WGSL, native and CUDA.

### Linear algebra

| Shader | Formula | Meta | Tensors | Workgroup |
|---|---|---|---|---|
| `matmul` | `C[m*N+n] = sum_k A[m*K+k] * B[k*N+n]` | `M N K` | A in, B in, C out | 16x16 |
| `matmul_add` | `C[m*N+n] += sum_k A[m*K+k] * B[k*N+n]` | `M N K` | A in, B in, C accumulate | 16x16 |
| `matmul_trp` | `C[m*N+n] = sum_k A[m*K+k] * B[n*K+k]` (B is NxK) | `M N K` | A in, B in, C out | 16x16 |
| `gemv` | `C[n] = sum_k A[k] * B[k*N+n]` | `M N K` (M ignored) | A in, B in, C out | 256 |
| `gemv_add` | `C[n] += sum_k A[k] * B[k*N+n]` | `M N K` (M ignored) | A in, B in, C accumulate | 256 |
| `transpose` | `dst[c*rows+r] = src[r*cols+c]` | `rows cols` | src in, dst out | 16x16 |
| `dot` | `partial[w] = sum a[i] * b[i]` over workgroup `w` | `n` | a in, b in, partial out | 256 |

### Elementwise

| Shader | Formula | Meta | Tensors | Workgroup |
|---|---|---|---|---|
| `add` | `x[i] += y[i]` | `n` | x accumulate, y in | 256 |
| `mul` | `x[i] *= y[i]` | `n` | x in-out, y in | 256 |
| `scale` | `x[i] *= factor` | `n factor` | x in-out | 256 |
| `max` | `x[i] = max(x[i], y[i])` | `n` | x in-out, y in | 256 |
| `min` | `x[i] = min(x[i], y[i])` | `n` | x in-out, y in | 256 |
| `clamp` | `x[i] = clamp(x[i], lo, hi)` | `n lo hi` | x in-out | 256 |

### Reduction

| Shader | Formula | Meta | Tensors | Workgroup |
|---|---|---|---|---|
| `sum` | `partial[w] = sum x[i]` over workgroup `w` | `n` | x in, partial out | 256 |

`sum` and `dot` reduce one workgroup's slice into one partial. On the GPU,
when more than one workgroup ran, `sum` is called again with the partials as
its input (and their count as `n`) until one value is left. Native code runs
as a single workgroup and writes the whole result to `partial[0]`.

### Buffer init

| Shader | Formula | Meta | Tensors | Workgroup |
|---|---|---|---|---|
| `zero_tensor` | `x[i] = 0` | `n` | x out | 256 |
| `fill_constant` | `x[i] = value` | `n value` | x out | 256 |
| `fill_random` | `x[i] = hash(i ^ seed) / 0xFFFFFFFF` | `n seed` | x out | 256 |

`fill_random` uses a cheap integer hash, identical in every format so their
outputs match. It isn't cryptographic or statistically rigorous.

## Not builtins

These are training- or model-specific and belong to sequexa-core:

- **AdamW**, **AdamWSchedule**: optimizer and schedule.
- **CausalMask**: autoregressive attention mask.

**MatMulWeightBwd** (`dB += A^T * dC`) isn't a separate primitive: `transpose`
followed by `matmul_add` covers it.
