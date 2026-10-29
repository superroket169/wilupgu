use super::*;
use crate::backend::shader::{BindingRole, Shader, ShaderBinary};

#[test]
fn all_lists_every_builtin_once() {
    assert_eq!(ALL.len(), 17);
    for (i, s) in ALL.iter().enumerate() {
        assert!(
            ALL[..i]
                .iter()
                .all(|o| o.signature.name != s.signature.name),
            "`{}` is listed twice",
            s.signature.name
        );
    }
}

// --- shader standards (docs/SHADERS.md) ---

fn native_source(s: &Signature) -> String {
    let path = format!(
        "{}/src/tools/builtins/codes/native/{}.rs",
        env!("CARGO_MANIFEST_DIR"),
        s.name
    );
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

// Every source this shader has, with the format's name.
fn sources(b: &Builtin) -> Vec<(&'static str, String)> {
    vec![
        ("wgsl", b.wgsl.0.to_string()),
        ("native", native_source(&b.signature)),
        ("cuda", b.cuda.0.to_string()),
    ]
}

// standard 2: exactly one meta, at slot 0
#[test]
fn every_builtin_has_meta() {
    for b in ALL {
        let s = &b.signature;
        assert!(!s.meta.is_empty(), "`{}` has no meta", s.name);
    }
}

// standard 9
#[test]
fn every_format_starts_with_the_same_formula() {
    for b in ALL {
        let s = &b.signature;
        let srcs = sources(b);
        let (first_format, first) = &srcs[0];
        let formula = first.lines().next().unwrap_or("");
        assert!(
            formula.starts_with("// "),
            "`{}` ({first_format}) doesn't start with a formula comment",
            s.name
        );
        for (format, src) in &srcs[1..] {
            assert_eq!(
                src.lines().next().unwrap_or(""),
                formula,
                "`{}`: {format}'s formula differs from {first_format}'s",
                s.name
            );
        }
    }
}

// standard 8, as far as a test can tell: no Turkish (or any non-ASCII) text
#[test]
fn sources_are_ascii() {
    for b in ALL {
        let s = &b.signature;
        for (format, src) in sources(b) {
            assert!(src.is_ascii(), "`{}` ({format}) has non-ASCII text", s.name);
        }
    }
}

// standard 4
#[test]
fn wgsl_never_uses_array_length() {
    for b in ALL {
        let s = &b.signature;
        assert!(
            !b.wgsl.0.contains("arrayLength"),
            "`{}` uses arrayLength",
            s.name
        );
    }
}

// standard 1 for native: the builtins table calls `native::<name>::entry`, so
// the compiler checks it.

// standards 1 and 6
#[test]
fn cuda_signature_follows_the_layout() {
    for b in ALL {
        let s = &b.signature;
        let src = &b.cuda.0;
        let head = "extern \"C\" __global__ void entry(";
        let start = src
            .find(head)
            .unwrap_or_else(|| panic!("`{}` has no `{head}`", s.name))
            + head.len();
        let params: Vec<&str> = src[start..start + src[start..].find(')').unwrap()]
            .split(',')
            .map(str::trim)
            .collect();
        let mut expected = vec!["const unsigned int* shader_meta"];
        for role in s.layout {
            expected.push(match role {
                BindingRole::Input(_) => "const float* ",
                _ => "float* ",
            });
        }
        assert_eq!(params.len(), expected.len(), "`{}` parameter count", s.name);
        assert_eq!(params[0], expected[0], "`{}` slot 0", s.name);
        for (i, (p, e)) in params.iter().zip(&expected).enumerate().skip(1) {
            assert!(
                p.starts_with(e),
                "`{}` slot {i}: `{p}`, expected `{e}...`",
                s.name
            );
        }
    }
}

// standards 1, 2, 3 and 5, read off the parsed WGSL
#[cfg(feature = "wgpu")]
#[test]
fn wgsl_follows_the_standards() {
    use naga::{AddressSpace, Scalar, StorageAccess, TypeInner};

    for b in ALL {
        let s = &b.signature;
        let name = s.name;
        let module = naga::front::wgsl::parse_str(&b.wgsl.0)
            .unwrap_or_else(|e| panic!("`{name}` doesn't parse: {e}"));

        assert_eq!(module.entry_points.len(), 1, "`{name}` entry points");
        let entry = &module.entry_points[0];
        assert_eq!(entry.name, "entry", "`{name}` entry point name");
        assert_eq!(
            entry.workgroup_size, s.workgroup_size,
            "`{name}` workgroup size"
        );

        let mut bound: Vec<_> = module
            .global_variables
            .iter()
            .filter_map(|(_, v)| v.binding.as_ref().map(|b| (b.binding, v)))
            .collect();
        bound.sort_by_key(|(b, _)| *b);
        let slots: Vec<u32> = bound.iter().map(|(b, _)| *b).collect();
        let expected: Vec<u32> = (0..=s.layout.len() as u32).collect();
        assert_eq!(slots, expected, "`{name}` binding slots");

        for (slot, var) in &bound {
            let AddressSpace::Storage { access } = var.space else {
                panic!("`{name}` slot {slot} isn't a storage buffer");
            };
            let read_only = access == StorageAccess::LOAD;
            if *slot == 0 {
                assert_eq!(
                    var.name.as_deref(),
                    Some("shader_meta"),
                    "`{name}` slot 0 name"
                );
                assert!(read_only, "`{name}` shader_meta must be read-only");
                let ty = &module.types[var.ty];
                assert_eq!(
                    ty.name.as_deref(),
                    Some("ShaderMeta"),
                    "`{name}` meta struct name"
                );
                let TypeInner::Struct { members, .. } = &ty.inner else {
                    panic!("`{name}` shader_meta isn't a struct");
                };
                assert_eq!(members.len(), s.meta.len(), "`{name}` meta field count");
                for (member, field) in members.iter().zip(s.meta) {
                    assert_eq!(
                        member.name.as_deref(),
                        Some(field.name),
                        "`{name}` meta field name"
                    );
                    let want = match field.ty {
                        MetaType::Uint => Scalar::U32,
                        MetaType::Float => Scalar::F32,
                    };
                    assert_eq!(
                        module.types[member.ty].inner,
                        TypeInner::Scalar(want),
                        "`{name}` meta field `{}` type",
                        field.name
                    );
                }
            } else {
                let role = &s.layout[*slot as usize - 1];
                let input = matches!(role, BindingRole::Input(_));
                assert_eq!(read_only, input, "`{name}` slot {slot} access vs {role:?}");
            }
        }
    }
}

// --- native code on the CPU backend ---

#[test]
fn add_runs_on_the_cpu_backend() {
    use crate::backend::shader::Workgroups;
    use crate::backend::{Access, Dispatch, Storage, Topology};
    use crate::backends::CpuBackend;

    let ctx = CpuBackend::attach(CpuBackend::choosable_devices().remove(0)).unwrap();
    let mut x = ctx.alloc_raw(12).unwrap();
    let mut y = ctx.alloc_raw(12).unwrap();
    // SAFETY: both buffers were made for exactly 12 bytes.
    unsafe {
        ctx.upload_raw(&mut x, 0, bytemuck::cast_slice(&[1.0f32, 2.0, 3.0]));
        ctx.upload_raw(&mut y, 0, bytemuck::cast_slice(&[10.0f32, 20.0, 30.0]));
    }
    let add = Shader::new(&ADD.signature, ShaderBinary::Native(ADD.native));
    let node = ctx.build_node(&add, &[3], Workgroups::linear(1)).unwrap();
    // SAFETY: ADD's layout is (x accumulate, y in), both hold 3 F32s, n = 3,
    // and nothing touches them before `synchronize`.
    unsafe {
        ctx.execute_raw(&node, &mut [Access::Write(&mut x), Access::Read(&y)])
            .unwrap();
    }
    ctx.synchronize();
    // SAFETY: 12 bytes fit in x.
    let out: Vec<f32> = bytemuck::pod_collect_to_vec(&unsafe { ctx.download_raw(&x, 0, 12) });
    assert_eq!(out, vec![11.0, 22.0, 33.0]);
}

// standard 10, for native code: each builtin runs twice on the CPU, on buffers of
// exactly `min_lens` and on padded ones. A `min_lens` smaller than what the code
// touches panics, changes the result or writes into the padding.
#[test]
fn min_lens_cover_what_native_code_touches() {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use crate::backend::dtype::{DataKind, DataType, F32};
    use crate::backend::shader::{MetaType, Workgroups};
    use crate::backend::Topology;
    use crate::backends::{CpuBackend, CpuInfo};
    use crate::safe::{Arg, ComputeDevice, DeviceConfig, MetaValue};

    let dev = ComputeDevice::new(
        CpuBackend::attach(CpuInfo).unwrap(),
        DeviceConfig::default(),
    );
    let wg = Workgroups::linear(2);
    for b in ALL {
        let s = &b.signature;
        // A different value per field, so a mixed-up M/N/K shows.
        let meta: Vec<MetaValue> = (2..)
            .zip(s.meta)
            .map(|(v, field)| match field.ty {
                MetaType::Uint => MetaValue::Uint(v),
                MetaType::Float => MetaValue::Float(0.5),
            })
            .collect();
        let words: Vec<u32> = meta
            .iter()
            .map(|v| match v {
                MetaValue::Uint(u) => *u,
                MetaValue::Float(f) => f.to_bits(),
            })
            .collect();
        let mut lens = vec![0; s.layout.len()];
        (s.min_lens)(&words, wg, &mut lens);

        // Every slot after one run, with `pad` extra elements of PAD.
        const PAD: f32 = 100.0;
        let run = |pad: usize| -> Vec<Vec<f32>> {
            let mut bufs: Vec<_> = lens
                .iter()
                .enumerate()
                .map(|(slot, &len)| {
                    let data = (0..len + pad)
                        .map(|i| {
                            if i < len {
                                1.0 + 0.25 * (slot * 7 + i) as f32
                            } else {
                                PAD
                            }
                        })
                        .collect();
                    let mut buf = dev.alloc(DataKind::F32, len + pad).unwrap();
                    buf.upload(&F32::wrap(data)).unwrap();
                    buf
                })
                .collect();
            let shader = Shader::new(s, ShaderBinary::Native(b.native));
            let mut node = dev.node(&shader, &meta, wg).unwrap();
            dev.scope(|sc| {
                let node = sc.bind_node(&mut node).unwrap();
                let args: Vec<Arg> = bufs
                    .iter_mut()
                    .zip(s.layout)
                    .map(|(buf, role)| {
                        let slot = sc.bind(buf).unwrap();
                        match role {
                            BindingRole::Input(_) => Arg::Read(slot),
                            _ => Arg::Write(slot),
                        }
                    })
                    .collect();
                sc.execute(node, &args).unwrap();
            });
            bufs.iter()
                .map(|buf| F32::unwrap(buf.download()).unwrap())
                .collect()
        };

        let exact = catch_unwind(AssertUnwindSafe(|| run(0)))
            .unwrap_or_else(|_| panic!("`{}`: native code goes past min_lens", s.name));
        let padded = run(5);
        for (slot, &len) in lens.iter().enumerate() {
            let (inside, padding) = padded[slot].split_at(len);
            assert_eq!(
                inside,
                exact[slot],
                "`{}` slot {}: native code reads past min_lens",
                s.name,
                slot + 1
            );
            assert!(
                padding.iter().all(|&v| v == PAD),
                "`{}` slot {}: native code writes past min_lens",
                s.name,
                slot + 1
            );
        }
    }
}

#[test]
fn shader_for_picks_the_backends_format() {
    use crate::backends::CpuBackend;
    for b in ALL {
        let shader = b.shader_for::<CpuBackend>().unwrap();
        assert!(matches!(shader.binary(), ShaderBinary::Native(_)));
    }
}
