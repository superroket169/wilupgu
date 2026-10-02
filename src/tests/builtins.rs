use super::*;
use crate::core::shader::{BindingRole, ShaderFormat};

#[test]
fn all_lists_every_builtin_once() {
    assert_eq!(ALL.len(), 17);
    for (i, s) in ALL.iter().enumerate() {
        assert!(
            ALL[..i].iter().all(|o| o.name != s.name),
            "`{}` is listed twice",
            s.name
        );
    }
}

#[test]
fn every_builtin_has_some_code() {
    let formats = [ShaderFormat::Wgsl, ShaderFormat::Native, ShaderFormat::Cuda];
    for s in ALL {
        assert!(
            formats.iter().any(|&f| s.shader_code.has(f)),
            "`{}` has no code in any format",
            s.name
        );
    }
}

#[test]
fn code_paths_follow_the_shader_name() {
    let wgsl = ADD.shader_code.wgsl.unwrap();
    assert_eq!(wgsl.path(), "src/shader-codes/wgsl/add.wgsl");
    assert!(wgsl.source().contains("@compute"));
    let cuda = ADD.shader_code.cuda.unwrap();
    assert_eq!(cuda.path(), "src/shader-codes/cuda/add.cu");
}

// --- shader standards (docs/TODO.md) ---

fn native_source(s: &Shader) -> String {
    let path = format!(
        "{}/src/shader-codes/native/{}.rs",
        env!("CARGO_MANIFEST_DIR"),
        s.name
    );
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

// Every source this shader has, with the format's name.
fn sources(s: &Shader) -> Vec<(&'static str, String)> {
    let mut out = Vec::new();
    if let Some(c) = s.shader_code.wgsl {
        out.push(("wgsl", c.source().to_string()));
    }
    if s.shader_code.native.is_some() {
        out.push(("native", native_source(s)));
    }
    if let Some(c) = s.shader_code.cuda {
        out.push(("cuda", c.source().to_string()));
    }
    out
}

// standard 2: exactly one meta, at slot 0
#[test]
fn every_builtin_has_meta() {
    for s in ALL {
        assert!(!s.meta.is_empty(), "`{}` has no meta", s.name);
    }
}

// standard 9
#[test]
fn every_format_starts_with_the_same_formula() {
    for s in ALL {
        let srcs = sources(s);
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
    for s in ALL {
        for (format, src) in sources(s) {
            assert!(src.is_ascii(), "`{}` ({format}) has non-ASCII text", s.name);
        }
    }
}

// standard 4
#[test]
fn wgsl_never_uses_array_length() {
    for s in ALL {
        if let Some(c) = s.shader_code.wgsl {
            assert!(
                !c.source().contains("arrayLength"),
                "`{}` uses arrayLength",
                s.name
            );
        }
    }
}

// standard 1 for native: the builtins table calls `native::<name>::entry`, so
// the compiler checks it.

// standards 1 and 6
#[test]
fn cuda_signature_follows_the_layout() {
    for s in ALL {
        let Some(c) = s.shader_code.cuda else {
            continue;
        };
        let src = c.source();
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

    for s in ALL {
        let Some(code) = s.shader_code.wgsl else {
            continue;
        };
        let name = s.name;
        let module = naga::front::wgsl::parse_str(code.source())
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
