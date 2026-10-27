use super::{compare, Case, Check, Report};
use crate::backend::shader::{ShaderBinary, Workgroups};
use crate::backend::{Backend, Topology};
use crate::error::Error;
use crate::safe::{ComputeDevice, MetaValue};
use crate::tools::builtins::{Builtin, ALL};

use MetaValue::{Float, Uint};

// Every builtin, every case of it, on `a` and on `b`.
pub fn compare_builtins<A: Backend + Topology, B: Backend + Topology>(
    a: &ComputeDevice<A>,
    b: &ComputeDevice<B>,
) -> Result<Vec<Report>, Error> {
    let mut reports = Vec::new();
    for builtin in ALL {
        let (sa, sb) = (builtin.shader_for::<A>()?, builtin.shader_for::<B>()?);
        let with_native = [&sa, &sb]
            .iter()
            .any(|s| matches!(s.binary(), ShaderBinary::Native(_)));
        for case in cases(builtin, with_native) {
            reports.push(compare((a, &sa), (b, &sb), &case)?);
        }
    }
    Ok(reports)
}

// Edge cases for each builtin: empty, one, and both sides of a workgroup's
// edge. A reduction's partials are compared as a `Total` when native code
// runs on a side, since it writes the whole sum to the first partial.
pub fn cases(builtin: &Builtin, with_native: bool) -> Vec<Case> {
    const EXACT: Check = close(1e-6, 0.0);
    let partials = if with_native {
        Check::Total {
            atol: 1e-3,
            rtol: 1e-5,
        }
    } else {
        close(1e-3, 1e-5)
    };
    let ignore = Check::Ignore;
    let products = close(1e-4, 1e-5);

    let cases: Vec<Case> = match builtin.signature.name {
        "matmul" | "matmul_trp" | "matmul_add" => {
            [(1, 1, 1), (16, 16, 16), (17, 15, 33), (5, 40, 3), (3, 4, 0)]
                .map(|(m, n, k)| {
                    Case::new(
                        &[Uint(m), Uint(n), Uint(k)],
                        Workgroups {
                            x: tiles(n, 16),
                            y: tiles(m, 16),
                            z: 1,
                        },
                        &[ignore, ignore, products],
                    )
                })
                .into()
        }
        "gemv" | "gemv_add" => [(1, 1), (255, 7), (257, 33), (300, 0)]
            .map(|(n, k)| {
                Case::new(
                    &[Uint(1), Uint(n), Uint(k)],
                    Workgroups::linear(tiles(n, 256)),
                    &[ignore, ignore, products],
                )
            })
            .into(),
        "transpose" => [(1, 1), (16, 16), (17, 33), (1, 300)]
            .map(|(rows, cols)| {
                Case::new(
                    &[Uint(rows), Uint(cols)],
                    Workgroups {
                        x: tiles(cols, 16),
                        y: tiles(rows, 16),
                        z: 1,
                    },
                    &[ignore, EXACT],
                )
            })
            .into(),
        "dot" => each_n(|_| vec![], &[ignore, ignore, partials]),
        "sum" => each_n(|_| vec![], &[ignore, partials]),
        "add" | "mul" | "max" | "min" => each_n(|_| vec![], &[EXACT, ignore]),
        "scale" => each_n(|_| vec![Float(-1.5)], &[EXACT]),
        // Inputs are in [-1, 1), so some of them are clamped and some aren't.
        "clamp" => each_n(|_| vec![Float(-0.5), Float(0.25)], &[EXACT]),
        "zero_tensor" => each_n(|_| vec![], &[EXACT]),
        "fill_constant" => each_n(|_| vec![Float(3.5)], &[EXACT]),
        // WGSL may round a division by up to 2.5 ULP.
        "fill_random" => each_n(|n| vec![Uint(n ^ 0x5EED)], &[close(0.0, 1e-6)]),
        _ => vec![],
    };
    cases
        .into_iter()
        .enumerate()
        .map(|(i, case)| case.with_seed(i as u64))
        .collect()
}

const fn close(atol: f32, rtol: f32) -> Check {
    Check::Close { atol, rtol }
}

// Workgroups of `size` threads to cover `n`; at least one.
fn tiles(n: u32, size: u32) -> u32 {
    n.div_ceil(size).max(1)
}

// A case for each `n`, with meta `[n, rest(n)..]` and one 256-thread
// workgroup per 256 elements.
fn each_n(rest: impl Fn(u32) -> Vec<MetaValue>, checks: &[Check]) -> Vec<Case> {
    [0, 1, 255, 256, 257, 1000]
        .map(|n| {
            let mut meta = vec![Uint(n)];
            meta.extend(rest(n));
            Case::new(&meta, Workgroups::linear(tiles(n, 256)), checks)
        })
        .into()
}

#[cfg(test)]
#[path = "../../tests/parity_builtins.rs"]
mod tests;
