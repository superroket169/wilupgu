use std::fmt;

use crate::backend::dtype::{DataKind, DataType, F32};
use crate::backend::shader::{BindingRole, Shader, Signature, Workgroups};
use crate::backend::{Backend, Topology};
use crate::error::Error;
use crate::safe::{Arg, ComputeDevice, MetaValue, Node};

// Written past `min_lens` in every slot before the run; a run that changes
// it wrote out of bounds. Compared by bits, and not a NaN.
const PAD: u32 = 0xDEAD_BEEF;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Check {
    Close { atol: f32, rtol: f32 },
    Total { atol: f32, rtol: f32 },
    Ignore,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Case {
    meta: Vec<MetaValue>,
    workgroups: Workgroups,
    checks: Vec<Check>,
    seed: u64,
}

impl Case {
    pub fn new(meta: &[MetaValue], workgroups: Workgroups, checks: &[Check]) -> Self {
        Self {
            meta: meta.to_vec(),
            workgroups,
            checks: checks.to_vec(),
            seed: 0,
        }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn meta(&self) -> &[MetaValue] {
        &self.meta
    }

    pub fn workgroups(&self) -> Workgroups {
        self.workgroups
    }

    pub fn checks(&self) -> &[Check] {
        &self.checks
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }
}

// The worst element of a `Close`, or the two sums of a `Total` (no index).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Diff {
    index: Option<usize>,
    values: [f32; 2],
    err: f32,
    tol: f32,
}

impl Diff {
    pub fn index(&self) -> Option<usize> {
        self.index
    }

    pub fn values(&self) -> [f32; 2] {
        self.values
    }

    pub fn err(&self) -> f32 {
        self.err
    }

    pub fn tol(&self) -> f32 {
        self.tol
    }

    // A NaN error never passes.
    pub fn passed(&self) -> bool {
        self.err <= self.tol
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SlotReport {
    role: BindingRole,
    check: Check,
    len: usize,
    diff: Option<Diff>,
    // Per side: the first element of an `Output` slot left NaN.
    unwritten: [Option<usize>; 2],
    // Per side: the first element past `min_lens` that changed.
    overrun: [Option<usize>; 2],
}

impl SlotReport {
    pub fn role(&self) -> &BindingRole {
        &self.role
    }

    pub fn check(&self) -> Check {
        self.check
    }

    // The slot's `min_lens`.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    // `None` for `Ignore`, and for a `Close` on an empty slot.
    pub fn diff(&self) -> Option<&Diff> {
        self.diff.as_ref()
    }

    pub fn unwritten(&self) -> [Option<usize>; 2] {
        self.unwritten
    }

    pub fn overrun(&self) -> [Option<usize>; 2] {
        self.overrun
    }

    pub fn passed(&self) -> bool {
        self.diff.is_none_or(|d| d.passed())
            && self.unwritten == [None; 2]
            && self.overrun == [None; 2]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Report {
    shader: &'static str,
    backends: [&'static str; 2],
    case: Case,
    slots: Vec<SlotReport>,
}

impl Report {
    pub fn shader(&self) -> &'static str {
        self.shader
    }

    pub fn backends(&self) -> [&'static str; 2] {
        self.backends
    }

    pub fn case(&self) -> &Case {
        &self.case
    }

    // Index 0 is slot 1.
    pub fn slots(&self) -> &[SlotReport] {
        &self.slots
    }

    pub fn passed(&self) -> bool {
        self.slots.iter().all(SlotReport::passed)
    }
}

// Runs `case` with `a`'s shader on `a`'s device and `b`'s on `b`'s, from the
// same data, and compares every slot by its check. Both shaders must have the
// same signature. Every slot holds `min_lens` elements of seeded data, then
// `PAD` words; an `Output` slot starts all NaN, or all 0 under `Total`.
pub fn compare<A: Backend + Topology, B: Backend + Topology>(
    a: (&ComputeDevice<A>, &Shader),
    b: (&ComputeDevice<B>, &Shader),
    case: &Case,
) -> Result<Report, Error> {
    let signature = a.1.signature();
    if !std::ptr::eq(signature, b.1.signature()) {
        return Err(Error::SignatureMismatch {
            a: signature.name,
            b: b.1.signature().name,
        });
    }
    if case.checks.len() != signature.layout.len() {
        return Err(Error::CheckCount {
            shader: signature.name,
            expected: signature.layout.len(),
            got: case.checks.len(),
        });
    }
    for (i, role) in signature.layout.iter().enumerate() {
        if role.kind() != DataKind::F32 {
            return Err(Error::ParityKind {
                shader: signature.name,
                slot: i + 1,
                kind: role.kind(),
            });
        }
    }

    let mut node_a = a.0.node(a.1, &case.meta, case.workgroups)?;
    let mut node_b = b.0.node(b.1, &case.meta, case.workgroups)?;
    let lens = node_a.min_lens().to_vec();
    let start = start_data(signature, case, &lens);
    let out = [
        run(a.0, &mut node_a, &start)?,
        run(b.0, &mut node_b, &start)?,
    ];

    let slots = signature
        .layout
        .iter()
        .zip(&case.checks)
        .zip(&lens)
        .enumerate()
        .map(|(i, ((role, &check), &len))| {
            let [ra, rb] = [&out[0][i], &out[1][i]];
            let diff = match check {
                Check::Close { atol, rtol } => worst(&ra[..len], &rb[..len], atol, rtol),
                Check::Total { atol, rtol } => Some(totals(&ra[..len], &rb[..len], atol, rtol)),
                Check::Ignore => None,
            };
            let canary =
                matches!(role, BindingRole::Output(_)) && !matches!(check, Check::Total { .. });
            SlotReport {
                role: role.clone(),
                check,
                len,
                diff,
                unwritten: [ra, rb].map(|r| {
                    if canary {
                        r[..len].iter().position(|v| v.is_nan())
                    } else {
                        None
                    }
                }),
                overrun: [ra, rb].map(|r| {
                    r[len..]
                        .iter()
                        .position(|v| v.to_bits() != PAD)
                        .map(|p| len + p)
                }),
            }
        })
        .collect();

    Ok(Report {
        shader: signature.name,
        backends: [A::NAME, B::NAME],
        case: case.clone(),
        slots,
    })
}

// One workgroup's threads: a shader that skips its bounds check writes at
// most that far past the end.
fn pad_len(signature: &Signature) -> usize {
    signature.workgroup_size.iter().product::<u32>().max(1) as usize
}

fn start_data(signature: &Signature, case: &Case, lens: &[usize]) -> Vec<Vec<f32>> {
    let pad = pad_len(signature);
    signature
        .layout
        .iter()
        .zip(&case.checks)
        .zip(lens)
        .enumerate()
        .map(|(i, ((role, check), &len))| {
            let mut data: Vec<f32> = match (role, check) {
                (BindingRole::Output(_), Check::Total { .. }) => vec![0.0; len],
                (BindingRole::Output(_), _) => vec![f32::NAN; len],
                _ => random(case.seed, i, len),
            };
            data.resize(len + pad, f32::from_bits(PAD));
            data
        })
        .collect()
}

// `len` values in [-1, 1), from splitmix64 seeded by `seed` and the slot.
fn random(seed: u64, slot: usize, len: usize) -> Vec<f32> {
    let mut state = seed ^ (slot as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..len)
        .map(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            // The top 24 bits, exact in an f32.
            (z >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

fn run<B: Backend + Topology>(
    dev: &ComputeDevice<B>,
    node: &mut Node<'_, B>,
    start: &[Vec<f32>],
) -> Result<Vec<Vec<f32>>, Error> {
    let mut tensors = start
        .iter()
        .map(|data| dev.tensor_from(&F32::wrap(data.clone())))
        .collect::<Result<Vec<_>, Error>>()?;
    let layout = node.signature().layout;
    dev.scope(|s| {
        let node = s.bind_node(node)?;
        let args = tensors
            .iter_mut()
            .zip(layout)
            .map(|(tensor, role)| {
                let slot = s.bind(tensor)?;
                Ok(match role {
                    BindingRole::Input(_) => Arg::Read(slot),
                    _ => Arg::Write(slot),
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        s.run(node, &args)
    })?;
    tensors.iter().map(|t| t.download_vec::<F32>()).collect()
}

fn tolerance(a: f32, b: f32, atol: f32, rtol: f32) -> f32 {
    atol + rtol * a.abs().max(b.abs())
}

// The element furthest over its tolerance; a NaN error counts as the furthest.
fn worst(a: &[f32], b: &[f32], atol: f32, rtol: f32) -> Option<Diff> {
    let over = |d: &Diff| {
        let o = d.err - d.tol;
        if o.is_nan() {
            f32::INFINITY
        } else {
            o
        }
    };
    a.iter()
        .zip(b)
        .enumerate()
        .map(|(i, (&x, &y))| Diff {
            index: Some(i),
            values: [x, y],
            err: (x - y).abs(),
            tol: tolerance(x, y, atol, rtol),
        })
        .reduce(|w, d| if over(&d) > over(&w) { d } else { w })
}

fn totals(a: &[f32], b: &[f32], atol: f32, rtol: f32) -> Diff {
    let sum = |v: &[f32]| v.iter().map(|&x| x as f64).sum::<f64>() as f32;
    let (x, y) = (sum(a), sum(b));
    Diff {
        index: None,
        values: [x, y],
        err: (x - y).abs(),
        tol: tolerance(x, y, atol, rtol),
    }
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let [a, b] = self.backends;
        let wg = self.case.workgroups;
        writeln!(
            f,
            "{}  meta={:?}  workgroups=[{},{},{}]  seed={}  {a} vs {b}",
            self.shader, self.case.meta, wg.x, wg.y, wg.z, self.case.seed
        )?;
        for (i, slot) in self.slots.iter().enumerate() {
            let verdict = if slot.passed() { "ok  " } else { "FAIL" };
            write!(
                f,
                "  slot {}  {:<18}  {verdict}",
                i + 1,
                format!("{:?}", slot.role)
            )?;
            match &slot.diff {
                Some(d) => {
                    let at = d.index.map_or("sum".to_string(), |i| format!("[{i}]"));
                    write!(
                        f,
                        "  worst {at}  {} vs {}  err {:.1e}  tol {:.1e}",
                        d.values[0], d.values[1], d.err, d.tol
                    )?;
                }
                None if slot.check == Check::Ignore => write!(f, "  skip")?,
                None => write!(f, "  empty")?,
            }
            writeln!(f)?;
            for (name, at) in self.backends.iter().zip(slot.unwritten) {
                if let Some(at) = at {
                    writeln!(f, "    {name} left [{at}] unwritten")?;
                }
            }
            for (name, at) in self.backends.iter().zip(slot.overrun) {
                if let Some(at) = at {
                    writeln!(f, "    {name} wrote past min_lens at [{at}]")?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../tests/parity.rs"]
mod tests;
