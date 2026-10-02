use super::common::{find, read_f32, read_u32, write_f32};
use crate::core::shader::CpuBinding;

// Unlike the wgsl version, the CPU reference needs no multi-pass reduction:
// one thread, one loop, done.
pub(crate) fn sum(bindings: &[CpuBinding]) {
    let x = read_f32(find(bindings, 0));
    let meta = read_u32(find(bindings, 2));
    let n = meta[0] as usize;
    let total: f32 = x[..n].iter().sum();
    write_f32(find(bindings, 1), &[total]);
}
