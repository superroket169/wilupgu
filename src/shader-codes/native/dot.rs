use super::common::{find, read_f32, read_u32, write_f32};
use crate::core::shader::CpuBinding;

pub(crate) fn dot(bindings: &[CpuBinding]) {
    let a = read_f32(find(bindings, 0));
    let b = read_f32(find(bindings, 1));
    let meta = read_u32(find(bindings, 3));
    let n = meta[0] as usize;
    let total: f32 = a[..n].iter().zip(b[..n].iter()).map(|(x, y)| x * y).sum();
    write_f32(find(bindings, 2), &[total]);
}
