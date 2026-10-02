use super::common::{find, read_f32, write_f32};
use crate::core::shader::CpuBinding;

pub(crate) fn add(bindings: &[CpuBinding]) {
    let mut x = read_f32(find(bindings, 0));
    let other = read_f32(find(bindings, 1));
    for (xi, oi) in x.iter_mut().zip(other.iter()) {
        *xi += oi;
    }
    write_f32(find(bindings, 0), &x);
}
