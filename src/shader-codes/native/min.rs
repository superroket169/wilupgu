use super::common::{find, read_f32, write_f32};
use crate::core::shader::CpuBinding;

pub(crate) fn min(bindings: &[CpuBinding]) {
    let mut x = read_f32(find(bindings, 0));
    let y = read_f32(find(bindings, 1));
    for (xi, yi) in x.iter_mut().zip(y.iter()) {
        *xi = xi.min(*yi);
    }
    write_f32(find(bindings, 0), &x);
}
