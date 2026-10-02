use super::common::{find, read_u32, write_f32};
use crate::core::shader::CpuBinding;

pub(crate) fn fill_constant(bindings: &[CpuBinding]) {
    let meta = read_u32(find(bindings, 1));
    let len = meta[0] as usize;
    let value = f32::from_bits(meta[1]);
    write_f32(find(bindings, 0), &vec![value; len]);
}
