use super::common::{find, read_u32, write_f32};
use crate::core::shader::CpuBinding;

pub(crate) fn zero_tensor(bindings: &[CpuBinding]) {
    let meta = read_u32(find(bindings, 1));
    let len = meta[0] as usize;
    write_f32(find(bindings, 0), &vec![0.0f32; len]);
}
