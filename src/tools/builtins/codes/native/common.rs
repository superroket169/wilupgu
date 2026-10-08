use crate::backend::shader::NativeBinding;

// Slots count from 1, like the WGSL bindings; `bindings[0]` is slot 1.
pub(super) fn read_f32(bindings: &[NativeBinding], slot: usize) -> Vec<f32> {
    let bytes: &[u8] = match &bindings[slot - 1] {
        NativeBinding::Read(b) => b,
        NativeBinding::Write(b) => b,
    };
    bytemuck::pod_collect_to_vec::<u8, f32>(bytes)
}

pub(super) fn write_f32(bindings: &mut [NativeBinding], slot: usize, data: &[f32]) {
    match &mut bindings[slot - 1] {
        NativeBinding::Write(b) => b.copy_from_slice(bytemuck::cast_slice(data)),
        NativeBinding::Read(_) => panic!("slot {slot} is bound for reading only"),
    }
}
