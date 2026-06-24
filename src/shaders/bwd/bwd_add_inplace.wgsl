@group(0) @binding(0) var<storage, read_write> target: array<f32>;
@group(0) @binding(1) var<storage, read> source: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    target[idx] = target[idx] + source[idx];
}
