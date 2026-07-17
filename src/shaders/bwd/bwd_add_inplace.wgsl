@group(0) @binding(0) var<storage, read_write> t: array<f32>;
@group(0) @binding(1) var<storage, read> source: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let idx = (wg_id.y * num_wg.x + wg_id.x) * 256u + local_id.x;

    if (idx >= arrayLength(&t) || idx >= arrayLength(&source)) {
        return;
    }

    t[idx] = t[idx] + source[idx];
}
