// x[i] = value   (i < n)
struct ShaderMeta {
    n: u32,
    value: f32,
}

@group(0) @binding(0) var<storage, read> shader_meta: ShaderMeta;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn entry(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let idx = (wg_id.y * num_wg.x + wg_id.x) * 256u + local_id.x;
    if (idx >= shader_meta.n) {
        return;
    }
    x[idx] = shader_meta.value;
}
