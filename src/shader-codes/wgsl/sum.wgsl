// partial[w] = sum of x[i] over workgroup w's slice   (i < n; native runs one workgroup)
struct ShaderMeta {
    n: u32,
}

@group(0) @binding(0) var<storage, read> shader_meta: ShaderMeta;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> partial: array<f32>;

var<workgroup> scratch: array<f32, 256>;

// One partial sum per workgroup. If more than one workgroup ran, call again
// with `partial` as the new `x` (and its length as the new `n`) until one
// value is left.
@compute @workgroup_size(256, 1, 1)
fn entry(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let idx = (wg_id.y * num_wg.x + wg_id.x) * 256u + local_id.x;
    scratch[local_id.x] = select(0.0, x[idx], idx < shader_meta.n);
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (local_id.x < stride) {
            scratch[local_id.x] = scratch[local_id.x] + scratch[local_id.x + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (local_id.x == 0u) {
        partial[wg_id.y * num_wg.x + wg_id.x] = scratch[0];
    }
}
