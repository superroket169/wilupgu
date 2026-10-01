struct Meta {
    n: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> partial: array<f32>;
@group(0) @binding(3) var<storage, read> config: Meta;

var<workgroup> scratch: array<f32, 256>;

// First pass of a dot product: multiply then reduce, one partial per
// workgroup. `sum.wgsl` finishes the job if more than one partial came out.
@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let idx = (wg_id.y * num_wg.x + wg_id.x) * 256u + local_id.x;
    scratch[local_id.x] = select(0.0, a[idx] * b[idx], idx < config.n);
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
