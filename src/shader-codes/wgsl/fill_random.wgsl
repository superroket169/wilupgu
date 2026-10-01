struct Meta {
    n: u32,
    seed: u32,
}

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> config: Meta;

// Not a cryptographic or statistically rigorous PRNG -- just a cheap,
// deterministic integer hash, so the CPU reference can reproduce it exactly.
fn hash(v: u32) -> u32 {
    var h = v;
    h = h ^ (h >> 16u);
    h = h * 0x7feb352du;
    h = h ^ (h >> 15u);
    h = h * 0x846ca68bu;
    h = h ^ (h >> 16u);
    return h;
}

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let idx = (wg_id.y * num_wg.x + wg_id.x) * 256u + local_id.x;
    if (idx >= config.n) {
        return;
    }
    x[idx] = f32(hash(idx ^ config.seed)) / 4294967295.0;
}
