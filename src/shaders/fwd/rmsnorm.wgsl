struct Meta {
    seq_len: u32,
    size: u32,
    eps: f32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> m: Meta;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= m.seq_len) {
        return;
    }
    let offset = row * m.size;

    var ss: f32 = 0.0;
    for (var i: u32 = 0u; i < m.size; i = i + 1u) {
        let val = x[offset + i];
        ss = ss + (val * val);
    }
    
    let rsqrt = 1.0 / sqrt((ss / f32(m.size)) + m.eps);

    for (var i: u32 = 0u; i < m.size; i = i + 1u) {
        output[offset + i] = x[offset + i] * rsqrt * weight[i];
    }
}
