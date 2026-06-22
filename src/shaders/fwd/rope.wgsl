struct Meta {
    seq_len: u32,
    dim: u32,
    head_dim: u32,
}

@group(0) @binding(0) var<storage, read_write> vec: array<f32>;
@group(0) @binding(1) var<storage, read> meta: Meta;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let token_idx = global_id.y;
    let dim_idx = global_id.x * 2u;

    if (token_idx >= meta.seq_len || dim_idx >= meta.head_dim) {
        return;
    }

    let num_heads = meta.dim / meta.head_dim;

    for (var h: u32 = 0u; h < num_heads; h = h + 1u) {
        let offset = token_idx * meta.dim + h * meta.head_dim + dim_idx;

        let x0 = vec[offset];
        let x1 = vec[offset + 1u];

        // TODO: will add inv_freq for optimization
        let freq = 1.0 / pow(10000.0, f32(dim_idx) / f32(meta.head_dim));
        let v_angle = f32(token_idx) * freq;

        let v_cos = cos(v_angle);
        let v_sin = sin(v_angle);

        vec[offset]       = x0 * v_cos - x1 * v_sin;
        vec[offset + 1u]  = x0 * v_sin + x1 * v_cos;
    }
}
