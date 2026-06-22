struct Meta {
    seq_len: u32,
}

@group(0) @binding(0) var<storage, read_write> attention_scores: array<f32>;
@group(0) @binding(1) var<storage, read> config: Meta;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;

    if (row >= config.seq_len || col >= config.seq_len) {
        return;
    }

    if (col > row) {
        let idx = row * config.seq_len + col;
        attention_scores[idx] = -1000000000.0;
    }
}
