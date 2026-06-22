struct Meta {
    M: u32,
    N: u32,
    K: u32,
}

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<storage, read> config: Meta;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;

    if (row >= config.M || col >= config.N) {
        return;
    }

    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < config.K; k = k + 1u) {
        let indexA = row * config.K + k;
        let indexB = k * config.N + col;
        sum = sum + A[indexA] * B[indexB];
    }

    let indexC = row * config.N + col;
    C[indexC] = sum;
}
