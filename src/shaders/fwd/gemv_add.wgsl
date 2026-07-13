// C[1,N] += A[1,K] @ B[K,N]; the m=1 counterpart of matmul_add
struct Meta {
    M: u32,
    N: u32,
    K: u32,
}

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<storage, read> config: Meta;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let col = global_id.x;
    if (col >= config.N) {
        return;
    }
    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < config.K; k = k + 1u) {
        sum = sum + A[k] * B[k * config.N + col];
    }
    C[col] = C[col] + sum;
}
