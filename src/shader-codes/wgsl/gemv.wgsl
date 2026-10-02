// C[n] = sum_k A[k] * B[k * N + n]   (M = 1; A: 1xK, B: KxN, C: 1xN)
struct ShaderMeta {
    M: u32,
    N: u32,
    K: u32,
}

@group(0) @binding(0) var<storage, read> shader_meta: ShaderMeta;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn entry(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let col = global_id.x;
    if (col >= shader_meta.N) {
        return;
    }
    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < shader_meta.K; k = k + 1u) {
        sum = sum + A[k] * B[k * shader_meta.N + col];
    }
    C[col] = sum;
}
