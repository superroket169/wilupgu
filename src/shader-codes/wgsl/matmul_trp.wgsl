struct Meta {
    M: u32,
    N: u32,
    K: u32,
}

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>; // transpoze alınıyor olan
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<storage, read> config: Meta;

const TILE: u32 = 16u;
var<workgroup> tile_A: array<f32, 256>;
var<workgroup> tile_B: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row = global_id.y;
    let col = global_id.x;
    let l_row = local_id.y;
    let l_col = local_id.x;

    var sum: f32 = 0.0;
    let num_tiles = (config.K + TILE - 1u) / TILE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let a_col = t * TILE + l_col;

        if (row < config.M && a_col < config.K) {
            tile_A[l_row * TILE + l_col] = A[row * config.K + a_col];
        } else { tile_A[l_row * TILE + l_col] = 0.0; }

        let b_col = t * TILE + l_row;
        if (col < config.N && b_col < config.K) {
            tile_B[l_row * TILE + l_col] = B[col * config.K + b_col];
        } else { tile_B[l_row * TILE + l_col] = 0.0; }

        workgroupBarrier();

        for (var k: u32 = 0u; k < TILE; k = k + 1u) {
            sum = sum + tile_A[l_row * TILE + k] * tile_B[k * TILE + l_col];
        }
        workgroupBarrier();
    }

    if (row < config.M && col < config.N) {
        C[row * config.N + col] = sum;
    }
}
