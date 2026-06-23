struct Meta {
    M: u32,
    N: u32,
    K: u32,
}

@group(0) @binding(0) var<storage, read> A: array<f32>;  // [M, K]
@group(0) @binding(1) var<storage, read> dC: array<f32>; // [M, N]
@group(0) @binding(2) var<storage, read_write> dB: array<f32>; // [K, N]
@group(0) @binding(3) var<storage, read> config: Meta;

const TILE: u32 = 16u;
var<workgroup> tile_A_trp: array<f32, 256>;
var<workgroup> tile_dC: array<f32, 256>;

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
    let num_tiles = (config.M + TILE - 1u) / TILE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let m_idx_a = t * TILE + l_col;

        if (m_idx_a < config.M && row < config.K) {
            tile_A_trp[l_row * TILE + l_col] = A[m_idx_a * config.K + row];
        } else { tile_A_trp[l_row * TILE + l_col] = 0.0; }

        let m_idx_dc = t * TILE + l_row;

        if (m_idx_dc < config.M && col < config.N) {
            tile_dC[l_row * TILE + l_col] = dC[m_idx_dc * config.N + col];
        } else { tile_dC[l_row * TILE + l_col] = 0.0; }

        workgroupBarrier();

        for (var k: u32 = 0u; k < TILE; k = k + 1u) {
            sum = sum + tile_A_trp[l_row * TILE + k] * tile_dC[k * TILE + l_col];
        }

        workgroupBarrier();
    }

    if (row < config.K && col < config.N) {
        let idx = row * config.N + col;
        dB[idx] = dB[idx] + sum;
    }
}
