// dst[c * rows + r] = src[r * cols + c]   (r < rows, c < cols)
struct ShaderMeta {
    rows: u32,
    cols: u32,
}

@group(0) @binding(0) var<storage, read> shader_meta: ShaderMeta;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn entry(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;
    if (row >= shader_meta.rows || col >= shader_meta.cols) {
        return;
    }
    dst[col * shader_meta.rows + row] = src[row * shader_meta.cols + col];
}
