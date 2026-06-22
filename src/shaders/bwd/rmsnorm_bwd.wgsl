struct Meta { size: u32, eps: f32, }

@group(0) @binding(0) var<storage, read> dY: array<f32>; 
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> Weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> dX: array<f32>;
@group(0) @binding(4) var<storage, read_write> dWeight: array<f32>;
@group(0) @binding(5) var<storage, read> config: Meta;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let offset = row * config.size;

    var ss: f32 = 0.0;
    for (var i: u32 = 0u; i < config.size; i = i + 1u) {
        ss = ss + (X[offset + i] * X[offset + i]);
    }
    let rsqrt = 1.0 / sqrt((ss / f32(config.size)) + config.eps);

    var sum_grad: f32 = 0.0;
    for (var i: u32 = 0u; i < config.size; i = i + 1u) {
        let norm_x = X[offset + i] * rsqrt;
        let dy_w = dY[offset + i] * Weight[i];
        sum_grad = sum_grad + (dy_w * norm_x);
        
        if (row == 0u) { dWeight[i] = dWeight[i] + (dY[offset + i] * norm_x); }
    }

    for (var i: u32 = 0u; i < config.size; i = i + 1u) {
        let norm_x = X[offset + i] * rsqrt; dy_w = dY[offset + i] * Weight[i];
        dX[offset + i] = rsqrt * (dy_w - (norm_x * sum_grad / f32(config.size)));
    }
}
