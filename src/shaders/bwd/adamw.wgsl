struct Meta {
    size: u32,
}

struct StepConfig {
    step: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
}

@group(0) @binding(0) var<storage, read_write> weights: array<f32>;
@group(0) @binding(1) var<storage, read> grads: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<storage, read> param_meta: Meta;
@group(0) @binding(5) var<storage, read> cfg: StepConfig;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= param_meta.size) {
        return;
    }

    let g = grads[idx];

    let m_new = cfg.beta1 * m[idx] + (1.0 - cfg.beta1) * g;
    let v_new = cfg.beta2 * v[idx] + (1.0 - cfg.beta2) * g * g;
    m[idx] = m_new;
    v[idx] = v_new;

    let bias_correction1 = 1.0 - pow(cfg.beta1, f32(cfg.step));
    let bias_correction2 = 1.0 - pow(cfg.beta2, f32(cfg.step));
    let m_hat = m_new / bias_correction1;
    let v_hat = v_new / bias_correction2;

    let theta = weights[idx];
    weights[idx] = theta - cfg.lr * (m_hat / (sqrt(v_hat) + cfg.eps) + cfg.weight_decay * theta);
}
