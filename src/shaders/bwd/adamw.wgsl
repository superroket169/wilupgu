struct Meta {
    size: u32,
    groups_x: u32,
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

// Vulkan caps each dispatch dimension at 65535 workgroups; large tensors
// (e.g. the 50257x768 embedding/lm_head, ~150772 workgroups of 256 threads)
// exceed that in a 1D dispatch, which a real driver doesn't reliably reject
// cleanly (manifests as "Parent device is lost" instead of a validation
// error). Dispatch a 2D grid instead and reconstruct the flat index here.
@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let idx = (wg_id.y * param_meta.groups_x + wg_id.x) * 256u + local_id.x;
    if (idx >= param_meta.size) {
        return;
    }

    let g = grads[idx];
    let clamped_grad = clamp(g, -1.0, 1.0);

    let m_new = cfg.beta1 * m[idx] + (1.0 - cfg.beta1) * clamped_grad;
    let v_new = cfg.beta2 * v[idx] + (1.0 - cfg.beta2) * clamped_grad * clamped_grad;
    m[idx] = m_new;
    v[idx] = v_new;

    let bias_correction1 = 1.0 - pow(cfg.beta1, f32(cfg.step));
    let bias_correction2 = 1.0 - pow(cfg.beta2, f32(cfg.step));
    let m_hat = m_new / bias_correction1;
    let v_hat = v_new / bias_correction2;

    let theta = weights[idx];
    weights[idx] = theta - cfg.lr * (m_hat / (sqrt(v_hat) + cfg.eps) + cfg.weight_decay * theta);
}
