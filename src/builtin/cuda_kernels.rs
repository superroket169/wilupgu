#![allow(dead_code)]

pub const ADD: &str = r#"
extern "C" __global__ void add_kernel(float* x, const float* residual, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { x[idx] = x[idx] + residual[idx]; }
}
"#;

pub const CAUSAL_MASK: &str = r#"
extern "C" __global__ void causal_mask_kernel(float* scores, unsigned int seq_len, float scale) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= seq_len || col >= seq_len) return;
    unsigned int idx = row * seq_len + col;
    if (col > row) {
        scores[idx] = -1000000000.0f;
    } else {
        scores[idx] = scores[idx] * scale;
    }
}
"#;

pub const BWD_ADD_INPLACE: &str = r#"
extern "C" __global__ void bwd_add_inplace_kernel(float* t, const float* source, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { t[idx] = t[idx] + source[idx]; }
}
"#;

pub const ZERO_TENSOR: &str = r#"
extern "C" __global__ void zero_tensor_kernel(float* x, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { x[idx] = 0.0f; }
}
"#;

pub const ADAMW: &str = r#"
extern "C" __global__ void adamw_kernel(
    float* weights, const float* grads, float* m, float* v,
    unsigned int size,
    unsigned int step, float lr, float beta1, float beta2, float eps, float weight_decay
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float g = grads[idx];
    float clamped_grad = fminf(fmaxf(g, -1.0f), 1.0f);

    float m_new = beta1 * m[idx] + (1.0f - beta1) * clamped_grad;
    float v_new = beta2 * v[idx] + (1.0f - beta2) * clamped_grad * clamped_grad;
    m[idx] = m_new;
    v[idx] = v_new;

    float bias_correction1 = 1.0f - powf(beta1, (float)step);
    float bias_correction2 = 1.0f - powf(beta2, (float)step);
    float m_hat = m_new / bias_correction1;
    float v_hat = v_new / bias_correction2;

    float theta = weights[idx];
    weights[idx] = theta - lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * theta);
}
"#;
