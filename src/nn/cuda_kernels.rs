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

pub const SILU: &str = r#"
extern "C" __global__ void silu_kernel(float* x, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        x[idx] = val / (1.0f + expf(-val));
    }
}
"#;

pub const SILU_BWD: &str = r#"
extern "C" __global__ void silu_bwd_kernel(const float* x, const float* dY, float* dX, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        float sig = 1.0f / (1.0f + expf(-val));
        float grad_silu = sig + val * sig * (1.0f - sig);
        dX[idx] = dY[idx] * grad_silu;
    }
}
"#;

pub const BWD_ADD_INPLACE: &str = r#"
extern "C" __global__ void bwd_add_inplace_kernel(float* t, const float* source, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { t[idx] = t[idx] + source[idx]; }
}
"#;

pub const EMBEDDING: &str = r#"
extern "C" __global__ void embedding_kernel(
    const unsigned int* tokens, const float* weight, float* output,
    unsigned int vocab_size, unsigned int embed_dim, unsigned int seq_len
) {
    unsigned int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int token_idx = blockIdx.y;
    if (token_idx >= seq_len || dim_idx >= embed_dim) return;

    unsigned int token_id = tokens[token_idx];
    if (token_id < vocab_size) {
        unsigned int weight_idx = token_id * embed_dim + dim_idx;
        unsigned int out_idx = token_idx * embed_dim + dim_idx;
        output[out_idx] = weight[weight_idx];
    }
}
"#;

pub const EMBEDDING_BWD: &str = r#"
extern "C" __global__ void embedding_bwd_kernel(
    const unsigned int* tokens, const float* grad_output, float* grad_table,
    unsigned int vocab_size, unsigned int embed_dim, unsigned int seq_len
) {
    unsigned int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int token_idx = blockIdx.y;
    if (token_idx >= seq_len || dim_idx >= embed_dim) return;

    unsigned int token_id = tokens[token_idx];
    if (token_id >= vocab_size) return;

    unsigned int target_idx = token_id * embed_dim + dim_idx;
    float grad_val = grad_output[token_idx * embed_dim + dim_idx];
    atomicAdd(&grad_table[target_idx], grad_val);
}
"#;

pub const ROPE: &str = r#"
extern "C" __global__ void rope_kernel(float* vec, unsigned int seq_len, unsigned int dim, unsigned int head_dim) {
    unsigned int dim_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2u;
    unsigned int token_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (token_idx >= seq_len || dim_idx >= head_dim) return;

    unsigned int num_heads = dim / head_dim;
    for (unsigned int h = 0; h < num_heads; h++) {
        unsigned int offset = token_idx * dim + h * head_dim + dim_idx;
        float x0 = vec[offset];
        float x1 = vec[offset + 1u];

        float freq = 1.0f / powf(10000.0f, (float)dim_idx / (float)head_dim);
        float v_angle = (float)token_idx * freq;
        float v_cos = cosf(v_angle);
        float v_sin = sinf(v_angle);

        vec[offset]      = x0 * v_cos - x1 * v_sin;
        vec[offset + 1u] = x0 * v_sin + x1 * v_cos;
    }
}
"#;

pub const ROPE_BWD: &str = r#"
extern "C" __global__ void rope_bwd_kernel(float* d_vec, unsigned int seq_len, unsigned int dim, unsigned int head_dim) {
    unsigned int dim_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2u;
    unsigned int token_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (token_idx >= seq_len || dim_idx >= head_dim) return;

    unsigned int num_heads = dim / head_dim;
    for (unsigned int h = 0; h < num_heads; h++) {
        unsigned int offset = token_idx * dim + h * head_dim + dim_idx;
        float dx0 = d_vec[offset];
        float dx1 = d_vec[offset + 1u];

        float freq = 1.0f / powf(10000.0f, (float)dim_idx / (float)head_dim);
        float v_angle = (float)token_idx * freq;
        float v_cos = cosf(v_angle);
        float v_sin = sinf(v_angle);

        d_vec[offset]      = dx0 * v_cos + dx1 * v_sin;
        d_vec[offset + 1u] = -dx0 * v_sin + dx1 * v_cos;
    }
}
"#;

pub const HEAD_GATHER: &str = r#"
extern "C" __global__ void head_gather_kernel(
    const float* src, float* dst,
    unsigned int seq_len, unsigned int full_dim, unsigned int head_dim, unsigned int head_offset
) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= seq_len || col >= head_dim) return;
    unsigned int src_idx = row * full_dim + head_offset + col;
    unsigned int dst_idx = row * head_dim + col;
    dst[dst_idx] = src[src_idx];
}
"#;

pub const HEAD_SCATTER: &str = r#"
extern "C" __global__ void head_scatter_kernel(
    const float* src, float* dst,
    unsigned int seq_len, unsigned int full_dim, unsigned int head_dim, unsigned int head_offset
) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= seq_len || col >= head_dim) return;
    unsigned int src_idx = row * head_dim + col;
    unsigned int dst_idx = row * full_dim + head_offset + col;
    dst[dst_idx] = src[src_idx];
}
"#;

pub const ZERO_TENSOR: &str = r#"
extern "C" __global__ void zero_tensor_kernel(float* x, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { x[idx] = 0.0f; }
}
"#;

pub const SOFTMAX: &str = r#"
extern "C" __global__ void softmax_kernel(float* x, unsigned int seq_len) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= seq_len) return;
    unsigned int offset = row * seq_len;

    float max_val = -1000000.0f;
    for (unsigned int i = 0; i < seq_len; i++) {
        float val = x[offset + i];
        if (val > max_val) max_val = val;
    }

    float sum_exp = 0.0f;
    for (unsigned int i = 0; i < seq_len; i++) {
        float e = expf(x[offset + i] - max_val);
        x[offset + i] = e;
        sum_exp += e;
    }

    for (unsigned int i = 0; i < seq_len; i++) {
        x[offset + i] = x[offset + i] / sum_exp;
    }
}
"#;

pub const SOFTMAX_BWD: &str = r#"
extern "C" __global__ void softmax_bwd_kernel(
    const float* Y, const float* dY, float* dX, unsigned int seq_len, float scale
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= seq_len) return;
    unsigned int offset = row * seq_len;

    float sum_ydy = 0.0f;
    for (unsigned int i = 0; i < seq_len; i++) {
        sum_ydy += Y[offset + i] * dY[offset + i];
    }

    for (unsigned int i = 0; i < seq_len; i++) {
        dX[offset + i] = Y[offset + i] * (dY[offset + i] - sum_ydy) * scale;
    }
}
"#;

pub const RMSNORM: &str = r#"
extern "C" __global__ void rmsnorm_kernel(
    const float* x, const float* weight, float* output,
    unsigned int seq_len, unsigned int size, float eps
) {
    __shared__ float partial[256];
    unsigned int row = blockIdx.x;
    if (row >= seq_len) return;
    unsigned int offset = row * size;
    unsigned int tid = threadIdx.x;

    float local_ss = 0.0f;
    for (unsigned int i = tid; i < size; i += 256u) {
        float val = x[offset + i];
        local_ss += val * val;
    }
    partial[tid] = local_ss;
    __syncthreads();

    for (unsigned int stride = 128u; stride > 0u; stride /= 2u) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }

    float rsqrt_v = 1.0f / sqrtf((partial[0] / (float)size) + eps);

    for (unsigned int i = tid; i < size; i += 256u) {
        output[offset + i] = x[offset + i] * rsqrt_v * weight[i];
    }
}
"#;

pub const RMSNORM_BWD: &str = r#"
extern "C" __global__ void rmsnorm_bwd_kernel(
    const float* dY, const float* X, const float* Weight,
    float* dX, float* rsqrt_cache,
    unsigned int seq_len, unsigned int size, float eps
) {
    __shared__ float partial[256];
    unsigned int row = blockIdx.x;
    if (row >= seq_len) return;
    unsigned int offset = row * size;
    unsigned int tid = threadIdx.x;

    float local_ss = 0.0f;
    for (unsigned int i = tid; i < size; i += 256u) {
        local_ss += X[offset + i] * X[offset + i];
    }
    partial[tid] = local_ss;
    __syncthreads();
    for (unsigned int stride = 128u; stride > 0u; stride /= 2u) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    float ss = partial[0];
    __syncthreads();

    float rsqrt_v = 1.0f / sqrtf((ss / (float)size) + eps);
    if (tid == 0u) rsqrt_cache[row] = rsqrt_v;

    float local_sum_grad = 0.0f;
    for (unsigned int i = tid; i < size; i += 256u) {
        float norm_x = X[offset + i] * rsqrt_v;
        float dy_w = dY[offset + i] * Weight[i];
        local_sum_grad += dy_w * norm_x;
    }
    partial[tid] = local_sum_grad;
    __syncthreads();
    for (unsigned int stride = 128u; stride > 0u; stride /= 2u) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    float sum_grad = partial[0];
    __syncthreads();

    for (unsigned int i = tid; i < size; i += 256u) {
        float norm_x = X[offset + i] * rsqrt_v;
        float dy_w = dY[offset + i] * Weight[i];
        dX[offset + i] = rsqrt_v * (dy_w - (norm_x * sum_grad / (float)size));
    }
}
"#;

pub const RMSNORM_WEIGHT_BWD: &str = r#"
extern "C" __global__ void rmsnorm_weight_bwd_kernel(
    const float* dY, const float* X, const float* rsqrt_cache, float* dWeight,
    unsigned int seq_len, unsigned int size
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    float acc = 0.0f;
    for (unsigned int row = 0; row < seq_len; row++) {
        unsigned int offset = row * size;
        float norm_x = X[offset + i] * rsqrt_cache[row];
        acc += dY[offset + i] * norm_x;
    }
    dWeight[i] = dWeight[i] + acc;
}
"#;

pub const CROSS_ENTROPY: &str = r#"
extern "C" __global__ void cross_entropy_kernel(
    const float* logits, const unsigned int* targets, float* probs, float* losses,
    unsigned int vocab_size, unsigned int num_rows
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;
    unsigned int offset = row * vocab_size;
    unsigned int target_id = targets[row];

    float max_val = -3.4028235e38f;
    for (unsigned int i = 0; i < vocab_size; i++) {
        max_val = fmaxf(max_val, logits[offset + i]);
    }

    float sum_exp = 0.0f;
    for (unsigned int i = 0; i < vocab_size; i++) {
        float e = expf(logits[offset + i] - max_val);
        probs[offset + i] = e;
        sum_exp += e;
    }

    float inv_sum = 1.0f / sum_exp;
    for (unsigned int i = 0; i < vocab_size; i++) {
        probs[offset + i] = probs[offset + i] * inv_sum;
    }

    float log_sum_exp = logf(sum_exp);
    losses[row] = -(logits[offset + target_id] - max_val - log_sum_exp);
}
"#;

pub const CROSS_ENTROPY_BWD: &str = r#"
extern "C" __global__ void cross_entropy_bwd_kernel(
    const float* probs, const unsigned int* targets, const float* d_losses, float* d_logits,
    unsigned int vocab_size, unsigned int num_rows
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;
    unsigned int offset = row * vocab_size;
    unsigned int target_id = targets[row];
    float grad_scale = d_losses[row];

    for (unsigned int i = 0; i < vocab_size; i++) {
        float indicator = (i == target_id) ? 1.0f : 0.0f;
        d_logits[offset + i] = (probs[offset + i] - indicator) * grad_scale;
    }
}
"#;

pub const SOFTMAX_RECT: &str = r#"
extern "C" __global__ void softmax_rect_kernel(float* x, unsigned int num_rows, unsigned int width, float scale) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;
    unsigned int offset = row * width;

    float max_val = -1000000.0f;
    for (unsigned int i = 0; i < width; i++) {
        float val = x[offset + i] * scale;
        if (val > max_val) max_val = val;
    }

    float sum_exp = 0.0f;
    for (unsigned int i = 0; i < width; i++) {
        float e = expf(x[offset + i] * scale - max_val);
        x[offset + i] = e;
        sum_exp += e;
    }

    for (unsigned int i = 0; i < width; i++) {
        x[offset + i] = x[offset + i] / sum_exp;
    }
}
"#;

pub const CAUSAL_SOFTMAX: &str = r#"
extern "C" __global__ void causal_softmax_kernel(float* x, unsigned int seq_len, float scale) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= seq_len) return;
    unsigned int offset = row * seq_len;

    float max_val = -1000000.0f;
    for (unsigned int i = 0; i < seq_len; i++) {
        float val = (i > row) ? -1000000000.0f : x[offset + i] * scale;
        if (val > max_val) max_val = val;
    }

    float sum_exp = 0.0f;
    for (unsigned int i = 0; i < seq_len; i++) {
        float val = (i > row) ? -1000000000.0f : x[offset + i] * scale;
        float e = expf(val - max_val);
        x[offset + i] = e;
        sum_exp += e;
    }

    for (unsigned int i = 0; i < seq_len; i++) {
        x[offset + i] = x[offset + i] / sum_exp;
    }
}
"#;

pub const CACHE_WRITE: &str = r#"
extern "C" __global__ void cache_write_kernel(
    const float* src, float* dst,
    unsigned int row_count, unsigned int width, unsigned int dst_row_offset
) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= row_count || col >= width) return;
    unsigned int src_idx = row * width + col;
    unsigned int dst_idx = (dst_row_offset + row) * width + col;
    dst[dst_idx] = src[src_idx];
}
"#;

pub const ROPE_OFFSET: &str = r#"
extern "C" __global__ void rope_offset_kernel(
    float* vec, unsigned int seq_len, unsigned int dim, unsigned int head_dim, unsigned int pos_offset
) {
    unsigned int dim_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2u;
    unsigned int token_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (token_idx >= seq_len || dim_idx >= head_dim) return;

    unsigned int num_heads = dim / head_dim;
    unsigned int abs_pos = token_idx + pos_offset;
    for (unsigned int h = 0; h < num_heads; h++) {
        unsigned int offset = token_idx * dim + h * head_dim + dim_idx;
        float x0 = vec[offset];
        float x1 = vec[offset + 1u];

        float freq = 1.0f / powf(10000.0f, (float)dim_idx / (float)head_dim);
        float v_angle = (float)abs_pos * freq;
        float v_cos = cosf(v_angle);
        float v_sin = sinf(v_angle);

        vec[offset]      = x0 * v_cos - x1 * v_sin;
        vec[offset + 1u] = x0 * v_sin + x1 * v_cos;
    }
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
