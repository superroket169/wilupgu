// C[n] = sum_k A[k] * B[k * N + n]   (M = 1; A: 1xK, B: KxN, C: 1xN)
// One thread per output column; M is ignored (one row).
extern "C" __global__ void entry(const unsigned int* shader_meta, const float* A, const float* B, float* C) {
    unsigned int N = shader_meta[1];
    unsigned int K = shader_meta[2];
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) {
        return;
    }
    float sum = 0.0f;
    for (unsigned int k = 0u; k < K; k++) {
        sum += A[k] * B[k * N + col];
    }
    C[col] = sum;
}
