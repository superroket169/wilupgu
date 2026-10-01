extern "C" __global__ void bwd_add_inplace_kernel(float* t, const float* source, unsigned int n) {
    unsigned int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) { t[idx] = t[idx] + source[idx]; }
}
