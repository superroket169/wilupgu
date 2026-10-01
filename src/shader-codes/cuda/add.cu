extern "C" __global__ void add_kernel(float* x, const float* residual, unsigned int n) {
    unsigned int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) { x[idx] = x[idx] + residual[idx]; }
}
