extern "C" __global__ void zero_tensor_kernel(float* x, const unsigned int* meta) {
    unsigned int n = meta[0];
    unsigned int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) { x[idx] = 0.0f; }
}
