// x[i] = 0   (i < n)
extern "C" __global__ void entry(const unsigned int* shader_meta, float* x) {
    unsigned int n = shader_meta[0];
    unsigned int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) { x[idx] = 0.0f; }
}
