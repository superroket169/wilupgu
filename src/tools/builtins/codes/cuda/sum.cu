// partial[w] = sum of x[i] over workgroup w's slice   (i < n; native runs one workgroup)
// One partial sum per block. If more than one block ran, call again with
// `partial` as the new `x` (and its length as the new `n`) until one value is left.
extern "C" __global__ void entry(const unsigned int* shader_meta, const float* x, float* partial) {
    __shared__ float scratch[256];
    unsigned int n = shader_meta[0];
    unsigned int wg = blockIdx.y * gridDim.x + blockIdx.x;
    unsigned int idx = wg * blockDim.x + threadIdx.x;
    scratch[threadIdx.x] = idx < n ? x[idx] : 0.0f;
    __syncthreads();

    for (unsigned int stride = 128u; stride > 0u; stride /= 2u) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0u) {
        partial[wg] = scratch[0];
    }
}
