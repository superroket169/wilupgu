// partial[w] = sum of a[i] * b[i] over workgroup w's slice   (i < n; native runs one workgroup)
// First pass of a dot product: multiply then reduce, one partial per block.
// `sum.cu` finishes the job if more than one partial came out.
extern "C" __global__ void entry(const unsigned int* shader_meta, const float* a, const float* b, float* partial) {
    __shared__ float scratch[256];
    unsigned int n = shader_meta[0];
    unsigned int wg = blockIdx.y * gridDim.x + blockIdx.x;
    unsigned int idx = wg * blockDim.x + threadIdx.x;
    scratch[threadIdx.x] = idx < n ? a[idx] * b[idx] : 0.0f;
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
