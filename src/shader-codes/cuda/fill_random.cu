// x[i] = hash(i ^ seed) / 0xFFFFFFFF   (i < n)
// Must match wgsl/fill_random.wgsl's `hash` exactly, so every format's output
// is comparable. Not a cryptographic or statistically rigorous PRNG.
__device__ unsigned int hash(unsigned int v) {
    unsigned int h = v;
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    h *= 0x846ca68bu;
    h ^= h >> 16;
    return h;
}

extern "C" __global__ void entry(const unsigned int* shader_meta, float* x) {
    unsigned int n = shader_meta[0];
    unsigned int seed = shader_meta[1];
    unsigned int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) { x[idx] = (float)hash(idx ^ seed) / 4294967295.0f; }
}
