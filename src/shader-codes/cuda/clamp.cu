// x[i] = clamp(x[i], lo, hi)   (i < n)
extern "C" __global__ void entry(const unsigned int* shader_meta, float* x) {
    unsigned int n = shader_meta[0];
    float lo = __uint_as_float(shader_meta[1]);
    float hi = __uint_as_float(shader_meta[2]);
    unsigned int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) { x[idx] = fminf(fmaxf(x[idx], lo), hi); }
}
