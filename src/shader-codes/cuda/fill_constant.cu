// x[i] = value   (i < n)
extern "C" __global__ void entry(const unsigned int* shader_meta, float* x) {
    unsigned int n = shader_meta[0];
    float value = __uint_as_float(shader_meta[1]);
    unsigned int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) { x[idx] = value; }
}
