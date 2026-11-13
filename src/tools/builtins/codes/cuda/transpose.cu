// dst[c * rows + r] = src[r * cols + c]   (r < rows, c < cols)
extern "C" __global__ void entry(const unsigned int* shader_meta, const float* src, float* dst) {
    unsigned int rows = shader_meta[0];
    unsigned int cols = shader_meta[1];
    unsigned long long row = (unsigned long long)blockIdx.y * blockDim.y + threadIdx.y;
    unsigned long long col = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || col >= cols) {
        return;
    }
    dst[col * rows + row] = src[row * cols + col];
}
