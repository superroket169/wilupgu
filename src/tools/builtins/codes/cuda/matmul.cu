// C[m * N + n] = sum_k A[m * K + k] * B[k * N + n]   (A: MxK, B: KxN, C: MxN)
#define TILE 16u

extern "C" __global__ void entry(const unsigned int* shader_meta, const float* A, const float* B, float* C) {
    __shared__ float tile_A[TILE * TILE];
    __shared__ float tile_B[TILE * TILE];
    unsigned int M = shader_meta[0];
    unsigned int N = shader_meta[1];
    unsigned int K = shader_meta[2];

    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int local_row = threadIdx.y;
    unsigned int local_col = threadIdx.x;

    float sum = 0.0f;
    unsigned int num_tiles = (K + TILE - 1u) / TILE;

    for (unsigned int t = 0u; t < num_tiles; t++) {
        unsigned int a_col = t * TILE + local_col;
        tile_A[local_row * TILE + local_col] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        unsigned int b_k = t * TILE + local_row;
        tile_B[local_row * TILE + local_col] =
            (b_k < K && col < N) ? B[b_k * N + col] : 0.0f;

        __syncthreads();

        for (unsigned int k = 0u; k < TILE; k++) {
            sum += tile_A[local_row * TILE + k] * tile_B[k * TILE + local_col];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
