#include "../../include/cuda/kernel_fma.cuh"

__global__ void cuda_fma_shared(float *C, const float *A, const float *B,
                                const int M, const int N, const int K,
                                const float alpha, const float beta)
{
    int i, j, k, K_;
    int tile_, tile_i, tile_j;
    float sum = 0.0;

    __shared__ float A_shared[TILE_SIZE][TILE_SIZE];
    __shared__ float B_shared[TILE_SIZE][TILE_SIZE];

    // Calculamos los índices i, j de la matriz D
    i = blockIdx.y * blockDim.y + threadIdx.y;
    j = blockIdx.x * blockDim.x + threadIdx.x;

    for (tile_ = 0; tile_ < (K - 1) / TILE_SIZE + 1; tile_++)
    {
        // Load de la submatriz A_shared
        tile_j = tile_ * TILE_SIZE + threadIdx.x;
        if (i < M && tile_j < K)
        {
            A_shared[threadIdx.y][threadIdx.x] = A[i * K + tile_j];
        }
        else
        {
            A_shared[threadIdx.y][threadIdx.x] = 0.0;
        }

        // Load de la submatriz B_shared
        tile_i = tile_ * TILE_SIZE + threadIdx.y;
        if (tile_i < K && j < N)
        {
            B_shared[threadIdx.y][threadIdx.x] = B[tile_i * N + j];
        }
        else
        {
            B_shared[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();

        // Ajuste en el bucle de multiplicación para manejar el caso de baldosas parciales
        K_ = (tile_ == (K - 1) / TILE_SIZE) ? K % TILE_SIZE : TILE_SIZE;
        for (k = 0; k < K_; k++)
        {
            sum += A_shared[threadIdx.y][k] * B_shared[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Escritura en mem. global de device (una sola vez)
    if (i < M && j < N)
    {
        C[i * N + j] = alpha * sum + beta * C[i * N + j];
    }
}
