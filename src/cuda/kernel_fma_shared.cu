#include "../../include/cuda/kernel_fma.cuh"

__global__ void cuda_fma_shared(float *A, float *B, float *C, float *D,
                                   int N, int M, int P)
{
    int i, j, k, K;
    int tile_, tile_i, tile_j;
    float sum = 0.0;

    // Inicializamos los tiles de A_ y B_
    __shared__ float A_shared[TILE_SIZE][TILE_SIZE];
    __shared__ float B_shared[TILE_SIZE][TILE_SIZE];

    // Calculamos los índices i, j de la matriz D
    i = blockIdx.y * blockDim.y + threadIdx.y;
    j = blockIdx.x * blockDim.x + threadIdx.x;

    for (tile_ = 0; tile_ < (M - 1) / TILE_SIZE + 1; tile_++)
    {
        // Load de la submatriz A_shared
        tile_j = tile_ * TILE_SIZE + threadIdx.x;
        if (i < N && tile_j < M)
        {
            A_shared[threadIdx.y][threadIdx.x] = A[i * M + tile_j];
        }
        else
        {
            A_shared[threadIdx.y][threadIdx.x] = 0.0;
        }

        // Load de la submatriz B_shared
        tile_i = tile_ * TILE_SIZE + threadIdx.y;
        if (tile_i < M && j < P)
        {
            B_shared[threadIdx.y][threadIdx.x] = B[tile_i * P + j];
        }
        else
        {
            B_shared[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();

        // Ajuste en el bucle de multiplicación para manejar el caso de baldosas parciales
        K = (tile_ == (M - 1) / TILE_SIZE) ? M % TILE_SIZE : TILE_SIZE;
        for (k = 0; k < K; k++)
        {
            sum += A_shared[threadIdx.y][k] * B_shared[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Escritura en mem. global de device (una sola vez)
    if (i < N && j < P)
    {
        D[i * P + j] = sum + C[i * P + j];
    }
}
