#include "../../include/cuda/kernel_fma.cuh"

__global__ void cuda_fma_shared(float *C, float *A, float *B,
                                const int M, const int N, const int K,
                                const float alpha, const float beta)
{
    int i, j, k;
    float sum = 0.0;

    __shared__ float A_shared[TILE_SIZE][TILE_SIZE];
    __shared__ float B_shared[TILE_SIZE][TILE_SIZE];

    // Calculamos los índices i, j de la matriz C
    i = blockIdx.y * blockDim.y + threadIdx.y;
    j = blockIdx.x * blockDim.x + threadIdx.x;

    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++)
    {
        // Índices dentro de la matriz completa
        int tile_i = tile * TILE_SIZE + threadIdx.y;
        int tile_j = tile * TILE_SIZE + threadIdx.x;

        // Carga condicional de A_shared y B_shared
        A_shared[threadIdx.y][threadIdx.x] = (i < M && tile_j < K) ? A[i * K + tile_j] : 0.0;
        B_shared[threadIdx.y][threadIdx.x] = (tile_i < K && j < N) ? B[tile_i * N + j] : 0.0;

        __syncthreads();

        // Cálculo de la suma parcial dentro de la baldosa (tile)
        // Ajuste en el bucle de multiplicación para manejar el caso de baldosas parciales
        int K_limit = (tile == (K + TILE_SIZE - 1) / TILE_SIZE - 1) ? K % TILE_SIZE : TILE_SIZE;
        for (k = 0; k < K_limit; k++)
        {
            sum += A_shared[threadIdx.y][k] * B_shared[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Escritura en memoria global del dispositivo (una sola vez)
    if (i < M && j < N)
    {
        C[i * N + j] = alpha * sum + beta * C[i * N + j];
    }
}
