#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>

#include "../include/utilities_CPU.h"

#include "../include/cuda/errchk_GPU.cuh"
#include "../include/cuda/matmul_sharedmem_GPU.cuh"

__global__ void cuda_fma_sharedmem(float *A_, float *B_, float *C_, float *D,
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
            A_shared[threadIdx.y][threadIdx.x] = A_[i * M + tile_j];
        }
        else
        {
            A_shared[threadIdx.y][threadIdx.x] = 0.0;
        }

        // Load de la submatriz B_shared
        tile_i = tile_ * TILE_SIZE + threadIdx.y;
        if (tile_i < M && j < P)
        {
            B_shared[threadIdx.y][threadIdx.x] = B_[tile_i * P + j];
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
        D[i * P + j] = sum + C_[i * P + j];
    }
}

double __fma_sharedmem_gpu(float *A_, float *B_, float *C_, float *D,
                           int N, int M, int P)
{
    /**
     * Medición de tiempos
     */
    cudaEvent_t start, stop;
    float exe_time_ms = 0.0;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    /**
     * Variables de mem. device
     */
    const size_t size_A = N * M * sizeof(float);
    const size_t size_B = M * P * sizeof(float);
    const size_t size_C = N * P * sizeof(float);
    float *d_A, *d_B, *d_C, *d_D;

    gpuErrchk(cudaMalloc((void **)&d_A, size_A));
    gpuErrchk(cudaMalloc((void **)&d_B, size_B));
    gpuErrchk(cudaMalloc((void **)&d_C, size_C));
    gpuErrchk(cudaMalloc((void **)&d_D, size_C));

    // Copiamos los datos necesarios para las matrices A, B y C
    gpuErrchk(cudaMemcpy((void *)d_A, (const void *)A_, size_A, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy((void *)d_B, (const void *)B_, size_B, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy((void *)d_C, (const void *)C_, size_C, cudaMemcpyHostToDevice));

    // Set execution configuration parameters
    //      threadsPerBlock: number of CUDA threads per grid block
    //      blocksPerGrid: number of blocks in grid
    dim3 threadsPerBlock(THR_PER_BLOCK, THR_PER_BLOCK);
    dim3 blocksPerGrid((P - 1) / threadsPerBlock.x + 1,
                       (N - 1) / threadsPerBlock.y + 1);

    // Launch kernel
    gpuErrchk(cudaEventRecord(start));
    cuda_fma_sharedmem<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, d_D, N, M, P);
    cudaCheckError(); // Check error after execution
    gpuErrchk(cudaEventRecord(stop));

    // Copy data from device array to host array
    gpuErrchk(cudaMemcpy((void *)D, (const void *)d_D, size_C, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaEventSynchronize(stop));
    gpuErrchk(cudaEventElapsedTime(&exe_time_ms, start, stop));

    /**
     * Free CUDA mem.
     */
    gpuErrchk(cudaFree(d_A));
    gpuErrchk(cudaFree(d_B));
    gpuErrchk(cudaFree(d_C));
    gpuErrchk(cudaFree(d_D));

    return (double)exe_time_ms;
}

double fma_sharedmem_GPU(float *A_, int N1, int M1,
                         float *B_, int N2, int M2,
                         float *C_, int N3, int M3,
                         float *D, int N, int M)
{
    if (!matrix_checkdims(N1, M1, N2, M2, N3, M3, N, M))
    {
        fprintf(stderr,
                "[DimError] La dimensiones de las matrices no coinciden: A(%d x %d) · B(%d x %d) + C(%d x %d) = D(%d x %d)\n",
                N1, M1, N2, M2, N3, M3, N, M);
        return 0.0; // Asum. que el checkeo no añade sobrecostes
    }

    return __fma_sharedmem_gpu(A_, B_, C_, D, N, M1, M);
}
