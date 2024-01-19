#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "../../include/cuda/fma.cuh"
#include "../../include/cuda/kernel_fma.cuh"

#include "../../include/utils.h"
#include "../../include/cuda/error.cuh"
#include "../../include/cuda/utils.cuh"

#define WARP_SIZE 32

double __fma_wmma_gpu(float *A_, float *B_, float *C_, float *D,
                      int M, int N, int K)
{
    size_t gpu_id;
    infoGPU_t device_prop;
    cudaEvent_t start, stop;
    float exe_time_ms = 0.0;

    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    // Matrices en el dispositivo.
    half *d_A, *d_B;
    float *d_C;

    // Calculamos el espacio de las matrices padded en el dispositivo
    // Calculamos el tamaño de las matrices padded
    const int M_padded = (M + WMMA_M - 1) / WMMA_M * WMMA_M;
    const int K_padded = (K + WMMA_K - 1) / WMMA_K * WMMA_K;
    const int N_padded = (N + WMMA_N - 1) / WMMA_N * WMMA_N;
    const size_t size_A_padded = M_padded * N_padded * sizeof(half);  // Matriz A (half)
    const size_t size_B_padded = N_padded * K_padded * sizeof(half);  // Matriz B (half)
    const size_t size_C_padded = M_padded * K_padded * sizeof(float); // Matrices C y D (float)

    // Generamos nuevas matrices padded, que se pasarán al kernel
    half *A_padded = (half *)malloc(size_A_padded);
    memset(A_padded, 0, size_A_padded);

    half *B_padded = (half *)malloc(size_B_padded);
    memset(B_padded, 0, size_B_padded);

    float *C_padded = (float *)malloc(size_C_padded);
    memset(C_padded, 0, size_C_padded);

    // Copiar los datos de A_, B_, C_ a A_padded, B_padded, C_padded, respectivamente
    // Asegurándose de hacer el cast de A_ y B_ a half
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            A_padded[i * N_padded + j] = __float2half(A_[i * N + j]);
        }
    }

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < K; ++j)
        {
            B_padded[i * K_padded + j] = __float2half(B_[i * K + j]);
        }
    }

    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < K; ++j)
        {
            C_padded[i * K_padded + j] = C_[i * K + j];
        }
    }

    // Reservamos memoria para las matrices del dispositivo
    gpuErrchk(cudaMalloc((void **)&d_A, size_A_padded));
    gpuErrchk(cudaMalloc((void **)&d_B, size_B_padded));
    gpuErrchk(cudaMalloc((void **)&d_C, size_C_padded));

    // Cambiar las copias de memoria y la llamada al kernel para usar las matrices padded
    gpuErrchk(cudaMemcpy((void *)d_A, (const void *)A_padded, size_A_padded, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy((void *)d_B, (const void *)B_padded, size_B_padded, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy((void *)d_C, (const void *)C_padded, size_C_padded, cudaMemcpyHostToDevice));

    // Set the CUDA layout
    dim3 blockDim;
    dim3 gridDim;

    // blockDim.x must be a multple of warpSize
    // 128x4 means we have 16 warps and a block computes a 64x64 output tile
    blockDim.x = 128;
    blockDim.y = 4;

    gridDim.x = (M_padded + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
    gridDim.y = (K_padded + WMMA_K * blockDim.y - 1) / (WMMA_K * blockDim.y);

    // Launch kernel
    printf("Running with wmma...\n");
    gpuErrchk(cudaEventRecord(start));
    cuda_fma_wmma<<<gridDim, blockDim>>>(d_A, d_B, d_C, M_padded, K_padded, N_padded, 1.0, 1.0);
    cudaCheckError();
    gpuErrchk(cudaEventRecord(stop));

    // Copy data from device array to host array
    printf("[+] Recuperando datos del dispositivo...\n");
    gpuErrchk(cudaMemcpy((void *)C_padded, (const void *)d_C, size_C_padded, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaEventSynchronize(stop));
    gpuErrchk(cudaEventElapsedTime(&exe_time_ms, start, stop));

    // Recuperamos los datos a la matriz D original (sin padding)
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < K; ++j)
        {
            D[i * K + j] = C_padded[i * K_padded + j];
        }
    }

    // Liberamos los recursos del dispositivo
    gpuErrchk(cudaFree(d_A));
    gpuErrchk(cudaFree(d_B));
    gpuErrchk(cudaFree(d_C));

    // Liberar la memoria de las matrices padded en el host
    free(A_padded);
    free(B_padded);
    free(C_padded);

    return (double)exe_time_ms;
}

double fma_wmma_gpu(float *A_, int N1, int M1,
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

    return __fma_wmma_gpu(A_, B_, C_, D, N, M1, M);
}
