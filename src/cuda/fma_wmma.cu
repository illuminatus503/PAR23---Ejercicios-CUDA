#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "../../include/cuda/fma.cuh"
#include "../../include/cuda/kernel_fma.cuh"

#include "../../include/utils.h"
#include "../../include/cuda/error.cuh"

double fma_wmma_gpu(float *A_, float *B_, float *C_, float *D,
                    int M, int N, int K)
{
    cudaEvent_t start, stop;
    float exe_time_ms = 0.0;

    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    half *d_A, *d_B;
    float *d_C;

    // Calculamos el espacio de las matrices padded en el dispositivo
    const int M_padded = (M + WMMA_M - 1) / WMMA_M * WMMA_M;
    const int K_padded = (K + WMMA_K - 1) / WMMA_K * WMMA_K;
    const int N_padded = (N + WMMA_N - 1) / WMMA_N * WMMA_N;

    // Generamos nuevas matrices padded, que se pasarán al kernel
    half *A_padded = (half *)calloc(M_padded * N_padded, sizeof(half));
    half *B_padded = (half *)calloc(N_padded * K_padded, sizeof(half));
    float *C_padded = (float *)calloc(M_padded * K_padded, sizeof(float));

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
    gpuErrchk(cudaMalloc((void **)&d_A, M_padded * N_padded * sizeof(half)));
    gpuErrchk(cudaMalloc((void **)&d_B, N_padded * K_padded * sizeof(half)));
    gpuErrchk(cudaMalloc((void **)&d_C, M_padded * K_padded * sizeof(float)));

    // Cambiar las copias de memoria y la llamada al kernel para usar las matrices padded
    gpuErrchk(cudaMemcpy((void *)d_A, (const void *)A_padded, M_padded * N_padded * sizeof(half), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy((void *)d_B, (const void *)B_padded, N_padded * K_padded * sizeof(half), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy((void *)d_C, (const void *)C_padded, M_padded * K_padded * sizeof(float), cudaMemcpyHostToDevice));

    // Set the CUDA layout
    dim3 blockDim(4 * WARP_SIZE, 4);
    dim3 gridDim((M_padded + (WMMA_M * blockDim.x / WARP_SIZE - 1)) / (WMMA_M * blockDim.x / WARP_SIZE),
                 (K_padded + WMMA_K * blockDim.y - 1) / (WMMA_K * blockDim.y));

    // Launch kernel
    gpuErrchk(cudaEventRecord(start));
    cuda_fma_wmma<<<gridDim, blockDim>>>(d_C, d_B, d_A, M_padded, K_padded, N_padded, 1.0, 1.0);
    cudaCheckError();
    gpuErrchk(cudaEventRecord(stop));

    // Copy data from device array to host array
    gpuErrchk(cudaMemcpy((void *)C_padded, (const void *)d_C, M_padded * K_padded * sizeof(float), cudaMemcpyDeviceToHost));
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
