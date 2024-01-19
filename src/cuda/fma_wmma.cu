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

double fma_gpu_wmma(float *D, const float *A, const float *B, const float *C,
                    const int M, const int N, const int K)
{
    cudaEvent_t start, stop;
    float exe_time_ms = 0.0;

    size_t free_mem, total_mem;
    dim3 gridDim, blockDim;

    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    // ! CUDA set device and check memory availability
    cudaSetDevice(0);
    cudaMemGetInfo(&free_mem, &total_mem);
    if (free_mem < ((M * K + K * N + M * N) * sizeof(float)))
    {
        fprintf(stderr, "[ERROR] Not enough memory available!\n");
        exit(1);
    }

    // ! CUDA global memory allocation
    // Declaramos las var. de memoria global
    float *d_f32_A, *d_f32_B, *d_C;
    half *d_f16_A, *d_f16_B;

    float *A_padded, *B_padded, *C_padded;
    int M_padded, N_padded, K_padded;

    // Padding, si es necesario.
    wmma_pad((float *)A, (float *)B, (float *)C, M, N, K,
             WMMA_M, WMMA_N, WMMA_K,
             &A_padded, &B_padded, &C_padded,
             &M_padded, &N_padded, &K_padded);

    // Reservamos mem. global
    gpuErrchk(cudaMalloc((void **)&d_f32_A, M_padded * K_padded * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&d_f32_B, K_padded * N_padded * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&d_f16_A, M_padded * K_padded * sizeof(half)));
    gpuErrchk(cudaMalloc((void **)&d_f16_B, K_padded * N_padded * sizeof(half)));
    gpuErrchk(cudaMalloc((void **)&d_C, M_padded * N_padded * sizeof(float)));

    // Copiamos las matrices A, B y C a GPU & convertimos a f16 A y B
    gpuErrchk(cudaMemcpy((void *)d_f32_A, (const void *)A_padded, M_padded * K_padded * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy((void *)d_f32_B, (const void *)B_padded, K_padded * N_padded * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy((void *)d_C, (const void *)C_padded, M_padded * N_padded * sizeof(float), cudaMemcpyHostToDevice));

    convertFp32ToFp16<<<(M_padded * K_padded + 255) / 256, 256>>>(d_f16_A, d_f32_A, M_padded * K_padded);
    cudaCheckError();
    convertFp32ToFp16<<<(K_padded * N_padded + 255) / 256, 256>>>(d_f16_B, d_f32_B, K_padded * N_padded);
    cudaCheckError();

    gpuErrchk(cudaFree(d_f32_A));
    gpuErrchk(cudaFree(d_f32_B));

    // ! CUDA layout design
    // Block dimension (in threads per dim)
    blockDim.x = 128;
    blockDim.y = 4;

    // Grid dimension (in blocks per dim)
    gridDim.x = (M_padded + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
    gridDim.y = (K_padded + WMMA_K * blockDim.y - 1) / (WMMA_K * blockDim.y);

    // ! CUDA launch kernel
    gpuErrchk(cudaEventRecord(start));
    cuda_gemm_wmma<<<gridDim, blockDim>>>(d_C, d_f16_A, d_f16_B, M_padded, N_padded, K_padded, 1.0, 1.0);
    cudaCheckError();

    // ! CUDA copy data to local mem.
    gpuErrchk(cudaMemcpy((void *)C_padded, (const void *)d_C, M_padded * K_padded * sizeof(float), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaEventRecord(stop));
    gpuErrchk(cudaEventSynchronize(stop));
    gpuErrchk(cudaEventElapsedTime(&exe_time_ms, start, stop));

    // Liberamos los recursos CUDA
    gpuErrchk(cudaFree(d_f16_A));
    gpuErrchk(cudaFree(d_f16_B));
    gpuErrchk(cudaFree(d_C));

    // Recuperamos los datos a la matriz resultado
    wmma_unpad(C_padded, M_padded, N_padded, D, M, N);

    // Liberamos los recursos locales, si fueron necesarios
    if (A_padded != A)
    {
        free(A_padded);
        free(B_padded);
        free(C_padded);
    }

    return (double)exe_time_ms;
}
